"""Agent-facing tools for the iterative script-improvement loop.

Two tools are exposed to ``create_agent``:

- ``eval_scripts(query_source, sort_sources, rationale, parent_iter=None)``
  snapshots the candidate to ``runs/<ts>/iter_NNN/``, runs a single-doc
  compile-check, evaluates against the full cohort on success, and
  appends one :class:`IterationRecord` to ``run.jsonl``. Each tool call
  advances the iteration counter; the JSONL line count maps to tool
  calls (not LangChain agent steps).
- ``read_history(last_n=5, include_sources=True)`` returns the most
  recent records (oldest-first), the baseline metrics, and the best
  iteration so far judged by the run's primary metric. Sources are
  inlined from the snapshot directories when requested so the agent
  can read its prior code without separate file lookups.

The outer iteration budget lives in :class:`ToolContext` and is
enforced inside ``eval_scripts`` — the harness does not poll the agent
loop; the agent simply receives a ``budget_exhausted`` tool response
when it is over the cap and terminates naturally.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from es_script_agent import config
from es_script_agent.data.schema import User
from es_script_agent.es.client import format_es_error
from es_script_agent.es.query import MAX_SORT_SCRIPTS, build_query
from es_script_agent.es.schemas import LOANS_INDEX
from es_script_agent.eval.runner import ScriptSet, evaluate
from es_script_agent.runlog import (
    IterationRecord,
    RunLog,
    snapshot_script_set,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Per-run state bound into the agent's tool closures.

    Attributes:
        es: Connected Elasticsearch client (or any object exposing
            ``search``). The tools never invoke any other ES API.
        run_dir: Per-run directory. Snapshots and the JSONL log live here.
        run_log: ``RunLog`` writer/reader bound to ``run_dir``.
        ground_truth: Filtered ground-truth dict (already restricted to
            indexed item ids).
        users_by_id: Per-user vectors keyed by user id. Used by both the
            compile-check (first user) and the full eval pass.
        indexed_item_ids: Item ids currently in the index. Reserved for future
            tool-level lookups; the runner already operates on
            ``ground_truth``.
        baseline_metrics: Metrics dict written by ``iter_000``. Disclosed
            to the agent through ``read_history`` and used to compute
            the per-record ``guardrail_held`` flag.
        objective: ``"ndcg"`` or ``"ild"``. Selects which metric is
            primary (and therefore which is the guardrail).
        max_iters: Outer budget. The ``iter_counter+1``-th call returns
            ``budget_exhausted`` without snapshotting.
        max_sort_scripts: Sort-script cap enforced inside
            ``eval_scripts`` before snapshotting.
        k: Top-K cutoff for evaluation.
        diversity_fields: Field set used for ILD and for
            ``_source.includes`` on each per-user search.
        index: Elasticsearch index to search.

    Mutable state:
        iter_counter: Next iteration index to assign. Starts at ``1``
            because ``iter_000`` is the baseline.
        last_success_iter: Iteration index of the most recent
            metrics-bearing record, used as the default
            ``parent_iter`` when the agent omits it. Defaults to ``0``
            (the baseline) so the first agent iteration descends from
            it, and is only overwritten by successful agent iterations.
    """

    es: Any
    run_dir: Path
    run_log: RunLog
    ground_truth: dict[str, set[str]]
    users_by_id: dict[str, User]
    indexed_item_ids: set[str]
    baseline_metrics: dict[str, float | None]
    objective: str = config.DEFAULT_OBJECTIVE
    max_iters: int = 20
    max_sort_scripts: int = MAX_SORT_SCRIPTS
    k: int = 10
    diversity_fields: Sequence[str] = field(default_factory=lambda: config.ILD_DIVERSITY_FIELDS)
    index: str = LOANS_INDEX
    iter_counter: int = 1
    last_success_iter: int | None = 0


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _primary_key(objective: str, k: int) -> str:
    return f"{objective}@{k}"


def _guardrail_key(objective: str, k: int) -> str:
    return f"{'ild' if objective == 'ndcg' else 'ndcg'}@{k}"


def _compile_check(ctx: ToolContext, script_set: ScriptSet) -> str | None:
    """Run a ``size=1`` search exercising the full script set.

    Returns ``None`` on success, or a stringified error message on
    failure. The compile-check uses the first user in ``users_by_id``
    (sorted by id for determinism) as a representative vector — the
    same shape Painless will see at full-eval time, so a body that
    fails this check would fail every per-user call.
    """
    if not ctx.users_by_id:
        return "no users available for compile-check"
    first_user_id = next(iter(sorted(ctx.users_by_id)))
    user = ctx.users_by_id[first_user_id]
    body = build_query(
        query_source=script_set.query_source,
        sort_sources=script_set.sort_sources,
        user_vector=user.vector,
        size=1,
    )
    body["_source"] = {"includes": list(ctx.diversity_fields)}
    try:
        ctx.es.search(index=ctx.index, body=body)
    except Exception as exc:
        return format_es_error(exc)
    return None


def _default_parent_iter(ctx: ToolContext) -> int | None:
    return ctx.last_success_iter


def _eval_scripts_impl(
    ctx: ToolContext,
    query_source: str,
    sort_sources: list[str],
    rationale: str,
    parent_iter: int | None,
) -> dict[str, Any]:
    if ctx.iter_counter > ctx.max_iters:
        return {
            "ok": False,
            "budget_exhausted": True,
            "iter": ctx.iter_counter,
            "max_iters": ctx.max_iters,
        }
    if len(sort_sources) > ctx.max_sort_scripts:
        return {
            "ok": False,
            "error": (
                f"too many sort scripts: {len(sort_sources)} > "
                f"max_sort_scripts={ctx.max_sort_scripts}"
            ),
        }

    try:
        script_set = ScriptSet(query_source=query_source, sort_sources=list(sort_sources))
    except Exception as exc:
        return {"ok": False, "error": f"invalid script set: {exc}"}

    iter_n = ctx.iter_counter
    ctx.iter_counter += 1
    timestamp = _utc_now_iso()
    parent = parent_iter if parent_iter is not None else _default_parent_iter(ctx)

    query_path, sort_paths = snapshot_script_set(ctx.run_dir, iter_n, script_set)

    compile_error = _compile_check(ctx, script_set)
    if compile_error is not None:
        record = IterationRecord(
            iter=iter_n,
            timestamp=timestamp,
            query_script_path=str(query_path),
            sort_script_paths=[str(p) for p in sort_paths],
            metrics=None,
            eval_users=0,
            eval_seconds=0.0,
            llm_rationale=rationale,
            parent_iter=parent,
            compile_error=compile_error,
            partial_failure=False,
            sample_error=None,
        )
        ctx.run_log.append(record)
        return {
            "ok": False,
            "iter": iter_n,
            "compile_error": compile_error,
            "failed_script": None,
        }

    try:
        result = evaluate(
            ctx.es,
            script_set,
            ctx.ground_truth,
            ctx.users_by_id,
            k=ctx.k,
            diversity_fields=ctx.diversity_fields,
            index=ctx.index,
        )
    except RuntimeError as exc:
        msg = str(exc)
        record = IterationRecord(
            iter=iter_n,
            timestamp=timestamp,
            query_script_path=str(query_path),
            sort_script_paths=[str(p) for p in sort_paths],
            metrics=None,
            eval_users=0,
            eval_seconds=0.0,
            llm_rationale=rationale,
            parent_iter=parent,
            compile_error=msg,
            partial_failure=False,
            sample_error=None,
        )
        ctx.run_log.append(record)
        return {"ok": False, "iter": iter_n, "compile_error": msg, "failed_script": None}

    record = IterationRecord(
        iter=iter_n,
        timestamp=timestamp,
        query_script_path=str(query_path),
        sort_script_paths=[str(p) for p in sort_paths],
        metrics=result.metrics,
        eval_users=result.eval_users,
        eval_seconds=result.eval_seconds,
        llm_rationale=rationale,
        parent_iter=parent,
        compile_error=None,
        partial_failure=result.partial_failure,
        sample_error=result.sample_error,
    )
    ctx.run_log.append(record)
    ctx.last_success_iter = iter_n
    return {
        "ok": True,
        "iter": iter_n,
        "metrics": result.metrics,
        "eval_users": result.eval_users,
        "eval_seconds": result.eval_seconds,
        "partial_failure": result.partial_failure,
        "sample_error": result.sample_error,
    }


def _inline_sources(record: IterationRecord) -> dict[str, Any]:
    """Read snapshot files for one record into a small dict."""
    query_source = Path(record.query_script_path).read_text()
    sort_sources = [Path(p).read_text() for p in record.sort_script_paths]
    return {"query_source": query_source, "sort_sources": sort_sources}


def _record_to_dict(
    record: IterationRecord,
    *,
    include_sources: bool,
    guardrail_key: str,
    baseline_metrics: dict[str, float | None],
) -> dict[str, Any]:
    metrics = record.metrics
    if metrics is None:
        guardrail_held = False
    else:
        baseline_value = baseline_metrics.get(guardrail_key)
        record_value = metrics.get(guardrail_key)
        if baseline_value is None:
            guardrail_held = True
        elif record_value is None:
            guardrail_held = False
        else:
            guardrail_held = record_value >= baseline_value
    payload: dict[str, Any] = {
        "iter": record.iter,
        "timestamp": record.timestamp,
        "metrics": metrics,
        "eval_users": record.eval_users,
        "eval_seconds": record.eval_seconds,
        "llm_rationale": record.llm_rationale,
        "parent_iter": record.parent_iter,
        "compile_error": record.compile_error,
        "partial_failure": record.partial_failure,
        "sample_error": record.sample_error,
        "guardrail_held": guardrail_held,
    }
    if include_sources:
        payload.update(_inline_sources(record))
    return payload


def _read_history_impl(
    ctx: ToolContext,
    last_n: int,
    include_sources: bool,
) -> dict[str, Any]:
    all_records = ctx.run_log.read_all()
    primary_key = _primary_key(ctx.objective, ctx.k)
    guardrail_key = _guardrail_key(ctx.objective, ctx.k)

    sliced = all_records[-last_n:] if last_n > 0 else []
    records_out = [
        _record_to_dict(
            r,
            include_sources=include_sources,
            guardrail_key=guardrail_key,
            baseline_metrics=ctx.baseline_metrics,
        )
        for r in sliced
    ]

    best_record: IterationRecord | None = None
    best_value: float = float("-inf")
    for r in all_records:
        if not r.metrics:
            continue
        value = r.metrics.get(primary_key)
        if value is None:
            continue
        if value > best_value:
            best_value = value
            best_record = r
    best_so_far = (
        _record_to_dict(
            best_record,
            include_sources=include_sources,
            guardrail_key=guardrail_key,
            baseline_metrics=ctx.baseline_metrics,
        )
        if best_record is not None
        else None
    )

    return {
        "records": records_out,
        "baseline_metrics": dict(ctx.baseline_metrics),
        "best_so_far": best_so_far,
        "primary_metric": primary_key,
        "guardrail_metric": guardrail_key,
    }


def make_tools(ctx: ToolContext) -> list[BaseTool]:
    """Build the agent's tools bound to ``ctx``.

    Each call produces fresh tool objects that close over ``ctx``; the
    same context (and therefore the same monotonic iteration counter)
    drives every tool invocation in a run.

    Args:
        ctx: Per-run :class:`ToolContext`. Mutated in place as the agent
            calls ``eval_scripts``.

    Returns:
        A list of LangChain ``BaseTool`` instances ready to pass to
        :func:`langchain.agents.create_agent`.
    """

    @tool
    def eval_scripts(
        query_source: str,
        sort_sources: list[str],
        rationale: str,
        parent_iter: int | None = None,
    ) -> dict[str, Any]:
        """Snapshot, compile-check, evaluate, and log a candidate script set.

        ``query_source`` is the ``script_score`` Painless body; supply
        zero or more ``sort_sources`` to append ``_script`` sort
        clauses (capped by the run's ``max_sort_scripts``). ``rationale``
        is a short free-text note that lands in the JSONL record so
        later iterations can read why each candidate was tried.

        On success returns ``{"ok": True, "metrics": {...}, "iter": N,
        ...}``. On compile or runtime failure returns ``{"ok": False,
        "compile_error": str, "iter": N}`` — the snapshot and a
        metrics-null JSONL line still land on disk. Budget exhaustion
        returns ``{"ok": False, "budget_exhausted": True}`` without
        snapshotting; the agent should stop.
        """
        return _eval_scripts_impl(
            ctx,
            query_source=query_source,
            sort_sources=sort_sources,
            rationale=rationale,
            parent_iter=parent_iter,
        )

    @tool
    def read_history(last_n: int = 5, include_sources: bool = True) -> dict[str, Any]:
        """Return the most recent iterations from this run's log.

        Records come back oldest-first (chronological). Each record
        carries its ``metrics`` (or ``None`` on compile failure), the
        agent's earlier ``llm_rationale``, and a derived
        ``guardrail_held`` flag computed against the run's baseline.
        When ``include_sources=True`` the Painless source for each
        record is inlined so no separate file lookup is needed.

        The response also includes ``baseline_metrics`` and the
        ``best_so_far`` record judged by the run's primary metric.
        """
        return _read_history_impl(ctx, last_n=last_n, include_sources=include_sources)

    return [eval_scripts, read_history]
