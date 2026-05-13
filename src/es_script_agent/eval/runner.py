"""End-to-end evaluation runner for a multi-script script set.

Wires the harness-owned query builder, an Elasticsearch client, the
ground-truth dict, and the metrics module into a single ``evaluate``
entry point that returns an aggregated :class:`EvalResult`.

The runner is the only place ``es.query`` and ``eval.metrics`` are
composed together. Keeping the composition here lets the CLI and the
agent tools share a single code path; tests can drive it with a fake
ES client that returns canned hits.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from es_script_agent import config
from es_script_agent.data.schema import User
from es_script_agent.es.client import format_es_error
from es_script_agent.es.query import MAX_SORT_SCRIPTS, build_query
from es_script_agent.es.schemas import LOANS_INDEX
from es_script_agent.eval.metrics import (
    aggregate,
    ild_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

logger = logging.getLogger(__name__)


class ScriptSet(BaseModel):
    """The agent-authored Painless surface for one iteration.

    Attributes:
        query_source: Single ``script_score`` body. Non-blank.
        sort_sources: Zero or more sort-script bodies, executed in list
            order before the ``_score desc`` tie-break. Bounded by
            :data:`~es_script_agent.es.query.MAX_SORT_SCRIPTS`.
    """

    model_config = ConfigDict(frozen=True)

    query_source: str
    sort_sources: list[str] = Field(default_factory=list)

    @field_validator("query_source")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query_source must be a non-empty Painless source string")
        return v

    @field_validator("sort_sources")
    @classmethod
    def _bounded(cls, v: list[str]) -> list[str]:
        if len(v) > MAX_SORT_SCRIPTS:
            raise ValueError(
                f"too many sort scripts: {len(v)} > MAX_SORT_SCRIPTS={MAX_SORT_SCRIPTS}"
            )
        return v


class EvalResult(BaseModel):
    """Aggregated result of one evaluation pass.

    Attributes:
        metrics: Aggregated metric dict (``ndcg@K``, ``recall@K``,
            ``precision@K``, ``ild@K``). Values may be ``None`` when no
            user contributed (e.g. every user had K<2 hits for ILD).
        eval_users: Users who returned hits successfully.
        failed_users: Users skipped because of a per-user ES error or a
            missing vector. Surfaces alongside ``partial_failure``.
        eval_seconds: Wall-clock time for the whole pass.
        partial_failure: ``True`` iff ``failed_users > 0`` but at least
            one user succeeded.
        sample_error: Stringified first exception, or ``None`` when no
            user failed. The runner records the first error verbatim;
            it is informational, not structured.
    """

    model_config = ConfigDict(frozen=True)

    metrics: dict[str, float | None]
    eval_users: int
    failed_users: int
    eval_seconds: float
    partial_failure: bool
    sample_error: str | None


def load_script_set(directory: Path) -> ScriptSet:
    """Load a script set from disk.

    The directory must contain exactly one ``query.painless`` plus any
    number of ``sort_NN.painless`` files; the latter are read in
    lexicographic filename order (zero-padded indices keep that order
    aligned with the intended execution order).

    Args:
        directory: Script-set directory.

    Returns:
        A :class:`ScriptSet` populated from the file contents.

    Raises:
        FileNotFoundError: If ``query.painless`` is missing.
    """
    query_path = directory / "query.painless"
    if not query_path.is_file():
        raise FileNotFoundError(f"missing required query script: {query_path}")
    sort_paths = sorted(directory.glob("sort_*.painless"))
    return ScriptSet(
        query_source=query_path.read_text(),
        sort_sources=[p.read_text() for p in sort_paths],
    )


def _extract_ranked(response: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Return ``(ids, sources)`` lists in rank order from an ES response."""
    hits = response.get("hits", {}).get("hits", [])
    ranked_ids = [h["_id"] for h in hits]
    ranked_sources = [h.get("_source", {}) or {} for h in hits]
    return ranked_ids, ranked_sources


def _per_user_metrics(
    ranked_ids: list[str],
    ranked_sources: list[dict[str, Any]],
    positives: set[str],
    diversity_fields: Sequence[str],
    k: int,
) -> dict[str, float | None]:
    return {
        f"ndcg@{k}": ndcg_at_k(ranked_ids, positives, k),
        f"recall@{k}": recall_at_k(ranked_ids, positives, k),
        f"precision@{k}": precision_at_k(ranked_ids, positives, k),
        f"ild@{k}": ild_at_k(ranked_sources, diversity_fields, k),
    }


def evaluate(
    es: Any,
    script_set: ScriptSet,
    ground_truth: dict[str, set[str]],
    users_by_id: dict[str, User],
    *,
    k: int = 10,
    diversity_fields: Sequence[str] = config.ILD_DIVERSITY_FIELDS,
    index: str = LOANS_INDEX,
) -> EvalResult:
    """Run ``script_set`` against every user in ``ground_truth``.

    For each user, the runner builds the harness search body via
    :func:`~es_script_agent.es.query.build_query`, executes ``_search``
    with ``_source.includes`` restricted to ``diversity_fields`` (the
    only attributes the runner needs back; ranking metrics use ``_id``
    only), and computes per-user metrics.

    Per-user failures are caught and counted; the loop continues. If
    *every* user fails the runner raises — the script set is broken,
    not partially flaky.

    Args:
        es: Connected Elasticsearch client (or a fake exposing
            ``search``).
        script_set: Painless sources to evaluate.
        ground_truth: ``{user_id: positives}`` mapping. Users absent
            from ``users_by_id`` are skipped and counted as failures.
        users_by_id: Per-user vectors. Each ``User.vector`` is passed
            verbatim as ``params.user_vector`` (a 10×D list).
        k: Top-K cutoff for every metric.
        diversity_fields: Field names used for ILD and for the
            ``_source.includes`` request.
        index: Elasticsearch index to search.

    Returns:
        :class:`EvalResult` with aggregated metrics, counts, timings,
        and a sample error if any user failed.

    Raises:
        ValueError: If ``ground_truth`` is empty.
        RuntimeError: If every user in ``ground_truth`` fails.
    """
    if not ground_truth:
        raise ValueError("ground_truth is empty; cannot evaluate")
    start = time.monotonic()
    per_user: list[dict[str, float | None]] = []
    failed = 0
    sample_error: str | None = None

    for user_id, positives in ground_truth.items():
        user = users_by_id.get(user_id)
        if user is None:
            failed += 1
            if sample_error is None:
                sample_error = f"user {user_id!r} missing from users_by_id"
            logger.warning("skipping user %r: no vector available", user_id)
            continue
        body = build_query(
            query_source=script_set.query_source,
            sort_sources=script_set.sort_sources,
            user_vector=user.vector,
            size=k,
        )
        body["_source"] = {"includes": list(diversity_fields)}
        try:
            response = es.search(index=index, body=body)
        except Exception as exc:  # pragma: no cover - exercised via fake
            failed += 1
            if sample_error is None:
                detail = format_es_error(exc)
                sample_error = f"user {user_id!r}: {detail}"
                logger.warning("user %r failed: %s", user_id, detail)
            else:
                logger.warning("user %r failed: %s", user_id, exc)
            continue
        ranked_ids, ranked_sources = _extract_ranked(response)
        per_user.append(
            _per_user_metrics(ranked_ids, ranked_sources, positives, diversity_fields, k)
        )

    if not per_user:
        raise RuntimeError(
            f"all users failed ({failed}/{len(ground_truth)}); last error: {sample_error}"
        )

    metrics = aggregate(per_user)
    eval_seconds = time.monotonic() - start
    return EvalResult(
        metrics=metrics,
        eval_users=len(per_user),
        failed_users=failed,
        eval_seconds=eval_seconds,
        partial_failure=failed > 0,
        sample_error=sample_error,
    )
