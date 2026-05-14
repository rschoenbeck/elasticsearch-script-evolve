"""Console-script entry points wired into ``pyproject.toml`` ``[project.scripts]``."""

from __future__ import annotations

import argparse
import logging
import random
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from es_script_agent import config
from es_script_agent.agent.lineage import assert_baseline_eligible
from es_script_agent.agent.llm import make_llm
from es_script_agent.agent.loop import build_initial_message, run_loop
from es_script_agent.agent.prompts import build_system_prompt
from es_script_agent.agent.tools import ToolContext
from es_script_agent.data import DatasetAdapter, load_dataset
from es_script_agent.data.schema import User
from es_script_agent.es.client import make_client
from es_script_agent.es.index import fetch_indexed_item_ids, setup_indices
from es_script_agent.es.query import DEFAULT_MAX_SORT_SCRIPTS
from es_script_agent.es.schemas import LOANS_INDEX
from es_script_agent.eval.interactions import (
    FilterStats,
    build_ground_truth,
    filter_to_indexed_items,
)
from es_script_agent.eval.runner import EvalResult, ScriptSet, evaluate, load_script_set
from es_script_agent.runlog import (
    IterationRecord,
    RunLog,
    check_guardrail,
    new_run_dir,
    snapshot_script_set,
)

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _setup_eval_cohort(
    adapter: DatasetAdapter,
    es: Any,
) -> tuple[dict[str, set[str]], dict[str, User], FilterStats, set[str]]:
    """Build the filtered ground truth + user lookup for an eval run.

    Pulls raw interactions, fetches the current indexed item ids,
    filters the ground truth to retrievable items, materialises the
    per-user vector dict, and logs the drop-rate. Shared between
    ``baseline_cmd`` and ``rl_loop_cmd``.

    Returns:
        ``(ground_truth, users_by_id, stats, indexed_ids)``.
    """
    raw_ground_truth = build_ground_truth(adapter.iter_interactions())
    indexed_ids = fetch_indexed_item_ids(es, index=LOANS_INDEX)
    ground_truth, stats = filter_to_indexed_items(raw_ground_truth, indexed_ids)
    logger.info(
        "ground-truth filter: kept %d/%d users, %d/%d positives (%.1f%% dropped)",
        stats.users_after,
        stats.users_before,
        stats.positives_after,
        stats.positives_before,
        stats.positives_dropped_pct,
    )
    users_by_id = {u.id: u for u in adapter.iter_users() if u.id in ground_truth}
    return ground_truth, users_by_id, stats, indexed_ids


def _snapshot_and_eval_baseline(
    es: Any,
    run_dir: Path,
    k: int,
    ground_truth: dict[str, set[str]],
    users_by_id: dict[str, User],
    baseline_dir: Path,
) -> tuple[EvalResult, Path, list[Path], str]:
    """Load ``baseline_dir``, snapshot it as iter_000, and evaluate.

    Returns:
        ``(result, query_path, sort_paths, started_at)``.
    """
    script_set = load_script_set(baseline_dir)
    query_path, sort_paths = snapshot_script_set(run_dir, 0, script_set)
    started_at = _utc_now_iso()
    result = evaluate(
        es,
        script_set,
        ground_truth,
        users_by_id,
        k=k,
        diversity_fields=config.ILD_DIVERSITY_FIELDS,
    )
    return result, query_path, sort_paths, started_at


def _make_iter_zero_record(
    *,
    result: EvalResult,
    started_at: str,
    query_path: Path,
    sort_paths: list[Path],
) -> IterationRecord:
    return IterationRecord(
        iter=0,
        timestamp=started_at,
        query_script_path=str(query_path),
        sort_script_paths=[str(p) for p in sort_paths],
        metrics=result.metrics,
        eval_users=result.eval_users,
        eval_seconds=result.eval_seconds,
        llm_rationale=None,
        parent_iter=None,
        compile_error=None,
        partial_failure=result.partial_failure,
        sample_error=result.sample_error,
    )


def _ground_truth_filter_meta(stats: FilterStats) -> dict[str, float | int]:
    return {
        "users_before": stats.users_before,
        "users_after": stats.users_after,
        "positives_before": stats.positives_before,
        "positives_after": stats.positives_after,
        "positives_dropped_pct": stats.positives_dropped_pct,
    }


def _base_meta_header(
    *,
    dataset: str,
    k: int,
    objective: str,
    cohort_size: int,
    max_sort_scripts: int,
    started_at: str,
    baseline_metrics: dict[str, float | None],
    stats: FilterStats,
) -> dict[str, Any]:
    """Build the keys ``meta.json`` carries on every run.

    Shared between :func:`baseline_cmd` and :func:`rl_loop_cmd` so the
    immutable header schema can't drift between commands. Callers
    append run-mode-specific keys (e.g. provider/model_id on rl-loop)
    after this returns.
    """
    return {
        "dataset": dataset,
        "relevance_threshold": config.RELEVANCE_THRESHOLD,
        "k": k,
        "cohort_size": cohort_size,
        "seed": 0,
        "max_sort_scripts": max_sort_scripts,
        "harness_version": pkg_version("elasticsearch-agentic-script-sorting"),
        "created_at": started_at,
        "ild_diversity_fields": list(config.ILD_DIVERSITY_FIELDS),
        "objective": objective,
        "baseline_metrics": baseline_metrics,
        "ground_truth_filter": _ground_truth_filter_meta(stats),
    }


def setup_indices_cmd() -> None:
    """Drop, recreate, and bulk-load the ``loans`` and ``users`` indices."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(prog="setup-indices")
    parser.add_argument(
        "--dataset",
        default="default",
        help="Registered dataset name (default: %(default)s).",
    )
    args = parser.parse_args()

    adapter = load_dataset(args.dataset)
    client = make_client()
    counts = setup_indices(client, adapter)

    for index_name, count in counts.items():
        print(f"{index_name}: {count} docs")


def baseline_cmd() -> None:
    """Run the baseline script set once and write a complete run directory.

    Loads ``scripts/baseline/``, builds the ground-truth dict from the
    selected adapter, filters it against the current index contents,
    evaluates, and persists ``meta.json`` + ``run.jsonl`` + ``iter_000``
    snapshot. The resulting baseline metrics seed the guardrail
    thresholds the agentic loop will respect.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(prog="baseline")
    parser.add_argument("--dataset", default="default")
    parser.add_argument("--objective", default=config.DEFAULT_OBJECTIVE, choices=("ndcg", "ild"))
    parser.add_argument("--k", type=_positive_int, default=10)
    parser.add_argument(
        "--baseline-dir",
        default=None,
        dest="baseline_dir",
        help="Directory containing query.painless + sort_NN.painless (default: scripts/baseline/).",
    )
    args = parser.parse_args()

    adapter = load_dataset(args.dataset)
    es = make_client()

    ground_truth, users_by_id, stats, _ = _setup_eval_cohort(adapter, es)

    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else config.SCRIPTS_DIR / "baseline"
    run_dir = new_run_dir(config.RUNS_DIR)
    result, query_path, sort_paths, started_at = _snapshot_and_eval_baseline(
        es, run_dir, args.k, ground_truth, users_by_id, baseline_dir
    )

    log = RunLog(run_dir)
    log.write_header(
        _base_meta_header(
            dataset=args.dataset,
            k=args.k,
            objective=args.objective,
            cohort_size=len(ground_truth),
            max_sort_scripts=DEFAULT_MAX_SORT_SCRIPTS,
            started_at=started_at,
            baseline_metrics=result.metrics,
            stats=stats,
        )
    )
    log.append(
        _make_iter_zero_record(
            result=result,
            started_at=started_at,
            query_path=query_path,
            sort_paths=sort_paths,
        )
    )

    print(f"run: {run_dir}")
    print(f"eval_users: {result.eval_users}")
    print(f"eval_seconds: {result.eval_seconds:.2f}")
    for key, value in sorted(result.metrics.items()):
        print(f"{key}: {value!r}" if value is None else f"{key}: {value:.4f}")


def _resolve_model_id(provider: str) -> str:
    return config.ANTHROPIC_MODEL if provider == "anthropic" else config.OPENAI_MODEL


def rl_loop_cmd() -> None:
    """Run the agentic improvement loop end-to-end.

    Builds a fresh run directory, runs the baseline as ``iter_000`` to
    seed the guardrail thresholds, then hands control to a
    ``create_agent`` instance backed by ``eval_scripts`` /
    ``read_history``. The harness only enforces the outer iteration
    budget (inside ``eval_scripts``) and writes the final agent
    message back into ``meta.json`` once the loop terminates.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(prog="rl-loop")
    parser.add_argument("--iters", type=_positive_int, required=True)
    parser.add_argument("--provider", default=None, choices=("anthropic", "openai"))
    parser.add_argument("--dataset", default="default")
    parser.add_argument(
        "--max-sort-scripts",
        type=_positive_int,
        default=DEFAULT_MAX_SORT_SCRIPTS,
        dest="max_sort_scripts",
    )
    parser.add_argument("--objective", default=config.DEFAULT_OBJECTIVE, choices=("ndcg", "ild"))
    parser.add_argument("--k", type=_positive_int, default=10)
    parser.add_argument(
        "--lineage",
        default="linear",
        choices=("linear", "evolutionary"),
        help=(
            "Parent-selection strategy. 'linear' (default) descends each iter from the "
            "previous successful one. 'evolutionary' samples from the archive proportional "
            "to fitness × 1/(1+children)."
        ),
    )
    parser.add_argument(
        "--baseline-dir",
        default=None,
        dest="baseline_dir",
        help="Directory containing query.painless + sort_NN.painless (default: scripts/baseline/).",
    )
    parser.add_argument(
        "--hint",
        default=None,
        help=(
            "Optional free-text hint inlined into the agent's kick-off message as "
            "'Hint: <content>'. Use to pass insights from prior runs."
        ),
    )
    args = parser.parse_args()

    provider = (args.provider or config.LLM_PROVIDER).lower()
    model_id = _resolve_model_id(provider)
    lineage_seed = random.randrange(2**32)
    rng = random.Random(lineage_seed)
    primary_key = f"{args.objective}@{args.k}"
    guardrail_key = f"{'ild' if args.objective == 'ndcg' else 'ndcg'}@{args.k}"

    adapter = load_dataset(args.dataset)
    es = make_client()

    ground_truth, users_by_id, stats, indexed_ids = _setup_eval_cohort(adapter, es)
    if not users_by_id:
        raise SystemExit("error: filtered ground truth is empty; nothing to evaluate")

    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else config.SCRIPTS_DIR / "baseline"
    run_dir = new_run_dir(config.RUNS_DIR)
    baseline_result, query_path, sort_paths, started_at = _snapshot_and_eval_baseline(
        es, run_dir, args.k, ground_truth, users_by_id, baseline_dir
    )

    log = RunLog(run_dir)
    meta = _base_meta_header(
        dataset=args.dataset,
        k=args.k,
        objective=args.objective,
        cohort_size=len(ground_truth),
        max_sort_scripts=args.max_sort_scripts,
        started_at=started_at,
        baseline_metrics=baseline_result.metrics,
        stats=stats,
    )
    meta["provider"] = provider
    meta["model_id"] = model_id
    meta["max_iters"] = args.iters
    meta["lineage"] = args.lineage
    meta["lineage_seed"] = lineage_seed
    meta["hint"] = args.hint
    log.write_header(meta)
    log.append(
        _make_iter_zero_record(
            result=baseline_result,
            started_at=started_at,
            query_path=query_path,
            sort_paths=sort_paths,
        )
    )

    if args.lineage == "evolutionary":
        try:
            assert_baseline_eligible(log.read_all(), baseline_result.metrics, guardrail_key)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from exc

    ctx = ToolContext(
        es=es,
        run_dir=run_dir,
        run_log=log,
        ground_truth=ground_truth,
        users_by_id=users_by_id,
        indexed_item_ids=indexed_ids,
        baseline_metrics=baseline_result.metrics,
        objective=args.objective,
        max_iters=args.iters,
        max_sort_scripts=args.max_sort_scripts,
        k=args.k,
        diversity_fields=config.ILD_DIVERSITY_FIELDS,
        index=LOANS_INDEX,
        lineage=args.lineage,
        rng=rng,
    )

    system_prompt = build_system_prompt(
        baseline_metrics=baseline_result.metrics,
        objective=args.objective,
        diversity_fields=config.ILD_DIVERSITY_FIELDS,
        max_sort_scripts=args.max_sort_scripts,
        attribute_fields=adapter.attribute_field_types,
        max_iters=args.iters,
        k=args.k,
    )
    initial_user_message = build_initial_message(load_script_set(baseline_dir), hint=args.hint)

    llm = make_llm(provider)

    run_error: BaseException | None = None
    try:
        result = run_loop(
            ctx=ctx,
            llm=llm,
            system_prompt=system_prompt,
            initial_user_message=initial_user_message,
        )
        final_message = result.final_message
        iters_attempted = result.iters_attempted
    except BaseException as exc:  # noqa: BLE001 — we re-raise after persisting
        run_error = exc
        final_message = f"<run failed: {type(exc).__name__}: {exc}>"
        iters_attempted = ctx.iter_counter - 1

    summary = _build_summary(
        records=log.read_all(),
        baseline_metrics=baseline_result.metrics,
        primary_key=primary_key,
        guardrail_key=guardrail_key,
        final_message=final_message,
        iters_attempted=iters_attempted,
    )
    meta["summary"] = summary
    log.write_header(meta)

    print(f"run: {run_dir}")
    print(f"iters_attempted: {iters_attempted}")
    print(f"best_iter: {summary['best_iter']}")
    if summary["best_primary"] is not None:
        print(f"best {primary_key}: {summary['best_primary']:.4f}")
    print(f"guardrail_held ({guardrail_key}): {summary['guardrail_held']}")

    if run_error is not None:
        raise run_error


def _build_summary(
    *,
    records: list[IterationRecord],
    baseline_metrics: dict[str, float | None],
    primary_key: str,
    guardrail_key: str,
    final_message: str,
    iters_attempted: int,
) -> dict[str, object]:
    """Compute the post-run summary block written into ``meta.json``.

    Selects the best iteration by ``primary_key`` and reuses
    :func:`runlog.check_guardrail` to decide whether its guardrail
    metric held against the baseline — matching the same semantics the
    agent's ``read_history`` tool exposes per record.
    """
    scored = [
        r for r in records if r.metrics is not None and r.metrics.get(primary_key) is not None
    ]
    best = max(scored, key=lambda r: r.metrics[primary_key]) if scored else None
    guardrail_held = (
        check_guardrail(best.metrics, guardrail_key, baseline_metrics) if best else False
    )
    return {
        "best_iter": best.iter if best else None,
        "best_primary": best.metrics[primary_key] if best else None,
        "guardrail_held": guardrail_held,
        "iters_attempted": iters_attempted,
        "final_message": final_message,
    }


def build_eval_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for ``uv run eval``.

    The parser exposes two mutually exclusive ways to point at a script
    set: ``--set <dir>`` (loads via :func:`load_script_set`) or
    ``--query <path>`` plus zero-or-more ``--sort <path>`` files. The
    parser enforces mutual exclusion between ``--set`` and ``--query``
    via an argparse group; the ``--sort``/``--set`` and
    ``--sort``-without-``--query`` cases are caught by
    :func:`_validate_eval_args` because ``--sort`` lives outside the
    group (argparse cannot express "sort follows query" natively).

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(prog="eval")
    parser.add_argument("--dataset", default="default")
    parser.add_argument("--k", type=_positive_int, default=10)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--set",
        dest="set_dir",
        help="Directory containing query.painless + sort_NN.painless.",
    )
    group.add_argument("--query", help="Path to a single query.painless source file.")

    parser.add_argument(
        "--sort",
        action="append",
        default=[],
        help="Sort script path; repeat for multiple. Only valid with --query.",
    )
    return parser


def _validate_eval_args(args: argparse.Namespace) -> None:
    """Reject ``--sort`` combined with ``--set`` or used without ``--query``."""
    if args.sort and args.set_dir is not None:
        raise SystemExit("error: --sort cannot be combined with --set")
    if args.sort and args.query is None:
        raise SystemExit("error: --sort requires --query")


def _load_script_set_from_args(args: argparse.Namespace) -> tuple[ScriptSet, Path, list[Path]]:
    """Resolve CLI args into a ``ScriptSet`` plus the paths it was read from.

    Returns:
        ``(script_set, query_path, sort_paths)`` where the paths are the
        files actually read (the query file inside ``--set`` for the
        dir-based form, or the explicit ``--query``/``--sort`` paths).
    """
    if args.set_dir is not None:
        set_dir = Path(args.set_dir)
        script_set = load_script_set(set_dir)
        sort_paths = sorted(set_dir.glob("sort_*.painless"))
        return script_set, set_dir / "query.painless", sort_paths

    query_path = Path(args.query)
    sort_paths = [Path(p) for p in args.sort]
    script_set = ScriptSet(
        query_source=query_path.read_text(),
        sort_sources=[p.read_text() for p in sort_paths],
    )
    return script_set, query_path, sort_paths


def eval_cmd() -> None:
    """Re-evaluate a saved script set against the current indices.

    One-shot inspection: prints metrics + timing + the resolved paths.
    Does **not** create a run directory or append to any run log.
    Exits non-zero if the script set fails to load or every user fails
    during evaluation.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = build_eval_parser()
    args = parser.parse_args()
    _validate_eval_args(args)

    script_set, query_path, sort_paths = _load_script_set_from_args(args)

    adapter = load_dataset(args.dataset)
    es = make_client()

    ground_truth, users_by_id, _stats, _indexed_ids = _setup_eval_cohort(adapter, es)

    try:
        result = evaluate(
            es,
            script_set,
            ground_truth,
            users_by_id,
            k=args.k,
            diversity_fields=config.ILD_DIVERSITY_FIELDS,
        )
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from exc

    if args.set_dir is not None:
        print(f"set: {args.set_dir}")
    print(f"query: {query_path}")
    for path in sort_paths:
        print(f"sort: {path}")
    print(f"eval_users: {result.eval_users}")
    print(f"eval_seconds: {result.eval_seconds:.2f}")
    for key, value in sorted(result.metrics.items()):
        print(f"{key}: {value!r}" if value is None else f"{key}: {value:.4f}")
    if result.partial_failure and result.sample_error is not None:
        print(f"sample_error: {result.sample_error}")
