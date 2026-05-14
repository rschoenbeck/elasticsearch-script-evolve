"""Console-script entry points wired into ``pyproject.toml`` ``[project.scripts]``."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

from es_script_agent import config
from es_script_agent.agent.llm import make_llm
from es_script_agent.agent.loop import run_loop
from es_script_agent.agent.prompts import build_system_prompt
from es_script_agent.agent.tools import ToolContext
from es_script_agent.data import load_dataset
from es_script_agent.es.client import make_client
from es_script_agent.es.index import fetch_indexed_item_ids, setup_indices
from es_script_agent.es.query import DEFAULT_MAX_SORT_SCRIPTS
from es_script_agent.es.schemas import LOANS_INDEX
from es_script_agent.eval.interactions import build_ground_truth, filter_to_indexed_items
from es_script_agent.eval.runner import ScriptSet, evaluate, load_script_set
from es_script_agent.runlog import IterationRecord, RunLog, new_run_dir, snapshot_script_set

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


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
    args = parser.parse_args()

    adapter = load_dataset(args.dataset)
    es = make_client()

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

    script_set = load_script_set(config.SCRIPTS_DIR / "baseline")
    run_dir = new_run_dir(config.RUNS_DIR)
    query_path, sort_paths = snapshot_script_set(run_dir, 0, script_set)
    started_at = _utc_now_iso()

    result = evaluate(
        es,
        script_set,
        ground_truth,
        users_by_id,
        k=args.k,
        diversity_fields=config.ILD_DIVERSITY_FIELDS,
    )

    log = RunLog(run_dir)
    log.write_header(
        {
            "dataset": args.dataset,
            "relevance_threshold": config.RELEVANCE_THRESHOLD,
            "k": args.k,
            "cohort_size": len(ground_truth),
            "seed": 0,
            "max_sort_scripts": DEFAULT_MAX_SORT_SCRIPTS,
            "harness_version": pkg_version("elasticsearch-agentic-script-sorting"),
            "created_at": started_at,
            "ild_diversity_fields": list(config.ILD_DIVERSITY_FIELDS),
            "objective": args.objective,
            "baseline_metrics": result.metrics,
            "ground_truth_filter": {
                "users_before": stats.users_before,
                "users_after": stats.users_after,
                "positives_before": stats.positives_before,
                "positives_after": stats.positives_after,
                "positives_dropped_pct": stats.positives_dropped_pct,
            },
        }
    )
    log.append(
        IterationRecord(
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
    args = parser.parse_args()

    provider = (args.provider or config.LLM_PROVIDER).lower()
    model_id = _resolve_model_id(provider)

    adapter = load_dataset(args.dataset)
    es = make_client()

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
    if not users_by_id:
        raise SystemExit("error: filtered ground truth is empty; nothing to evaluate")

    baseline_set = load_script_set(config.SCRIPTS_DIR / "baseline")
    run_dir = new_run_dir(config.RUNS_DIR)
    started_at = _utc_now_iso()

    query_path, sort_paths = snapshot_script_set(run_dir, 0, baseline_set)
    baseline_result = evaluate(
        es,
        baseline_set,
        ground_truth,
        users_by_id,
        k=args.k,
        diversity_fields=config.ILD_DIVERSITY_FIELDS,
    )

    log = RunLog(run_dir)
    meta = {
        "dataset": args.dataset,
        "relevance_threshold": config.RELEVANCE_THRESHOLD,
        "k": args.k,
        "cohort_size": len(ground_truth),
        "seed": 0,
        "max_sort_scripts": args.max_sort_scripts,
        "harness_version": pkg_version("elasticsearch-agentic-script-sorting"),
        "created_at": started_at,
        "ild_diversity_fields": list(config.ILD_DIVERSITY_FIELDS),
        "objective": args.objective,
        "baseline_metrics": baseline_result.metrics,
        "provider": provider,
        "model_id": model_id,
        "max_iters": args.iters,
        "ground_truth_filter": {
            "users_before": stats.users_before,
            "users_after": stats.users_after,
            "positives_before": stats.positives_before,
            "positives_after": stats.positives_after,
            "positives_dropped_pct": stats.positives_dropped_pct,
        },
    }
    log.write_header(meta)
    log.append(
        IterationRecord(
            iter=0,
            timestamp=started_at,
            query_script_path=str(query_path),
            sort_script_paths=[str(p) for p in sort_paths],
            metrics=baseline_result.metrics,
            eval_users=baseline_result.eval_users,
            eval_seconds=baseline_result.eval_seconds,
            llm_rationale=None,
            parent_iter=None,
            compile_error=None,
            partial_failure=baseline_result.partial_failure,
            sample_error=baseline_result.sample_error,
        )
    )

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
    )

    system_prompt = build_system_prompt(
        baseline_metrics=baseline_result.metrics,
        objective=args.objective,
        diversity_fields=config.ILD_DIVERSITY_FIELDS,
        max_sort_scripts=args.max_sort_scripts,
        reference_dir=config.SCRIPTS_DIR / "reference",
        attribute_fields=getattr(adapter, "attribute_field_types", {}),
        k=args.k,
    )

    llm = make_llm(provider)
    result = run_loop(ctx=ctx, llm=llm, system_prompt=system_prompt)

    records = log.read_all()
    primary_key = f"{args.objective}@{args.k}"
    guardrail_key = f"{'ild' if args.objective == 'ndcg' else 'ndcg'}@{args.k}"
    scored = [
        r for r in records if r.metrics is not None and r.metrics.get(primary_key) is not None
    ]
    best = max(scored, key=lambda r: r.metrics[primary_key]) if scored else None
    baseline_guardrail = baseline_result.metrics.get(guardrail_key)
    if best is not None and baseline_guardrail is not None:
        best_guardrail = best.metrics.get(guardrail_key)
        guardrail_held = best_guardrail is not None and best_guardrail >= baseline_guardrail
    else:
        guardrail_held = best is not None

    summary = {
        "best_iter": best.iter if best else None,
        "best_primary": best.metrics[primary_key] if best else None,
        "guardrail_held": guardrail_held,
        "iters_attempted": result.iters_attempted,
        "final_message": result.final_message,
    }
    meta["summary"] = summary
    log.write_header(meta)

    print(f"run: {run_dir}")
    print(f"iters_attempted: {result.iters_attempted}")
    print(f"best_iter: {summary['best_iter']}")
    if best is not None:
        print(f"best {primary_key}: {best.metrics[primary_key]:.4f}")
    print(f"guardrail_held ({guardrail_key}): {guardrail_held}")


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
