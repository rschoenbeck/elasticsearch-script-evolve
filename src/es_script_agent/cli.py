"""Console-script entry points wired into ``pyproject.toml`` ``[project.scripts]``."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version

from es_script_agent import config
from es_script_agent.data import load_dataset
from es_script_agent.es.client import make_client
from es_script_agent.es.load import fetch_indexed_item_ids, setup_indices
from es_script_agent.es.query import MAX_SORT_SCRIPTS
from es_script_agent.es.schemas import LOANS_INDEX
from es_script_agent.eval.interactions import build_ground_truth, filter_to_indexed_items
from es_script_agent.eval.runner import evaluate, load_script_set
from es_script_agent.runlog import IterationRecord, RunLog, new_run_dir, snapshot_script_set

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    parser.add_argument("--k", type=int, default=10)
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
            "max_sort_scripts": MAX_SORT_SCRIPTS,
            "harness_version": pkg_version("elasticsearch-agentic-script-sorting"),
            "created_at": _utc_now_iso(),
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
            timestamp=_utc_now_iso(),
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


def rl_loop_cmd() -> None:
    raise NotImplementedError("rl-loop is not implemented yet (Task 16)")


def eval_cmd() -> None:
    raise NotImplementedError("eval is not implemented yet (Task 13)")
