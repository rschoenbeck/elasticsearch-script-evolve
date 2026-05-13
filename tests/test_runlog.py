"""Tests for the per-run JSONL writer + reader and the snapshot helper.

The run log is the artifact every later tool consumes (analysis notebook,
agent history). Schema drift here breaks cross-run comparability, so the
shape is asserted aggressively.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from es_script_agent.eval.runner import ScriptSet
from es_script_agent.runlog import (
    IterationRecord,
    RunLog,
    new_run_dir,
    snapshot_script_set,
)


def _record(iter_n: int = 0, **overrides: object) -> IterationRecord:
    defaults: dict[str, object] = {
        "iter": iter_n,
        "timestamp": "2026-05-11T18:42:13Z",
        "query_script_path": f"runs/x/iter_{iter_n:03d}/query.painless",
        "sort_script_paths": [],
        "metrics": {"ndcg@10": 0.5, "ild@10": 0.4},
        "eval_users": 100,
        "eval_seconds": 8.4,
        "llm_rationale": None,
        "parent_iter": None,
        "compile_error": None,
        "partial_failure": False,
        "sample_error": None,
    }
    defaults.update(overrides)
    return IterationRecord(**defaults)  # type: ignore[arg-type]


# --- IterationRecord schema --------------------------------------------


def test_iteration_record_round_trips_through_json() -> None:
    record = _record(iter_n=3, llm_rationale="tried a different pooling")
    raw = record.model_dump_json()
    loaded = IterationRecord.model_validate_json(raw)
    assert loaded == record


def test_iteration_record_accepts_null_metrics_with_compile_error() -> None:
    record = _record(
        iter_n=4,
        metrics=None,
        compile_error="painless: unexpected token",
    )
    assert record.metrics is None
    assert record.compile_error is not None


# --- snapshot helper ---------------------------------------------------


def test_snapshot_writes_query_and_sort_files(tmp_path: Path) -> None:
    script_set = ScriptSet(
        query_source="return 1.0;",
        sort_sources=["return 0.0;", "return _score;"],
    )
    query_path, sort_paths = snapshot_script_set(tmp_path, 7, script_set)
    assert query_path == tmp_path / "iter_007" / "query.painless"
    assert query_path.read_text() == "return 1.0;"
    assert [p.name for p in sort_paths] == ["sort_00.painless", "sort_01.painless"]
    assert sort_paths[0].read_text() == "return 0.0;"
    assert sort_paths[1].read_text() == "return _score;"


def test_snapshot_handles_zero_sort_scripts(tmp_path: Path) -> None:
    query_path, sort_paths = snapshot_script_set(
        tmp_path, 0, ScriptSet(query_source="Q")
    )
    assert query_path.exists()
    assert sort_paths == []
    assert not any(tmp_path.glob("iter_000/sort_*.painless"))


def test_snapshot_refuses_to_overwrite(tmp_path: Path) -> None:
    snapshot_script_set(tmp_path, 0, ScriptSet(query_source="Q"))
    with pytest.raises(FileExistsError):
        snapshot_script_set(tmp_path, 0, ScriptSet(query_source="Q2"))


# --- RunLog ------------------------------------------------------------


def test_runlog_write_header_persists_meta_json(tmp_path: Path) -> None:
    log = RunLog(tmp_path)
    meta = {
        "dataset": "default",
        "relevance_threshold": 1,
        "k": 10,
        "cohort_size": 412,
        "seed": 0,
        "max_sort_scripts": 5,
        "harness_version": "0.1.0",
        "created_at": "2026-05-11T18:42:13Z",
        "ild_diversity_fields": ["sector", "country", "partnerId"],
        "objective": "ndcg",
        "baseline_metrics": {
            "ndcg@10": 0.21,
            "recall@10": 0.11,
            "precision@10": 0.02,
            "ild@10": 0.41,
        },
    }
    log.write_header(meta)
    on_disk = json.loads((tmp_path / "meta.json").read_text())
    assert on_disk == meta


def test_runlog_append_then_read_round_trips(tmp_path: Path) -> None:
    log = RunLog(tmp_path)
    r0 = _record(iter_n=0)
    r1 = _record(iter_n=1, llm_rationale="follow-up")
    log.append(r0)
    log.append(r1)
    loaded = log.read_all()
    assert loaded == [r0, r1]


def test_runlog_jsonl_is_one_record_per_line(tmp_path: Path) -> None:
    log = RunLog(tmp_path)
    log.append(_record(iter_n=0))
    log.append(_record(iter_n=1))
    lines = (tmp_path / "run.jsonl").read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # each line is valid JSON


def test_runlog_read_all_on_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    assert RunLog(tmp_path).read_all() == []


# --- new_run_dir -------------------------------------------------------


def test_new_run_dir_creates_timestamped_subdir(tmp_path: Path) -> None:
    p = new_run_dir(tmp_path)
    assert p.parent == tmp_path
    assert p.exists() and p.is_dir()
    # YYYYMMDD-HHMMSS, UTC
    name = p.name
    assert len(name) == 15
    assert name[8] == "-"
    assert name[:8].isdigit()
    assert name[9:].isdigit()
