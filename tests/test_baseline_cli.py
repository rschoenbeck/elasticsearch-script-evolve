"""baseline CLI: wires adapter → ground truth → ScriptSet → evaluate → run log.

These tests use a fake ES client and a fake adapter to drive the whole
pipeline without depending on a live cluster.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from es_script_agent import cli, config
from es_script_agent.data.schema import Interaction, Item, User


class _FakeAdapter:
    vector_dim = 4
    required_attributes: list[str] = []
    attribute_field_types: dict[str, dict[str, Any]] = {}

    def __init__(self) -> None:
        self._items = [
            Item(id="l1", vector=[0.1] * 4, attributes={"sector": "S"}),
            Item(id="l2", vector=[0.2] * 4, attributes={"sector": "T"}),
        ]
        self._users = [
            User(id="u1", vector=[[0.1] * 4] * 10, attributes={}),
            User(id="u2", vector=[[0.2] * 4] * 10, attributes={}),
        ]
        self._interactions = [
            Interaction(user_id="u1", item_id="l1", weight=2),
            Interaction(user_id="u2", item_id="l2", weight=2),
        ]

    def iter_items(self) -> Any:
        return iter(self._items)

    def iter_users(self) -> Any:
        return iter(self._users)

    def iter_interactions(self) -> Any:
        return iter(self._interactions)


class _FakeClient:
    """Records `search` calls and returns canned hits; rejects other APIs."""

    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> Any:
        self.search_calls.append(kwargs)
        body = kwargs["body"]
        marker = body["query"]["script_score"]["script"]["params"]["user_vector"][0][0]
        # Return whichever loan id matches the user's marker.
        item_id = "l1" if marker == 0.1 else "l2"
        return {"hits": {"hits": [{"_id": item_id, "_source": {"sector": "S"}}]}}


@pytest.fixture
def stubbed_baseline_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Any]:
    """Stand the baseline CLI's deps on stubs and a tmp scripts/baseline dir."""
    # Stub scripts/baseline/ to a tmp dir we control.
    scripts_dir = tmp_path / "scripts" / "baseline"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "query.painless").write_text("return 1.0;")

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    monkeypatch.setattr(cli.config, "SCRIPTS_DIR", tmp_path / "scripts")
    monkeypatch.setattr(cli.config, "RUNS_DIR", runs_dir)

    adapter = _FakeAdapter()
    monkeypatch.setattr(cli, "load_dataset", lambda name: adapter)
    monkeypatch.setattr(cli, "make_client", lambda url=None: _FakeClient())
    # Indexed-id fetch returns both adapter item ids so ground truth is unfiltered.
    monkeypatch.setattr(cli, "fetch_indexed_item_ids", lambda es, index: {"l1", "l2"})

    monkeypatch.setattr("sys.argv", ["baseline"])

    return {"runs_dir": runs_dir, "scripts_dir": scripts_dir, "adapter": adapter}


# --- end-to-end CLI ----------------------------------------------------


def test_baseline_creates_run_dir_with_meta_jsonl_and_snapshot(
    stubbed_baseline_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.baseline_cmd()
    runs_dir: Path = stubbed_baseline_env["runs_dir"]
    children = sorted(runs_dir.iterdir())
    assert len(children) == 1
    run_dir = children[0]

    assert (run_dir / "meta.json").is_file()
    assert (run_dir / "run.jsonl").is_file()
    assert (run_dir / "iter_000" / "query.painless").read_text() == "return 1.0;"


def test_baseline_meta_records_objective_and_baseline_metrics(
    stubbed_baseline_env: dict[str, Any],
) -> None:
    cli.baseline_cmd()
    runs_dir: Path = stubbed_baseline_env["runs_dir"]
    run_dir = next(runs_dir.iterdir())
    meta = json.loads((run_dir / "meta.json").read_text())

    assert meta["dataset"] == "default"
    assert meta["objective"] == config.DEFAULT_OBJECTIVE
    assert meta["k"] == 10
    assert meta["ild_diversity_fields"] == list(config.ILD_DIVERSITY_FIELDS)
    assert "baseline_metrics" in meta
    # baseline_metrics must contain all four metric keys at K=10.
    assert {"ndcg@10", "recall@10", "precision@10", "ild@10"} <= set(meta["baseline_metrics"])
    assert meta["relevance_threshold"] == config.RELEVANCE_THRESHOLD


def test_baseline_jsonl_has_one_record_pointing_at_snapshot(
    stubbed_baseline_env: dict[str, Any],
) -> None:
    cli.baseline_cmd()
    runs_dir: Path = stubbed_baseline_env["runs_dir"]
    run_dir = next(runs_dir.iterdir())
    lines = (run_dir / "run.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["iter"] == 0
    assert record["query_script_path"].endswith("iter_000/query.painless")
    assert record["sort_script_paths"] == []
    assert record["metrics"] is not None
    assert record["compile_error"] is None


def test_baseline_prints_all_four_metrics(
    stubbed_baseline_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.baseline_cmd()
    out = capsys.readouterr().out
    for key in ("ndcg@10", "recall@10", "precision@10", "ild@10"):
        assert key in out


def test_baseline_objective_flag_recorded_in_meta(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_baseline_env: dict[str, Any],
) -> None:
    monkeypatch.setattr("sys.argv", ["baseline", "--objective", "ild"])
    cli.baseline_cmd()
    runs_dir: Path = stubbed_baseline_env["runs_dir"]
    meta = json.loads((next(runs_dir.iterdir()) / "meta.json").read_text())
    assert meta["objective"] == "ild"


def test_baseline_k_flag_threads_into_metrics_and_meta(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_baseline_env: dict[str, Any],
) -> None:
    monkeypatch.setattr("sys.argv", ["baseline", "--k", "5"])
    cli.baseline_cmd()
    runs_dir: Path = stubbed_baseline_env["runs_dir"]
    meta = json.loads((next(runs_dir.iterdir()) / "meta.json").read_text())
    assert meta["k"] == 5
    # baseline_metrics keys carry the configured K, not a hardcoded 10.
    assert "ndcg@5" in meta["baseline_metrics"]


def test_baseline_rejects_non_positive_k(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_baseline_env: dict[str, Any],
) -> None:
    monkeypatch.setattr("sys.argv", ["baseline", "--k", "0"])
    with pytest.raises(SystemExit):
        cli.baseline_cmd()


def test_baseline_drops_unindexed_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_baseline_env: dict[str, Any],
) -> None:
    """If `fetch_indexed_item_ids` returns a subset, positives outside it must be dropped."""
    monkeypatch.setattr(cli, "fetch_indexed_item_ids", lambda es, index: {"l1"})
    cli.baseline_cmd()
    runs_dir: Path = stubbed_baseline_env["runs_dir"]
    run_dir = next(runs_dir.iterdir())
    meta = json.loads((run_dir / "meta.json").read_text())
    # cohort_size reflects the filtered ground truth (u2 dropped because l2 isn't indexed).
    assert meta["cohort_size"] == 1
