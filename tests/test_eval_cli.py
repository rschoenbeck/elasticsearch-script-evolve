"""eval CLI: re-evaluates a saved script set; prints metrics, writes no run log.

These tests drive the CLI with a fake adapter and fake ES so they stay
unit-level. The mocking pattern mirrors ``tests/test_baseline_cli.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from es_script_agent import cli
from es_script_agent.data.schema import Interaction, Item, User
from es_script_agent.eval.runner import EvalResult, ScriptSet


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
    def search(self, **kwargs: Any) -> Any:
        body = kwargs["body"]
        marker = body["query"]["script_score"]["script"]["params"]["user_vector"][0][0]
        item_id = "l1" if marker == 0.1 else "l2"
        return {"hits": {"hits": [{"_id": item_id, "_source": {"sector": "S"}}]}}


def _make_set_dir(tmp_path: Path) -> Path:
    set_dir = tmp_path / "myset"
    set_dir.mkdir()
    (set_dir / "query.painless").write_text("return 1.0;")
    (set_dir / "sort_00.painless").write_text("return 0.0;")
    return set_dir


# --- parser-level tests ------------------------------------------------


def test_parser_requires_one_of_set_or_query() -> None:
    parser = cli.build_eval_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_rejects_mixing_set_and_query(tmp_path: Path) -> None:
    parser = cli.build_eval_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--set", str(tmp_path), "--query", str(tmp_path / "q.painless")])


def test_parser_rejects_sort_without_query(tmp_path: Path) -> None:
    parser = cli.build_eval_parser()
    args = parser.parse_args(["--set", str(tmp_path), "--sort", str(tmp_path / "s.painless")])
    # The mix is caught at validation time in eval_cmd, not by argparse itself,
    # because --sort defaults to a separate option. The parser must surface it.
    with pytest.raises(SystemExit):
        cli._validate_eval_args(args)


def test_parser_sort_with_query_ok(tmp_path: Path) -> None:
    parser = cli.build_eval_parser()
    args = parser.parse_args(
        [
            "--query",
            str(tmp_path / "q.painless"),
            "--sort",
            str(tmp_path / "s0.painless"),
            "--sort",
            str(tmp_path / "s1.painless"),
        ]
    )
    assert args.sort == [str(tmp_path / "s0.painless"), str(tmp_path / "s1.painless")]


def test_parser_defaults() -> None:
    parser = cli.build_eval_parser()
    args = parser.parse_args(["--set", "/tmp/x"])
    assert args.dataset == "default"
    assert args.k == 10
    assert args.sort == []


def test_parser_rejects_non_positive_k() -> None:
    parser = cli.build_eval_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--set", "/tmp/x", "--k", "0"])


# --- end-to-end CLI ----------------------------------------------------


@pytest.fixture
def stubbed_eval_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Any]:
    adapter = _FakeAdapter()
    monkeypatch.setattr(cli, "load_dataset", lambda name: adapter)
    monkeypatch.setattr(cli, "make_client", lambda url=None: _FakeClient())
    monkeypatch.setattr(cli, "fetch_indexed_item_ids", lambda es, index: {"l1", "l2"})
    return {"tmp_path": tmp_path, "adapter": adapter}


def test_eval_with_set_dir_runs_and_prints_metrics(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_eval_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp_path: Path = stubbed_eval_env["tmp_path"]
    set_dir = _make_set_dir(tmp_path)
    monkeypatch.setattr("sys.argv", ["eval", "--set", str(set_dir)])

    cli.eval_cmd()
    out = capsys.readouterr().out

    assert str(set_dir) in out
    assert "eval_users" in out
    assert "eval_seconds" in out
    for key in ("ndcg@10", "recall@10", "precision@10", "ild@10"):
        assert key in out


def test_eval_with_query_and_sort_constructs_script_set_from_files(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_eval_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp_path: Path = stubbed_eval_env["tmp_path"]
    qpath = tmp_path / "q.painless"
    qpath.write_text("return 1.0;")
    s0 = tmp_path / "s0.painless"
    s0.write_text("return 0.0;")

    captured: dict[str, Any] = {}

    def fake_evaluate(*args: Any, **kwargs: Any) -> EvalResult:
        captured["script_set"] = args[1]
        return EvalResult(
            metrics={"ndcg@10": 0.5, "recall@10": 0.5, "precision@10": 0.5, "ild@10": 0.0},
            eval_users=2,
            failed_users=0,
            eval_seconds=0.01,
            partial_failure=False,
            sample_error=None,
        )

    monkeypatch.setattr(cli, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        "sys.argv", ["eval", "--query", str(qpath), "--sort", str(s0)]
    )

    cli.eval_cmd()
    out = capsys.readouterr().out

    script_set: ScriptSet = captured["script_set"]
    assert script_set.query_source == "return 1.0;"
    assert script_set.sort_sources == ["return 0.0;"]
    assert str(qpath) in out
    assert str(s0) in out


def test_eval_query_with_zero_sort_scripts(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_eval_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp_path: Path = stubbed_eval_env["tmp_path"]
    qpath = tmp_path / "q.painless"
    qpath.write_text("return 1.0;")

    captured: dict[str, Any] = {}

    def fake_evaluate(*args: Any, **kwargs: Any) -> EvalResult:
        captured["script_set"] = args[1]
        return EvalResult(
            metrics={"ndcg@10": 1.0, "recall@10": 1.0, "precision@10": 1.0, "ild@10": 0.0},
            eval_users=2,
            failed_users=0,
            eval_seconds=0.01,
            partial_failure=False,
            sample_error=None,
        )

    monkeypatch.setattr(cli, "evaluate", fake_evaluate)
    monkeypatch.setattr("sys.argv", ["eval", "--query", str(qpath)])

    cli.eval_cmd()
    capsys.readouterr()

    assert captured["script_set"].sort_sources == []


def test_eval_mixing_set_and_query_errors(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_eval_env: dict[str, Any],
) -> None:
    tmp_path: Path = stubbed_eval_env["tmp_path"]
    set_dir = _make_set_dir(tmp_path)
    qpath = tmp_path / "q.painless"
    qpath.write_text("return 1.0;")
    monkeypatch.setattr(
        "sys.argv", ["eval", "--set", str(set_dir), "--query", str(qpath)]
    )
    with pytest.raises(SystemExit):
        cli.eval_cmd()


def test_eval_sort_without_query_errors(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_eval_env: dict[str, Any],
) -> None:
    tmp_path: Path = stubbed_eval_env["tmp_path"]
    set_dir = _make_set_dir(tmp_path)
    sortpath = tmp_path / "s0.painless"
    sortpath.write_text("return 0.0;")
    monkeypatch.setattr(
        "sys.argv", ["eval", "--set", str(set_dir), "--sort", str(sortpath)]
    )
    with pytest.raises(SystemExit):
        cli.eval_cmd()


def test_eval_propagates_evaluate_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_eval_env: dict[str, Any],
) -> None:
    tmp_path: Path = stubbed_eval_env["tmp_path"]
    set_dir = _make_set_dir(tmp_path)

    def raising_evaluate(*args: Any, **kwargs: Any) -> EvalResult:
        raise RuntimeError("all users failed (2/2); last error: boom")

    monkeypatch.setattr(cli, "evaluate", raising_evaluate)
    monkeypatch.setattr("sys.argv", ["eval", "--set", str(set_dir)])

    with pytest.raises(SystemExit) as exc:
        cli.eval_cmd()
    assert exc.value.code != 0


def test_eval_prints_sample_error_on_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_eval_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp_path: Path = stubbed_eval_env["tmp_path"]
    set_dir = _make_set_dir(tmp_path)

    def fake_evaluate(*args: Any, **kwargs: Any) -> EvalResult:
        return EvalResult(
            metrics={"ndcg@10": 0.5, "recall@10": 0.5, "precision@10": 0.5, "ild@10": 0.0},
            eval_users=1,
            failed_users=1,
            eval_seconds=0.01,
            partial_failure=True,
            sample_error="user 'u2': kaboom",
        )

    monkeypatch.setattr(cli, "evaluate", fake_evaluate)
    monkeypatch.setattr("sys.argv", ["eval", "--set", str(set_dir)])

    cli.eval_cmd()
    out = capsys.readouterr().out
    assert "kaboom" in out


def test_eval_k_flag_threads_through(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_eval_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    tmp_path: Path = stubbed_eval_env["tmp_path"]
    set_dir = _make_set_dir(tmp_path)

    captured: dict[str, Any] = {}

    def fake_evaluate(*args: Any, **kwargs: Any) -> EvalResult:
        captured["k"] = kwargs["k"]
        return EvalResult(
            metrics={"ndcg@5": 1.0, "recall@5": 1.0, "precision@5": 1.0, "ild@5": 0.0},
            eval_users=2,
            failed_users=0,
            eval_seconds=0.01,
            partial_failure=False,
            sample_error=None,
        )

    monkeypatch.setattr(cli, "evaluate", fake_evaluate)
    monkeypatch.setattr("sys.argv", ["eval", "--set", str(set_dir), "--k", "5"])

    cli.eval_cmd()
    out = capsys.readouterr().out
    assert captured["k"] == 5
    assert "ndcg@5" in out
