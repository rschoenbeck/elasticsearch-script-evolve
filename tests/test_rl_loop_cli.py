"""rl-loop CLI: wires baseline → ToolContext → create_agent → meta summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from es_script_agent import cli, config
from es_script_agent.data.schema import Interaction, Item, User


class _ToolCallingFakeLLM(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ToolCallingFakeLLM":  # noqa: D401
        return self


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
    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> Any:
        self.search_calls.append(kwargs)
        body = kwargs["body"]
        marker = body["query"]["script_score"]["script"]["params"]["user_vector"][0][0]
        item_id = "l1" if marker == 0.1 else "l2"
        return {"hits": {"hits": [{"_id": item_id, "_source": {"sector": "S"}}]}}


@pytest.fixture
def stubbed_rl_loop_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Any]:
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
    monkeypatch.setattr(cli, "fetch_indexed_item_ids", lambda es, index: {"l1", "l2"})

    # The system prompt builder reads ``scripts/reference/`` — make it a no-op.
    monkeypatch.setattr(cli, "build_system_prompt", lambda **kwargs: "sys")

    # Default fake LLM stops immediately. Individual tests override this
    # by re-patching ``cli.make_llm``.
    monkeypatch.setattr(
        cli,
        "make_llm",
        lambda provider=None: _ToolCallingFakeLLM(responses=[AIMessage(content="done")]),
    )

    monkeypatch.setattr("sys.argv", ["rl-loop", "--iters", "3"])

    return {"runs_dir": runs_dir, "scripts_dir": scripts_dir, "adapter": adapter}


def _eval_call(query: str, *, tid: str = "t", sort: list[str] | None = None) -> dict:
    return {
        "name": "eval_scripts",
        "id": tid,
        "args": {
            "query_source": query,
            "sort_sources": sort or [],
            "rationale": "test",
        },
    }


# --- baseline wiring ---------------------------------------------------


def test_rl_loop_writes_meta_with_baseline_and_iters(
    stubbed_rl_loop_env: dict[str, Any],
) -> None:
    cli.rl_loop_cmd()
    run_dir = next(stubbed_rl_loop_env["runs_dir"].iterdir())
    meta = json.loads((run_dir / "meta.json").read_text())

    assert meta["max_iters"] == 3
    assert meta["dataset"] == "default"
    assert meta["objective"] == config.DEFAULT_OBJECTIVE
    assert {"ndcg@10", "ild@10"} <= set(meta["baseline_metrics"])
    assert (run_dir / "iter_000" / "query.painless").read_text() == "return 1.0;"


def test_rl_loop_jsonl_starts_with_baseline_iter_000(
    stubbed_rl_loop_env: dict[str, Any],
) -> None:
    cli.rl_loop_cmd()
    run_dir = next(stubbed_rl_loop_env["runs_dir"].iterdir())
    lines = (run_dir / "run.jsonl").read_text().splitlines()
    assert lines, "expected at least the baseline record"
    record = json.loads(lines[0])
    assert record["iter"] == 0
    assert record["metrics"] is not None


def test_rl_loop_records_provider_and_model_id_in_meta(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_rl_loop_env: dict[str, Any],
) -> None:
    monkeypatch.setattr(cli.config, "ANTHROPIC_MODEL", "claude-x")
    monkeypatch.setattr("sys.argv", ["rl-loop", "--iters", "3", "--provider", "anthropic"])
    cli.rl_loop_cmd()
    run_dir = next(stubbed_rl_loop_env["runs_dir"].iterdir())
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["provider"] == "anthropic"
    assert meta["model_id"] == "claude-x"


# --- agent loop & summary ----------------------------------------------


def test_rl_loop_appends_agent_iterations_after_baseline(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_rl_loop_env: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        cli,
        "make_llm",
        lambda provider=None: _ToolCallingFakeLLM(
            responses=[
                AIMessage(content="", tool_calls=[_eval_call("return 2.0;", tid="a")]),
                AIMessage(content="", tool_calls=[_eval_call("return 3.0;", tid="b")]),
                AIMessage(content="wrap-up"),
            ]
        ),
    )

    cli.rl_loop_cmd()

    run_dir = next(stubbed_rl_loop_env["runs_dir"].iterdir())
    iters = [json.loads(line)["iter"] for line in (run_dir / "run.jsonl").read_text().splitlines()]
    assert iters == [0, 1, 2]


def test_rl_loop_summary_records_final_message_and_best(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_rl_loop_env: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        cli,
        "make_llm",
        lambda provider=None: _ToolCallingFakeLLM(
            responses=[
                AIMessage(content="", tool_calls=[_eval_call("return 2.0;")]),
                AIMessage(content="all done"),
            ]
        ),
    )

    cli.rl_loop_cmd()

    run_dir = next(stubbed_rl_loop_env["runs_dir"].iterdir())
    meta = json.loads((run_dir / "meta.json").read_text())
    summary = meta["summary"]
    assert summary["final_message"] == "all done"
    assert summary["iters_attempted"] == 1
    # best_iter is the iter with the highest primary-metric value seen;
    # with only iter_000 + iter_001 (both feeding the same fake) ties are
    # broken by `max`'s first-occurrence rule, so any non-None index is OK.
    assert summary["best_iter"] in (0, 1)


def test_rl_loop_budget_one_admits_one_agent_iter(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_rl_loop_env: dict[str, Any],
) -> None:
    monkeypatch.setattr("sys.argv", ["rl-loop", "--iters", "1"])
    monkeypatch.setattr(
        cli,
        "make_llm",
        lambda provider=None: _ToolCallingFakeLLM(
            responses=[
                AIMessage(content="", tool_calls=[_eval_call("return 2.0;", tid="a")]),
                AIMessage(content="", tool_calls=[_eval_call("return 3.0;", tid="b")]),
                AIMessage(content="stopping"),
            ]
        ),
    )

    cli.rl_loop_cmd()

    run_dir = next(stubbed_rl_loop_env["runs_dir"].iterdir())
    iters = [json.loads(line)["iter"] for line in (run_dir / "run.jsonl").read_text().splitlines()]
    # iter_000 baseline + exactly one successful agent iter (iter_001);
    # the second eval_scripts call should have been budget-rejected.
    assert iters == [0, 1]


# --- arg validation ----------------------------------------------------


def test_rl_loop_requires_iters(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_rl_loop_env: dict[str, Any],
) -> None:
    monkeypatch.setattr("sys.argv", ["rl-loop"])
    with pytest.raises(SystemExit):
        cli.rl_loop_cmd()


def test_rl_loop_rejects_unknown_objective(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_rl_loop_env: dict[str, Any],
) -> None:
    monkeypatch.setattr("sys.argv", ["rl-loop", "--iters", "3", "--objective", "bogus"])
    with pytest.raises(SystemExit):
        cli.rl_loop_cmd()


def test_rl_loop_objective_ild_recorded_in_meta(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_rl_loop_env: dict[str, Any],
) -> None:
    monkeypatch.setattr("sys.argv", ["rl-loop", "--iters", "3", "--objective", "ild"])
    cli.rl_loop_cmd()
    run_dir = next(stubbed_rl_loop_env["runs_dir"].iterdir())
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["objective"] == "ild"


def test_rl_loop_max_sort_scripts_flag_recorded(
    monkeypatch: pytest.MonkeyPatch,
    stubbed_rl_loop_env: dict[str, Any],
) -> None:
    monkeypatch.setattr("sys.argv", ["rl-loop", "--iters", "3", "--max-sort-scripts", "2"])
    cli.rl_loop_cmd()
    run_dir = next(stubbed_rl_loop_env["runs_dir"].iterdir())
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["max_sort_scripts"] == 2
