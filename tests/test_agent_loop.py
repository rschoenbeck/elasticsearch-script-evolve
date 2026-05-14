"""Tests for ``agent.loop.run_loop``.

The loop is thin glue around ``langchain.agents.create_agent``: drive
the agent with a scripted fake chat model, confirm tool calls land in
the run log via the bound ``ToolContext``, and confirm the returned
``AgentRunResult`` reflects the trace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from es_script_agent.agent.loop import (
    AgentRunResult,
    build_initial_message,
    run_loop,
)
from es_script_agent.agent.tools import ToolContext
from es_script_agent.data.schema import User
from es_script_agent.eval.runner import ScriptSet
from es_script_agent.runlog import RunLog


class _ToolCallingFakeLLM(FakeMessagesListChatModel):
    """``FakeMessagesListChatModel`` that survives ``bind_tools``."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ToolCallingFakeLLM":  # noqa: D401
        return self


class _FakeES:
    """Minimal ES fake: returns one hit on every search."""

    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> Any:
        self.search_calls.append(kwargs)
        return {
            "hits": {
                "hits": [{"_id": "item_pos", "_source": {"sector": "S"}}],
            }
        }

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"loop called non-search ES API: {name!r}")


def _user(uid: str = "u1") -> User:
    return User(id=uid, vector=[[0.1] * 4] * 10, attributes={})


def _make_ctx(tmp_path: Path, *, max_iters: int = 3) -> ToolContext:
    run_dir = tmp_path / "run_x"
    run_dir.mkdir()
    log = RunLog(run_dir)
    log.write_header({"k": 10, "objective": "ndcg"})
    return ToolContext(
        es=_FakeES(),
        run_dir=run_dir,
        run_log=log,
        ground_truth={"u1": {"item_pos"}},
        users_by_id={"u1": _user()},
        indexed_item_ids={"item_pos"},
        baseline_metrics={
            "ndcg@10": 0.30,
            "recall@10": 0.20,
            "precision@10": 0.10,
            "ild@10": 0.50,
        },
        objective="ndcg",
        max_iters=max_iters,
        k=10,
    )


def _eval_tool_call(query: str, sort: list[str] | None = None, *, tool_id: str = "t") -> dict:
    return {
        "name": "eval_scripts",
        "id": tool_id,
        "args": {
            "query_source": query,
            "sort_sources": sort or [],
            "rationale": "test",
        },
    }


def _read_history_call(tool_id: str = "t") -> dict:
    return {
        "name": "read_history",
        "id": tool_id,
        "args": {"last_n": 5, "include_sources": False},
    }


def test_run_loop_returns_final_message_and_iter_count(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    llm = _ToolCallingFakeLLM(
        responses=[
            AIMessage(content="", tool_calls=[_eval_tool_call("return 1.0;")]),
            AIMessage(content="all done"),
        ]
    )

    result = run_loop(ctx=ctx, llm=llm, system_prompt="sys")

    assert isinstance(result, AgentRunResult)
    assert result.final_message == "all done"
    assert result.iters_attempted == 1


def test_run_loop_appends_jsonl_per_eval_call(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    llm = _ToolCallingFakeLLM(
        responses=[
            AIMessage(content="", tool_calls=[_eval_tool_call("return 1.0;", tool_id="a")]),
            AIMessage(content="", tool_calls=[_eval_tool_call("return 2.0;", tool_id="b")]),
            AIMessage(content="wrap-up"),
        ]
    )

    run_loop(ctx=ctx, llm=llm, system_prompt="sys")

    records = ctx.run_log.read_all()
    assert [r.iter for r in records] == [1, 2]
    assert all(r.metrics is not None for r in records)


def test_run_loop_records_read_history_without_advancing_iter(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    llm = _ToolCallingFakeLLM(
        responses=[
            AIMessage(content="", tool_calls=[_eval_tool_call("return 1.0;")]),
            AIMessage(content="", tool_calls=[_read_history_call()]),
            AIMessage(content="done"),
        ]
    )

    result = run_loop(ctx=ctx, llm=llm, system_prompt="sys")

    # read_history does not consume budget; only the one eval_scripts call did.
    assert result.iters_attempted == 1
    assert len(ctx.run_log.read_all()) == 1


def test_run_loop_handles_budget_exhaustion(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, max_iters=1)
    llm = _ToolCallingFakeLLM(
        responses=[
            AIMessage(content="", tool_calls=[_eval_tool_call("return 1.0;", tool_id="a")]),
            AIMessage(content="", tool_calls=[_eval_tool_call("return 2.0;", tool_id="b")]),
            AIMessage(content="stopping"),
        ]
    )

    result = run_loop(ctx=ctx, llm=llm, system_prompt="sys")

    # The second eval_scripts call returned budget_exhausted without
    # advancing the counter, so only one iteration landed in the log.
    assert result.iters_attempted == 1
    assert len(ctx.run_log.read_all()) == 1
    assert result.final_message == "stopping"


def test_run_loop_extracts_text_from_last_ai_message(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    llm = _ToolCallingFakeLLM(responses=[AIMessage(content="ready")])

    result = run_loop(ctx=ctx, llm=llm, system_prompt="sys")

    assert result.final_message == "ready"
    assert result.iters_attempted == 0


# --- build_initial_message ---------------------------------------------


def test_build_initial_message_inlines_baseline_query_source() -> None:
    baseline = ScriptSet(query_source="return 1.23;", sort_sources=[])
    msg = build_initial_message(baseline)
    assert "return 1.23;" in msg
    assert "query.painless" in msg


def test_build_initial_message_inlines_all_baseline_sort_sources() -> None:
    baseline = ScriptSet(
        query_source="return 1.0;",
        sort_sources=["return _score * 2;", "return doc['x'].value;"],
    )
    msg = build_initial_message(baseline)
    assert "return _score * 2;" in msg
    assert "return doc['x'].value;" in msg
    # Each sort script is labeled by its position so the agent can
    # reference the execution order.
    assert "sort_00" in msg
    assert "sort_01" in msg


def test_build_initial_message_handles_zero_sort_scripts() -> None:
    """Baseline with no sort scripts should not render an empty sort block."""
    baseline = ScriptSet(query_source="return 1.0;", sort_sources=[])
    msg = build_initial_message(baseline)
    assert "sort_00" not in msg


def test_build_initial_message_drops_clear_winner_language() -> None:
    """The new termination policy lives in the system prompt; the kick-off
    must not reintroduce 'stop on clear winner' framing."""
    baseline = ScriptSet(query_source="return 1.0;", sort_sources=[])
    msg = build_initial_message(baseline)
    assert "clear winner" not in msg.lower()
    # The kick-off should still point the agent at the budget rule.
    assert "budget_exhausted" in msg


def test_build_initial_message_omits_hint_block_when_none() -> None:
    """Default invocation must be byte-equal to passing hint=None — no
    accidental whitespace drift when the flag is absent."""
    baseline = ScriptSet(query_source="return 1.0;", sort_sources=[])
    msg_default = build_initial_message(baseline)
    msg_explicit_none = build_initial_message(baseline, hint=None)
    assert msg_default == msg_explicit_none
    assert "Hint:" not in msg_default


def test_build_initial_message_inlines_hint_when_supplied() -> None:
    baseline = ScriptSet(query_source="return 1.0;", sort_sources=[])
    msg = build_initial_message(baseline, hint="prefer Math.log1p over raw multiplication")
    assert "Hint: prefer Math.log1p over raw multiplication" in msg
    # Hint sits between the intro and the baseline source block.
    assert msg.index("Hint:") < msg.index("## Baseline query.painless")


def test_build_initial_message_treats_whitespace_only_hint_as_absent() -> None:
    baseline = ScriptSet(query_source="return 1.0;", sort_sources=[])
    msg = build_initial_message(baseline, hint="   \n  ")
    assert "Hint:" not in msg
