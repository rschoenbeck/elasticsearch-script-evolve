"""AgentState defaults, round-trip, and nested IterationRecord validation."""

from __future__ import annotations

from es_script_agent.agent.state import AgentState
from es_script_agent.runlog import IterationRecord


def _record(iter_n: int = 0) -> IterationRecord:
    return IterationRecord(
        iter=iter_n,
        timestamp="2026-05-13T00:00:00Z",
        query_script_path=f"runs/x/iter_{iter_n:03d}/query.painless",
        sort_script_paths=[],
        metrics={"ndcg@10": 0.5, "ild@10": 0.4},
        eval_users=100,
        eval_seconds=1.0,
        llm_rationale=None,
        parent_iter=None,
        compile_error=None,
        partial_failure=False,
        sample_error=None,
    )


def test_agent_state_defaults() -> None:
    state = AgentState()
    assert state.iter == 0
    assert state.history == []
    assert state.current_query_source is None
    assert state.current_sort_sources is None
    assert state.current_rationale is None
    assert state.parent_iter is None


def test_agent_state_round_trip_empty() -> None:
    state = AgentState()
    dumped = state.model_dump()
    loaded = AgentState.model_validate(dumped)
    assert loaded == state


def test_agent_state_round_trip_with_history() -> None:
    state = AgentState(
        iter=2,
        history=[_record(0), _record(1)],
        current_query_source="return 1.0;",
        current_sort_sources=["return doc['popularityScore'].value;"],
        current_rationale="trying mean pooling",
        parent_iter=1,
    )
    dumped = state.model_dump()
    loaded = AgentState.model_validate(dumped)
    assert loaded == state
    assert isinstance(loaded.history[0], IterationRecord)


def test_agent_state_history_json_round_trip() -> None:
    state = AgentState(history=[_record(0)])
    raw = state.model_dump_json()
    loaded = AgentState.model_validate_json(raw)
    assert loaded == state
