"""Agent loop state.

:class:`AgentState` is the Pydantic value threaded through the LangGraph
nodes. It is intentionally minimal at this stage: a single proposed
Painless source plus the run history. Multi-script proposals are
introduced in a later task.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from es_script_agent.runlog import IterationRecord


class AgentState(BaseModel):
    """Mutable-by-replacement state for the LangGraph agent loop.

    Attributes:
        iter: Current iteration index. ``0`` corresponds to the baseline
            slot; the agent starts proposing at ``1``.
        history: Append-only log of completed iterations, oldest first.
        current_source: The Painless source the agent has just proposed
            but not yet evaluated. ``None`` between iterations.
        current_rationale: Free-text rationale paired with
            ``current_source``. ``None`` between iterations.
        parent_iter: Iteration index the current proposal descends from.
            ``None`` when the proposal has no parent (e.g. first move).
    """

    iter: int = 0
    history: list[IterationRecord] = Field(default_factory=list)
    current_source: str | None = None
    current_rationale: str | None = None
    parent_iter: int | None = None
