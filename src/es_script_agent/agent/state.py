"""Agent loop state.

:class:`AgentState` is the Pydantic value threaded through the agent
loop. It mirrors the two-script-type contract: each iteration emits one
query Painless source plus zero-to-N sort sources.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from es_script_agent.runlog import IterationRecord


class AgentState(BaseModel):
    """Mutable-by-replacement state for the agent loop.

    Attributes:
        iter: Current iteration index. ``0`` corresponds to the baseline
            slot; the agent starts proposing at ``1``.
        history: Append-only log of completed iterations, oldest first.
        current_query_source: The ``script_score`` Painless source the
            agent has just proposed but not yet evaluated. ``None``
            between iterations.
        current_sort_sources: The ordered list of ``_script`` sort
            Painless sources accompanying ``current_query_source``. May
            be an empty list (no sort scripts) or ``None`` between
            iterations.
        current_rationale: Free-text rationale paired with the current
            proposal. ``None`` between iterations.
        parent_iter: Iteration index the current proposal descends from.
            ``None`` when the proposal has no parent (e.g. first move).
    """

    iter: int = 0
    history: list[IterationRecord] = Field(default_factory=list)
    current_query_source: str | None = None
    current_sort_sources: list[str] | None = None
    current_rationale: str | None = None
    parent_iter: int | None = None
