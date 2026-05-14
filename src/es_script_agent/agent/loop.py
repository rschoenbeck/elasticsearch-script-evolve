"""Agentic improvement loop.

Wraps :func:`langchain.agents.create_agent` and invokes it with the
tools, system prompt, and retry middleware required by the harness.
The outer iteration budget is enforced inside the ``eval_scripts``
tool (see :mod:`es_script_agent.agent.tools`); this module owns only
the invocation surface and the post-run summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from es_script_agent.agent.tools import ToolContext, make_tools

logger = logging.getLogger(__name__)


_DEFAULT_INITIAL_MESSAGE = (
    "Begin improving the scoring scripts. Read the system prompt for the run "
    "objective and baseline metrics, then propose your first candidate via "
    "the eval_scripts tool. Stop when you have produced a clear winner "
    "against the primary metric while holding the guardrail, or when the "
    "iteration budget is exhausted."
)


@dataclass(frozen=True)
class AgentRunResult:
    """Outcome of one :func:`run_loop` invocation.

    Attributes:
        final_message: Text of the agent's last assistant message.
        iters_attempted: Number of ``eval_scripts`` tool calls that
            advanced the iteration counter (successful + compile-failed
            combined). Excludes budget-rejected calls.
    """

    final_message: str
    iters_attempted: int


def _extract_final_text(messages: list) -> str:
    """Return the text of the most recent ``AIMessage`` in ``messages``.

    ``create_agent`` always concludes with an AI message; the
    no-AIMessage branch only fires on a degenerate trace and returns
    ``""`` after a warning rather than leaking a raw message repr into
    ``meta.json``.
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            return str(content)
    logger.warning(
        "agent trace ended without an AIMessage (len=%d); returning empty final_message",
        len(messages),
    )
    return ""


def run_loop(
    *,
    ctx: ToolContext,
    llm: BaseChatModel,
    system_prompt: str,
    initial_user_message: str = _DEFAULT_INITIAL_MESSAGE,
    middleware_max_retries: int = 3,
) -> AgentRunResult:
    """Invoke ``create_agent`` against ``ctx`` and return the final message.

    Args:
        ctx: Per-run tool context. Mutated as ``eval_scripts`` is called.
        llm: Bound chat model to drive the agent.
        system_prompt: Pre-rendered system prompt.
        initial_user_message: Kick-off user turn fed to the agent.
        middleware_max_retries: Retries on transient model errors.

    Returns:
        :class:`AgentRunResult` carrying the agent's final assistant
        message and the count of completed iterations (per the JSONL
        log).
    """
    tools = make_tools(ctx)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            ModelRetryMiddleware(
                max_retries=middleware_max_retries,
                backoff_factor=2.0,
                initial_delay=1.0,
            )
        ],
    )

    result = agent.invoke({"messages": [HumanMessage(initial_user_message)]})
    messages = result["messages"] if isinstance(result, dict) else result.messages
    final_text = _extract_final_text(messages)
    iters_attempted = ctx.iter_counter - 1
    return AgentRunResult(final_message=final_text, iters_attempted=iters_attempted)
