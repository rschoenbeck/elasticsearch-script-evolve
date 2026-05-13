"""Provider-agnostic LangChain chat model factory.

The agent loop only depends on :class:`BaseChatModel`; the choice of
provider is a CLI-time decision threaded down via this factory. Each
provider's API key is read from :mod:`es_script_agent.config` so a
missing key surfaces as a single explicit ``ValueError`` rather than a
late authentication failure inside the loop.
"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from es_script_agent import config

_SUPPORTED_PROVIDERS = ("anthropic", "openai")


def make_llm(provider: str | None = None, model: str | None = None) -> BaseChatModel:
    """Build a LangChain chat model for the requested provider.

    Args:
        provider: Provider name, one of ``"anthropic"`` or ``"openai"``.
            Defaults to :data:`config.LLM_PROVIDER` when ``None``.
        model: Model id override. Defaults to :data:`config.ANTHROPIC_MODEL`
            or :data:`config.OPENAI_MODEL` depending on ``provider``.

    Returns:
        A configured :class:`BaseChatModel` produced by
        :func:`langchain.chat_models.init_chat_model`.

    Raises:
        ValueError: If ``provider`` is unknown or the matching API key
            is missing from the environment.
    """
    name = (provider or config.LLM_PROVIDER).lower()
    if name not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"unknown LLM provider {provider!r}; supported: {_SUPPORTED_PROVIDERS}"
        )
    if name == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set in the environment")
        model_id = model or config.ANTHROPIC_MODEL
    else:
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in the environment")
        model_id = model or config.OPENAI_MODEL
    return init_chat_model(model=model_id, model_provider=name)
