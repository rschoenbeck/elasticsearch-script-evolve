"""make_llm factory: provider dispatch, defaults, and error handling."""

from __future__ import annotations

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from es_script_agent import config
from es_script_agent.agent.llm import make_llm


def test_make_llm_anthropic_returns_chat_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-anthropic-key")
    llm = make_llm("anthropic")
    assert isinstance(llm, ChatAnthropic)


def test_make_llm_openai_returns_chat_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-openai-key")
    llm = make_llm("openai")
    assert isinstance(llm, ChatOpenAI)


def test_make_llm_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="bogus"):
        make_llm("bogus")


def test_make_llm_missing_anthropic_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        make_llm("anthropic")


def test_make_llm_missing_openai_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make_llm("openai")


def test_make_llm_defaults_to_config_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-anthropic-key")
    llm = make_llm()
    assert isinstance(llm, ChatAnthropic)


def test_make_llm_anthropic_uses_config_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setattr(config, "ANTHROPIC_MODEL", "claude-from-config")
    llm = make_llm("anthropic")
    assert llm.model == "claude-from-config"


def test_make_llm_openai_uses_config_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(config, "OPENAI_MODEL", "gpt-from-config")
    llm = make_llm("openai")
    assert llm.model_name == "gpt-from-config"


def test_make_llm_model_param_overrides_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setattr(config, "ANTHROPIC_MODEL", "claude-from-config")
    llm = make_llm("anthropic", model="claude-from-arg")
    assert llm.model == "claude-from-arg"
