"""config.py exports the expected constants with correct types and defaults."""

from pathlib import Path

from es_script_agent import config


def test_es_url_is_str() -> None:
    assert isinstance(config.ES_URL, str)
    assert config.ES_URL  # non-empty


def test_llm_provider_is_str() -> None:
    assert isinstance(config.LLM_PROVIDER, str)
    assert config.LLM_PROVIDER in {"openai", "anthropic"}


def test_api_key_constants_exist() -> None:
    # Keys may be empty strings in CI; just assert presence + type.
    assert isinstance(config.ANTHROPIC_API_KEY, str)
    assert isinstance(config.OPENAI_API_KEY, str)


def test_path_constants_are_paths() -> None:
    for name in ("DATA_DIR", "RUNS_DIR", "SCRIPTS_DIR"):
        value = getattr(config, name)
        assert isinstance(value, Path), f"{name} must be a Path"


def test_relevance_threshold_default() -> None:
    assert config.RELEVANCE_THRESHOLD == 1


def test_ild_diversity_fields_default() -> None:
    assert config.ILD_DIVERSITY_FIELDS == ("sector", "country", "partnerId")
    assert isinstance(config.ILD_DIVERSITY_FIELDS, tuple)


def test_default_objective() -> None:
    assert config.DEFAULT_OBJECTIVE == "ndcg"
