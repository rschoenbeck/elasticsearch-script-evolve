"""Tests for the ES client factory and response-side error rendering."""

from typing import Any

from elasticsearch import Elasticsearch

from es_script_agent.es import client as es_client
from es_script_agent.es.client import format_es_error


def test_make_client_default_url() -> None:
    client = es_client.make_client()
    assert isinstance(client, Elasticsearch)


def test_make_client_explicit_url() -> None:
    client = es_client.make_client("http://example.invalid:9200")
    assert isinstance(client, Elasticsearch)


def test_no_module_level_client() -> None:
    # The module exposes a factory, not a pre-built client.
    public_attrs = {n for n in dir(es_client) if not n.startswith("_")}
    assert "make_client" in public_attrs
    for name in public_attrs:
        value = getattr(es_client, name)
        assert not isinstance(value, Elasticsearch), (
            f"es_client.{name} is a module-level Elasticsearch instance; "
            "callers should build their own via make_client()."
        )


# --- format_es_error ----------------------------------------------------


class _ApiErrorLike(Exception):
    """Mimics elasticsearch.ApiError: carries a ``body`` dict; ``str()`` collapses to a short reason."""

    def __init__(self, body: dict[str, Any], short: str = "compile error") -> None:
        super().__init__(short)
        self.body = body
        self._short = short

    def __str__(self) -> str:
        return self._short


def _painless_error_body() -> dict[str, Any]:
    return {
        "error": {
            "root_cause": [
                {
                    "type": "script_exception",
                    "reason": "compile error",
                    "script_stack": [
                        "... cosineSimilarity(pooled, 'item_vector') + ...",
                        "                    ^---- HERE",
                    ],
                    "caused_by": {
                        "type": "class_cast_exception",
                        "reason": "Cannot cast from [double[]] to [java.util.List].",
                    },
                }
            ]
        }
    }


def test_format_es_error_extracts_script_stack_and_caused_by_chain() -> None:
    rendered = format_es_error(_ApiErrorLike(_painless_error_body()))
    assert "script_exception: compile error" in rendered
    assert "script_stack:" in rendered
    assert "cosineSimilarity" in rendered
    assert "caused_by class_cast_exception" in rendered
    assert "Cannot cast from [double[]] to [java.util.List]." in rendered


def test_format_es_error_walks_nested_caused_by() -> None:
    body = {
        "error": {
            "root_cause": [
                {
                    "type": "outer",
                    "reason": "outer reason",
                    "caused_by": {
                        "type": "mid",
                        "reason": "mid reason",
                        "caused_by": {"type": "inner", "reason": "root cause here"},
                    },
                }
            ]
        }
    }
    rendered = format_es_error(_ApiErrorLike(body))
    assert "caused_by mid: mid reason" in rendered
    assert "caused_by inner: root cause here" in rendered


def test_format_es_error_falls_back_for_non_api_exceptions() -> None:
    assert format_es_error(RuntimeError("boom")) == "boom"


def test_format_es_error_falls_back_on_unexpected_body_shape() -> None:
    assert format_es_error(_ApiErrorLike({"unrelated": "shape"}, short="x")) == "x"
