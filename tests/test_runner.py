"""Tests for the multi-script evaluation runner.

These pin the runner's contract with mocked ES: it produces an
``EvalResult`` aggregating per-user metrics for a ``ScriptSet`` against a
ground-truth dict, fetches ``_source`` for the configured diversity
fields, propagates failures into ``partial_failure`` + ``sample_error``,
and raises only when every user fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from es_script_agent.data.schema import User
from es_script_agent.es.schemas import LOANS_INDEX
from es_script_agent.eval.runner import (
    EvalResult,
    ScriptSet,
    evaluate,
    load_script_set,
)


# --- Fakes --------------------------------------------------------------


class _FakeES:
    """Returns canned hits keyed by the first user_vector cell.

    The runner injects ``params.user_vector`` verbatim; the fake reads
    ``user_vector[0][0]`` as a stable per-user marker so tests can
    distinguish users without a test-only kwarg on the public API.
    """

    def __init__(self, responses_by_marker: dict[float, Any]) -> None:
        self._responses = responses_by_marker
        self.search_calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> Any:
        self.search_calls.append(kwargs)
        body = kwargs["body"]
        marker = body["query"]["script_score"]["script"]["params"]["user_vector"][0][0]
        response = self._responses[marker]
        if isinstance(response, Exception):
            raise response
        return response

    def __getattr__(self, name: str) -> Any:
        # Any non-search API call is a contract violation.
        raise AssertionError(f"runner called non-search ES API: {name!r}")


def _hits(ranked: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Build an ES search response from ``(id, source)`` pairs in rank order."""
    return {
        "hits": {
            "hits": [
                {"_id": item_id, "_source": source} for item_id, source in ranked
            ]
        }
    }


def _user(user_id: str, marker: float) -> User:
    return User(id=user_id, vector=[[marker] * 4] * 10, attributes={})


# --- ScriptSet + load_script_set ---------------------------------------


def test_script_set_default_has_no_sort_sources() -> None:
    s = ScriptSet(query_source="return 1.0;")
    assert s.sort_sources == []


def test_script_set_rejects_blank_query_source() -> None:
    with pytest.raises(ValueError):
        ScriptSet(query_source="   ")


def test_script_set_carries_arbitrary_sort_source_count() -> None:
    """ScriptSet is a pure value type; sort-script cap enforcement is policy
    and lives at the agent tool layer, not on this model."""
    s = ScriptSet(query_source="Q", sort_sources=["S"] * 12)
    assert len(s.sort_sources) == 12


def test_load_script_set_reads_query_only(tmp_path: Path) -> None:
    (tmp_path / "query.painless").write_text("return 1.0;")
    s = load_script_set(tmp_path)
    assert s.query_source == "return 1.0;"
    assert s.sort_sources == []


def test_load_script_set_orders_sort_scripts_lexicographically(tmp_path: Path) -> None:
    (tmp_path / "query.painless").write_text("Q")
    (tmp_path / "sort_01.painless").write_text("S1")
    (tmp_path / "sort_00.painless").write_text("S0")
    (tmp_path / "sort_02.painless").write_text("S2")
    s = load_script_set(tmp_path)
    assert s.sort_sources == ["S0", "S1", "S2"]


def test_load_script_set_missing_query_raises(tmp_path: Path) -> None:
    (tmp_path / "sort_00.painless").write_text("S0")
    with pytest.raises(FileNotFoundError):
        load_script_set(tmp_path)


# --- evaluate ----------------------------------------------------------


def test_evaluate_aggregates_metrics_across_users() -> None:
    # Two users; ground-truth puts user_a's positive at rank 0 (perfect),
    # user_b's positive at rank 2.
    responses = {
        1.0: _hits(
            [
                ("item_pos", {"sector": "S", "country": "C"}),
                ("item_x", {"sector": "T", "country": "D"}),
            ]
        ),
        2.0: _hits(
            [
                ("item_x", {"sector": "S", "country": "C"}),
                ("item_y", {"sector": "T", "country": "D"}),
                ("item_pos_b", {"sector": "S", "country": "D"}),
            ]
        ),
    }
    es = _FakeES(responses)
    ground_truth = {"user_a": {"item_pos"}, "user_b": {"item_pos_b"}}
    users_by_id = {"user_a": _user("user_a", 1.0), "user_b": _user("user_b", 2.0)}

    result = evaluate(
        es,
        ScriptSet(query_source="return 1.0;"),
        ground_truth,
        users_by_id,
        k=10,
        diversity_fields=("sector", "country"),
    )

    assert isinstance(result, EvalResult)
    assert result.eval_users == 2
    assert result.failed_users == 0
    assert result.partial_failure is False
    assert result.sample_error is None
    assert set(result.metrics.keys()) == {"ndcg@10", "recall@10", "precision@10", "ild@10"}
    # Per-user NDCG@10 = 1.0 for user_a, 0.5 for user_b → mean = 0.75.
    assert result.metrics["ndcg@10"] == pytest.approx(0.75, abs=1e-9)
    # Diversity-field _source includes propagated into every search.
    for call in es.search_calls:
        body = call["body"]
        includes = body["_source"]["includes"]
        assert set(includes) == {"sector", "country"}


def test_evaluate_propagates_index_and_size() -> None:
    es = _FakeES({1.0: _hits([("item_pos", {"sector": "S"})])})
    evaluate(
        es,
        ScriptSet(query_source="return 1.0;"),
        {"u1": {"item_pos"}},
        {"u1": _user("u1", 1.0)},
        k=5,
        diversity_fields=("sector",),
        index="custom_index",
    )
    call = es.search_calls[0]
    assert call["index"] == "custom_index"
    assert call["body"]["size"] == 5


def test_evaluate_default_index_is_loans() -> None:
    es = _FakeES({1.0: _hits([("item_pos", {"sector": "S"})])})
    evaluate(
        es,
        ScriptSet(query_source="return 1.0;"),
        {"u1": {"item_pos"}},
        {"u1": _user("u1", 1.0)},
        k=10,
        diversity_fields=("sector",),
    )
    assert es.search_calls[0]["index"] == LOANS_INDEX


def test_evaluate_partial_failure_keeps_running() -> None:
    responses: dict[float, Any] = {
        1.0: _hits([("item_pos", {"sector": "S"})]),
        2.0: RuntimeError("script_exception: bad bad bad"),
    }
    es = _FakeES(responses)
    ground_truth = {"user_ok": {"item_pos"}, "user_bad": {"item_pos"}}
    users_by_id = {"user_ok": _user("user_ok", 1.0), "user_bad": _user("user_bad", 2.0)}

    result = evaluate(
        es,
        ScriptSet(query_source="return 1.0;"),
        ground_truth,
        users_by_id,
        k=10,
        diversity_fields=("sector",),
    )
    assert result.eval_users == 1
    assert result.failed_users == 1
    assert result.partial_failure is True
    assert result.sample_error is not None
    assert "bad bad bad" in result.sample_error


def test_evaluate_raises_when_every_user_fails() -> None:
    responses: dict[float, Any] = {
        1.0: RuntimeError("boom"),
        2.0: RuntimeError("boom"),
    }
    es = _FakeES(responses)
    ground_truth = {"u1": {"x"}, "u2": {"y"}}
    users_by_id = {"u1": _user("u1", 1.0), "u2": _user("u2", 2.0)}

    with pytest.raises(RuntimeError, match="all users failed"):
        evaluate(
            es,
            ScriptSet(query_source="return 1.0;"),
            ground_truth,
            users_by_id,
            k=10,
            diversity_fields=("sector",),
        )


def test_evaluate_raises_on_empty_ground_truth() -> None:
    es = _FakeES({})
    with pytest.raises(ValueError, match="empty"):
        evaluate(
            es,
            ScriptSet(query_source="return 1.0;"),
            {},
            {},
            k=10,
            diversity_fields=("sector",),
        )


def test_evaluate_with_sort_scripts_calls_es_with_sort_clauses() -> None:
    responses = {1.0: _hits([("item_pos", {"sector": "S"})])}
    es = _FakeES(responses)
    result = evaluate(
        es,
        ScriptSet(query_source="return 1.0;", sort_sources=["return 0.0;", "return 1.0;"]),
        {"u1": {"item_pos"}},
        {"u1": _user("u1", 1.0)},
        k=10,
        diversity_fields=("sector",),
    )
    assert result.eval_users == 1
    body = es.search_calls[0]["body"]
    # 2 sort scripts + 1 _score tie-break
    assert len(body["sort"]) == 3
    assert body["sort"][-1] == {"_score": "desc"}


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
                    "caused_by": {
                        "type": "class_cast_exception",
                        "reason": "Cannot cast from [double[]] to [java.util.List].",
                    },
                }
            ]
        }
    }


def test_evaluate_surfaces_painless_detail_in_sample_error() -> None:
    es = _FakeES({1.0: _ApiErrorLike(_painless_error_body())})
    with pytest.raises(RuntimeError, match="all users failed"):
        evaluate(
            es,
            ScriptSet(query_source="return 1.0;"),
            {"u1": {"x"}},
            {"u1": _user("u1", 1.0)},
            k=10,
            diversity_fields=("sector",),
        )


def test_evaluate_partial_failure_sample_error_includes_caused_by() -> None:
    responses: dict[float, Any] = {
        1.0: _hits([("item_pos", {"sector": "S"})]),
        2.0: _ApiErrorLike(_painless_error_body()),
    }
    es = _FakeES(responses)
    result = evaluate(
        es,
        ScriptSet(query_source="return 1.0;"),
        {"user_ok": {"item_pos"}, "user_bad": {"item_pos"}},
        {"user_ok": _user("user_ok", 1.0), "user_bad": _user("user_bad", 2.0)},
        k=10,
        diversity_fields=("sector",),
    )
    assert result.sample_error is not None
    assert "Cannot cast from [double[]] to [java.util.List]." in result.sample_error


def test_evaluate_counts_users_missing_from_users_by_id_as_failed() -> None:
    es = _FakeES({1.0: _hits([("y", {"sector": "S"})])})
    result = evaluate(
        es,
        ScriptSet(query_source="return 1.0;"),
        {"u_missing": {"x"}, "u_ok": {"y"}},
        {"u_ok": _user("u_ok", 1.0)},
        k=10,
        diversity_fields=("sector",),
    )
    assert result.eval_users == 1
    assert result.failed_users == 1
    assert result.partial_failure is True
    assert result.sample_error is not None
    assert "u_missing" in result.sample_error
