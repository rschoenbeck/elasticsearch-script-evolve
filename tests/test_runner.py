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
from es_script_agent.eval.runner import (
    EvalResult,
    ScriptSet,
    evaluate,
    load_script_set,
)


# --- Fakes --------------------------------------------------------------


class _FakeES:
    """Records calls and returns canned hits keyed by the user_vector param.

    The fake honours the harness contract: it reads ``params.user_vector``
    out of the request body and looks up the canned response from the
    constructor map. Tests can therefore control per-user output without
    inspecting Painless bodies.
    """

    def __init__(self, responses_by_user_id: dict[str, Any]) -> None:
        self._responses = responses_by_user_id
        self.search_calls: list[dict[str, Any]] = []
        self.non_search_calls: list[str] = []

    def search(self, **kwargs: Any) -> Any:
        self.search_calls.append(kwargs)
        body = kwargs.get("body", kwargs)
        params = body["query"]["script_score"]["script"]["params"]
        user_id = params.get("__user_id_for_test")
        response = self._responses[user_id]
        if isinstance(response, Exception):
            raise response
        return response

    def __getattr__(self, name: str) -> Any:
        # Any other API access is a contract violation; record + fail loudly.
        self.non_search_calls.append(name)
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


def _user(user_id: str) -> User:
    return User(id=user_id, vector=[[0.0] * 4] * 10, attributes={})


# --- ScriptSet + load_script_set ---------------------------------------


def test_script_set_default_has_no_sort_sources() -> None:
    s = ScriptSet(query_source="return 1.0;")
    assert s.sort_sources == []


def test_script_set_rejects_blank_query_source() -> None:
    with pytest.raises(ValueError):
        ScriptSet(query_source="   ")


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
        "user_a": _hits(
            [
                ("item_pos", {"sector": "S", "country": "C"}),
                ("item_x", {"sector": "T", "country": "D"}),
            ]
        ),
        "user_b": _hits(
            [
                ("item_x", {"sector": "S", "country": "C"}),
                ("item_y", {"sector": "T", "country": "D"}),
                ("item_pos_b", {"sector": "S", "country": "D"}),
            ]
        ),
    }
    es = _FakeES(responses)
    ground_truth = {"user_a": {"item_pos"}, "user_b": {"item_pos_b"}}
    users_by_id = {"user_a": _user("user_a"), "user_b": _user("user_b")}

    result = evaluate(
        es,
        ScriptSet(query_source="return 1.0;"),
        ground_truth,
        users_by_id,
        k=10,
        diversity_fields=("sector", "country"),
        extra_params_per_user=lambda uid: {"__user_id_for_test": uid},
    )

    assert isinstance(result, EvalResult)
    assert result.eval_users == 2
    assert result.failed_users == 0
    assert result.partial_failure is False
    assert result.sample_error is None
    assert "ndcg@10" in result.metrics
    assert "ild@10" in result.metrics
    # Per-user NDCG@10 = 1.0 for user_a, 0.5 for user_b (log2(4)/log2(2)*1)
    # mean = 0.75
    assert result.metrics["ndcg@10"] == pytest.approx(0.75, abs=1e-9)
    # Diversity-field _source includes propagated into every search.
    for call in es.search_calls:
        body = call.get("body", call)
        includes = body["_source"]["includes"]
        assert set(includes) == {"sector", "country"}


def test_evaluate_partial_failure_keeps_running() -> None:
    responses: dict[str, Any] = {
        "user_ok": _hits([("item_pos", {"sector": "S"})]),
        "user_bad": RuntimeError("script_exception: bad bad bad"),
    }
    es = _FakeES(responses)
    ground_truth = {"user_ok": {"item_pos"}, "user_bad": {"item_pos"}}
    users_by_id = {"user_ok": _user("user_ok"), "user_bad": _user("user_bad")}

    result = evaluate(
        es,
        ScriptSet(query_source="return 1.0;"),
        ground_truth,
        users_by_id,
        k=10,
        diversity_fields=("sector",),
        extra_params_per_user=lambda uid: {"__user_id_for_test": uid},
    )
    assert result.eval_users == 1
    assert result.failed_users == 1
    assert result.partial_failure is True
    assert result.sample_error is not None
    assert "bad bad bad" in result.sample_error


def test_evaluate_raises_when_every_user_fails() -> None:
    responses: dict[str, Any] = {
        "u1": RuntimeError("boom"),
        "u2": RuntimeError("boom"),
    }
    es = _FakeES(responses)
    ground_truth = {"u1": {"x"}, "u2": {"y"}}
    users_by_id = {"u1": _user("u1"), "u2": _user("u2")}

    with pytest.raises(RuntimeError, match="all users failed"):
        evaluate(
            es,
            ScriptSet(query_source="return 1.0;"),
            ground_truth,
            users_by_id,
            k=10,
            diversity_fields=("sector",),
            extra_params_per_user=lambda uid: {"__user_id_for_test": uid},
        )


def test_evaluate_with_sort_scripts_calls_es_with_sort_clauses() -> None:
    responses = {
        "u1": _hits([("item_pos", {"sector": "S"})]),
    }
    es = _FakeES(responses)
    result = evaluate(
        es,
        ScriptSet(query_source="return 1.0;", sort_sources=["return 0.0;", "return 1.0;"]),
        {"u1": {"item_pos"}},
        {"u1": _user("u1")},
        k=10,
        diversity_fields=("sector",),
        extra_params_per_user=lambda uid: {"__user_id_for_test": uid},
    )
    assert result.eval_users == 1
    body = es.search_calls[0].get("body", es.search_calls[0])
    # 2 sort scripts + 1 _score tie-break
    assert len(body["sort"]) == 3
    assert body["sort"][-1] == {"_score": "desc"}


def test_evaluate_skips_users_missing_from_users_by_id() -> None:
    """A ground-truth user with no vector entry cannot be evaluated; skip + count as failed."""
    es = _FakeES({"u_ok": _hits([("y", {"sector": "S"})])})
    result = evaluate(
        es,
        ScriptSet(query_source="return 1.0;"),
        {"u_missing": {"x"}, "u_ok": {"y"}},
        {"u_ok": _user("u_ok")},
        k=10,
        diversity_fields=("sector",),
        extra_params_per_user=lambda uid: {"__user_id_for_test": uid},
    )
    assert result.eval_users == 1
    assert result.failed_users == 1
