"""Tests for the multi-script query builder.

Snapshot-style tests pin the harness-owned JSON shape: any divergence between
the builder output and the committed fixture is a change to the contract that
the agent depends on, and must be reviewed deliberately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from es_script_agent.eval.query import MAX_SORT_SCRIPTS, build_query


FIXTURES = Path(__file__).parent / "fixtures"


def _deterministic_user_vector() -> list[list[float]]:
    """Return a stable 10x32 user_vector used across snapshot tests."""
    return [[round(i + j / 100, 2) for j in range(32)] for i in range(10)]


# --- Snapshot tests ------------------------------------------------------


def test_basic_shape_matches_snapshot() -> None:
    out = build_query(
        query_source="return 1.0;",
        sort_sources=[],
        user_vector=_deterministic_user_vector(),
    )
    expected = json.loads((FIXTURES / "query_snapshot_basic.json").read_text())
    assert out == expected


def test_with_sort_matches_snapshot() -> None:
    out = build_query(
        query_source="return 1.0;",
        sort_sources=[
            "return doc['popularityScore'].value;",
            "return doc['fundedAmount'].value;",
        ],
        user_vector=_deterministic_user_vector(),
    )
    expected = json.loads((FIXTURES / "query_snapshot_with_sort.json").read_text())
    assert out == expected


def test_query_block_invariant_between_snapshots() -> None:
    """The query block must be byte-identical across iterations; only sort changes."""
    basic = json.loads((FIXTURES / "query_snapshot_basic.json").read_text())
    with_sort = json.loads((FIXTURES / "query_snapshot_with_sort.json").read_text())
    assert json.dumps(basic["query"], sort_keys=True) == json.dumps(
        with_sort["query"], sort_keys=True
    )
    assert basic["sort"] != with_sort["sort"]


# --- Parameter propagation ----------------------------------------------


def test_user_vector_injected_into_all_script_params() -> None:
    user_vector = _deterministic_user_vector()
    out = build_query(
        query_source="return 1.0;",
        sort_sources=["return 0.0;", "return _score;"],
        user_vector=user_vector,
    )
    assert out["query"]["script_score"]["script"]["params"]["user_vector"] == user_vector
    sort_clauses = out["sort"][:-1]  # last entry is _score tie-break
    assert len(sort_clauses) == 2
    for clause in sort_clauses:
        assert clause["_script"]["script"]["params"]["user_vector"] == user_vector


def test_extra_params_merged_into_every_script() -> None:
    out = build_query(
        query_source="return 1.0;",
        sort_sources=["return 0.0;"],
        user_vector=_deterministic_user_vector(),
        extra_params={"foo": 1},
    )
    assert out["query"]["script_score"]["script"]["params"]["foo"] == 1
    assert out["sort"][0]["_script"]["script"]["params"]["foo"] == 1


def test_extra_params_cannot_shadow_user_vector() -> None:
    with pytest.raises(ValueError, match="user_vector"):
        build_query(
            query_source="return 1.0;",
            sort_sources=[],
            user_vector=_deterministic_user_vector(),
            extra_params={"user_vector": [[0.0] * 32] * 10},
        )


# --- Size and shape options ---------------------------------------------


def test_size_propagates() -> None:
    out = build_query(
        query_source="return 1.0;",
        sort_sources=[],
        user_vector=_deterministic_user_vector(),
        size=5,
    )
    assert out["size"] == 5


def test_empty_sort_yields_score_tiebreak_only() -> None:
    out = build_query(
        query_source="return 1.0;",
        sort_sources=[],
        user_vector=_deterministic_user_vector(),
    )
    assert out["sort"] == [{"_score": "desc"}]


# --- Validation ---------------------------------------------------------


def test_max_sort_scripts_allowed() -> None:
    out = build_query(
        query_source="return 1.0;",
        sort_sources=["return 0.0;"] * MAX_SORT_SCRIPTS,
        user_vector=_deterministic_user_vector(),
    )
    assert len(out["sort"]) == MAX_SORT_SCRIPTS + 1  # + _score tie-break


def test_exceeding_max_sort_scripts_raises() -> None:
    with pytest.raises(ValueError, match="sort"):
        build_query(
            query_source="return 1.0;",
            sort_sources=["return 0.0;"] * (MAX_SORT_SCRIPTS + 1),
            user_vector=_deterministic_user_vector(),
        )


def test_empty_query_source_raises() -> None:
    with pytest.raises(ValueError, match="query_source"):
        build_query(
            query_source="   ",
            sort_sources=[],
            user_vector=_deterministic_user_vector(),
        )
