"""Tests for IR metrics: NDCG@K, Recall@K, Precision@K, ILD@K, and aggregate."""

from __future__ import annotations

import math

import pytest

from es_script_agent.eval.metrics import (
    aggregate,
    ild_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# --- NDCG@K --------------------------------------------------------------


def test_ndcg_perfect_ranking_is_one() -> None:
    ranked = ["a", "b", "c", "d", "e"]
    relevant = {"a", "b", "c"}
    assert ndcg_at_k(ranked, relevant, k=5) == 1.0


def test_ndcg_no_relevant_in_topk_is_zero() -> None:
    ranked = ["x", "y", "z"]
    relevant = {"a", "b"}
    assert ndcg_at_k(ranked, relevant, k=3) == 0.0


def test_ndcg_known_value() -> None:
    # Relevant at positions 1 and 3 (0-indexed: 0 and 2).
    # DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 1/log2(3)
    ranked = ["a", "x", "b", "y"]
    relevant = {"a", "b"}
    expected_dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    expected_idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k(ranked, relevant, k=4) == expected_dcg / expected_idcg


def test_ndcg_k_larger_than_ranked() -> None:
    # K exceeds the ranked list length — should not raise.
    ranked = ["a", "b"]
    relevant = {"a", "b"}
    assert ndcg_at_k(ranked, relevant, k=10) == 1.0


def test_ndcg_empty_ranked_is_zero() -> None:
    assert ndcg_at_k([], {"a"}, k=10) == 0.0


def test_ndcg_truncates_to_k() -> None:
    # Relevant item exists past position k — must not contribute.
    ranked = ["x", "y", "z", "a"]
    relevant = {"a"}
    assert ndcg_at_k(ranked, relevant, k=3) == 0.0


def test_ndcg_ties_preserve_input_order() -> None:
    # Caller decides the order. NDCG must not reorder ties.
    ranked = ["x", "a", "y", "b"]  # relevant items at positions 1 and 3.
    relevant = {"a", "b"}
    expected_dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    expected_idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k(ranked, relevant, k=4) == expected_dcg / expected_idcg


# --- Recall@K ------------------------------------------------------------


def test_recall_all_retrieved_is_one() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_recall_none_retrieved_is_zero() -> None:
    assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0


def test_recall_half_retrieved() -> None:
    assert recall_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5


def test_recall_k_larger_than_ranked() -> None:
    assert recall_at_k(["a"], {"a", "b"}, k=10) == 0.5


def test_recall_truncates_to_k() -> None:
    # Only positions < k count.
    assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


# --- Precision@K ---------------------------------------------------------


def test_precision_all_relevant() -> None:
    assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_precision_none_relevant() -> None:
    assert precision_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_precision_partial() -> None:
    assert precision_at_k(["a", "x", "b"], {"a", "b"}, k=3) == 2 / 3


def test_precision_k_larger_than_ranked() -> None:
    # Denominator caps at min(k, len(ranked)) — otherwise short result lists
    # are unfairly penalised.
    assert precision_at_k(["a"], {"a", "b"}, k=10) == 1.0


# --- ILD@K ---------------------------------------------------------------


def test_ild_all_identical_is_zero() -> None:
    items = [
        {"sector": "Agriculture", "country": "KE"},
        {"sector": "Agriculture", "country": "KE"},
        {"sector": "Agriculture", "country": "KE"},
    ]
    assert ild_at_k(items, fields=("sector", "country"), k=3) == 0.0


def test_ild_all_distinct_is_one() -> None:
    items = [
        {"sector": "Agriculture", "country": "KE"},
        {"sector": "Retail", "country": "PH"},
        {"sector": "Services", "country": "PE"},
    ]
    assert ild_at_k(items, fields=("sector", "country"), k=3) == 1.0


def test_ild_mixed_handcomputed() -> None:
    # Three items, two fields. Pair distances:
    #   (0,1): sector differs, country differs → 2/2 = 1.0
    #   (0,2): sector matches, country differs → 1/2 = 0.5
    #   (1,2): sector differs, country differs → 2/2 = 1.0
    # Mean = (1.0 + 0.5 + 1.0) / 3 = 2.5/3
    items = [
        {"sector": "Agriculture", "country": "KE"},
        {"sector": "Retail", "country": "PH"},
        {"sector": "Agriculture", "country": "PE"},
    ]
    assert ild_at_k(items, fields=("sector", "country"), k=3) == 2.5 / 3


def test_ild_k_one_is_none() -> None:
    items = [{"sector": "Agriculture"}]
    assert ild_at_k(items, fields=("sector",), k=1) is None


def test_ild_k_zero_is_none() -> None:
    assert ild_at_k([], fields=("sector",), k=0) is None


def test_ild_truncates_to_k() -> None:
    # Distinct items past position k must not contribute.
    items = [
        {"sector": "Agriculture"},
        {"sector": "Agriculture"},
        {"sector": "Retail"},  # ignored at k=2
    ]
    assert ild_at_k(items, fields=("sector",), k=2) == 0.0


def test_ild_missing_field_is_its_own_category() -> None:
    # None == None → equal; None != "foo" → different.
    items = [
        {"sector": None},
        {"sector": None},
        {"sector": "Retail"},
    ]
    # Pair distances on the single "sector" field:
    #   (0,1): both None → 0
    #   (0,2): None vs "Retail" → 1
    #   (1,2): None vs "Retail" → 1
    # Mean = 2/3
    assert ild_at_k(items, fields=("sector",), k=3) == 2 / 3


def test_ild_missing_key_treated_same_as_none() -> None:
    items = [
        {},
        {"sector": None},
        {"sector": "Retail"},
    ]
    assert ild_at_k(items, fields=("sector",), k=3) == 2 / 3


def test_ild_k_larger_than_items() -> None:
    items = [
        {"sector": "Agriculture"},
        {"sector": "Retail"},
    ]
    assert ild_at_k(items, fields=("sector",), k=10) == 1.0


# --- aggregate -----------------------------------------------------------


def test_aggregate_averages_each_key() -> None:
    per_user = [
        {"ndcg@10": 1.0, "recall@10": 1.0, "precision@10": 1.0, "ild@10": 0.5},
        {"ndcg@10": 0.0, "recall@10": 0.0, "precision@10": 0.0, "ild@10": 0.7},
    ]
    result = aggregate(per_user)
    assert result["ndcg@10"] == 0.5
    assert result["recall@10"] == 0.5
    assert result["precision@10"] == 0.5
    assert result["ild@10"] == 0.6


def test_aggregate_skips_none_ild_values() -> None:
    per_user = [
        {"ndcg@10": 1.0, "ild@10": None},
        {"ndcg@10": 0.5, "ild@10": 0.8},
        {"ndcg@10": 0.0, "ild@10": 0.4},
    ]
    result = aggregate(per_user)
    assert result["ndcg@10"] == 0.5
    # Mean of 0.8 and 0.4 over two contributors, not three.
    assert result["ild@10"] == pytest.approx(0.6)


def test_aggregate_all_none_ild() -> None:
    per_user = [
        {"ndcg@10": 1.0, "ild@10": None},
        {"ndcg@10": 0.0, "ild@10": None},
    ]
    result = aggregate(per_user)
    assert result["ndcg@10"] == 0.5
    assert result["ild@10"] is None


def test_aggregate_empty_input() -> None:
    assert aggregate([]) == {}
