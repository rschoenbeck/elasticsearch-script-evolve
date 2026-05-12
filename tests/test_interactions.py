"""Ground-truth construction from raw Interaction signals (Task 8)."""

from __future__ import annotations

import pytest

from es_script_agent import config
from es_script_agent.data.schema import Interaction
from es_script_agent.eval.interactions import (
    FilterStats,
    build_ground_truth,
    filter_to_indexed_items,
)


def _ix(user_id: str, item_id: str, weight: float) -> Interaction:
    return Interaction(user_id=user_id, item_id=item_id, weight=weight)


def test_threshold_includes_pairs_at_or_above_threshold() -> None:
    interactions = [
        _ix("u1", "i1", 1.0),
        _ix("u1", "i2", 2.0),
    ]
    gt = build_ground_truth(interactions, threshold=1.0)
    assert gt == {"u1": {"i1", "i2"}}


def test_threshold_excludes_pairs_below_threshold() -> None:
    interactions = [
        _ix("u1", "i1", 0.0),
        _ix("u1", "i2", 0.5),
        _ix("u1", "i3", 1.0),
    ]
    gt = build_ground_truth(interactions, threshold=1.0)
    assert gt == {"u1": {"i3"}}


def test_duplicate_pairs_have_weights_summed_then_thresholded() -> None:
    # Two sub-threshold rows that sum to clear the threshold should count.
    interactions = [
        _ix("u1", "i1", 0.4),
        _ix("u1", "i1", 0.7),  # 0.4 + 0.7 = 1.1 >= 1.0
    ]
    gt = build_ground_truth(interactions, threshold=1.0)
    assert gt == {"u1": {"i1"}}


def test_duplicate_pairs_summing_below_threshold_are_excluded() -> None:
    interactions = [
        _ix("u1", "i1", 0.3),
        _ix("u1", "i1", 0.4),  # 0.7 < 1.0
    ]
    gt = build_ground_truth(interactions, threshold=1.0)
    assert gt == {}


def test_user_with_no_passing_items_is_dropped() -> None:
    interactions = [
        _ix("u1", "i1", 1.0),
        _ix("u2", "i1", 0.0),  # u2 has no positives → dropped
        _ix("u2", "i2", 0.0),
    ]
    gt = build_ground_truth(interactions, threshold=1.0)
    assert "u2" not in gt
    assert gt == {"u1": {"i1"}}


def test_returns_sets_so_duplicate_items_collapse() -> None:
    interactions = [
        _ix("u1", "i1", 1.0),
        _ix("u1", "i1", 1.0),  # same pair: still one entry in set
    ]
    gt = build_ground_truth(interactions, threshold=1.0)
    assert gt == {"u1": {"i1"}}


def test_threshold_defaults_to_config_relevance_threshold() -> None:
    # With config default (1), a weight of 1 should clear.
    interactions = [_ix("u1", "i1", config.RELEVANCE_THRESHOLD)]
    gt = build_ground_truth(interactions)
    assert gt == {"u1": {"i1"}}


def test_empty_input_returns_empty_dict() -> None:
    assert build_ground_truth([], threshold=1.0) == {}


def test_accepts_iterable_not_just_list() -> None:
    def gen():
        yield _ix("u1", "i1", 1.0)
        yield _ix("u2", "i2", 2.0)

    gt = build_ground_truth(gen(), threshold=1.0)
    assert gt == {"u1": {"i1"}, "u2": {"i2"}}


# --- filter_to_indexed_items (Task 8a) -------------------------------------


def test_filter_noop_when_every_positive_is_indexed() -> None:
    gt = {"u1": {"i1", "i2"}, "u2": {"i3"}}
    indexed = {"i1", "i2", "i3", "i4"}
    filtered, stats = filter_to_indexed_items(gt, indexed)
    assert filtered == gt
    assert filtered is not gt  # returns a new dict, doesn't mutate the input
    assert stats == FilterStats(
        users_before=2, users_after=2, positives_before=3, positives_after=3
    )
    assert stats.positives_dropped_pct == 0.0


def test_filter_drops_some_positives_but_keeps_user() -> None:
    gt = {"u1": {"i1", "i2", "i3"}}
    indexed = {"i1", "i2"}
    filtered, stats = filter_to_indexed_items(gt, indexed)
    assert filtered == {"u1": {"i1", "i2"}}
    assert stats.users_before == 1
    assert stats.users_after == 1
    assert stats.positives_before == 3
    assert stats.positives_after == 2
    assert stats.positives_dropped_pct == pytest.approx(100 / 3)


def test_filter_drops_user_whose_positives_all_become_empty() -> None:
    gt = {"u1": {"i1", "i2"}, "u2": {"i3"}}
    indexed = {"i1", "i2"}  # u2's only positive is unindexed
    filtered, stats = filter_to_indexed_items(gt, indexed)
    assert filtered == {"u1": {"i1", "i2"}}
    assert "u2" not in filtered
    assert stats == FilterStats(
        users_before=2, users_after=1, positives_before=3, positives_after=2
    )


def test_filter_with_empty_indexed_ids_drops_everything() -> None:
    gt = {"u1": {"i1"}, "u2": {"i2", "i3"}}
    filtered, stats = filter_to_indexed_items(gt, set())
    assert filtered == {}
    assert stats == FilterStats(
        users_before=2, users_after=0, positives_before=3, positives_after=0
    )
    assert stats.positives_dropped_pct == 100.0


def test_filter_empty_ground_truth_yields_zero_stats() -> None:
    filtered, stats = filter_to_indexed_items({}, {"i1", "i2"})
    assert filtered == {}
    assert stats == FilterStats(
        users_before=0, users_after=0, positives_before=0, positives_after=0
    )
    # No positives to drop → defined as 0.0 (avoids divide-by-zero).
    assert stats.positives_dropped_pct == 0.0


def test_filter_stats_arithmetic_hand_checked() -> None:
    # 3 users, 7 positives total; 4 survive → 3 dropped → ~42.857% drop.
    gt = {
        "u1": {"i1", "i2", "i3"},  # keep i1, i2; drop i3
        "u2": {"i4", "i5"},  # keep i4; drop i5
        "u3": {"i6", "i7"},  # drop both → user removed
    }
    indexed = {"i1", "i2", "i4"}
    filtered, stats = filter_to_indexed_items(gt, indexed)
    assert filtered == {"u1": {"i1", "i2"}, "u2": {"i4"}}
    assert stats.users_before == 3
    assert stats.users_after == 2
    assert stats.positives_before == 7
    assert stats.positives_after == 3
    assert stats.positives_dropped_pct == pytest.approx(100 * 4 / 7)