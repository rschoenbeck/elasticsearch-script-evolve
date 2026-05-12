"""Ground-truth construction from raw Interaction signals (Task 8)."""

from __future__ import annotations

from es_script_agent import config
from es_script_agent.data.schema import Interaction
from es_script_agent.eval.interactions import build_ground_truth


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