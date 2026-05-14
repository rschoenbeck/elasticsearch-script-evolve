"""Tests for the pure parent-selection module ``agent.lineage``.

The lineage module is the entire algorithmic surface of the evolutionary
strategy; the CLI/ToolContext wiring in Task 20 only routes inputs and
outputs. So everything that could plausibly go wrong with parent picks
— filters, offspring penalty, RNG determinism, fallback behavior — is
covered here against the pure function, with no ES, no LLM, and no
disk.
"""

from __future__ import annotations

import logging
import random

import pytest

from es_script_agent.agent.lineage import (
    assert_baseline_eligible,
    select_parent_evolutionary,
)
from es_script_agent.runlog import IterationRecord


def _record(
    iter_n: int,
    *,
    primary: float | None = 0.5,
    guardrail: float | None = 0.4,
    parent_iter: int | None = None,
    compile_error: str | None = None,
    metrics_none: bool = False,
    primary_key: str = "ndcg@10",
    guardrail_key: str = "ild@10",
) -> IterationRecord:
    metrics: dict[str, float | None] | None
    if metrics_none:
        metrics = None
    else:
        metrics = {primary_key: primary, guardrail_key: guardrail}
    return IterationRecord(
        iter=iter_n,
        timestamp="2026-05-14T12:00:00Z",
        query_script_path=f"runs/x/iter_{iter_n:03d}/query.painless",
        sort_script_paths=[],
        metrics=metrics,
        eval_users=100,
        eval_seconds=1.0,
        llm_rationale=None,
        parent_iter=parent_iter,
        compile_error=compile_error,
        partial_failure=False,
        sample_error=None,
    )


PRIMARY = "ndcg@10"
GUARDRAIL = "ild@10"
BASELINE_METRICS: dict[str, float | None] = {PRIMARY: 0.4, GUARDRAIL: 0.3}


# --- select_parent_evolutionary ----------------------------------------


def test_single_candidate_pool_returns_baseline() -> None:
    records = [_record(0, primary=0.4, guardrail=0.3)]
    rng = random.Random(123)
    assert select_parent_evolutionary(records, BASELINE_METRICS, PRIMARY, GUARDRAIL, rng) == 0


def test_deterministic_with_same_seed() -> None:
    records = [
        _record(0, primary=0.4, guardrail=0.3),
        _record(1, primary=0.5, guardrail=0.4, parent_iter=0),
        _record(2, primary=0.55, guardrail=0.35, parent_iter=0),
        _record(3, primary=0.6, guardrail=0.4, parent_iter=1),
    ]
    a = select_parent_evolutionary(
        records, BASELINE_METRICS, PRIMARY, GUARDRAIL, random.Random(42)
    )
    b = select_parent_evolutionary(
        records, BASELINE_METRICS, PRIMARY, GUARDRAIL, random.Random(42)
    )
    assert a == b
    # Across a small seed grid, the pick should change at least once —
    # otherwise the RNG isn't actually being consulted.
    picks = {
        select_parent_evolutionary(
            records, BASELINE_METRICS, PRIMARY, GUARDRAIL, random.Random(s)
        )
        for s in range(20)
    }
    assert len(picks) > 1


def test_offspring_penalty_penalizes_over_exploited_parent() -> None:
    # Two equally fit candidates; one already has 5 children. The
    # heavily-parented one should be picked strictly less often.
    records = [
        _record(0, primary=0.4, guardrail=0.3),
        _record(1, primary=0.6, guardrail=0.4, parent_iter=0),  # 0 children
        _record(2, primary=0.6, guardrail=0.4, parent_iter=0),  # 5 children
        # Pad children of iter 2:
        _record(3, primary=0.6, guardrail=0.4, parent_iter=2),
        _record(4, primary=0.6, guardrail=0.4, parent_iter=2),
        _record(5, primary=0.6, guardrail=0.4, parent_iter=2),
        _record(6, primary=0.6, guardrail=0.4, parent_iter=2),
        _record(7, primary=0.6, guardrail=0.4, parent_iter=2),
    ]
    rng = random.Random(0)
    trials = 1000
    picks_of_2 = sum(
        1
        for _ in range(trials)
        if select_parent_evolutionary(records, BASELINE_METRICS, PRIMARY, GUARDRAIL, rng) == 2
    )
    # iter 2 has 5 children → weight 0.6 * 1/6 ≈ 0.10
    # iter 1 has 0 children → weight 0.6 * 1/1 = 0.60
    # Expected share for iter 2 is well under 35% in a large sample.
    assert picks_of_2 < trials * 0.35


def test_guardrail_filter_excludes_high_primary_low_guardrail() -> None:
    # A candidate with a primary value above all others but a guardrail
    # below baseline must never be picked.
    records = [
        _record(0, primary=0.4, guardrail=0.3),
        _record(1, primary=0.5, guardrail=0.35, parent_iter=0),
        _record(2, primary=0.99, guardrail=0.05, parent_iter=0),  # busts guardrail
    ]
    rng = random.Random(0)
    picks = [
        select_parent_evolutionary(records, BASELINE_METRICS, PRIMARY, GUARDRAIL, rng)
        for _ in range(200)
    ]
    assert 2 not in picks


def test_compile_failure_and_metrics_none_are_never_picked() -> None:
    records = [
        _record(0, primary=0.4, guardrail=0.3),
        _record(1, compile_error="boom", metrics_none=True, parent_iter=0),
        _record(2, metrics_none=True, parent_iter=0),
        _record(3, primary=0.5, guardrail=0.4, parent_iter=0),
    ]
    rng = random.Random(0)
    picks = {
        select_parent_evolutionary(records, BASELINE_METRICS, PRIMARY, GUARDRAIL, rng)
        for _ in range(200)
    }
    assert picks.issubset({0, 3})


def test_empty_pool_falls_back_to_baseline_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Every non-baseline record is ineligible, and the baseline itself
    # is also filtered (its guardrail value is below the recorded
    # baseline_metrics — a synthetic scenario, but it's exactly the
    # all-zero-weight case the fallback exists for).
    baseline = {PRIMARY: 1.0, GUARDRAIL: 1.0}
    records = [
        _record(0, primary=0.0, guardrail=0.0),
        _record(1, compile_error="boom", metrics_none=True, parent_iter=0),
        _record(2, primary=0.99, guardrail=0.05, parent_iter=0),  # busts guardrail
    ]
    with caplog.at_level(logging.WARNING, logger="es_script_agent.agent.lineage"):
        chosen = select_parent_evolutionary(
            records, baseline, PRIMARY, GUARDRAIL, random.Random(0)
        )
    assert chosen == 0
    assert any("empty" in rec.message.lower() or "fallback" in rec.message.lower()
               for rec in caplog.records)


def test_equal_weights_are_uniform_within_ci() -> None:
    # Three eligible candidates with equal fitness and zero children
    # each (no record carries a parent_iter, so no one accrues offspring
    # penalty). Expected share each is 1/3; trials × 0.05 margin keeps
    # the test well outside binomial flake territory.
    records = [
        _record(0, primary=0.5, guardrail=0.4),
        _record(1, primary=0.5, guardrail=0.4),
        _record(2, primary=0.5, guardrail=0.4),
    ]
    rng = random.Random(0)
    trials = 1500
    counts = {0: 0, 1: 0, 2: 0}
    for _ in range(trials):
        counts[
            select_parent_evolutionary(records, BASELINE_METRICS, PRIMARY, GUARDRAIL, rng)
        ] += 1
    expected = trials / 3
    margin = trials * 0.05
    for c in counts.values():
        assert abs(c - expected) < margin


def test_children_count_includes_failed_records() -> None:
    # A compile-failed child still increments its parent's child count.
    # We verify by comparing two scenarios that differ only in whether
    # iter 1 has a compile-failed child (iter 2): iter 1 must be picked
    # less often in the scenario where it does.
    base = [
        _record(0, primary=0.4, guardrail=0.3),
        _record(1, primary=0.6, guardrail=0.4, parent_iter=0),
        _record(2, primary=0.6, guardrail=0.4, parent_iter=0),
    ]
    with_failed_child = [
        *base,
        _record(3, compile_error="boom", metrics_none=True, parent_iter=1),
    ]
    trials = 2000
    rng_a = random.Random(7)
    picks_1_no_failed = sum(
        1
        for _ in range(trials)
        if select_parent_evolutionary(base, BASELINE_METRICS, PRIMARY, GUARDRAIL, rng_a) == 1
    )
    rng_b = random.Random(7)
    picks_1_with_failed = sum(
        1
        for _ in range(trials)
        if select_parent_evolutionary(
            with_failed_child, BASELINE_METRICS, PRIMARY, GUARDRAIL, rng_b
        )
        == 1
    )
    assert picks_1_with_failed < picks_1_no_failed


# --- assert_baseline_eligible ------------------------------------------


def test_assert_baseline_eligible_passes_on_healthy_baseline() -> None:
    records = [_record(0, primary=0.4, guardrail=0.3)]
    # Should not raise.
    assert_baseline_eligible(records, BASELINE_METRICS, GUARDRAIL)


def test_assert_baseline_eligible_raises_on_compile_error() -> None:
    records = [_record(0, compile_error="boom", metrics_none=True)]
    with pytest.raises(ValueError, match="baseline"):
        assert_baseline_eligible(records, BASELINE_METRICS, GUARDRAIL)


def test_assert_baseline_eligible_raises_on_guardrail_bust() -> None:
    records = [_record(0, primary=0.4, guardrail=0.01)]
    with pytest.raises(ValueError, match="guardrail"):
        assert_baseline_eligible(records, BASELINE_METRICS, GUARDRAIL)


def test_assert_baseline_eligible_raises_on_metrics_none() -> None:
    records = [_record(0, metrics_none=True)]
    with pytest.raises(ValueError, match="baseline"):
        assert_baseline_eligible(records, BASELINE_METRICS, GUARDRAIL)


def test_assert_baseline_eligible_raises_on_empty_records() -> None:
    with pytest.raises(ValueError):
        assert_baseline_eligible([], BASELINE_METRICS, GUARDRAIL)
