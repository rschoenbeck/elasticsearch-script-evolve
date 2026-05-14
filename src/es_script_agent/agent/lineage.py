"""Pure parent-selection logic for the evolutionary lineage strategy.

Treats a run's :class:`IterationRecord` list as a growing archive. A
parent for the next iteration is sampled from past iterations
proportional to ``fitness * 1/(1+children)`` — the simplest weighting
that combines a fitness signal with an offspring-count penalty without
requiring extra knobs (alpha, novelty, lineage depth). The penalty does
the diversification work when fitness values cluster tightly, which is
the common case once a run has been running for a while.

Nothing in this module touches Elasticsearch, the LLM, the clock, or
the filesystem. The CLI/ToolContext wiring in Task 20 is responsible
for assembling the seeded RNG and the records list; the algorithm
itself stays a pure function so it can be reasoned about and tested in
isolation.
"""

from __future__ import annotations

import logging
import random
from collections import Counter

from es_script_agent.runlog import IterationRecord, check_guardrail

logger = logging.getLogger(__name__)


def _is_eligible(
    record: IterationRecord,
    baseline_metrics: dict[str, float | None],
    primary_key: str,
    guardrail_key: str,
) -> bool:
    if record.compile_error is not None or record.metrics is None:
        return False
    if record.metrics.get(primary_key) is None:
        return False
    return check_guardrail(record.metrics, guardrail_key, baseline_metrics)


def select_parent_evolutionary(
    records: list[IterationRecord],
    baseline_metrics: dict[str, float | None],
    primary_key: str,
    guardrail_key: str,
    rng: random.Random,
) -> int:
    """Sample a parent iteration from the archive by fitness × offspring penalty.

    Args:
        records: Every iteration written to the run log so far,
            baseline first.
        baseline_metrics: Metrics dict from ``iter_000`` — the guardrail
            floor.
        primary_key: Metric key the run is optimizing
            (``"ndcg@10"`` or ``"ild@10"``).
        guardrail_key: The other metric in the objective pair; serves
            as the eligibility floor.
        rng: Seeded ``random.Random`` from the CLI entry. The only
            source of nondeterminism in this function.

    Returns:
        The ``iter`` field of the chosen parent record.

    Notes:
        Empty-pool fallback: when every record is filtered out (e.g.
        every non-baseline iteration compile-failed or busted the
        guardrail, and the baseline itself is somehow ineligible too),
        returns ``0`` and logs at WARNING. The fallback is reachable
        in practice during pathological runs and silent failure here
        would make a degrading run hard to diagnose; cf.
        ``assert_baseline_eligible`` for the pre-run guard.
    """
    children_count = Counter(
        r.parent_iter for r in records if r.parent_iter is not None
    )

    weights: list[float] = []
    for r in records:
        if not _is_eligible(r, baseline_metrics, primary_key, guardrail_key):
            weights.append(0.0)
            continue
        fitness = r.metrics[primary_key]  # type: ignore[index]
        assert fitness is not None  # guarded by _is_eligible
        weights.append(fitness / (1.0 + children_count[r.iter]))

    if sum(weights) <= 0.0:
        logger.warning(
            "lineage: empty candidate pool — falling back to baseline (iter 0)"
        )
        return 0

    chosen = rng.choices(records, weights=weights, k=1)[0]

    eligible = [(records[i].iter, weights[i]) for i in range(len(records)) if weights[i] > 0]
    top3 = sorted(eligible, key=lambda t: t[1], reverse=True)[:3]
    logger.info(
        "lineage: chose parent_iter=%d; top3 candidates=%s",
        chosen.iter,
        top3,
    )
    return chosen.iter


def assert_baseline_eligible(
    records: list[IterationRecord],
    baseline_metrics: dict[str, float | None],
    guardrail_key: str,
) -> None:
    """Raise if the baseline cannot serve as a fallback parent.

    Called once at CLI entry (after the baseline record is appended,
    before the agent starts) under ``--lineage evolutionary``. The
    empty-pool fallback in :func:`select_parent_evolutionary` returns
    ``0`` unconditionally, so a corrupt or guardrail-busting baseline
    would silently become every iteration's parent. Fail loudly here
    instead.

    Args:
        records: The run log so far. Expects at least one row.
        baseline_metrics: The metrics dict from ``iter_000``.
        guardrail_key: The objective's guardrail metric key.

    Raises:
        ValueError: If the baseline record is absent, lacks metrics,
            has a compile error, or fails the guardrail.
    """
    if not records:
        raise ValueError("baseline record missing — cannot start evolutionary lineage")
    baseline = records[0]
    if baseline.compile_error is not None or baseline.metrics is None:
        raise ValueError(
            f"baseline (iter {baseline.iter}) has no metrics — cannot start "
            f"evolutionary lineage; compile_error={baseline.compile_error!r}"
        )
    if not check_guardrail(baseline.metrics, guardrail_key, baseline_metrics):
        raise ValueError(
            f"baseline (iter {baseline.iter}) fails its own guardrail "
            f"({guardrail_key}); cannot start evolutionary lineage"
        )
