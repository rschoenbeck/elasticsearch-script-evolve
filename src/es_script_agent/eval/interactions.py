"""Ground-truth construction from raw ``Interaction`` signals.

Binary relevance: an item is positive for a user when the user's summed
interaction weight for it meets ``threshold``. Users with no qualifying
items are dropped, since they cannot be evaluated.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from es_script_agent import config
from es_script_agent.data.schema import Interaction


@dataclass(frozen=True)
class FilterStats:
    """Summary of a ground-truth filter pass against an indexed-id set.

    Attributes:
        users_before: Distinct users in the input ground truth.
        users_after: Users remaining after filtering (positives non-empty).
        positives_before: Total ``(user, item)`` positives in the input.
        positives_after: Positives whose item id is in the indexed set.
    """

    users_before: int
    users_after: int
    positives_before: int
    positives_after: int

    @property
    def positives_dropped_pct(self) -> float:
        """Percentage of positives that referenced unindexed items.

        Defined as ``0.0`` when there were no positives to drop, so callers
        can format the value uniformly without guarding against divide-by-zero.
        """
        if self.positives_before == 0:
            return 0.0
        dropped = self.positives_before - self.positives_after
        return 100.0 * dropped / self.positives_before


def build_ground_truth(
    interactions: Iterable[Interaction],
    threshold: float = config.RELEVANCE_THRESHOLD,
) -> dict[str, set[str]]:
    """Collapse raw interactions into per-user positive item sets.

    Duplicate ``(user_id, item_id)`` pairs are summed before the
    threshold is applied, so multi-row signals aggregate naturally.

    Args:
        interactions: Raw signals yielded by a dataset adapter.
        threshold: Minimum summed weight for an item to count as
            positive. Defaults to ``config.RELEVANCE_THRESHOLD``.

    Returns:
        Mapping from user id to the set of item ids whose summed weight
        meets ``threshold``. Users with no qualifying items are omitted.
    """
    sums: dict[str, dict[str, float]] = defaultdict(dict)
    for ix in interactions:
        per_user = sums[ix.user_id]
        per_user[ix.item_id] = per_user.get(ix.item_id, 0.0) + ix.weight

    ground_truth: dict[str, set[str]] = {}
    for user_id, item_weights in sums.items():
        positives = {item for item, w in item_weights.items() if w >= threshold}
        if positives:
            ground_truth[user_id] = positives
    return ground_truth


def filter_to_indexed_items(
    ground_truth: dict[str, set[str]],
    indexed_ids: set[str],
) -> tuple[dict[str, set[str]], FilterStats]:
    """Drop ground-truth positives whose item id is absent from the index.

    The interaction export and the loan export don't share a time window,
    so the raw ground truth references item ids that the ``loans`` index
    doesn't contain. Those positives can never be retrieved and silently
    depress Recall/NDCG denominators; filtering them out keeps the metrics
    honest. Users whose positive set becomes empty are dropped, matching
    the contract of :func:`build_ground_truth`.

    Pure: no ES calls, no I/O. The caller is responsible for fetching
    ``indexed_ids``.

    Args:
        ground_truth: Per-user positive item sets to filter.
        indexed_ids: Item ids known to exist in the index.

    Returns:
        ``(filtered, stats)`` where ``filtered`` is a fresh dict (the input
        is not mutated) and ``stats`` captures before/after counts for
        logging into the run-log header.
    """
    filtered: dict[str, set[str]] = {}
    positives_before = 0
    positives_after = 0
    for user_id, positives in ground_truth.items():
        positives_before += len(positives)
        kept = positives & indexed_ids
        if kept:
            filtered[user_id] = kept
            positives_after += len(kept)
    stats = FilterStats(
        users_before=len(ground_truth),
        users_after=len(filtered),
        positives_before=positives_before,
        positives_after=positives_after,
    )
    return filtered, stats
