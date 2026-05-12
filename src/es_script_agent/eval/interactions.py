"""Ground-truth construction from raw ``Interaction`` signals.

Binary relevance: an item is positive for a user when the user's summed
interaction weight for it meets ``threshold``. Users with no qualifying
items are dropped, since they cannot be evaluated.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from es_script_agent import config
from es_script_agent.data.schema import Interaction


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
