"""Pure-function IR metrics: NDCG@K, Recall@K, Precision@K, ILD@K.

All ranking metrics consume a ``ranked_ids`` list (caller-determined order,
ties preserved) and a ``relevant_ids`` set (binary relevance). ILD consumes
the top-K item attribute dicts and a fixed field set, computing pairwise
field-equality distance and averaging over all unordered pairs.

Aggregation across users skips ``None`` ILD values so users with K<2 hits
do not depress the diversity mean.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import Any


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary-relevance NDCG with the standard ``log2(i+2)`` discount.

    Args:
        ranked_ids: Item ids in ranked order. Ties are preserved as given.
        relevant_ids: The set of truly-relevant item ids for this user.
        k: Cutoff position.

    Returns:
        NDCG in [0.0, 1.0]. Zero when no relevant items appear in the top-K
        or when the relevant set is empty.
    """
    topk = ranked_ids[:k]
    dcg = sum(
        1.0 / math.log2(i + 2) for i, item_id in enumerate(topk) if item_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant items recovered in the top-K.

    Args:
        ranked_ids: Item ids in ranked order.
        relevant_ids: The set of truly-relevant item ids.
        k: Cutoff position.

    Returns:
        Recall in [0.0, 1.0]. Zero when the relevant set is empty.
    """
    if not relevant_ids:
        return 0.0
    hits = sum(1 for item_id in ranked_ids[:k] if item_id in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-K that is relevant.

    The denominator is ``min(k, len(ranked_ids))`` so a short result list
    is not unfairly penalised against the requested K.

    Args:
        ranked_ids: Item ids in ranked order.
        relevant_ids: The set of truly-relevant item ids.
        k: Cutoff position.

    Returns:
        Precision in [0.0, 1.0]. Zero when no items are returned.
    """
    denom = min(k, len(ranked_ids))
    if denom == 0:
        return 0.0
    hits = sum(1 for item_id in ranked_ids[:k] if item_id in relevant_ids)
    return hits / denom


def ild_at_k(
    ranked_items: list[dict[str, Any]],
    fields: Sequence[str],
    k: int,
) -> float | None:
    """Intra-list distance over the top-K (mean pairwise field-equality distance).

    Distance between two items is the fraction of ``fields`` on which they
    disagree. Missing keys and explicit ``None`` are treated as the same
    "absent" category — so two items both missing a field are *equal* on it,
    while a present value differs from absence.

    Args:
        ranked_items: Per-hit attribute dicts (caller restricts to the
            configured diversity fields when fetching ``_source``).
        fields: Diversity field names to compare on. Must be non-empty
            and constant across an experiment.
        k: Cutoff position.

    Returns:
        Mean pairwise distance in [0.0, 1.0], or ``None`` when fewer than
        two hits are available (diversity is undefined). Callers should
        exclude ``None`` from aggregation.
    """
    topk = ranked_items[:k]
    if len(topk) < 2:
        return None
    field_list = list(fields)
    n_fields = len(field_list)
    pair_distances = [
        sum(1 for f in field_list if a.get(f) != b.get(f)) / n_fields
        for a, b in combinations(topk, 2)
    ]
    return sum(pair_distances) / len(pair_distances)


def aggregate(per_user_metrics: Iterable[dict[str, float | None]]) -> dict[str, float | None]:
    """Mean each metric across users, skipping ``None`` contributors.

    A ``None`` contribution (e.g. ILD when the user had K<2 hits) is
    excluded from both the numerator and the denominator. If every user
    is ``None`` for a given key, the key resolves to ``None`` rather than
    NaN or zero so downstream consumers can distinguish "no data" from
    "zero diversity".

    Args:
        per_user_metrics: One dict per user, keyed by metric name
            (e.g. ``"ndcg@10"``). Missing keys are tolerated.

    Returns:
        Dict with the same keys; each value is the mean of the non-``None``
        contributors, or ``None`` if no user contributed a value.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for user_metrics in per_user_metrics:
        for key, value in user_metrics.items():
            seen.add(key)
            if value is None:
                continue
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: (sums[key] / counts[key] if counts.get(key, 0) > 0 else None) for key in seen}
