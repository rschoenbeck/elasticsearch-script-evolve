"""Normalized internal types that cross the adapter boundary (SPEC §3a).

Outside ``data/adapters/``, code consumes only these shapes — never
dataset-specific field names. New datasets ship as new adapters that
yield instances of these types.
"""

from __future__ import annotations
from typing import Any

from pydantic import BaseModel, ConfigDict


class Item(BaseModel):
    """A single rankable item.

    Attributes:
        id: Stable identifier used as the ES document id.
        vector: Single dense embedding, or ``None`` if the adapter
            cannot supply one (such items will be filtered at index time).
        attributes: Free-form per-item fields. Keys consumed by the
            index mapping or by Painless scripts must be declared in
            the adapter's ``required_attributes``.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    vector: list[float] | None
    attributes: dict[str, Any]


class User(BaseModel):
    """A user represented by one or more dense vectors of equal dimension.

    The vector is 2D so a single user can carry multiple embeddings.
    Single-vector adapters wrap as ``[vec]``. The harness passes the
    whole 2D list as ``params.user_vector``; pooling strategies live
    inside the Painless source.

    Attributes:
        id: Stable identifier used as the ES document id.
        vector: List of dense vectors. All inner vectors must have the
            same length as ``DatasetAdapter.vector_dim``.
        attributes: Free-form per-user fields.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    vector: list[list[float]]
    attributes: dict[str, Any]


class Interaction(BaseModel):
    """A raw (user, item, weight) signal — no thresholding applied.

    Binary-relevance thresholding is applied later in
    ``eval/interactions.py`` so adapters stay decoupled from eval policy.

    Attributes:
        user_id: Foreign key into ``User.id``.
        item_id: Foreign key into ``Item.id``.
        weight: Raw signal magnitude (e.g. interaction count).
    """

    model_config = ConfigDict(frozen=True)

    user_id: str
    item_id: str
    weight: float
