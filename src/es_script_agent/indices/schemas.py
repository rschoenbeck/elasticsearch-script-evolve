"""Elasticsearch index mappings for the ``loans`` and ``users`` indices.

The mappings are built from the normalized field names only — never from
dataset-specific names. Item vectors land in ``item_vector`` and user
vectors land in ten flat fields ``user_vector_0..user_vector_9`` because
Painless cannot read a 2D ``doc[…]`` array directly; the primary scoring
path passes the full 10×32 matrix via ``params.user_vector`` and the
indexed flat fields are an alternative the agent may join against.
"""

from __future__ import annotations

from typing import Any

LOANS_INDEX: str = "loans"
USERS_INDEX: str = "users"

NUM_USER_VECTORS: int = 10
USER_VECTOR_FIELDS: tuple[str, ...] = tuple(
    f"user_vector_{i}" for i in range(NUM_USER_VECTORS)
)
ITEM_VECTOR_FIELD: str = "item_vector"

# Field types for declared item attributes. Diversity fields (sector,
# country, partnerId) must be keyword so equality comparisons in ILD —
# and any future term-level filters — work without analyzer surprises.
ATTRIBUTE_FIELD_TYPES: dict[str, dict[str, Any]] = {
    "activity": {"type": "keyword"},
    "amountLeft": {"type": "double"},
    "borrowerCount": {"type": "integer"},
    "country": {"type": "keyword"},
    "distributionModel": {"type": "keyword"},
    "fundraisingDate": {"type": "date", "format": "basic_date_time_no_millis"},
    "gender": {"type": "keyword"},
    "isMatchable": {"type": "boolean"},
    "loanAmount": {"type": "double"},
    "partnerId": {"type": "keyword"},
    "popularityScore": {"type": "integer"},
    "sector": {"type": "keyword"},
    "tagsIds": {"type": "integer"},
    "themesIds": {"type": "integer"},
}


def _dense_vector(dim: int) -> dict[str, Any]:
    return {
        "type": "dense_vector",
        "dims": dim,
        "similarity": "cosine",
        "index": True,
    }


def loans_mapping(
    vector_dim: int, attribute_fields: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Build the ``loans`` index mapping.

    Args:
        vector_dim: Dimensionality of ``item_vector``.
        attribute_fields: Mapping of attribute name → field definition.
            Each entry is inserted under ``properties`` as-is.

    Returns:
        A full ES create-index body containing ``mappings.properties``.
    """
    properties: dict[str, Any] = {
        "id": {"type": "keyword"},
        ITEM_VECTOR_FIELD: _dense_vector(vector_dim),
    }
    properties.update(attribute_fields)
    return {"mappings": {"properties": properties}}


def users_mapping(vector_dim: int) -> dict[str, Any]:
    """Build the ``users`` index mapping with ten flat dense_vector fields.

    Args:
        vector_dim: Dimensionality of each ``user_vector_N``.

    Returns:
        A full ES create-index body containing ``mappings.properties``.
    """
    properties: dict[str, Any] = {"id": {"type": "keyword"}}
    for name in USER_VECTOR_FIELDS:
        properties[name] = _dense_vector(vector_dim)
    return {"mappings": {"properties": properties}}
