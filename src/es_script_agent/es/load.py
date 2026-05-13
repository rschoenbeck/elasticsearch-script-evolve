"""Bulk loader for the ``loans`` and ``users`` indices.

``setup_indices`` is the orchestrator: it drops + recreates both indices
from the adapter's vector dim, then bulk-loads items and users via the
pure ``item_actions`` / ``user_actions`` generators. The generators are
exported separately so they can be unit-tested without ES.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Iterable, Iterator, TypeVar

from elasticsearch.helpers import bulk

from es_script_agent.data import DatasetAdapter
from es_script_agent.data.schema import Item, User
from es_script_agent.es.schemas import (
    ITEM_VECTOR_FIELD,
    LOANS_INDEX,
    NUM_USER_VECTORS,
    USER_VECTOR_FIELDS,
    USERS_INDEX,
    loans_mapping,
    users_mapping,
)

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE: int = 500


def item_actions(items: Iterable[Item], index_name: str) -> Iterator[dict[str, Any]]:
    """Yield ES bulk-index actions for items.

    Skips items without a vector (they can't be retrieved by vector
    search and would invalidate any cosine score). Drops ``None``-valued
    attributes from ``_source`` so they don't index as the literal type
    ``null`` against a typed field.

    Args:
        items: Normalized items from a ``DatasetAdapter``.
        index_name: Target index.

    Yields:
        Bulk action dicts with ``_index``, ``_id``, ``_source``.
    """
    for item in items:
        if item.vector is None:
            continue
        source: dict[str, Any] = {
            "id": item.id,
            ITEM_VECTOR_FIELD: item.vector,
        }
        for key, value in item.attributes.items():
            if value is None:
                continue
            source[key] = value
        yield {"_index": index_name, "_id": item.id, "_source": source}


def user_actions(users: Iterable[User], index_name: str) -> Iterator[dict[str, Any]]:
    """Yield ES bulk-index actions for users, flattening the 10×D matrix.

    Each ``User.vector[i]`` lands in ``user_vector_i``. Raises if the
    adapter yields a user whose vector list doesn't have exactly
    ``NUM_USER_VECTORS`` entries — that indicates an adapter bug, not
    a recoverable per-record skip.

    Args:
        users: Normalized users from a ``DatasetAdapter``.
        index_name: Target index.

    Yields:
        Bulk action dicts with ``_index``, ``_id``, ``_source``.

    Raises:
        ValueError: If a user vector matrix has the wrong number of rows.
    """
    for user in users:
        if len(user.vector) != NUM_USER_VECTORS:
            raise ValueError(
                f"user {user.id!r} has {len(user.vector)} vectors; "
                f"adapter must yield exactly {NUM_USER_VECTORS}"
            )
        source: dict[str, Any] = {"id": user.id}
        for field, vec in zip(USER_VECTOR_FIELDS, user.vector):
            source[field] = vec
        yield {"_index": index_name, "_id": user.id, "_source": source}


def _drop_and_create(es: Any, index_name: str, body: dict[str, Any]) -> None:
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        logger.info("dropped existing index %s", index_name)
    es.indices.create(index=index_name, **body)
    logger.info("created index %s", index_name)


_T = TypeVar("_T")


def _peek_first(source: Iterable[_T], *, kind: str) -> tuple[_T, Iterator[_T]]:
    """Pull the first element, returning it and a re-chained iterator."""
    iterator = iter(source)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError(f"adapter yielded no {kind}; cannot infer vector dim from empty data")
    return first, itertools.chain([first], iterator)


def setup_indices(
    es: Any,
    adapter: DatasetAdapter,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, int]:
    """Drop, recreate, and bulk-load the ``loans`` and ``users`` indices.

    Vector dim is inferred from the first item the adapter yields and
    asserted equal against the first user-vector slot. A mismatch fails
    loudly before any data is loaded.

    Args:
        es: Connected Elasticsearch client.
        adapter: Dataset adapter producing normalized ``Item`` / ``User``.
        chunk_size: ``elasticsearch.helpers.bulk`` chunk size.

    Returns:
        Mapping of index name → final doc count, as reported by ``_count``.

    Raises:
        ValueError: If the adapter is empty or item/user dims disagree.
    """
    first_item, items = _peek_first(adapter.iter_items(), kind="items")
    if first_item.vector is None:
        raise ValueError(f"first item {first_item.id!r} has no vector; cannot infer dim")
    item_dim = len(first_item.vector)

    first_user, users = _peek_first(adapter.iter_users(), kind="users")
    if len(first_user.vector) != NUM_USER_VECTORS:
        raise ValueError(
            f"first user {first_user.id!r} has {len(first_user.vector)} vectors; "
            f"adapter must yield exactly {NUM_USER_VECTORS}"
        )
    user_dim = len(first_user.vector[0])
    if user_dim != item_dim:
        raise ValueError(f"vector dim mismatch: items={item_dim}, users={user_dim}")

    _drop_and_create(es, LOANS_INDEX, loans_mapping(item_dim, adapter.attribute_field_types))
    _drop_and_create(es, USERS_INDEX, users_mapping(item_dim))

    # refresh="wait_for" so the count() below sees the just-loaded docs;
    # without it, ES's 1s default refresh interval makes the CLI lie.
    bulk(es, item_actions(items, LOANS_INDEX), chunk_size=chunk_size, refresh="wait_for")
    bulk(es, user_actions(users, USERS_INDEX), chunk_size=chunk_size, refresh="wait_for")

    return {
        LOANS_INDEX: es.count(index=LOANS_INDEX)["count"],
        USERS_INDEX: es.count(index=USERS_INDEX)["count"],
    }
