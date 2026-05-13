"""Bulk action generators + ``setup_indices`` orchestration."""

from __future__ import annotations

from typing import Any

import pytest

from es_script_agent.data.schema import Item, User
from es_script_agent.es.load import (
    LOANS_INDEX,
    USERS_INDEX,
    item_actions,
    setup_indices,
    user_actions,
)


def _item(id_: str, vector: list[float], **attrs: Any) -> Item:
    return Item(id=id_, vector=vector, attributes=attrs)


def _user(id_: str, vectors: list[list[float]]) -> User:
    return User(id=id_, vector=vectors, attributes={})


def test_item_actions_shape() -> None:
    items = [
        _item("L1", [0.1] * 4, sector="Agriculture", country="Kenya", partnerId=42),
    ]
    actions = list(item_actions(items, index_name="loans-x"))
    assert len(actions) == 1
    action = actions[0]

    assert action["_index"] == "loans-x"
    assert action["_id"] == "L1"
    source = action["_source"]
    assert source["id"] == "L1"
    assert source["item_vector"] == [0.1] * 4
    assert source["sector"] == "Agriculture"
    assert source["country"] == "Kenya"
    assert source["partnerId"] == 42


def test_item_actions_skips_none_attributes() -> None:
    # None attribute values are omitted from _source so ES doesn't index them
    # as the literal string "None" or fail on type coercion.
    items = [_item("L1", [0.0] * 4, sector=None, country="Kenya")]
    [action] = list(item_actions(items, index_name="loans"))
    assert "sector" not in action["_source"]
    assert action["_source"]["country"] == "Kenya"


def test_item_actions_skips_items_without_vector() -> None:
    items = [
        Item(id="L1", vector=None, attributes={}),
        _item("L2", [0.0] * 4),
    ]
    actions = list(item_actions(items, index_name="loans"))
    assert [a["_id"] for a in actions] == ["L2"]


def test_user_actions_flattens_2d_vector() -> None:
    vectors = [[float(i + j * 0.01) for j in range(4)] for i in range(10)]
    users = [_user("U1", vectors)]
    [action] = list(user_actions(users, index_name="users-x"))

    assert action["_index"] == "users-x"
    assert action["_id"] == "U1"
    source = action["_source"]
    assert source["id"] == "U1"
    for i in range(10):
        assert source[f"user_vector_{i}"] == vectors[i]


def test_user_actions_raises_when_wrong_number_of_vectors() -> None:
    users = [_user("U1", [[0.0] * 4] * 5)]  # only 5 vectors instead of 10
    with pytest.raises(ValueError, match="10"):
        list(user_actions(users, index_name="users"))


class FakeIndicesClient:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.created: list[tuple[str, dict[str, Any]]] = []
        self._exists: set[str] = set()

    def exists(self, *, index: str) -> bool:
        return index in self._exists

    def delete(self, *, index: str) -> None:
        self._exists.discard(index)
        self.deleted.append(index)

    def create(self, *, index: str, **body: Any) -> None:
        self._exists.add(index)
        self.created.append((index, body))


class FakeES:
    """Bare-bones fake ES capturing only the calls setup_indices makes."""

    def __init__(self, doc_counts: dict[str, int] | None = None) -> None:
        self.indices = FakeIndicesClient()
        self.bulk_calls: list[tuple[str, int]] = []
        self._doc_counts = doc_counts or {}

    def count(self, *, index: str) -> dict[str, int]:
        return {"count": self._doc_counts.get(index, 0)}


class DummyAdapter:
    vector_dim = 4
    required_attributes = ["sector", "country"]
    attribute_field_types: dict[str, dict[str, Any]] = {
        "sector": {"type": "keyword"},
        "country": {"type": "keyword"},
    }

    def __init__(
        self,
        items: list[Item],
        users: list[User],
    ) -> None:
        self._items = items
        self._users = users

    def iter_items(self):
        return iter(self._items)

    def iter_users(self):
        return iter(self._users)

    def iter_interactions(self):
        return iter([])


def test_setup_indices_drops_creates_and_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    items = [_item(f"L{i}", [0.1] * 4, sector="X", country="Y") for i in range(3)]
    users = [_user(f"U{i}", [[0.2] * 4 for _ in range(10)]) for i in range(2)]
    adapter = DummyAdapter(items, users)
    es = FakeES(doc_counts={LOANS_INDEX: 3, USERS_INDEX: 2})

    bulk_invocations: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []

    def fake_bulk(client: Any, actions: Any, **kwargs: Any) -> tuple[int, list]:
        materialized = list(actions)
        index_name = materialized[0]["_index"] if materialized else "<empty>"
        bulk_invocations.append((index_name, materialized, kwargs))
        return (len(materialized), [])

    monkeypatch.setattr("es_script_agent.es.load.bulk", fake_bulk)

    counts = setup_indices(es, adapter)

    assert counts == {LOANS_INDEX: 3, USERS_INDEX: 2}
    assert es.indices.created[0][0] == LOANS_INDEX
    assert es.indices.created[1][0] == USERS_INDEX
    assert [name for name, _, _ in bulk_invocations] == [LOANS_INDEX, USERS_INDEX]
    assert len(bulk_invocations[0][1]) == 3
    assert len(bulk_invocations[1][1]) == 2
    # Both bulk calls must wait for refresh so the count() that follows
    # in setup_indices sees the just-loaded docs (not stale 0s).
    for _, _, kwargs in bulk_invocations:
        assert kwargs.get("refresh") == "wait_for"


def test_setup_indices_drops_existing_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = DummyAdapter(
        items=[_item("L1", [0.0] * 4)],
        users=[_user("U1", [[0.0] * 4] * 10)],
    )
    es = FakeES()
    es.indices._exists = {LOANS_INDEX, USERS_INDEX}

    monkeypatch.setattr(
        "es_script_agent.es.load.bulk", lambda c, a, **kw: (len(list(a)), [])
    )

    setup_indices(es, adapter)
    assert es.indices.deleted == [LOANS_INDEX, USERS_INDEX]


def test_setup_indices_uses_adapter_attribute_field_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The adapter owns dataset-specific field names and types; setup_indices
    # must build the loans mapping from adapter.attribute_field_types, not
    # from any constant living next to the index code.
    adapter = DummyAdapter(
        items=[_item("L1", [0.0] * 4, sector="X", country="Y")],
        users=[_user("U1", [[0.0] * 4] * 10)],
    )
    adapter.attribute_field_types = {
        "sector": {"type": "keyword"},
        "country": {"type": "keyword"},
        "amount": {"type": "double"},
    }
    es = FakeES()
    monkeypatch.setattr(
        "es_script_agent.es.load.bulk", lambda c, a, **kw: (len(list(a)), [])
    )

    setup_indices(es, adapter)

    loans_create_body = es.indices.created[0][1]
    props = loans_create_body["mappings"]["properties"]
    assert props["sector"] == {"type": "keyword"}
    assert props["country"] == {"type": "keyword"}
    assert props["amount"] == {"type": "double"}


def test_setup_indices_rejects_user_vector_dim_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First item is 4-dim but user vectors are 8-dim — must fail before bulk.
    adapter = DummyAdapter(
        items=[_item("L1", [0.0] * 4)],
        users=[_user("U1", [[0.0] * 8] * 10)],
    )
    es = FakeES()
    monkeypatch.setattr(
        "es_script_agent.es.load.bulk", lambda c, a, **kw: (len(list(a)), [])
    )

    with pytest.raises(ValueError, match="dim"):
        setup_indices(es, adapter)


def test_setup_indices_rejects_empty_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = DummyAdapter(items=[], users=[])
    es = FakeES()
    monkeypatch.setattr(
        "es_script_agent.es.load.bulk", lambda c, a, **kw: (0, [])
    )

    with pytest.raises(ValueError, match="empty"):
        setup_indices(es, adapter)
