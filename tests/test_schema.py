"""Normalized dataset types and adapter protocol — schema-level contract."""

from __future__ import annotations

from typing import get_type_hints

import pytest

from es_script_agent.data import DatasetAdapter, load_dataset
from es_script_agent.data.schema import Interaction, Item, User


def test_item_round_trip() -> None:
    item = Item(id="loan-1", vector=[0.1, 0.2, 0.3], attributes={"sector": "Agriculture"})
    assert item.id == "loan-1"
    assert item.vector == [0.1, 0.2, 0.3]
    assert item.attributes == {"sector": "Agriculture"}


def test_item_vector_may_be_none() -> None:
    item = Item(id="loan-1", vector=None, attributes={})
    assert item.vector is None


def test_user_vector_is_two_dimensional() -> None:
    user = User(
        id="user-1",
        vector=[[0.1] * 32, [0.2] * 32, [0.3] * 32],
        attributes={},
    )
    assert len(user.vector) == 3
    assert all(len(v) == 32 for v in user.vector)


def test_interaction_fields() -> None:
    interaction = Interaction(user_id="user-1", item_id="loan-1", weight=2.0)
    assert interaction.user_id == "user-1"
    assert interaction.item_id == "loan-1"
    assert interaction.weight == 2.0


def test_dataset_adapter_protocol_surface() -> None:
    # The Protocol must declare the documented attributes and methods.
    hints = get_type_hints(DatasetAdapter)
    assert "vector_dim" in hints
    assert "required_attributes" in hints
    for name in ("iter_items", "iter_users", "iter_interactions"):
        assert callable(getattr(DatasetAdapter, name, None)), f"{name} not declared"


def test_load_dataset_unknown_raises() -> None:
    with pytest.raises(ValueError):
        load_dataset("does-not-exist")


def test_load_dataset_default_is_registered() -> None:
    # The adapter implementation arrives later; for now we only require
    # that "default" is a known name (no ValueError), distinct from the
    # generic unknown-name failure mode.
    try:
        load_dataset("default")
    except ValueError:
        pytest.fail("'default' should be a registered dataset name")
    except NotImplementedError:
        pass  # acceptable until the adapter lands
