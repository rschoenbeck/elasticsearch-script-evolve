"""Mapping builders for the ``loans`` and ``users`` indices."""

from __future__ import annotations

from es_script_agent.indices.schemas import (
    ATTRIBUTE_FIELD_TYPES,
    USER_VECTOR_FIELDS,
    loans_mapping,
    users_mapping,
)


def test_loans_mapping_id_and_vector() -> None:
    mapping = loans_mapping(vector_dim=32, attribute_fields=ATTRIBUTE_FIELD_TYPES)
    props = mapping["mappings"]["properties"]

    assert props["id"] == {"type": "keyword"}
    assert props["item_vector"] == {
        "type": "dense_vector",
        "dims": 32,
        "similarity": "cosine",
        "index": True,
    }


def test_loans_mapping_includes_every_attribute_field() -> None:
    mapping = loans_mapping(vector_dim=32, attribute_fields=ATTRIBUTE_FIELD_TYPES)
    props = mapping["mappings"]["properties"]

    for name, type_def in ATTRIBUTE_FIELD_TYPES.items():
        assert props[name] == type_def, f"attribute {name!r} mapping mismatch"


def test_loans_mapping_ild_diversity_fields_are_keyword() -> None:
    # ILD computes equality across categorical fields — they must be keyword
    # so _source returns them unanalyzed for the runner.
    mapping = loans_mapping(vector_dim=32, attribute_fields=ATTRIBUTE_FIELD_TYPES)
    props = mapping["mappings"]["properties"]

    for diversity_field in ("sector", "country", "partnerId"):
        assert props[diversity_field]["type"] == "keyword"


def test_loans_mapping_respects_custom_dim() -> None:
    mapping = loans_mapping(vector_dim=64, attribute_fields=ATTRIBUTE_FIELD_TYPES)
    assert mapping["mappings"]["properties"]["item_vector"]["dims"] == 64


def test_users_mapping_has_ten_vector_fields() -> None:
    mapping = users_mapping(vector_dim=32)
    props = mapping["mappings"]["properties"]

    assert props["id"] == {"type": "keyword"}
    assert len(USER_VECTOR_FIELDS) == 10
    for name in USER_VECTOR_FIELDS:
        assert props[name] == {
            "type": "dense_vector",
            "dims": 32,
            "similarity": "cosine",
            "index": True,
        }


def test_users_mapping_respects_custom_dim() -> None:
    mapping = users_mapping(vector_dim=16)
    for name in USER_VECTOR_FIELDS:
        assert mapping["mappings"]["properties"][name]["dims"] == 16
