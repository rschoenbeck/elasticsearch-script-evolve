"""Golden-file contract test for the default adapter (SPEC §5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from es_script_agent.data.adapters.default import DefaultAdapter
from es_script_agent.data.schema import Interaction, Item

FIXTURES = Path(__file__).parent / "fixtures" / "default"


def _vec(seed: int) -> list[float]:
    return [round((seed + i) * 0.01, 4) for i in range(32)]


@pytest.fixture
def adapter() -> DefaultAdapter:
    return DefaultAdapter(
        loans_path=FIXTURES / "loans.jsonl",
        users_path=FIXTURES / "users.jsonl",
        interactions_path=FIXTURES / "interactions.csv",
    )


def test_vector_dim_is_32(adapter: DefaultAdapter) -> None:
    assert adapter.vector_dim == 32


def test_required_attributes_declared(adapter: DefaultAdapter) -> None:
    expected = {
        "sector",
        "country",
        "loanAmount",
        "partnerId",
        "activityId",
        "themesIds",
        "gender",
        "borrowerCount",
        "fundraisingDate",
        "partnerRiskRating",
        "researchScore",
        "popularityScore",
    }
    assert set(adapter.required_attributes) == expected


def test_iter_items_golden(adapter: DefaultAdapter) -> None:
    items = list(adapter.iter_items())
    assert len(items) == 3

    assert items[0] == Item(
        id="L1",
        vector=_vec(0),
        attributes={
            "sector": "Agriculture",
            "country": "Kenya",
            "loanAmount": 500,
            "partnerId": 42,
            "activityId": 31,
            "themesIds": [1, 2],
            "gender": "FEMALE",
            "borrowerCount": 1,
            "fundraisingDate": "20260101T000000Z",
            "partnerRiskRating": 1.5,
            "researchScore": 10.0,
            "popularityScore": 3,
        },
    )

    # L3 has only id + vector; every optional attribute resolves to None.
    assert items[2].id == "L3"
    assert items[2].vector == _vec(100)
    assert items[2].attributes == {k: None for k in adapter.required_attributes}


def test_iter_users_golden(adapter: DefaultAdapter) -> None:
    users = list(adapter.iter_users())
    assert len(users) == 2

    u1 = users[0]
    assert u1.id == "U1"
    assert len(u1.vector) == 10
    assert all(len(v) == 32 for v in u1.vector)
    # vector1 was built with seed=10, vector2 with seed=20, ..., vector10 with seed=100.
    assert u1.vector[0] == _vec(10)
    assert u1.vector[9] == _vec(100)

    assert users[1].id == "U2"


def test_iter_interactions_passes_through_raw_weights(adapter: DefaultAdapter) -> None:
    interactions = list(adapter.iter_interactions())
    # 5 rows in the CSV; the blank-weight row is skipped → 4 interactions yielded.
    assert interactions == [
        Interaction(user_id="U1", item_id="L1", weight=1.0),
        Interaction(user_id="U1", item_id="L2", weight=0.0),
        Interaction(user_id="U2", item_id="L1", weight=2.0),
        Interaction(user_id="U2", item_id="L2", weight=3.0),
    ]


def test_missing_loan_id_raises(tmp_path: Path, adapter: DefaultAdapter) -> None:
    bad = tmp_path / "loans.jsonl"
    bad.write_text('{"_source": {"vectorVersionB": {"itemVector": [0.0]}}}\n')
    a = DefaultAdapter(
        loans_path=bad,
        users_path=adapter.users_path,
        interactions_path=adapter.interactions_path,
    )
    with pytest.raises(ValueError, match="loanId"):
        list(a.iter_items())


def test_missing_item_vector_raises(tmp_path: Path, adapter: DefaultAdapter) -> None:
    bad = tmp_path / "loans.jsonl"
    bad.write_text('{"_source": {"loanId": "X"}}\n')
    a = DefaultAdapter(
        loans_path=bad,
        users_path=adapter.users_path,
        interactions_path=adapter.interactions_path,
    )
    with pytest.raises(ValueError, match="itemVector"):
        list(a.iter_items())


def test_missing_user_vector_slot_raises(tmp_path: Path, adapter: DefaultAdapter) -> None:
    # User missing vector5 → hard fail (vectors are required).
    bad = tmp_path / "users.jsonl"
    src = {
        "_source": {
            "userId": "U-bad",
            "vectorVersionB": {f"vector{i}": [0.0] * 32 for i in range(1, 11) if i != 5},
        }
    }
    import json as _json

    bad.write_text(_json.dumps(src) + "\n")
    a = DefaultAdapter(
        loans_path=adapter.loans_path,
        users_path=bad,
        interactions_path=adapter.interactions_path,
    )
    with pytest.raises(ValueError, match="vector5"):
        list(a.iter_users())


def test_load_dataset_default_returns_adapter() -> None:
    # The registered entry point ("default") must return a working DefaultAdapter;
    # we only assert the type — the default paths point at the user's local data dir.
    from es_script_agent.data import load_dataset

    a = load_dataset("default")
    assert isinstance(a, DefaultAdapter)
    assert a.vector_dim == 32
