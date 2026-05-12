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
        "activity",
        "amountLeft",
        "borrowerCount",
        "country",
        "distributionModel",
        "fundraisingDate",
        "gender",
        "isMatchable",
        "loanAmount",
        "partnerId",
        "popularityScore",
        "sector",
        "tagsIds",
        "themesIds",
    }
    assert set(adapter.required_attributes) == expected


def test_iter_items_golden(adapter: DefaultAdapter) -> None:
    items = list(adapter.iter_items())
    assert len(items) == 3

    assert items[0] == Item(
        id="L1",
        vector=_vec(0),
        attributes={
            "activity": "Farming",
            "amountLeft": 200.0,
            "borrowerCount": 1,
            "country": "Kenya",
            "distributionModel": "field_partner",
            "fundraisingDate": "20260101T000000Z",
            "gender": "FEMALE",
            "isMatchable": True,
            "loanAmount": 500,
            "partnerId": 42,
            "popularityScore": 3,
            "sector": "Agriculture",
            "tagsIds": [11, 12],
            "themesIds": [1, 2],
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


def test_items_missing_item_vector_are_skipped_with_warning(
    tmp_path: Path,
    adapter: DefaultAdapter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Item dumps include partial records sourced upstream; one missing
    # vector shouldn't abort the whole load. Skip the record, warn once.
    bad = tmp_path / "loans.jsonl"
    good = (
        '{"_source": {"loanId": "L1", "vectorVersionB": {"itemVector": '
        + str([0.0] * 32)
        + "}}}\n"
    )
    missing = '{"_source": {"loanId": "L2"}}\n'
    bad.write_text(good + missing)
    a = DefaultAdapter(
        loans_path=bad,
        users_path=adapter.users_path,
        interactions_path=adapter.interactions_path,
    )

    with caplog.at_level("WARNING", logger="es_script_agent.data.adapters.default"):
        items = list(a.iter_items())

    assert [i.id for i in items] == ["L1"]
    assert any("skipped 1 loan record" in r.getMessage() for r in caplog.records)


def test_users_missing_vector_slot_are_skipped_with_warning(
    tmp_path: Path,
    adapter: DefaultAdapter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import json as _json

    good_src = {
        "_source": {
            "userId": "U-good",
            "vectorVersionB": {f"vector{i}": [0.0] * 32 for i in range(1, 11)},
        }
    }
    bad_src = {
        "_source": {
            "userId": "U-bad",
            "vectorVersionB": {
                f"vector{i}": [0.0] * 32 for i in range(1, 11) if i != 5
            },
        }
    }
    bad = tmp_path / "users.jsonl"
    bad.write_text(_json.dumps(good_src) + "\n" + _json.dumps(bad_src) + "\n")
    a = DefaultAdapter(
        loans_path=adapter.loans_path,
        users_path=bad,
        interactions_path=adapter.interactions_path,
    )

    with caplog.at_level("WARNING", logger="es_script_agent.data.adapters.default"):
        users = list(a.iter_users())

    assert [u.id for u in users] == ["U-good"]
    assert any("skipped 1 user record" in r.getMessage() for r in caplog.records)


def test_load_dataset_default_returns_adapter() -> None:
    # The registered entry point ("default") must return a working DefaultAdapter;
    # we only assert the type — the default paths point at the user's local data dir.
    from es_script_agent.data import load_dataset

    a = load_dataset("default")
    assert isinstance(a, DefaultAdapter)
    assert a.vector_dim == 32
