"""Default adapter — normalizes the local loan/user/interaction dump."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

from es_script_agent.config import (
    DEFAULT_INTERACTIONS_PATH,
    DEFAULT_LOANS_PATH,
    DEFAULT_USERS_PATH,
)
from es_script_agent.data.schema import Interaction, Item, User

_REQUIRED_ATTRIBUTES: tuple[str, ...] = (
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
)

# Use these keys to extract user and item vectors
_VECTOR_VERSION_KEY = "vectorVersionB"
_ITEM_VECTOR_KEY = "itemVector"
_USER_VECTOR_KEYS: tuple[str, ...] = tuple(f"vector{i}" for i in range(1, 11))

_CSV_LOAN_COL = "Loan Details Loan ID"
_CSV_USER_COL = "User Details Login ID"
_CSV_COUNT_COL = "Measures: Actions Number of 'Add to Basket' Sessions"


class DefaultAdapter:
    """Adapter for the locally-staged loan / user / interaction dump.

    Loans and users are JSONL files where each line is an Elasticsearch hit
    (``_source`` holds the record). Interactions live in a CSV with one
    (loan id, user id, count) row per impression. Raw counts are passed
    through as ``Interaction.weight`` — the binary-relevance threshold is
    applied later in ``eval/interactions.py``.

    Attributes:
        vector_dim: Common item/user-vector dimensionality (32).
        required_attributes: Item-attribute keys the adapter populates; missing
            optional values resolve to ``None``.
        loans_path: JSONL of loan hits.
        users_path: JSONL of user hits.
        interactions_path: CSV of (loan id, user id, count) rows.
    """

    vector_dim: int = 32
    required_attributes: list[str] = list(_REQUIRED_ATTRIBUTES)

    def __init__(
        self,
        loans_path: Path | None = None,
        users_path: Path | None = None,
        interactions_path: Path | None = None,
    ) -> None:
        self.loans_path = Path(loans_path) if loans_path else DEFAULT_LOANS_PATH
        self.users_path = Path(users_path) if users_path else DEFAULT_USERS_PATH
        self.interactions_path = (
            Path(interactions_path) if interactions_path else DEFAULT_INTERACTIONS_PATH
        )

    def iter_items(self) -> Iterator[Item]:
        for src in _iter_jsonl_sources(self.loans_path):
            yield _item_from_source(src)

    def iter_users(self) -> Iterator[User]:
        for src in _iter_jsonl_sources(self.users_path):
            yield _user_from_source(src)

    def iter_interactions(self) -> Iterator[Interaction]:
        with self.interactions_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                weight_raw = (row.get(_CSV_COUNT_COL) or "").strip()
                if not weight_raw:
                    continue
                try:
                    weight = float(weight_raw)
                except ValueError:
                    continue
                yield Interaction(
                    user_id=str(row[_CSV_USER_COL]),
                    item_id=str(row[_CSV_LOAN_COL]),
                    weight=weight,
                )


def _iter_jsonl_sources(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = rec.get("_source") if isinstance(rec, dict) else None
            if src is None:
                raise ValueError(f"{path.name}: record missing '_source'")
            yield src


def _item_from_source(src: dict[str, Any]) -> Item:
    loan_id = src.get("loanId")
    if loan_id is None:
        raise ValueError("loan record missing required field 'loanId'")
    vector_block = src.get(_VECTOR_VERSION_KEY) or {}
    vector = vector_block.get(_ITEM_VECTOR_KEY)
    if vector is None:
        raise ValueError(f"loan {loan_id!r} missing required 'vectorVersionB.itemVector'")
    attributes = {key: src.get(key) for key in _REQUIRED_ATTRIBUTES}
    return Item(id=str(loan_id), vector=[float(x) for x in vector], attributes=attributes)


def _user_from_source(src: dict[str, Any]) -> User:
    user_id = src.get("userId")
    if user_id is None:
        raise ValueError("user record missing required field 'userId'")
    vector_block = src.get(_VECTOR_VERSION_KEY) or {}
    vectors: list[list[float]] = []
    for key in _USER_VECTOR_KEYS:
        v = vector_block.get(key)
        if v is None:
            raise ValueError(f"user {user_id!r} missing required 'vectorVersionB.{key}'")
        vectors.append([float(x) for x in v])
    return User(id=str(user_id), vector=vectors, attributes={})
