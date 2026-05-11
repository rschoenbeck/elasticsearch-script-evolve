"""Dataset adapter layer (SPEC §3a / boundary §6.6).

All dataset access flows through ``DatasetAdapter`` implementations
that yield the normalized types in ``data.schema``. Eval, indexing,
and agent code import only from ``data`` — never from ``data.adapters``
directly.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from es_script_agent.data.schema import Interaction, Item, User

__all__ = ["DatasetAdapter", "Interaction", "Item", "User", "load_dataset"]

# Names registered here without an importable adapter module are
# accepted by `load_dataset` (no ValueError) but raise NotImplementedError
# until their adapter lands. Lets schema-level tests run before adapters
# exist.
_REGISTERED: frozenset[str] = frozenset({"flss"})


@runtime_checkable
class DatasetAdapter(Protocol):
    """Contract every dataset adapter must satisfy.

    Attributes:
        vector_dim: Common dimensionality of item and user vectors.
            Asserted equal across both at index time.
        required_attributes: Item-attribute keys the adapter promises
            to populate; consumed by ``indices/schemas.py`` when
            building the ES mapping.
    """

    vector_dim: int
    required_attributes: list[str]

    def iter_items(self) -> Iterable[Item]: ...

    def iter_users(self) -> Iterable[User]: ...

    def iter_interactions(self) -> Iterable[Interaction]: ...


def load_dataset(name: str) -> DatasetAdapter:
    """Return the registered adapter for ``name``.

    Args:
        name: Registered dataset name (e.g. ``"flss"``).

    Returns:
        An instantiated ``DatasetAdapter``.

    Raises:
        ValueError: If ``name`` is not a registered dataset.
        NotImplementedError: If ``name`` is registered but its adapter
            module has not been built yet.
    """
    if name not in _REGISTERED:
        raise ValueError(
            f"Unknown dataset {name!r}. Registered: {sorted(_REGISTERED)}"
        )
    if name == "flss":
        try:
            from es_script_agent.data.adapters.flss import FlssAdapter
        except ImportError as exc:
            raise NotImplementedError(
                "FLSS adapter not yet implemented (Task 5)"
            ) from exc
        return FlssAdapter()
    raise NotImplementedError(f"Adapter for {name!r} is registered but not wired up")
