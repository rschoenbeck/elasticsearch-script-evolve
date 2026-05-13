"""Build the Elasticsearch search body for a script set.

The harness owns the surrounding JSON shape: ``match_all`` candidate set,
``script_score`` query clause, sort-script ordering, and the ``_score desc``
tie-break. The agent edits only the Painless ``source`` strings and the count
of sort scripts (bounded by :data:`MAX_SORT_SCRIPTS`).

``params.user_vector`` is harness-owned and injected identically into the
query script and every sort script, so a sort script can reference it without
re-plumbing.
"""

from __future__ import annotations

from typing import Any


MAX_SORT_SCRIPTS: int = 5


def build_query(
    query_source: str,
    sort_sources: list[str],
    user_vector: list[list[float]],
    size: int = 10,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a complete ES search body for a one-query + N-sort script set.

    Args:
        query_source: Painless source for the single ``script_score`` clause.
            Must be non-empty (non-whitespace).
        sort_sources: Ordered Painless sources for sort scripts. Length must
            be ``<= MAX_SORT_SCRIPTS``. May be empty.
        user_vector: The 2D user representation (e.g. 10 vectors of 32 dims).
            Injected verbatim into every script's ``params``.
        size: Top-K to return from ES.
        extra_params: Optional additional params merged into every script.
            Must not contain the reserved key ``"user_vector"``.

    Returns:
        A dict matching the harness-owned search body shape. ``query`` is
        identical across iterations for the same ``query_source``; sort
        clauses appear in declared order followed by a ``{"_score": "desc"}``
        tie-break.

    Raises:
        ValueError: If ``query_source`` is blank, ``sort_sources`` exceeds
            ``MAX_SORT_SCRIPTS``, or ``extra_params`` tries to shadow the
            reserved ``user_vector`` key.
    """
    if not query_source or not query_source.strip():
        raise ValueError("query_source must be a non-empty Painless source string")
    if len(sort_sources) > MAX_SORT_SCRIPTS:
        raise ValueError(
            f"too many sort scripts: {len(sort_sources)} > MAX_SORT_SCRIPTS={MAX_SORT_SCRIPTS}"
        )
    if extra_params is not None and "user_vector" in extra_params:
        raise ValueError(
            "extra_params must not contain reserved key 'user_vector'; the harness owns that param"
        )

    params: dict[str, Any] = {"user_vector": user_vector}
    if extra_params:
        params.update(extra_params)

    sort: list[dict[str, Any]] = [
        {
            "_script": {
                "type": "number",
                "order": "desc",
                "script": {"source": s, "params": params},
            }
        }
        for s in sort_sources
    ]
    sort.append({"_score": "desc"})

    return {
        "size": size,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {"source": query_source, "params": params},
            }
        },
        "sort": sort,
    }
