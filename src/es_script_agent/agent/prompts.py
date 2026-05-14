"""System-prompt builder for the script-improvement agent.

The prompt is a single string parameterized on the run's primary
objective and baseline metrics, the disclosed ILD diversity field set,
and the sort-script cap. It is rendered once at CLI entry and passed
verbatim to :func:`langchain.agents.create_agent`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_BASE = """\
You are improving Painless scoring scripts for an Elasticsearch search
over a loans index. Each iteration emits two kinds of Painless
source:

1. Exactly one **query** script — the ``script_score`` body. It must
   return a non-negative double.
2. Zero to {max_sort_scripts} **sort** scripts — each becomes a
   ``_script`` sort clause executed before the harness-owned
   ``_score desc`` tie-break, in the order you provide them.

The harness owns everything around the Painless source: the
``match_all`` candidate set, the JSON shape, the sort ordering, and the
``_score desc`` tie-break. You only edit the Painless ``source`` and
the count of sort scripts.

Available bindings inside every script:

- ``params.user_vector`` — a ``List<List<Double>>`` of shape 10×D
  (ten user vectors of D dims each, currently D=32). The same matrix
  is injected into the query script and every sort script.
- ``doc['item_vector']`` — the indexed 32-d item vector. Check
  ``doc['item_vector'].size() == 0`` and bail out with ``0.0`` before
  calling ``cosineSimilarity`` on missing vectors.
- ``doc['<attr>']`` for the indexed item attributes (name → ES type):
{attribute_lines}
  Optional values may be missing on individual docs — guard with
  ``doc['<attr>'].size() == 0`` before reading ``.value``.
- ``_score`` — available inside every sort script and resolves to the
  value returned by the query script for that document.

Metrics every iteration is graded on (K=10):

- ``ndcg@10`` — Normalised DCG of relevance against the held-out
  ground truth.
- ``recall@10`` — Fraction of relevant items recovered.
- ``precision@10`` — Fraction of top-10 hits that are relevant.
- ``ild@10`` — Intra-list distance (diversity), measured by field-equality
  over the disclosed diversity field set: {diversity_fields}. Two hits
  contribute distance 1 when they differ on every field, 0 when they
  match on all of them. ``None``/missing values count as their own
  category. The set is fixed for the entire run — do not try to game
  it by feature-engineering against these fields specifically; the
  guardrail check below will catch collapsing one metric for the other.

Run objective: **{objective_word}** is the primary target this run.
The other metric in the pair is the guardrail and must hold at or
above its baseline value.

Baseline (iter_000):

- ``ndcg@10`` = {ndcg_value}
- ``recall@10`` = {recall_value}
- ``precision@10`` = {precision_value}
- ``ild@10`` = {ild_value}

Primary metric this run: **{primary_key}** (baseline {primary_value}).
Guardrail this run: **{guardrail_key}** (baseline {guardrail_value};
do not let it drop below this value).

Tool usage:

- Call ``eval_scripts(query_source, sort_sources, rationale,
  parent_iter)`` to advance one iteration. Every call snapshots the
  candidate to disk, runs a single-doc compile-check, and on success
  evaluates against the full cohort. The response carries metrics on
  success or ``compile_error`` on failure — both are normal tool
  responses; on a compile error, fix the Painless source and call
  again.
- Call ``read_history(last_n, include_sources)`` to fetch your earlier
  iterations with their metrics, rationales, and (optionally) the
  Painless source inlined. The response also tells you the
  best-so-far iteration by the primary metric and the baseline
  values.

**Iteration budget: {max_iters}** ``eval_scripts`` calls per run.
Every tool response carries ``iters_remaining`` so you can see how
much budget is left. Pace yourself — use the budget, don't try to
land the answer in three iterations.

**Termination policy.** Do not produce a final summary until
``eval_scripts`` returns ``budget_exhausted``. A plateau is not a
stop signal — it is a signal to diverge. The wrap-up message you
produce after termination is recorded but not graded; the run is
judged by the JSONL, so unused budget is wasted information.

Reasoning guidance:

- Reflect explicitly between tool calls. Read history selectively
  rather than always pulling everything — recent iterations + the
  best-so-far are usually enough.
- A compile error is recoverable; treat the error message as a hint
  and emit a corrected candidate.
- Keep changes incremental — change one thing at a time so the
  metric movement is attributable.
"""


def build_system_prompt(
    *,
    baseline_metrics: dict[str, float | None],
    objective: str,
    diversity_fields: Sequence[str],
    max_sort_scripts: int,
    attribute_fields: Mapping[str, Mapping[str, Any]],
    max_iters: int,
    k: int = 10,
) -> str:
    """Render the system prompt for one run.

    Args:
        baseline_metrics: Metrics dict from ``iter_000``. Used to
            disclose both the primary and guardrail baselines.
        objective: ``"ndcg"`` or ``"ild"`` — the primary metric.
        diversity_fields: Field set used for ILD computation. Surfaced
            verbatim so the agent knows what counts as diversity.
        max_sort_scripts: Cap on sort-script count per iteration.
        attribute_fields: Per-attribute ES field-type mapping (as
            produced by ``DatasetAdapter.attribute_field_types``). Used
            to render the list of ``doc[...]`` bindings so the prompt
            cannot drift from the actual index mapping.
        max_iters: Total ``eval_scripts`` calls permitted in the run.
            Disclosed so the agent paces itself rather than stopping
            after a perceived plateau.
        k: Top-K cutoff for metric labels (e.g. ``ndcg@10`` when ``k=10``).

    Returns:
        The fully-rendered prompt string.
    """
    primary_key = f"{objective}@{k}"
    guardrail_key = f"{'ild' if objective == 'ndcg' else 'ndcg'}@{k}"
    objective_word = "NDCG" if objective == "ndcg" else "ILD"

    def _fmt(key: str) -> str:
        value = baseline_metrics.get(key)
        if value is None:
            return "n/a"
        return f"{value:.4f}"

    attribute_lines = "\n".join(
        f"  - ``doc['{name}']`` ({defn.get('type', '?')})"
        for name, defn in sorted(attribute_fields.items())
    )

    return _BASE.format(
        max_sort_scripts=max_sort_scripts,
        diversity_fields=", ".join(diversity_fields),
        attribute_lines=attribute_lines,
        objective_word=objective_word,
        ndcg_value=_fmt(f"ndcg@{k}"),
        recall_value=_fmt(f"recall@{k}"),
        precision_value=_fmt(f"precision@{k}"),
        ild_value=_fmt(f"ild@{k}"),
        primary_key=primary_key,
        primary_value=_fmt(primary_key),
        guardrail_key=guardrail_key,
        guardrail_value=_fmt(guardrail_key),
        max_iters=max_iters,
    )
