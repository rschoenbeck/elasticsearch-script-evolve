# AGENTS.md

Project-specific guidance for working in this repo. The authoritative documents are `plans/SPEC.md` (what we're building and why) and `tasks/plan.md` (how we're sequencing the build). Read them before any non-trivial change.

## What this is

A single-user research harness: a LangGraph agent iteratively rewrites Painless scoring scripts and is graded on IR metrics (NDCG@10, Recall@10, Precision@10, ILD@10) against a local Elasticsearch index of Kiva loans. Each iteration emits **one query script + zero-to-N sort scripts**; the harness owns the surrounding JSON shape.

Out of scope: production deployment, multi-user serving, model training, online learning, distributed ES. Don't add infrastructure for any of these.

## Stack

- Python 3.13, managed with `uv` (already initialized). Run everything through `uv run …`.
- Elasticsearch 8.13.4, single node, security off, via `docker compose`. Client pinned to `elasticsearch==8.13.*` (matches server major).
- LangGraph + LangChain for the agent loop; `langchain-anthropic` and `langchain-openai` as provider adapters.
- Pydantic for state types (`IterationRecord`, `EvalResult`, `AgentState`); dataclasses fine elsewhere.
- `numpy` for metric math, `pandas` for CSV only, `pytest` for tests, `jupyterlab` for analysis notebooks.
- Format + lint with `ruff` (single tool). Config lives in `pyproject.toml`.

Keep dependencies minimal — if you find yourself adding a package, check that it's not already covered by the above.

## Commands

| Command | Purpose |
|---|---|
| `docker compose up -d` / `down -v` | Start / stop+wipe local ES |
| `uv sync` | Install deps |
| `uv run setup-indices [--dataset flss]` | Drop+recreate `loans` and `users` indices, bulk-load from `data/*.json` |
| `uv run baseline` | Run baseline script set once; print + log metrics |
| `uv run rl-loop --iters N [--provider …] [--max-sort-scripts N] [--objective ndcg\|ild]` | Agentic improvement loop |
| `uv run eval --query <path> [--sort <path> …]` | Re-evaluate a saved script set |
| `pytest` / `pytest -m integration` | Unit tests / integration (needs ES up + `setup-indices`) |

Env vars (`.env`, gitignored): `ES_URL`, `LLM_PROVIDER`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.

## Code style

- Fully type-annotated. `pathlib.Path` everywhere, never `os.path`.
- Pure functions for metrics + query building; side-effectful I/O isolated to `es_client.py`, `runlog.py`, `indices/load.py`.
- No global mutable state. ES client created once at the CLI entry, threaded through.
- `logging`, not `print`, except for one-shot CLI status lines.
- Comments only where a non-obvious *why* matters (Painless quirks, ES version-specific behavior, metric edge cases). Don't narrate what the code does.
- Docstrings use Google format (`Args:` / `Returns:` / `Raises:` / `Attributes:` sections), not reStructuredText or NumPy style.

## Architectural invariants (the things easy to break)

- **Dataset access goes through `data/adapters/`.** Eval, indexing, and agent code consume normalized `Item`/`User`/`Interaction` only — no Elasticsearch field names outside the adapter. New datasets ship as a new adapter module.
- **Harness owns the query/sort JSON shape.** The agent edits only Painless `source` strings + the count of sort scripts (up to `--max-sort-scripts`). Tie-break is `_score desc`. `params.user_vector` is the user representation and is not renamed or stripped.
- **`params.user_vector` is 10 × 32.** Users carry 10 vectors of 32 dims each; items carry one 32-dim vector indexed as `item_vector`. The baseline mean-pools inside Painless; the agent can experiment with other pooling strategies inside the source body.
- **Run snapshots are immutable.** Before evaluating, snapshot `query.painless` + `sort_NN.painless` to `runs/<ts>/iter_NNN/`. JSONL points at the snapshot paths. Compile failures still snapshot the failing source.
- **ILD diversity field set is harness-owned and immutable mid-run.** Lives in `config.ILD_DIVERSITY_FIELDS`, recorded in `meta.json`. Disclosed to the agent in the prompt — the guardrail mechanism, not obscurity, prevents gaming.
- **Objective is symmetric.** `--objective ndcg` (default) → primary NDCG, guardrail ILD. `--objective ild` → flipped. All four metrics are logged every iteration regardless.
- **Tools exposed to the agent are read-only against ES (search) + write-only against per-run snapshot dirs.** No mapping changes, no admin APIs, no writes outside the run dir.

If you're about to change any of the above, that's a SPEC §6 "ask first" — surface it before editing.

## Testing

- Unit tests run by default. Integration tests (`pytest -m integration`) require ES up and `setup-indices` already run; they are the pre-experiment gate, not part of the default loop.
- After any harness change, `uv run baseline` should produce metrics identical to the last recorded baseline within float tolerance. Drift there means the eval harness moved and cross-run comparisons are invalid — investigate before proceeding.

## Repo conventions

- Specs and planning docs live in `plans/`, not the repo root.
- No `tests/__init__.py` — pytest + src layout discovers tests without it.
- Conventional commits with branch prefixes: `chore/…`, `feat/…`, `fix/…`. The first commit on a branch leads with the matching type (`chore:`, `feat:`, `fix:`).
- Gitignored and never committed: `data/`, `runs/`, `scripts/reference/`, `.env`, any API keys. Verify before commit.
