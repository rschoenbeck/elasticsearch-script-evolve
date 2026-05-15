# Evolutionary agentic retrieval script optimization for Elasticsearch

## About

Most recommendation systems combine learned item and user vectors with
non-vector signals — popularity, recency, diversity, business rules — at a
scoring layer that sits *after* model training. That layer is usually
hand-tuned: an engineer edits a scoring formula, evaluates it against
held-out interactions, edits again. This repo is a research harness for
running that loop with an LLM agent in the driver's seat.

Each iteration the agent proposes a Painless scoring script (one query
script plus zero-to-N sort scripts); the harness evaluates it against a
fixed user cohort on NDCG@10, Recall@10, Precision@10, and ILD@10
(intra-list distance); the resulting metrics and the agent's own
rationale feed back into the next iteration's prompt. The aim is to
compress the post-training tuning cycle while keeping enough structural
guardrails that the agent has to genuinely improve metrics rather than
game the eval.

Day-to-day rules for contributors (humans or agents) live in `AGENTS.md`.

## Design highlights

- **Code harness maintains control over query/sort scripts.** The agent edits only
  Painless `source` strings and the count of sort scripts. It cannot
  change the eval cohort, the ground truth, the metric implementations,
  or the params — the only way to win is to write a better script.
- **Symmetric objective + guardrail.** Each run picks a primary metric
  (`ndcg` or `ild`); the other becomes a guardrail that must hold above
  baseline. A ranking that wins on one axis by collapsing the other is
  not a win. All four metrics are logged every iteration regardless of
  which is primary.
- **Immutable per-iteration snapshots.** Before evaluation, the candidate
  scripts are written to `runs/<ts>/iter_NNN/` and the path is recorded
  in the JSONL log. Compile failures still snapshot the failing source,
  so the agent can read its own mistake on the next turn.
- **Lineage flag for evolutionary code development.** `--lineage linear` always edits
  the previous iteration; `--lineage evolutionary` lets the agent pick a
  parent from the run history, with a penalty for repeated picks to encourage
  diverse attempts.

## Prerequisites

- macOS or Linux.
- [`uv`](https://docs.astral.sh/uv/) for Python 3.13 environment management.
- Docker (Compose v2) for the local Elasticsearch node.
- One of: an Anthropic API key, an OpenAI API key. The agent uses LangChain
  provider adapters; either works, but you'll need at least one to run the
  improvement loop.

## Setup

### 1. Install Python dependencies

```bash
uv sync
```

This pins Python 3.13 and installs the dependencies declared in
`pyproject.toml` (Elasticsearch client `8.13.*`, LangGraph/LangChain,
Pydantic, numpy, pytest, JupyterLab, ruff).

### 2. Configure environment

Copy the example file and fill in keys:

```bash
cp .env.example .env
$EDITOR .env
```

Keys used:

| Variable | Purpose |
|---|---|
| `ES_URL` | Elasticsearch endpoint (default `http://localhost:9200`). |
| `LLM_PROVIDER` | `anthropic` or `openai`. CLI `--provider` overrides. |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | At least one is required for `rl-loop`. |
| `ANTHROPIC_MODEL` / `OPENAI_MODEL` | Optional model overrides. |
| `DEFAULT_LOANS_PATH`, `DEFAULT_USERS_PATH`, `DEFAULT_INTERACTIONS_PATH` | Paths the default adapter reads from (relative to `data/`). |

`.env` is gitignored. Never commit API keys.

### 3. Provide a dataset

The repo ships **without** data. Drop the three files referenced above into
`data/` (gitignored). The default adapter expects:

- `loans.jsonl` — one item per line, with at minimum an `id`, a dense
  vector field, and the categorical attributes used by the ILD diversity
  field set.
- `users.jsonl` — one user per line, each with a `10 × 32` user vector.
- `interactions.csv` — `(item_id, user_id, weight)` rows; weights are
  thresholded to binary relevance during eval.

To wire in a different dataset, add a new module under
`src/es_script_agent/data/adapters/` that yields normalized `Item` / `User`
/ `Interaction` records (see `src/es_script_agent/data/schema.py`).

### 4. Start Elasticsearch

```bash
docker compose up -d
```

This brings up a single-node ES 8.13.4 on `localhost:9200` with security
disabled and a relaxed Painless compilation-rate limit (the agent
recompiles many scripts per run). Tear down with `docker compose down -v`
to wipe the data volume.

### 5. Create indices and bulk-load

```bash
uv run setup-indices              # uses --dataset default
uv run setup-indices --dataset foo  # alternative adapter
```

This is idempotent — it drops and recreates the `loans` and `users`
indices each time. Vector dimensionality is inferred from the first user
record and asserted constant across both indices.

## Running experiments

### Baseline

Evaluate the committed baseline script set (`scripts/baseline/`) once and
write a complete `runs/<timestamp>/` directory with `meta.json`,
`run.jsonl`, and a snapshot of the scripts under `iter_000/`:

```bash
uv run baseline
uv run baseline --objective ild   # flip primary/guardrail
```

The baseline's metrics seed the guardrail thresholds for the agentic loop,
so re-run this after any harness change to confirm metrics haven't drifted.

### Agentic improvement loop

```bash
uv run rl-loop --iters 20
uv run rl-loop --iters 20 --provider anthropic
uv run rl-loop --iters 20 --objective ild --max-sort-scripts 3
uv run rl-loop --iters 20 --lineage evolutionary
uv run rl-loop --iters 20 --hint "try mean-pooling the top-3 user vectors"
```

Flags:

- `--iters` (required): outer iteration budget.
- `--provider`: `anthropic` | `openai` (overrides `LLM_PROVIDER`).
- `--dataset`: registered adapter name (default `default`).
- `--max-sort-scripts`: cap on sort scripts per iteration (default 5).
- `--objective`: `ndcg` (default) or `ild`. The other metric becomes the
  guardrail and must not collapse below baseline.
- `--lineage`: `linear` (default) or `evolutionary` parent selection.
- `--baseline-dir`: alternative starting script set.
- `--hint`: free-text hint inlined into the agent's kick-off message.

Every run writes:

```
runs/<YYYYMMDD-HHMMSS>/
  meta.json            # dataset, threshold, K, cohort, ILD field set, objective, summary
  run.jsonl            # one line per iteration (metrics, rationale, parent, errors)
  iter_000/            # baseline snapshot
    query.painless
    sort_NN.painless   # zero or more
  iter_001/ ...        # one directory per agent iteration
```

Snapshots are immutable — compile failures still get a snapshot so the
agent can read its own mistake.

A `run.jsonl` line is one self-contained record per iteration:

```jsonc
{"iter": 2,
 "metrics": {"ndcg@10": 0.0915, "ild@10": 0.3931,
             "precision@10": 0.0289, "recall@10": 0.1379},
 "llm_rationale": "Start from baseline mean-pooled vector and add a small
                   popularity log boost; tests whether global item popularity
                   improves ranking without using diversity-specific fields.",
 "parent_iter": 0, "compile_error": null, ...}
```

`meta.json` records the run-level configuration plus a final summary:

```jsonc
{
  "objective": "ndcg",
  "lineage": "evolutionary",
  "baseline_metrics": {"ndcg@10": 0.0404, "ild@10": 0.2423, ...},
  "ild_diversity_fields": ["sector", "country", "partnerId"],
  "summary": {
    "best_iter": 10,
    "best_primary": 0.0939,
    "guardrail_held": true,
    "iters_attempted": 10,
    "final_message": "Best iteration was iter_10 — NDCG 0.0939 (baseline
                      0.0404), ILD 0.3945 (baseline 0.2423)..."
  }
}
```

### Re-evaluating a saved script set

```bash
uv run eval --set runs/20260513-184000/iter_007
uv run eval --query path/to/query.painless --sort path/to/sort_00.painless
```

`--set` points at any directory shaped like a script set (one
`query.painless` + zero or more `sort_NN.painless`). `--query` /
`--sort` let you mix-and-match individual files. `eval` prints metrics
but does **not** write a run directory.

### Analysis notebooks

```bash
uv run jupyter lab
```

`notebooks/analysis.ipynb` loads `runs/*/run.jsonl` into a dataframe and
charts metric trajectories across iterations.

## Script-set contract

A script set is a directory with exactly one `query.painless` and zero or
more `sort_NN.painless` files (zero-padded two-digit index = execution
order). The harness wraps these into the query JSON; the agent only edits
the Painless `source` bodies and the count of sort scripts.

Every script receives `params.user_vector`, a `10 × 32`
`List<List<Double>>`. The baseline mean-pools across the outer list
before cosine similarity against the indexed `item_vector` field — see
`scripts/baseline/query.painless`. Alternative pooling is fair game
inside the script body. Scripts must return a non-negative `double`.

Full contract docs: `scripts/README.md`.

## Testing

```bash
pytest
```

All current tests are unit tests — fast, hermetic, no Elasticsearch
required. The SPEC reserves an `integration` pytest marker for tests
that hit a live ES node, but none are implemented yet.

The real pre-experiment gate is `uv run baseline`: after any harness
change, re-run it and confirm metrics match the last recorded baseline
within float tolerance. Drift there means the eval harness moved and
cross-run comparisons are invalid.

## Project layout

```
src/es_script_agent/
  cli.py              # console-script entry points
  config.py           # env + path constants, ILD diversity field set
  agent/              # LangGraph loop, prompts, tools, lineage
  data/               # adapter protocol + normalized Item/User/Interaction
    adapters/         # one module per dataset
  es/                 # client factory, index setup, query builder, mappings
  eval/               # metrics, ground-truth filtering, evaluation runner
  runlog.py           # JSONL writer + reader, snapshotting
scripts/baseline/     # committed baseline script set
scripts/reference/    # gitignored; user-supplied few-shot examples
runs/                 # gitignored; one subdir per run
data/                 # gitignored; dataset files
tests/                # unit + integration tests
notebooks/            # analysis notebooks
```

## Troubleshooting

- **`circuit_breaking_exception` on bulk load.** The Docker container is
  capped at 1 GB heap. Drop the dataset size, or raise `ES_JAVA_OPTS` in
  `docker-compose.yml`.
- **`compilation_limit_exceeded`.** The compose file already raises the
  Painless compile-rate limit to `10000/1m`. If you still hit it, the
  agent is probably looping on broken scripts — check `compile_error` in
  the most recent `run.jsonl`.
- **`ground-truth filter: kept 0/N users`.** Interactions point at items
  that aren't in the `loans` index. Re-run `setup-indices` and verify the
  adapter is producing items with ids that match the interaction file.
- **Baseline metrics drift between runs on the same dataset.** Something
  in the eval path or the index has changed — investigate before
  trusting any subsequent agentic-run comparison.
