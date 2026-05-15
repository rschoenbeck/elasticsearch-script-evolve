# Evolutionary agentic retrieval script optimization for Elasticsearch

A single-user research harness where a LangGraph agent iteratively rewrites
Painless scoring scripts and is graded on IR metrics (NDCG@10, Recall@10,
Precision@10, ILD@10) against a local Elasticsearch index. Each iteration
emits **one query script + zero-to-N sort scripts**; the harness owns the
surrounding query JSON, the eval cohort, the ground truth, and the run log.

The authoritative design docs are in `plans/`:

- `plans/SPEC.md` — what we're building and why.
- `plans/SPEC_EVOLUTIONARY_LINEAGE.md` — the evolutionary parent-selection mode.

Day-to-day rules for contributors (humans or agents) live in `AGENTS.md`.

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
- `--lineage`: `linear` (default) or `evolutionary` parent selection. See
  `plans/SPEC_EVOLUTIONARY_LINEAGE.md`.
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
pytest                  # unit tests; fast, hermetic
pytest -m integration   # needs ES up and `setup-indices` already run
```

Integration tests are the pre-experiment gate, not part of the default
loop. After any harness change, re-run `uv run baseline` and confirm
metrics match the last recorded baseline within float tolerance — drift
there means the eval harness moved and cross-run comparisons are invalid.

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
plans/                # specs
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
