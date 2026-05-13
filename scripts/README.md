# Painless script sets

A **script set** is a directory containing exactly one `query.painless` and
zero or more `sort_NN.painless` files, where `NN` is a zero-padded two-digit
index. The numeric prefix is the execution order — because the index is
zero-padded, lexicographic sort equals numeric sort, and lookups stay simple.

```
scripts/baseline/
  query.painless          # required, exactly one
  sort_00.painless        # optional, executed first
  sort_01.painless        # optional, executed second
  ...
```

The same contract applies to three places in the repo:

- `scripts/baseline/` — the generic, committed reference script set used as
  the starting point for the agent loop and as the comparison baseline for
  every experiment. Tracked in git.
- `scripts/reference/<set-name>/` — private, user-supplied script sets the
  agent may see as few-shot examples. This whole directory is gitignored;
  drop any number of subdirectories here to seed the agent without leaking
  proprietary scoring logic into version control.
- `runs/<timestamp>/iter_NNN/` — immutable per-iteration snapshots written
  by the harness. Same shape, written automatically before each evaluation.

## What the script bodies own (and what they don't)

Authors edit only the Painless `source` body of each file. The harness owns
the surrounding JSON shape: `script_score` wrapping for the query, sort
direction and `_score desc` tie-break for sort scripts, and the params bag
passed in at query time. Scripts must not reference fields outside the
normalized item schema.

Every script — query and sort alike — receives `params.user_vector`, a
10×32 list of doubles representing the user's ten learned representations.
The baseline mean-pools across the outer list before computing cosine
similarity against the indexed `item_vector` field; alternative pooling
strategies are fair game inside the script body.

Both query and sort scripts must return a non-negative `double` (the query
script feeds `script_score`, and sort scripts default to descending order).
Return `0.0` when the document is missing the inputs the script needs.
