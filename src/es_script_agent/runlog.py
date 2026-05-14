"""Per-run JSONL log + per-iteration script snapshot.

Each run lives in ``runs/<YYYYMMDD-HHMMSS>/`` and contains:

- ``meta.json`` — immutable run header (dataset, K, objective, diversity
  field set, baseline metrics, harness version). The agent and the
  analysis notebook read this to interpret iterations.
- ``run.jsonl`` — one :class:`IterationRecord` per line, in iteration
  order. Append-only.
- ``iter_NNN/`` — per-iteration Painless snapshots written *before*
  evaluation so a compile failure still leaves the failing source on
  disk. The JSONL record's ``query_script_path`` / ``sort_script_paths``
  point here.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from es_script_agent.eval.runner import ScriptSet


class IterationRecord(BaseModel):
    """One row of ``run.jsonl`` — the full record for a single iteration.

    Compile failures keep ``metrics`` as ``None`` while ``compile_error``
    carries the message and the script paths still point at the
    snapshotted (failing) source.

    Attributes:
        iter: Zero-based iteration index. ``0`` is always the baseline.
        timestamp: ISO-8601 UTC timestamp with trailing ``Z``.
        query_script_path: Path to the snapshotted query script.
        sort_script_paths: Paths to the snapshotted sort scripts, in
            execution order.
        metrics: Aggregated metric dict, or ``None`` on compile failure.
        eval_users: Users that returned hits during this iteration.
        eval_seconds: Wall-clock seconds for the evaluation pass.
        llm_rationale: Free-text rationale from the agent. ``None`` for
            non-agentic iterations (e.g. baseline).
        parent_iter: The iteration this candidate descends from. ``None``
            when there's no parent (baseline).
        compile_error: Captured compile / runtime error, or ``None``.
        partial_failure: ``True`` iff some users failed during eval.
        sample_error: First captured per-user error, if any.
    """

    model_config = ConfigDict(frozen=True)

    iter: int
    timestamp: str
    query_script_path: str
    sort_script_paths: list[str]
    metrics: dict[str, float | None] | None
    eval_users: int
    eval_seconds: float
    llm_rationale: str | None
    parent_iter: int | None
    compile_error: str | None
    partial_failure: bool
    sample_error: str | None


class RunLog:
    """Writer/reader for a single run's ``meta.json`` + ``run.jsonl``.

    Attributes:
        dir: Run directory. Must already exist when ``RunLog`` is
            constructed; :func:`new_run_dir` is the canonical way to
            create one.
    """

    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir
        self._meta_path = run_dir / "meta.json"
        self._jsonl_path = run_dir / "run.jsonl"

    def write_header(self, meta: dict[str, Any]) -> None:
        """Persist the immutable run header. Overwrites any prior file."""
        self._meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    def read_meta(self) -> dict[str, Any]:
        """Load the run header. Raises if absent."""
        return json.loads(self._meta_path.read_text())

    def append(self, record: IterationRecord) -> None:
        """Append one iteration record as a JSON line."""
        with self._jsonl_path.open("a") as f:
            f.write(record.model_dump_json() + "\n")

    def read_all(self) -> list[IterationRecord]:
        """Load every record in order. Empty list if the file is absent."""
        if not self._jsonl_path.exists():
            return []
        return [
            IterationRecord.model_validate_json(line)
            for line in self._jsonl_path.read_text().splitlines()
            if line.strip()
        ]


def check_guardrail(
    metrics: dict[str, float | None] | None,
    key: str,
    baseline: dict[str, float | None],
) -> bool:
    """Return ``True`` iff ``metrics[key]`` is at least ``baseline[key]``.

    A missing baseline is treated as vacuously holding; a missing
    record value (or ``metrics is None``) is treated as not holding —
    failures shouldn't be allowed to claim the guardrail.
    """
    if not metrics:
        return False
    baseline_value = baseline.get(key)
    if baseline_value is None:
        return True
    record_value = metrics.get(key)
    if record_value is None:
        return False
    return record_value >= baseline_value


def snapshot_script_set(
    run_dir: Path,
    iter_n: int,
    script_set: ScriptSet,
) -> tuple[Path, list[Path]]:
    """Write the script set into ``<run_dir>/iter_NNN/`` before evaluation.

    Snapshots are immutable: this helper refuses to overwrite an
    existing iteration directory, since duplicated iter numbers would
    silently corrupt cross-iteration comparison.

    Args:
        run_dir: The parent run directory.
        iter_n: Zero-based iteration index.
        script_set: The Painless sources to snapshot.

    Returns:
        ``(query_path, sort_paths)`` — absolute paths to the written files.

    Raises:
        FileExistsError: If ``iter_NNN/`` already exists.
    """
    iter_dir = run_dir / f"iter_{iter_n:03d}"
    iter_dir.mkdir(parents=True, exist_ok=False)
    query_path = iter_dir / "query.painless"
    query_path.write_text(script_set.query_source)
    sort_paths: list[Path] = []
    for i, source in enumerate(script_set.sort_sources):
        path = iter_dir / f"sort_{i:02d}.painless"
        path.write_text(source)
        sort_paths.append(path)
    return query_path, sort_paths


def new_run_dir(base: Path) -> Path:
    """Create a fresh ``<base>/<YYYYMMDD-HHMMSS>/`` directory and return it."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = base / timestamp
    path.mkdir(parents=True, exist_ok=False)
    return path
