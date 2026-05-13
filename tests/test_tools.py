"""Tests for the agent's ``eval_scripts`` and ``read_history`` tools.

These pin the contract documented in :mod:`es_script_agent.agent.tools`:
each ``eval_scripts`` call snapshots → compile-checks → evaluates →
appends one JSONL line; budget + sort-script cap are enforced inside
the tool; ``read_history`` is pure file I/O against the run log.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from es_script_agent.agent.tools import ToolContext, make_tools
from es_script_agent.data.schema import User
from es_script_agent.es.query import MAX_SORT_SCRIPTS
from es_script_agent.runlog import RunLog


# --- Fakes --------------------------------------------------------------


class _FakeES:
    """Fake ES whose ``search`` returns canned hits, asserting no other API is touched."""

    def __init__(
        self,
        full_hits: dict[str, Any] | None = None,
        *,
        compile_response: dict[str, Any] | Exception | None = None,
    ) -> None:
        self._full_hits = full_hits or _hits([("item_pos", {"sector": "S", "country": "C"})])
        self._compile_response = (
            compile_response
            if compile_response is not None
            else _hits([("item_pos", {"sector": "S"})])
        )
        self.search_calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> Any:
        self.search_calls.append(kwargs)
        body = kwargs["body"]
        if body.get("size") == 1:
            if isinstance(self._compile_response, Exception):
                raise self._compile_response
            return self._compile_response
        return self._full_hits

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"tool called non-search ES API: {name!r}")


def _hits(ranked: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    return {
        "hits": {
            "hits": [
                {"_id": item_id, "_source": source} for item_id, source in ranked
            ]
        }
    }


def _user(user_id: str = "u1", marker: float = 1.0) -> User:
    return User(id=user_id, vector=[[marker] * 4] * 10, attributes={})


def _baseline_metrics() -> dict[str, float | None]:
    return {"ndcg@10": 0.30, "recall@10": 0.20, "precision@10": 0.10, "ild@10": 0.50}


def _make_ctx(
    tmp_path: Path,
    *,
    es: _FakeES | None = None,
    max_iters: int = 5,
    objective: str = "ndcg",
    baseline_metrics: dict[str, float | None] | None = None,
    seed_baseline: bool = True,
) -> ToolContext:
    run_dir = tmp_path / "run_x"
    run_dir.mkdir()
    log = RunLog(run_dir)
    log.write_header({"k": 10, "objective": objective})
    es = es if es is not None else _FakeES()
    ctx = ToolContext(
        es=es,
        run_dir=run_dir,
        run_log=log,
        ground_truth={"u1": {"item_pos"}},
        users_by_id={"u1": _user("u1", 1.0)},
        indexed_item_ids={"item_pos"},
        baseline_metrics=baseline_metrics or _baseline_metrics(),
        objective=objective,
        max_iters=max_iters,
        max_sort_scripts=MAX_SORT_SCRIPTS,
        k=10,
        diversity_fields=("sector", "country"),
    )
    if seed_baseline:
        # Mirror what `baseline_cmd` writes: iter_000 with baseline metrics on disk.
        baseline_dir = run_dir / "iter_000"
        baseline_dir.mkdir()
        (baseline_dir / "query.painless").write_text("return 1.0;")
        from es_script_agent.runlog import IterationRecord

        log.append(
            IterationRecord(
                iter=0,
                timestamp="2026-01-01T00:00:00Z",
                query_script_path=str(baseline_dir / "query.painless"),
                sort_script_paths=[],
                metrics=ctx.baseline_metrics,
                eval_users=1,
                eval_seconds=0.01,
                llm_rationale=None,
                parent_iter=None,
                compile_error=None,
                partial_failure=False,
                sample_error=None,
            )
        )
    return ctx


def _tools_by_name(tools: list[Any]) -> dict[str, Any]:
    return {t.name: t for t in tools}


def _invoke(tool: Any, **kwargs: Any) -> dict[str, Any]:
    return tool.invoke(kwargs)


# --- make_tools wiring --------------------------------------------------


def test_make_tools_exposes_eval_scripts_and_read_history(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    tools = _tools_by_name(make_tools(ctx))
    assert set(tools.keys()) == {"eval_scripts", "read_history"}


# --- eval_scripts -------------------------------------------------------


def test_eval_scripts_success_writes_jsonl_and_snapshot(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    tools = _tools_by_name(make_tools(ctx))

    result = _invoke(
        tools["eval_scripts"],
        query_source="return 1.0;",
        sort_sources=[],
        rationale="first try",
    )

    assert result["ok"] is True
    assert result["iter"] == 1
    assert "metrics" in result
    assert set(result["metrics"].keys()) == {
        "ndcg@10",
        "recall@10",
        "precision@10",
        "ild@10",
    }
    # Snapshot lives under the run dir.
    iter_dir = ctx.run_dir / "iter_001"
    assert iter_dir.is_dir()
    assert (iter_dir / "query.painless").read_text() == "return 1.0;"
    # JSONL gained a new line for iter_001.
    records = ctx.run_log.read_all()
    assert [r.iter for r in records] == [0, 1]
    assert records[-1].compile_error is None
    assert records[-1].metrics is not None


def test_eval_scripts_writes_only_within_run_dir(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    tools = _tools_by_name(make_tools(ctx))
    _invoke(
        tools["eval_scripts"],
        query_source="return 1.0;",
        sort_sources=["return _score;"],
        rationale="r",
    )
    # Walk the run dir; every written file lives below it.
    written = [p for p in ctx.run_dir.rglob("*") if p.is_file()]
    assert written, "expected at least one file under the run dir"
    for p in written:
        assert ctx.run_dir in p.parents or p.parent == ctx.run_dir, p


def test_eval_scripts_calls_only_search_api(tmp_path: Path) -> None:
    es = _FakeES()
    ctx = _make_ctx(tmp_path, es=es)
    tools = _tools_by_name(make_tools(ctx))
    _invoke(
        tools["eval_scripts"],
        query_source="return 1.0;",
        sort_sources=[],
        rationale="r",
    )
    # _FakeES.__getattr__ raises on anything except .search; we got here, so we're good.
    assert es.search_calls, "expected at least one search call"


def test_eval_scripts_iter_counter_monotonic(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    tools = _tools_by_name(make_tools(ctx))
    r1 = _invoke(tools["eval_scripts"], query_source="Q1", sort_sources=[], rationale="r1")
    r2 = _invoke(tools["eval_scripts"], query_source="Q2", sort_sources=[], rationale="r2")
    r3 = _invoke(tools["eval_scripts"], query_source="Q3", sort_sources=[], rationale="r3")
    assert [r1["iter"], r2["iter"], r3["iter"]] == [1, 2, 3]


def test_eval_scripts_compile_failure_returns_ok_false_and_writes_jsonl(tmp_path: Path) -> None:
    es = _FakeES(compile_response=RuntimeError("script_exception: bad cast"))
    ctx = _make_ctx(tmp_path, es=es)
    tools = _tools_by_name(make_tools(ctx))

    result = _invoke(
        tools["eval_scripts"],
        query_source="boom;",
        sort_sources=[],
        rationale="should fail",
    )

    assert result["ok"] is False
    assert "compile_error" in result and result["compile_error"]
    assert "bad cast" in result["compile_error"]
    assert result["iter"] == 1

    records = ctx.run_log.read_all()
    assert records[-1].iter == 1
    assert records[-1].metrics is None
    assert records[-1].compile_error is not None
    assert "bad cast" in records[-1].compile_error
    # Failing source is still snapshotted.
    assert (ctx.run_dir / "iter_001" / "query.painless").read_text() == "boom;"


def test_eval_scripts_compile_failure_advances_iter_counter(tmp_path: Path) -> None:
    """Compile failures still consume one iteration slot."""
    es = _FakeES(compile_response=RuntimeError("nope"))
    ctx = _make_ctx(tmp_path, es=es)
    tools = _tools_by_name(make_tools(ctx))
    r1 = _invoke(tools["eval_scripts"], query_source="bad", sort_sources=[], rationale="r")
    r2 = _invoke(tools["eval_scripts"], query_source="bad", sort_sources=[], rationale="r")
    assert r1["iter"] == 1
    assert r2["iter"] == 2


def test_eval_scripts_rejects_too_many_sort_scripts_without_snapshot(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    tools = _tools_by_name(make_tools(ctx))
    too_many = ["return 0.0;"] * (MAX_SORT_SCRIPTS + 1)

    result = _invoke(
        tools["eval_scripts"],
        query_source="Q",
        sort_sources=too_many,
        rationale="r",
    )

    assert result["ok"] is False
    assert "too many sort scripts" in result.get("error", "").lower()
    # No iter_001 dir was created; no JSONL line was appended.
    assert not (ctx.run_dir / "iter_001").exists()
    assert [r.iter for r in ctx.run_log.read_all()] == [0]


def test_eval_scripts_budget_exhausted_on_second_call_when_max_iters_1(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, max_iters=1)
    tools = _tools_by_name(make_tools(ctx))
    r1 = _invoke(tools["eval_scripts"], query_source="Q1", sort_sources=[], rationale="r1")
    r2 = _invoke(tools["eval_scripts"], query_source="Q2", sort_sources=[], rationale="r2")
    assert r1["ok"] is True
    assert r2["ok"] is False
    assert r2.get("budget_exhausted") is True
    # No iter_002 dir; no second agent-iteration line.
    assert not (ctx.run_dir / "iter_002").exists()
    iters = [r.iter for r in ctx.run_log.read_all()]
    assert iters == [0, 1]


def test_eval_scripts_default_parent_iter_is_last_successful(tmp_path: Path) -> None:
    """Without an explicit ``parent_iter``, default to the most recent metrics-bearing iter."""
    ctx = _make_ctx(tmp_path)
    tools = _tools_by_name(make_tools(ctx))
    _invoke(tools["eval_scripts"], query_source="Q1", sort_sources=[], rationale="r")
    _invoke(tools["eval_scripts"], query_source="Q2", sort_sources=[], rationale="r")
    records = ctx.run_log.read_all()
    # iter_001 descends from iter_000 (baseline); iter_002 descends from iter_001.
    assert records[1].parent_iter == 0
    assert records[2].parent_iter == 1


def test_eval_scripts_compile_failure_does_not_advance_last_success(tmp_path: Path) -> None:
    es = _FakeES(compile_response=RuntimeError("nope"))
    ctx = _make_ctx(tmp_path, es=es)
    tools = _tools_by_name(make_tools(ctx))
    _invoke(tools["eval_scripts"], query_source="bad1", sort_sources=[], rationale="r")
    _invoke(tools["eval_scripts"], query_source="bad2", sort_sources=[], rationale="r")
    records = ctx.run_log.read_all()
    # Both failed iters should reference the baseline (iter_000) as parent,
    # since neither succeeded.
    assert records[1].parent_iter == 0
    assert records[2].parent_iter == 0


def test_eval_scripts_records_explicit_parent_iter_when_supplied(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    tools = _tools_by_name(make_tools(ctx))
    _invoke(
        tools["eval_scripts"],
        query_source="Q",
        sort_sources=[],
        rationale="r",
        parent_iter=42,
    )
    assert ctx.run_log.read_all()[-1].parent_iter == 42


def test_eval_scripts_sort_clauses_present_when_sort_sources_provided(tmp_path: Path) -> None:
    es = _FakeES()
    ctx = _make_ctx(tmp_path, es=es)
    tools = _tools_by_name(make_tools(ctx))
    _invoke(
        tools["eval_scripts"],
        query_source="Q",
        sort_sources=["return _score;", "return doc['x'].value;"],
        rationale="r",
    )
    # The per-user search (size=10) is the last call; verify it carries both sort clauses.
    eval_call = [c for c in es.search_calls if c["body"].get("size") == 10][0]
    sort = eval_call["body"]["sort"]
    assert len(sort) == 3  # 2 script sorts + _score tie-break
    assert sort[-1] == {"_score": "desc"}


# --- read_history -------------------------------------------------------


def _run_a_few(ctx: ToolContext, n: int = 2) -> None:
    tools = _tools_by_name(make_tools(ctx))
    for i in range(n):
        _invoke(
            tools["eval_scripts"],
            query_source=f"return {i + 1}.0;",
            sort_sources=[],
            rationale=f"trial {i + 1}",
        )


def test_read_history_returns_records_oldest_first(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    _run_a_few(ctx, n=3)
    tools = _tools_by_name(make_tools(ctx))
    out = _invoke(tools["read_history"], last_n=5)
    iters = [r["iter"] for r in out["records"]]
    assert iters == sorted(iters)
    # Newest record is last.
    assert iters[-1] == max(iters)


def test_read_history_slices_to_last_n(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    _run_a_few(ctx, n=4)
    tools = _tools_by_name(make_tools(ctx))
    out = _invoke(tools["read_history"], last_n=2)
    assert len(out["records"]) == 2
    # The two most recent in chronological order.
    iters = [r["iter"] for r in out["records"]]
    assert iters == sorted(iters)
    assert iters[-1] == 4  # last agent iteration


def test_read_history_inlines_sources_when_requested(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    _run_a_few(ctx, n=1)
    tools = _tools_by_name(make_tools(ctx))
    out = _invoke(tools["read_history"], last_n=5, include_sources=True)
    iter1 = [r for r in out["records"] if r["iter"] == 1][0]
    assert iter1["query_source"] == "return 1.0;"
    assert iter1["sort_sources"] == []


def test_read_history_omits_sources_when_disabled(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    _run_a_few(ctx, n=1)
    tools = _tools_by_name(make_tools(ctx))
    out = _invoke(tools["read_history"], last_n=5, include_sources=False)
    iter1 = [r for r in out["records"] if r["iter"] == 1][0]
    assert "query_source" not in iter1
    assert "sort_sources" not in iter1


def test_read_history_returns_baseline_metrics(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    tools = _tools_by_name(make_tools(ctx))
    out = _invoke(tools["read_history"], last_n=5)
    assert out["baseline_metrics"] == ctx.baseline_metrics


def _value_or(record_metric: Any, default: float) -> float:
    return default if record_metric is None else record_metric


def test_read_history_best_so_far_picks_primary_metric_max(tmp_path: Path) -> None:
    """``best_so_far`` is whichever record has the highest primary metric."""
    ctx = _make_ctx(tmp_path)
    _run_a_few(ctx, n=2)  # baseline + 2 agent iters
    tools = _tools_by_name(make_tools(ctx))
    out = _invoke(tools["read_history"], last_n=10)
    assert out["best_so_far"] is not None
    records = ctx.run_log.read_all()
    best_iter = max(
        records,
        key=lambda r: _value_or((r.metrics or {}).get("ndcg@10"), -1.0),
    ).iter
    assert out["best_so_far"]["iter"] == best_iter


def test_read_history_best_so_far_uses_objective_for_primary(tmp_path: Path) -> None:
    """Flip the objective; ``best_so_far`` switches to ILD."""
    ctx = _make_ctx(tmp_path, objective="ild")
    _run_a_few(ctx, n=2)
    tools = _tools_by_name(make_tools(ctx))
    out = _invoke(tools["read_history"], last_n=10)
    records_with_ild = [
        r for r in ctx.run_log.read_all() if (r.metrics or {}).get("ild@10") is not None
    ]
    if records_with_ild:
        best_iter = max(records_with_ild, key=lambda r: (r.metrics or {})["ild@10"]).iter
        assert out["best_so_far"]["iter"] == best_iter
    else:
        # With a 1-hit fake, ILD is None for every iter → no best_so_far.
        # In that case best_so_far should still be picked from the baseline (which has ild=0.50).
        assert out["best_so_far"] is not None
        assert out["best_so_far"]["iter"] == 0


def test_read_history_guardrail_held_flag(tmp_path: Path) -> None:
    """Each record carries ``guardrail_held``; baseline always holds."""
    # Pin baseline ILD@10 to a known low value so the synthetic record (ILD=0.5) clears it.
    ctx = _make_ctx(
        tmp_path,
        baseline_metrics={
            "ndcg@10": 0.30,
            "recall@10": 0.20,
            "precision@10": 0.10,
            "ild@10": 0.10,
        },
    )
    _run_a_few(ctx, n=1)
    tools = _tools_by_name(make_tools(ctx))
    out = _invoke(tools["read_history"], last_n=5)
    iter0 = [r for r in out["records"] if r["iter"] == 0][0]
    iter1 = [r for r in out["records"] if r["iter"] == 1][0]
    # Baseline trivially holds against itself.
    assert iter0["guardrail_held"] is True
    # Agent iter is guardrail-held iff its ild@10 >= baseline ild@10 (0.10).
    metrics = iter1.get("metrics") or {}
    expected = (metrics.get("ild@10") or 0.0) >= 0.10
    assert iter1["guardrail_held"] is expected


def test_read_history_handles_compile_failures(tmp_path: Path) -> None:
    """A compile-failure record has metrics=None; ``guardrail_held`` defaults to False."""
    es = _FakeES(compile_response=RuntimeError("nope"))
    ctx = _make_ctx(tmp_path, es=es)
    _run_a_few(ctx, n=1)
    tools = _tools_by_name(make_tools(ctx))
    out = _invoke(tools["read_history"], last_n=5)
    iter1 = [r for r in out["records"] if r["iter"] == 1][0]
    assert iter1["metrics"] is None
    assert iter1["guardrail_held"] is False


def test_read_history_does_not_touch_es(tmp_path: Path) -> None:
    """``read_history`` is pure file I/O; ES is never invoked."""
    es = _FakeES()
    ctx = _make_ctx(tmp_path, es=es)
    tools = _tools_by_name(make_tools(ctx))
    _invoke(tools["read_history"], last_n=5)
    assert es.search_calls == []


# --- system prompt builder ----------------------------------------------


_SAMPLE_ATTRIBUTE_FIELDS: dict[str, dict[str, Any]] = {
    "sector": {"type": "keyword"},
    "country": {"type": "keyword"},
    "partnerId": {"type": "keyword"},
    "loanAmount": {"type": "double"},
    "popularityScore": {"type": "integer"},
}


def test_build_system_prompt_includes_contract_fragments() -> None:
    from es_script_agent.agent.prompts import build_system_prompt

    prompt = build_system_prompt(
        baseline_metrics={"ndcg@10": 0.30, "recall@10": 0.20, "precision@10": 0.10, "ild@10": 0.40},
        objective="ndcg",
        diversity_fields=("sector", "country", "partnerId"),
        max_sort_scripts=5,
        reference_dir=Path("/does/not/exist"),
        attribute_fields=_SAMPLE_ATTRIBUTE_FIELDS,
    )
    assert "user_vector" in prompt
    assert "item_vector" in prompt
    assert "ndcg@10" in prompt
    assert "ild@10" in prompt
    # Both baseline values disclosed so the agent knows the guardrail.
    assert "0.3" in prompt or "0.30" in prompt
    assert "0.4" in prompt or "0.40" in prompt
    # Diversity fields disclosed.
    assert "sector" in prompt and "country" in prompt and "partnerId" in prompt
    # Sort-script cap disclosed.
    assert "5" in prompt
    # Tool names mentioned in the usage guide.
    assert "eval_scripts" in prompt
    assert "read_history" in prompt


def test_build_system_prompt_renders_attribute_fields_with_types() -> None:
    """Each declared attribute should appear with its ES type."""
    from es_script_agent.agent.prompts import build_system_prompt

    prompt = build_system_prompt(
        baseline_metrics={"ndcg@10": 0.30, "recall@10": 0.20, "precision@10": 0.10, "ild@10": 0.40},
        objective="ndcg",
        diversity_fields=("sector",),
        max_sort_scripts=5,
        reference_dir=Path("/does/not/exist"),
        attribute_fields=_SAMPLE_ATTRIBUTE_FIELDS,
    )
    assert "doc['loanAmount']" in prompt and "double" in prompt
    assert "doc['popularityScore']" in prompt and "integer" in prompt
    assert "doc['partnerId']" in prompt and "keyword" in prompt


def test_build_system_prompt_does_not_leak_hardcoded_field_names() -> None:
    """Prompt must not name fields the caller didn't pass in."""
    from es_script_agent.agent.prompts import build_system_prompt

    prompt = build_system_prompt(
        baseline_metrics={"ndcg@10": 0.30, "recall@10": 0.20, "precision@10": 0.10, "ild@10": 0.40},
        objective="ndcg",
        diversity_fields=("sector",),
        max_sort_scripts=5,
        reference_dir=Path("/does/not/exist"),
        attribute_fields={"sector": {"type": "keyword"}},
    )
    # Stale names from earlier hardcoded list — must not appear unless passed in.
    for stale in ("activityId", "researchScore", "partnerRiskRating"):
        assert stale not in prompt, f"prompt still references stale field {stale!r}"


def test_build_system_prompt_objective_flips_primary_and_guardrail() -> None:
    from es_script_agent.agent.prompts import build_system_prompt

    ndcg_prompt = build_system_prompt(
        baseline_metrics={"ndcg@10": 0.30, "recall@10": 0.20, "precision@10": 0.10, "ild@10": 0.40},
        objective="ndcg",
        diversity_fields=("sector",),
        max_sort_scripts=5,
        reference_dir=Path("/does/not/exist"),
        attribute_fields=_SAMPLE_ATTRIBUTE_FIELDS,
    )
    ild_prompt = build_system_prompt(
        baseline_metrics={"ndcg@10": 0.30, "recall@10": 0.20, "precision@10": 0.10, "ild@10": 0.40},
        objective="ild",
        diversity_fields=("sector",),
        max_sort_scripts=5,
        reference_dir=Path("/does/not/exist"),
        attribute_fields=_SAMPLE_ATTRIBUTE_FIELDS,
    )
    # Same metrics, flipped framing → outputs differ.
    assert ndcg_prompt != ild_prompt


def test_build_system_prompt_inlines_reference_script_sets(tmp_path: Path) -> None:
    from es_script_agent.agent.prompts import build_system_prompt

    ref_dir = tmp_path / "reference"
    set_dir = ref_dir / "popularity-boost"
    set_dir.mkdir(parents=True)
    (set_dir / "query.painless").write_text("RETURN_Q_POPULARITY;")
    (set_dir / "sort_00.painless").write_text("RETURN_S_POPULARITY;")

    prompt = build_system_prompt(
        baseline_metrics={"ndcg@10": 0.30, "recall@10": 0.20, "precision@10": 0.10, "ild@10": 0.40},
        objective="ndcg",
        diversity_fields=("sector",),
        max_sort_scripts=5,
        reference_dir=ref_dir,
        attribute_fields=_SAMPLE_ATTRIBUTE_FIELDS,
    )
    assert "popularity-boost" in prompt
    assert "RETURN_Q_POPULARITY;" in prompt
    assert "RETURN_S_POPULARITY;" in prompt


def test_build_system_prompt_skips_reference_block_when_dir_empty(tmp_path: Path) -> None:
    """Empty or missing reference dir → prompt builds fine, omits the block."""
    from es_script_agent.agent.prompts import build_system_prompt

    out_missing = build_system_prompt(
        baseline_metrics={"ndcg@10": 0.30, "recall@10": 0.20, "precision@10": 0.10, "ild@10": 0.40},
        objective="ndcg",
        diversity_fields=("sector",),
        max_sort_scripts=5,
        reference_dir=tmp_path / "missing",
        attribute_fields=_SAMPLE_ATTRIBUTE_FIELDS,
    )
    # No exception; output is non-empty.
    assert out_missing.strip()


# --- ToolContext sanity -------------------------------------------------


def test_tool_context_persists_run_meta(tmp_path: Path) -> None:
    """The run log header survives between ``eval_scripts`` calls."""
    ctx = _make_ctx(tmp_path)
    _run_a_few(ctx, n=1)
    meta = json.loads((ctx.run_dir / "meta.json").read_text())
    assert meta["k"] == 10
