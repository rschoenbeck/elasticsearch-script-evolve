"""Elasticsearch client factory and response-side error rendering.

One client per process, instantiated at the CLI entry and threaded through.
No module-level client; callers own the lifecycle.
"""

from __future__ import annotations

from elasticsearch import Elasticsearch

from es_script_agent import config


def make_client(url: str | None = None) -> Elasticsearch:
    return Elasticsearch(url or config.ES_URL)


def format_es_error(exc: Exception) -> str:
    """Render an ES exception with Painless ``script_stack`` and ``caused_by`` detail.

    ``elasticsearch.ApiError.__str__`` collapses to ``root_cause[0].reason``,
    which for Painless failures is the unhelpful ``"compile error"``. The
    actual compiler message lives in ``body.error.root_cause[0].caused_by``
    and the offending source span is in ``script_stack``. Falls back to
    ``str(exc)`` for non-API exceptions or unexpected body shapes.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return str(exc)
    error = body.get("error")
    if not isinstance(error, dict):
        return str(exc)
    root = error.get("root_cause")
    rc = root[0] if isinstance(root, list) and root else error
    if not isinstance(rc, dict):
        return str(exc)
    parts = [f"{rc.get('type', '?')}: {rc.get('reason', '?')}"]
    stack = rc.get("script_stack")
    if isinstance(stack, list) and stack:
        parts.append("script_stack: " + " | ".join(str(s).strip() for s in stack))
    cause = rc.get("caused_by")
    while isinstance(cause, dict):
        parts.append(f"caused_by {cause.get('type', '?')}: {cause.get('reason', '?')}")
        cause = cause.get("caused_by")
    return " ".join(parts)
