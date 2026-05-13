"""Elasticsearch client factory.

One client per process, instantiated at the CLI entry and threaded through.
No module-level client; callers own the lifecycle.
"""

from __future__ import annotations

from elasticsearch import Elasticsearch

from es_script_agent import config


def make_client(url: str | None = None) -> Elasticsearch:
    return Elasticsearch(url or config.ES_URL)
