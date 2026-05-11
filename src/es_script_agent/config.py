"""Centralized configuration: env-var loading + path constants.

Loaded once at import time. Callers read attributes directly; do not mutate.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")

ES_URL: str = os.getenv("ES_URL", "http://localhost:9200")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

DATA_DIR: Path = _REPO_ROOT / "data"
RUNS_DIR: Path = _REPO_ROOT / "runs"
SCRIPTS_DIR: Path = _REPO_ROOT / "scripts"

RELEVANCE_THRESHOLD: float = 1
ILD_DIVERSITY_FIELDS: tuple[str, ...] = ("sector", "country", "partnerId")
DEFAULT_OBJECTIVE: str = "ndcg"
