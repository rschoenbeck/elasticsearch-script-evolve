"""Centralized configuration: env-var loading + path constants.

Loaded once at import time. Callers read attributes directly; do not mutate.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _find_repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "pyproject.toml").is_file():
            return p
    raise RuntimeError("repo root not found (no pyproject.toml in any parent)")


_REPO_ROOT = _find_repo_root()
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
