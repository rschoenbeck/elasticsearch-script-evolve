"""Skeleton checks: package imports cleanly."""

import importlib


def test_package_imports() -> None:
    importlib.import_module("es_script_agent")
