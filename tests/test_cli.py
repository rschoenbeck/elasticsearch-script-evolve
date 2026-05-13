"""Skeleton checks: package imports and unimplemented CLI stubs still raise."""

import importlib

import pytest


def test_package_imports() -> None:
    importlib.import_module("es_script_agent")


@pytest.mark.parametrize(
    "func_name",
    ["rl_loop_cmd", "eval_cmd"],
)
def test_cli_stub_raises_not_implemented(func_name: str) -> None:
    cli = importlib.import_module("es_script_agent.cli")
    func = getattr(cli, func_name)
    with pytest.raises(NotImplementedError):
        func()
