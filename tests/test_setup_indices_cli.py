"""setup-indices CLI: wires adapter → ES client → bulk load, prints counts."""

from __future__ import annotations

from typing import Any

import pytest

from es_script_agent import cli


class _FakeClient: ...


def test_setup_indices_cmd_invokes_loader_and_prints_counts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_args: dict[str, Any] = {}

    def fake_make_client(url: str | None = None) -> Any:
        captured_args["client_url"] = url
        return _FakeClient()

    fake_adapter = object()

    def fake_load_dataset(name: str) -> Any:
        captured_args["dataset"] = name
        return fake_adapter

    def fake_setup(es: Any, adapter: Any) -> dict[str, int]:
        captured_args["client"] = es
        captured_args["adapter"] = adapter
        return {"loans": 17, "users": 5}

    monkeypatch.setattr(cli, "make_client", fake_make_client)
    monkeypatch.setattr(cli, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(cli, "setup_indices", fake_setup)
    monkeypatch.setattr("sys.argv", ["setup-indices"])

    cli.setup_indices_cmd()

    assert captured_args["dataset"] == "default"
    assert isinstance(captured_args["client"], _FakeClient)
    assert captured_args["adapter"] is fake_adapter

    out = capsys.readouterr().out
    assert "loans: 17" in out
    assert "users: 5" in out


def test_setup_indices_cmd_accepts_dataset_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, Any] = {}

    def fake_load_dataset(name: str) -> Any:
        seen["dataset"] = name
        return object()

    monkeypatch.setattr(cli, "make_client", lambda url=None: _FakeClient())
    monkeypatch.setattr(cli, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(cli, "setup_indices", lambda es, a: {"loans": 0, "users": 0})
    monkeypatch.setattr("sys.argv", ["setup-indices", "--dataset", "default"])

    cli.setup_indices_cmd()
    assert seen["dataset"] == "default"
