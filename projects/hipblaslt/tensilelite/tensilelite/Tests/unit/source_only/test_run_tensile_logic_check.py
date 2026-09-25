# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

_HIPBLASLT_ROOT = Path(__file__).resolve().parents[5]
_SCRIPT = _HIPBLASLT_ROOT / "scripts/run_tensile_logic_check.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_tensile_logic_check", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_logic_check_uses_installed_command_and_required_options(tmp_path, monkeypatch):
    module = _load_script()
    logic_root = tmp_path / "library"
    logic_root.mkdir()
    calls = []
    monkeypatch.setattr(module, "_hipblaslt_root", lambda: tmp_path)
    monkeypatch.setattr(module.sys, "executable", "/selected/python")
    monkeypatch.setattr(module.subprocess, "call", lambda command: calls.append(command) or 7)

    assert module.main([]) == 7
    assert calls == [
        [
            "/selected/python",
            "-m",
            "tensilelite",
            "logic",
            str(logic_root.resolve()),
            "--use-bundled-known-bugs",
            "--check-all",
        ]
    ]


@pytest.mark.parametrize(
    "known_bugs_option",
    [
        ["--known-bugs", "custom.yaml"],
        ["--known-bugs=custom.yaml"],
        ["--use-bundled-known-bugs"],
    ],
)
def test_explicit_logic_path_and_known_bugs_option_are_preserved(
    tmp_path, monkeypatch, known_bugs_option
):
    module = _load_script()
    logic_root = tmp_path / "logic"
    logic_root.mkdir()
    calls = []
    monkeypatch.setattr(module, "_hipblaslt_root", lambda: tmp_path)
    monkeypatch.setattr(module.sys, "executable", "/selected/python")
    monkeypatch.setattr(module.subprocess, "call", lambda command: calls.append(command) or 0)

    assert module.main([str(logic_root), *known_bugs_option, "--check-all"]) == 0
    command = calls[0]
    assert command[:5] == [
        "/selected/python",
        "-m",
        "tensilelite",
        "logic",
        str(logic_root.resolve()),
    ]
    assert command[5:] == [*known_bugs_option, "--check-all"]
