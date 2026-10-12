# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Progress reporting for the byte-identity gate."""

import importlib.util
from pathlib import Path
import subprocess


def test_differential_gate_streams_unbuffered_progress(monkeypatch, tmp_path):
    tools = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    path = tools / "check_byte_identity.py"
    spec = importlib.util.spec_from_file_location("check_byte_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:2] == ["cmake", "--build"]:
            (tmp_path / "librocke_core.a").touch()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "_cxx", lambda: None)
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["check_byte_identity.py", "--build-root", str(tmp_path)],
    )

    assert module.main() == 0
    gate_calls = [call for call in calls if str(module.RUN_DIFF) in call[0]]
    assert len(gate_calls) == 1
    command, kwargs = gate_calls[0]
    assert command[:3] == [module.sys.executable, "-u", str(module.RUN_DIFF)]
    assert "capture_output" not in kwargs
    assert "stdout" not in kwargs
    assert "stderr" not in kwargs
