# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Progress reporting for the byte-identity gate."""

import importlib.util
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize(
    ("flavor", "supports_target_ops", "expected_gate_calls"),
    [(None, False, 2), ("llvm23", True, 1), ("llvm24", True, 1)],
)
def test_differential_gate_streams_unbuffered_progress(
    monkeypatch, tmp_path, flavor, supports_target_ops, expected_gate_calls
):
    if flavor is None:
        monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)
    else:
        monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
    tools = Path(__file__).resolve().parents[3] / "tools"
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
        module, "_selected_profile_supports_target_ops", lambda: supports_target_ops
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["check_byte_identity.py", "--build-root", str(tmp_path)],
    )

    assert module.main() == 0
    gate_calls = [call for call in calls if str(module.RUN_DIFF) in call[0]]
    assert len(gate_calls) == expected_gate_calls
    if not supports_target_ops:
        primary, supplemental = gate_calls
        assert "--exclude" in primary[0]
        assert module.TARGET_OP_PROFILE_FAMILIES[0] in primary[0]
        assert supplemental[1]["env"]["ROCKE_LLVM_FLAVOR"] == "llvm23"
        only_index = supplemental[0].index("--only")
        supplemental_families = supplemental[0][only_index + 1].split(",")
        assert module.GFX1250_FAMILY in supplemental_families
        assert module.TARGET_OP_PROFILE_FAMILIES[0] in supplemental_families
    for command, kwargs in gate_calls:
        assert command[:3] == [module.sys.executable, "-u", str(module.RUN_DIFF)]
        if supports_target_ops:
            assert "--exclude" not in command
        assert "capture_output" not in kwargs
        assert "stdout" not in kwargs
        assert "stderr" not in kwargs
