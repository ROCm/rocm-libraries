# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the superbuild skills' stage_shadowed_dlls.py helper.

Staging is Windows-only, so each test pretends to be on Windows and disables PE
version lookup; staging decisions then fall back to the content comparison,
which behaves the same on every host.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "skills"
    / "hipdnn-superbuild"
    / "scripts"
    / "stage_shadowed_dlls.py"
)


@pytest.fixture
def stage_mod(monkeypatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("hipdnn_stage_shadowed_dlls", SCRIPT)
    assert spec and spec.loader, f"could not load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(module, "dll_version", lambda path: None)
    return module


def _rocm_bin(tmp_path: Path, *names: str) -> Path:
    rocm_bin = tmp_path / "rocm" / "bin"
    rocm_bin.mkdir(parents=True)
    for name in names:
        (rocm_bin / name).write_bytes(f"wheel {name}".encode())
    return rocm_bin


def test_stages_comgr_and_hip_runtime_only(stage_mod, tmp_path):
    """The HIP runtime is shadowed like comgr; staging only comgr let the driver's
    System32 amdhip64 crash the wheel's rocBLAS. Unshadowed DLLs stay put."""
    rocm_bin = _rocm_bin(tmp_path, "amd_comgr.dll", "amdhip64_7.dll", "rocblas.dll")
    dest = tmp_path / "build" / "bin"

    actions = stage_mod.stage_shadowed_dlls(rocm_bin, dest)

    assert actions == {"amd_comgr.dll": "copied", "amdhip64_7.dll": "copied"}
    assert sorted(p.name for p in dest.iterdir()) == ["amd_comgr.dll", "amdhip64_7.dll"]
    assert (dest / "amdhip64_7.dll").read_bytes() == b"wheel amdhip64_7.dll"


def test_missing_hip_runtime_is_reported_not_skipped(stage_mod, tmp_path):
    """A wheel with no amdhip64_*.dll must fail loudly, or the stale System32
    runtime would silently load."""
    rocm_bin = _rocm_bin(tmp_path, "amd_comgr.dll")

    actions = stage_mod.stage_shadowed_dlls(rocm_bin, tmp_path / "bin")

    assert actions[stage_mod.HIP_RUNTIME_GLOB] == "missing-source"
    assert actions["amd_comgr.dll"] == "copied"


def test_restages_only_when_wheel_copy_changes(stage_mod, tmp_path):
    rocm_bin = _rocm_bin(tmp_path, "amd_comgr.dll", "amdhip64_7.dll")
    dest = tmp_path / "bin"
    stage_mod.stage_shadowed_dlls(rocm_bin, dest)

    assert set(stage_mod.stage_shadowed_dlls(rocm_bin, dest).values()) == {"up-to-date"}

    (rocm_bin / "amdhip64_7.dll").write_bytes(b"newer wheel runtime")
    actions = stage_mod.stage_shadowed_dlls(rocm_bin, dest)

    assert actions == {"amd_comgr.dll": "up-to-date", "amdhip64_7.dll": "copied"}
    assert (dest / "amdhip64_7.dll").read_bytes() == b"newer wheel runtime"
