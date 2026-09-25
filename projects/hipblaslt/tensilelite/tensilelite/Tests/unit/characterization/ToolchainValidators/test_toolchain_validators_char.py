################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization coverage for selected-installation toolchain validation."""

from pathlib import Path

import pytest

from tensilelite.Toolchain import Validators


pytestmark = pytest.mark.unit


def test_supported_component_matches_basename():
    assert Validators._supportedComponent("/a/b/amdclang", ["amdclang"])
    assert not Validators._supportedComponent("amdclang", ["clang"])


@pytest.mark.parametrize(
    "predicate,name,expected",
    [
        (Validators.supportedCxxCompiler, "amdclang++", True),
        (Validators.supportedCxxCompiler, "clang++", True),
        (Validators.supportedCxxCompiler, "/opt/rocm/bin/amdclang++", True),
        (Validators.supportedCxxCompiler, "amdclang", False),
        (Validators.supportedCxxCompiler, "g++", False),
        (Validators.supportedCCompiler, "amdclang", True),
        (Validators.supportedCCompiler, "clang", True),
        (Validators.supportedCCompiler, "/usr/bin/clang", True),
        (Validators.supportedCCompiler, "amdclang++", False),
        (Validators.supportedCCompiler, "gcc", False),
        (Validators.supportedOffloadBundler, "clang-offload-bundler", True),
        (Validators.supportedOffloadBundler, "clang", False),
        (Validators.supportedOffloadBundler, "clang-offload-bundlerx", False),
        (Validators.supportedHip, "hipcc", True),
        (Validators.supportedHip, "hipconfig", True),
        (Validators.supportedHip, "hipcc.exe", False),
        (Validators.supportedDeviceEnumerator, "offload-arch", True),
        (Validators.supportedDeviceEnumerator, "rocm_agent_enumerator", True),
        (Validators.supportedDeviceEnumerator, "amdgpu-arch", True),
        (Validators.supportedDeviceEnumerator, "/opt/rocm/bin/amdgpu-arch", True),
        (Validators.supportedDeviceEnumerator, "hipinfo", False),
        (Validators.supportedDeviceEnumerator, "device-enumerator", False),
    ],
)
def test_current_supported_component_predicates(predicate, name, expected):
    assert predicate(name) is expected


def test_executable_exists_and_missing(tmp_path):
    executable = _executable(tmp_path, "amdclang++")

    assert Validators._exeExists(executable)
    assert not Validators._exeExists(tmp_path / "missing")


def test_validate_toolchain_resolves_a_relative_component(monkeypatch, tmp_path):
    _executable(tmp_path, "amdclang")
    monkeypatch.setattr(Validators, "executable_search_paths", lambda: [tmp_path])

    assert Validators.validateToolchain("amdclang") == str(tmp_path / "amdclang")


def test_validate_toolchain_rejects_missing_relative_component(monkeypatch, tmp_path):
    monkeypatch.setattr(Validators, "executable_search_paths", lambda: [tmp_path])

    with pytest.raises(FileNotFoundError):
        Validators.validateToolchain("amdclang++")


def test_validate_executable_preserves_absolute_and_rejection_contracts(tmp_path):
    executable = _executable(tmp_path, "amdclang++")

    assert Validators._validateExecutable(str(executable), []) == str(executable)
    with pytest.raises(FileNotFoundError):
        Validators._validateExecutable(str(tmp_path / "missing" / "amdclang++"), [])
    with pytest.raises(ValueError):
        Validators._validateExecutable(str(_executable(tmp_path, "g++")), [tmp_path])


def test_validate_toolchain_preserves_zero_scalar_and_tuple_contracts(monkeypatch, tmp_path):
    _executable(tmp_path, "amdclang++")
    _executable(tmp_path, "amdclang")
    monkeypatch.setattr(Validators, "executable_search_paths", lambda: [tmp_path])

    with pytest.raises(ValueError):
        Validators.validateToolchain()
    assert Validators.validateToolchain("amdclang++") == str(tmp_path / "amdclang++")
    assert Validators.validateToolchain("amdclang++", "amdclang") == (
        str(tmp_path / "amdclang++"),
        str(tmp_path / "amdclang"),
    )


def test_toolchain_defaults_and_os_selection_match_posix():
    defaults = Validators.ToolchainDefaults

    assert defaults.CXX_COMPILER == "amdclang++"
    assert defaults.C_COMPILER == "amdclang"
    assert defaults.OFFLOAD_BUNDLER == "clang-offload-bundler"
    assert defaults.ASSEMBLER == "amdclang++"
    assert defaults.HIP_CONFIG == "hipconfig"
    assert defaults.DEVICE_ENUMERATOR == "offload-arch"
    assert Validators.osSelect(linux="linux", windows="windows") == "linux"


def test_windows_extensions_are_rejected_on_posix():
    with pytest.raises(ValueError):
        Validators._windowsWithExtensions("amdclang++")


@pytest.mark.nt_path_simulation
def test_windows_extensions_and_supported_components(monkeypatch):
    monkeypatch.setattr(Validators.os, "name", "nt")
    monkeypatch.setenv("PATHEXT", ".EXE;.BAT")

    assert Validators._windowsWithExtensions("amdclang++") == [
        "amdclang++",
        "amdclang++.exe",
        "amdclang++.bat",
    ]
    assert Validators._supportedComponent("amdclang++", ["amdclang++"])
    assert Validators._supportedComponent("amdclang++.exe", ["amdclang++"])
    assert Validators.supportedDeviceEnumerator("hipinfo")
    assert Validators.supportedDeviceEnumerator("hipInfo")
    assert not Validators.supportedDeviceEnumerator("amdgpu-arch")


def test_device_enumerator_candidates_use_rhel_compatibility_fallback(monkeypatch):
    calls = []

    def validate(name):
        calls.append(name)
        if name in ("offload-arch", "amdgpu-arch"):
            raise FileNotFoundError(name)
        return f"/selected/{name}"

    monkeypatch.setattr(Validators, "validateToolchain", validate)
    monkeypatch.setattr(Validators, "isRhel8", lambda: True)
    monkeypatch.setattr(Validators.ToolchainDefaults, "inFFMEnv", False)

    assert Validators.deviceEnumeratorCandidates() == ("/selected/rocm_agent_enumerator",)
    assert calls == ["offload-arch", "amdgpu-arch", "rocm_agent_enumerator"]


def _executable(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path
