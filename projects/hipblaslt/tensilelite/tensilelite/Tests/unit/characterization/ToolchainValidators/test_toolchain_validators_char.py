################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization coverage for selected-installation toolchain validation."""

from pathlib import Path

import pytest

from tensilelite.Toolchain import Validators


pytestmark = pytest.mark.unit


def _executable(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_supported_component_matches_basename():
    assert Validators._supportedComponent("/a/b/amdclang", ["amdclang"])
    assert not Validators._supportedComponent("amdclang", ["clang"])


def test_current_supported_component_predicates():
    assert Validators.supportedCxxCompiler("amdclang++")
    assert Validators.supportedCxxCompiler("clang++")
    assert not Validators.supportedCxxCompiler("g++")
    assert Validators.supportedCCompiler("amdclang")
    assert Validators.supportedCCompiler("clang")
    assert not Validators.supportedCCompiler("gcc")
    assert Validators.supportedOffloadBundler("clang-offload-bundler")
    assert not Validators.supportedOffloadBundler("clang-offload-bundlerx")
    assert Validators.supportedHip("hipcc")
    assert Validators.supportedHip("hipconfig")
    assert not Validators.supportedHip("hipcc.exe")
    assert Validators.supportedDeviceEnumerator("offload-arch")
    assert Validators.supportedDeviceEnumerator("amdgpu-arch")
    assert not Validators.supportedDeviceEnumerator("device-enumerator")


def test_validate_toolchain_resolves_a_relative_component(monkeypatch, tmp_path):
    _executable(tmp_path, "amdclang")
    monkeypatch.setattr(Validators, "executable_search_paths", lambda: [tmp_path])

    assert Validators.validateToolchain("amdclang") == str(tmp_path / "amdclang")


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
