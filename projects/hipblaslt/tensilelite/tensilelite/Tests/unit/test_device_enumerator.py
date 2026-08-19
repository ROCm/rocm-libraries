# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from tensilelite.Common import Architectures
from tensilelite.Common.Types import IsaVersion
from tensilelite.Toolchain import Validators


pytestmark = pytest.mark.unit


def test_device_enumerator_candidates_prefer_offload_arch(monkeypatch):
    calls = []

    def validate(name):
        calls.append(name)
        return f"/selected/{name}"

    monkeypatch.setattr(Validators, "validateToolchain", validate)
    monkeypatch.setattr(Validators, "isRhel8", lambda: False)

    assert Validators.deviceEnumeratorCandidates() == (
        "/selected/offload-arch",
        "/selected/amdgpu-arch",
    )
    assert calls == ["offload-arch", "amdgpu-arch"]


def test_device_enumerator_candidates_keep_amdgpu_arch_as_fallback(monkeypatch):
    calls = []

    def validate(name):
        calls.append(name)
        if name == "offload-arch":
            raise FileNotFoundError(name)
        return f"/selected/{name}"

    monkeypatch.setattr(Validators, "validateToolchain", validate)
    monkeypatch.setattr(Validators, "isRhel8", lambda: False)

    assert Validators.deviceEnumeratorCandidates() == ("/selected/amdgpu-arch",)
    assert calls == ["offload-arch", "amdgpu-arch"]


def test_detect_current_isa_tries_amdgpu_arch_after_offload_arch(monkeypatch):
    calls = []

    def ask(tool):
        calls.append(tool)
        # offload-arch answers nothing; amdgpu-arch names the device.
        return [] if tool == "/selected/offload-arch" else ["gfx942"]

    monkeypatch.setattr(Architectures, "_fromEnumerator", ask)
    monkeypatch.setattr(Architectures, "restore_steppings", lambda archs: archs)

    assert Architectures.detectGlobalCurrentISA(
        0,
        ("/selected/offload-arch", "/selected/amdgpu-arch"),
    ) == IsaVersion(9, 4, 2)
    assert calls == ["/selected/offload-arch", "/selected/amdgpu-arch"]
