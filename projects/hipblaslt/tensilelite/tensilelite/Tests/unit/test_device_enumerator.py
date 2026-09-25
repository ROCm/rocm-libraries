# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse

import pytest

from tensilelite.Common import Architectures
from tensilelite.Common.Types import IsaVersion
from tensilelite.Toolchain import Validators
from tensilelite import tensilelite


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("option", ["--device-enumerator", "--rocm-agent-enumerator"])
def test_device_enumerator_cli_aliases_share_one_value(option):
    parser = argparse.ArgumentParser()
    tensilelite.addCommonArguments(parser)

    args = parser.parse_args([option, "/selected/offload-arch"])

    assert args.device_enumerator == "/selected/offload-arch"


def test_device_enumerator_cli_defaults_to_fallback_selection():
    parser = argparse.ArgumentParser()
    tensilelite.addCommonArguments(parser)

    assert parser.parse_args([]).device_enumerator is None


def test_device_enumerator_cli_value_is_forwarded(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tensilelite,
        "deviceEnumeratorCandidates",
        lambda explicit: calls.append(explicit) or ("/selected/enumerator",),
    )

    assert tensilelite._device_enumerator_candidates(
        None, "/requested/enumerator"
    ) == ("/selected/enumerator",)
    assert calls == ["/requested/enumerator"]


def test_explicit_gpu_targets_do_not_resolve_an_enumerator(monkeypatch):
    monkeypatch.setattr(
        tensilelite,
        "deviceEnumeratorCandidates",
        lambda explicit: (_ for _ in ()).throw(AssertionError("enumerator was resolved")),
    )

    assert (
        tensilelite._device_enumerator_candidates("gfx942", "/requested/enumerator") is None
    )


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

    def detect(tool, device_id):
        calls.append((tool, device_id))
        if tool == "/selected/offload-arch":
            return 1
        return IsaVersion(9, 4, 2)

    monkeypatch.setattr(Architectures, "_detectGlobalCurrentISA", detect)

    assert Architectures.detectGlobalCurrentISA(
        0,
        ("/selected/offload-arch", "/selected/amdgpu-arch"),
    ) == IsaVersion(9, 4, 2)
    assert calls == [
        ("/selected/offload-arch", 0),
        ("/selected/amdgpu-arch", 0),
    ]
