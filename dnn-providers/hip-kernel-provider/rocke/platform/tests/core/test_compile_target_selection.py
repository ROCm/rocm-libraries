# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from rocke.core.ir import KernelDef, Region
from rocke.runtime.comgr import ComgrTimings

compile_module = importlib.import_module("rocke.helpers.compile")


@pytest.fixture(autouse=True)
def forbid_device_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    for target in (
        "rocke.runtime.hip_module.get_device_arch",
        "rocke.runtime.hip_module.get_device_target_id",
        "rocke.runtime.device_info.get_device_info",
    ):
        monkeypatch.setattr(
            target,
            mock.Mock(side_effect=AssertionError("device discovery must not run")),
        )


@pytest.fixture
def kernel() -> KernelDef:
    return KernelDef("target_contract", [], Region("entry"))


@pytest.mark.parametrize("argument", ["arch", "isa"])
@pytest.mark.parametrize(
    ("target_id", "base_arch", "compiler_target"),
    [
        ("gfx1250", "gfx1250", "gfx1250"),
        ("gfx1250-strict", "gfx1250", "gfx1250"),
        ("gfx942:sramecc+:xnack-", "gfx942", "gfx942:sramecc+:xnack-"),
    ],
)
def test_compile_kernel_selects_targets(
    kernel: KernelDef,
    argument: str,
    target_id: str,
    base_arch: str,
    compiler_target: str,
) -> None:
    value = target_id if argument == "arch" else f"amdgcn-amd-amdhsa--{target_id}"
    with (
        mock.patch.object(
            compile_module, "_lower_llvm_via_backend", return_value="llvm"
        ) as lower,
        mock.patch.object(
            compile_module,
            "build_hsaco_from_llvm_ir",
            return_value=(b"hsaco", ComgrTimings()),
        ) as build,
    ):
        artifact = compile_module.compile_kernel(kernel, **{argument: value})

    assert lower.call_args.kwargs["arch"] == base_arch
    compiler_isa = f"amdgcn-amd-amdhsa--{compiler_target}"
    assert build.call_args.kwargs["isa"] == compiler_isa
    assert artifact.isa == compiler_isa


@pytest.mark.parametrize(
    ("target_id", "base_arch", "compiler_target"),
    [
        ("gfx1250-strict", "gfx1250", "gfx1250"),
        ("gfx942:sramecc+:xnack-", "gfx942", "gfx942:sramecc+:xnack-"),
    ],
)
def test_hipcc_selects_targets(
    kernel: KernelDef, target_id: str, base_arch: str, compiler_target: str
) -> None:
    def run_hipcc(args, **kwargs):
        Path(args[args.index("-o") + 1]).write_bytes(b"hsaco")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    with (
        mock.patch.object(
            compile_module, "lower_kernel_to_hip", return_value="hip"
        ) as lower,
        mock.patch.object(
            compile_module.subprocess, "run", side_effect=run_hipcc
        ) as run,
    ):
        artifact = compile_module.compile_kernel_via_hipcc(kernel, arch=target_id)

    lower.assert_called_once_with(kernel, arch=base_arch)
    assert f"--offload-arch={compiler_target}" in run.call_args.args[0]
    assert artifact.isa == f"amdgcn-amd-amdhsa--{compiler_target}"


def test_hipcc_ir_maps_runtime_profile_to_compiler_target(kernel: KernelDef) -> None:
    def run_hipcc(args, **kwargs):
        Path(args[args.index("-o") + 1]).write_text(
            'target datalayout = "test"\n', encoding="utf-8"
        )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    with (
        mock.patch.object(
            compile_module, "lower_kernel_to_hip", return_value="hip"
        ) as lower,
        mock.patch.object(
            compile_module.subprocess, "run", side_effect=run_hipcc
        ) as run,
    ):
        llvm_ir = compile_module.emit_device_llvm_ir_via_hipcc(
            kernel, arch="gfx1250-strict"
        )

    lower.assert_called_once_with(kernel, arch="gfx1250")
    assert "--offload-arch=gfx1250" in run.call_args.args[0]
    assert 'target datalayout = "test"' in llvm_ir
