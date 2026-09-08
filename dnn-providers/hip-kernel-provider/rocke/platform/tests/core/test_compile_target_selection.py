# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rocke.core.codegen_policy import CodegenPolicy
from rocke.runtime.comgr import ComgrTimings

compile_module = importlib.import_module("rocke.helpers.compile")


def _kernel() -> SimpleNamespace:
    return SimpleNamespace(name="target_contract", attrs={})


def test_compile_kernel_preserves_exact_isa_without_device_discovery() -> None:
    isa = "amdgcn-amd-amdhsa--gfx1250-strict:xnack-"
    kernel = _kernel()

    with (
        mock.patch.object(compile_module, "print_ir", return_value="ir"),
        mock.patch.object(
            compile_module, "_lower_llvm_via_backend", return_value="llvm"
        ) as lower,
        mock.patch.object(
            compile_module,
            "build_hsaco_from_llvm_ir",
            return_value=(b"hsaco", ComgrTimings()),
        ) as build,
        mock.patch(
            "rocke.runtime.hip_module.get_device_arch", autospec=True
        ) as device_arch,
    ):
        artifact = compile_module.compile_kernel(kernel, isa=isa)

    assert lower.call_args.kwargs["arch"] == "gfx1250"
    assert build.call_args.kwargs["isa"] == isa
    assert artifact.isa == isa
    device_arch.assert_not_called()


def test_compile_kernel_base_arch_derives_ordinary_isa() -> None:
    kernel = _kernel()

    with (
        mock.patch.object(compile_module, "print_ir", return_value="ir"),
        mock.patch.object(
            compile_module, "_lower_llvm_via_backend", return_value="llvm"
        ) as lower,
        mock.patch.object(
            compile_module,
            "build_hsaco_from_llvm_ir",
            return_value=(b"hsaco", ComgrTimings()),
        ) as build,
    ):
        artifact = compile_module.compile_kernel(kernel, arch="gfx1250")

    assert lower.call_args.kwargs["arch"] == "gfx1250"
    assert build.call_args.kwargs["isa"] == "amdgcn-amd-amdhsa--gfx1250"
    assert artifact.isa == "amdgcn-amd-amdhsa--gfx1250"


def test_hipcc_preserves_profile_but_lowers_for_base_arch() -> None:
    target_id = "gfx1250-strict:xnack-"
    kernel = _kernel()

    def run_hipcc(args, **kwargs):
        output = Path(args[args.index("-o") + 1])
        output.write_bytes(b"hsaco")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    with (
        mock.patch.object(
            compile_module, "codegen_policy_for_kernel", return_value=CodegenPolicy()
        ),
        mock.patch.object(compile_module, "print_ir", return_value="ir"),
        mock.patch.object(
            compile_module, "lower_kernel_to_hip", return_value="hip"
        ) as lower,
        mock.patch.object(
            compile_module.subprocess, "run", side_effect=run_hipcc
        ) as run,
        mock.patch(
            "rocke.runtime.hip_module.get_device_arch", autospec=True
        ) as device_arch,
    ):
        artifact = compile_module.compile_kernel_via_hipcc(kernel, arch=target_id)

    lower.assert_called_once_with(kernel, arch="gfx1250")
    assert f"--offload-arch={target_id}" in run.call_args.args[0]
    assert artifact.isa == f"amdgcn-amd-amdhsa--{target_id}"
    device_arch.assert_not_called()
