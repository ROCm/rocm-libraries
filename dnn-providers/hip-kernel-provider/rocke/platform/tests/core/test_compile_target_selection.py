# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib
import subprocess
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from rocke.core.codegen_policy import CodegenPolicy
from rocke.runtime.comgr import ComgrTimings

compile_module = importlib.import_module("rocke.helpers.compile")


def _kernel() -> SimpleNamespace:
    return SimpleNamespace(name="target_contract", attrs={})


@contextmanager
def _forbid_device_discovery() -> Iterator[None]:
    with ExitStack() as stack:
        for target in (
            "rocke.runtime.hip_module.get_device_arch",
            "rocke.runtime.hip_module.get_device_target_id",
            "rocke.runtime.device_info.get_device_info",
        ):
            stack.enter_context(
                mock.patch(
                    target,
                    side_effect=AssertionError("device discovery must not run"),
                )
            )
        yield


def test_compile_kernel_maps_runtime_profile_to_compiler_isa() -> None:
    isa = "amdgcn-amd-amdhsa--gfx1250-strict"
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
        _forbid_device_discovery(),
    ):
        artifact = compile_module.compile_kernel(kernel, isa=isa)

    assert lower.call_args.kwargs["arch"] == "gfx1250"
    compiler_isa = "amdgcn-amd-amdhsa--gfx1250"
    assert build.call_args.kwargs["isa"] == compiler_isa
    assert artifact.isa == compiler_isa


def test_compile_kernel_preserves_supported_target_features() -> None:
    isa = "amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"
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
        _forbid_device_discovery(),
    ):
        artifact = compile_module.compile_kernel(kernel, isa=isa)

    assert lower.call_args.kwargs["arch"] == "gfx942"
    assert build.call_args.kwargs["isa"] == isa
    assert artifact.isa == isa


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


def test_compile_kernel_arch_maps_runtime_profile_to_compiler_target() -> None:
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
        _forbid_device_discovery(),
    ):
        artifact = compile_module.compile_kernel(kernel, arch="gfx1250-strict")

    compiler_isa = "amdgcn-amd-amdhsa--gfx1250"
    assert lower.call_args.kwargs["arch"] == "gfx1250"
    assert build.call_args.kwargs["isa"] == compiler_isa
    assert artifact.isa == compiler_isa


@pytest.mark.parametrize(
    ("target_id", "lower_arch", "compiler_target"),
    [
        ("gfx1250-strict", "gfx1250", "gfx1250"),
        ("gfx942:sramecc+:xnack-", "gfx942", "gfx942:sramecc+:xnack-"),
    ],
)
def test_hipcc_uses_compiler_target_for_runtime_identity(
    target_id: str, lower_arch: str, compiler_target: str
) -> None:
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
        _forbid_device_discovery(),
    ):
        artifact = compile_module.compile_kernel_via_hipcc(kernel, arch=target_id)

    lower.assert_called_once_with(kernel, arch=lower_arch)
    assert f"--offload-arch={compiler_target}" in run.call_args.args[0]
    assert artifact.isa == f"amdgcn-amd-amdhsa--{compiler_target}"


def test_hipcc_ir_maps_runtime_profile_to_compiler_target() -> None:
    target_id = "gfx1250-strict"
    kernel = _kernel()

    def run_hipcc(args, **kwargs):
        output = Path(args[args.index("-o") + 1])
        output.write_text('target datalayout = "test"\n', encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    with (
        mock.patch.object(
            compile_module, "lower_kernel_to_hip", return_value="hip"
        ) as lower,
        mock.patch.object(
            compile_module.subprocess, "run", side_effect=run_hipcc
        ) as run,
        _forbid_device_discovery(),
    ):
        llvm_ir = compile_module.emit_device_llvm_ir_via_hipcc(kernel, arch=target_id)

    lower.assert_called_once_with(kernel, arch="gfx1250")
    assert "--offload-arch=gfx1250" in run.call_args.args[0]
    assert 'target datalayout = "test"' in llvm_ir
