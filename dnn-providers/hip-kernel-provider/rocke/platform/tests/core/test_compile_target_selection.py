# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
import importlib
import os
import subprocess
import warnings
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


@pytest.fixture(autouse=True)
def llvm23_compiler():
    """The compile target depends on the compiler's LLVM flavor; these cases are
    written for llvm23, whose compiler accepts gfx1250 but not gfx1154."""
    with mock.patch.dict(os.environ, {"ROCKE_LLVM_FLAVOR": "llvm23"}):
        yield


def _hipcc(clang_major: int, output: str):
    """A stand-in for ``hipcc`` whose clang is LLVM ``clang_major``: answers
    ``--version`` and writes ``output`` to the ``-o`` path of a compile."""

    def run(args, **kwargs):
        if args[1:] == ["--version"]:
            version = f"AMD clang version {clang_major}.0.0git\n"
            return subprocess.CompletedProcess(args, 0, stdout=version, stderr="")
        Path(args[args.index("-o") + 1]).write_text(output, encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    return run


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
        ("gfx1101", "gfx1101", "gfx1101"),
        ("gfx1101:unknown+", "gfx1101", "gfx1101:unknown+"),
        ("gfx1101-strict", "gfx1101", "gfx1101"),
        ("gfx1154", "gfx1154", "gfx11-generic"),
        ("gfx1154-strict:xnack-", "gfx1154", "gfx11-generic:xnack-"),
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
        _falls_back() if target_id.startswith("gfx1154") else _no_warning(),
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
        ("gfx1101", "gfx1101", "gfx1101"),
        ("gfx1101:unknown+", "gfx1101", "gfx1101:unknown+"),
        ("gfx1101-strict", "gfx1101", "gfx1101"),
        ("gfx1154", "gfx1154", "gfx11-generic"),
    ],
)
def test_hipcc_selects_targets(
    kernel: KernelDef, target_id: str, base_arch: str, compiler_target: str
) -> None:
    with (
        mock.patch.object(
            compile_module, "lower_kernel_to_hip", return_value="hip"
        ) as lower,
        mock.patch.object(
            compile_module.subprocess, "run", side_effect=_hipcc(23, "hsaco")
        ) as run,
        _falls_back() if target_id == "gfx1154" else _no_warning(),
    ):
        artifact = compile_module.compile_kernel_via_hipcc(kernel, arch=target_id)

    lower.assert_called_once_with(kernel, arch=base_arch)
    assert f"--offload-arch={compiler_target}" in run.call_args.args[0]
    assert artifact.isa == f"amdgcn-amd-amdhsa--{compiler_target}"


def test_hipcc_ir_maps_runtime_profile_to_compiler_target(kernel: KernelDef) -> None:
    with (
        mock.patch.object(
            compile_module, "lower_kernel_to_hip", return_value="hip"
        ) as lower,
        mock.patch.object(
            compile_module.subprocess,
            "run",
            side_effect=_hipcc(23, 'target datalayout = "test"\n'),
        ) as run,
    ):
        llvm_ir = compile_module.emit_device_llvm_ir_via_hipcc(
            kernel, arch="gfx1250-strict"
        )

    lower.assert_called_once_with(kernel, arch="gfx1250")
    assert "--offload-arch=gfx1250" in run.call_args.args[0]
    assert 'target datalayout = "test"' in llvm_ir


def test_a_target_the_compiler_cannot_take_is_refused_before_lowering(
    kernel: KernelDef,
) -> None:
    with (
        mock.patch.dict(os.environ, {"ROCKE_LLVM_FLAVOR": "llvm20"}),
        mock.patch.object(compile_module, "_lower_llvm_via_backend") as lower,
        pytest.raises(
            ValueError, match="llvm20 compiler does not accept the processor 'gfx1250'"
        ),
    ):
        compile_module.compile_kernel(kernel, arch="gfx1250")
    lower.assert_not_called()


@contextlib.contextmanager
def _no_warning():
    """A context in which any warning fails the test."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _falls_back():
    return pytest.warns(UserWarning, match="compiling for 'gfx11-generic' instead")


def test_hipcc_chooses_the_target_for_its_own_clang(kernel: KernelDef) -> None:
    """hipcc's clang decides, not the compiler COMGR loads: with
    ROCKE_LLVM_FLAVOR=llvm23 and a clang-20 hipcc, gfx1250 is refused before
    lowering because the llvm20 compiler does not accept it."""
    with (
        mock.patch.object(compile_module, "lower_kernel_to_hip") as lower,
        mock.patch.object(
            compile_module.subprocess, "run", side_effect=_hipcc(20, "hsaco")
        ),
        pytest.raises(
            ValueError, match="llvm20 compiler does not accept the processor 'gfx1250'"
        ),
    ):
        compile_module.compile_kernel_via_hipcc(kernel, arch="gfx1250")
    lower.assert_not_called()


def test_an_unreadable_hipcc_version_is_refused(kernel: KernelDef) -> None:
    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="hipcc\n", stderr="")

    with (
        mock.patch.object(compile_module.subprocess, "run", side_effect=run),
        pytest.raises(RuntimeError, match="cannot read the LLVM version of hipcc"),
    ):
        compile_module.compile_kernel_via_hipcc(kernel, arch="gfx942")
