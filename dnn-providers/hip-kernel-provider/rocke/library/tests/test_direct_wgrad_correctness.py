# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""End-to-end GPU correctness of the direct-wgrad kernel family, with the REAL
output dtype (fp16/bf16) the hipDNN integration dispatches against -- not the
fp32 scratch buffer every earlier test in this tree used.

Context: the direct-wgrad kernels atomic-add / fold their partial sums in
fp32. ``direct_wgrad`` / ``direct_wgrad_tiled`` wrote that fp32 value straight
into their "D" kernarg; the five folded variants' shared
``direct_wgrad_fold`` helper did the same on its own final store. Pointed at
rocke's own fp32 scratch (as every pre-existing test here does), that is
correct -- rel_err ~1e-7. Pointed at the hipDNN graph's real dW tensor, which
is fp16/bf16 (the same dtype as the activations, per standard mixed-precision
convention), it is a 2x-vs-4x element-size mismatch: every write corrupts
memory, which is exactly the "100% element mismatch, max_abs_diff/max_rel_diff
= inf" the hipDNN integration hit on real graphs. See WGRAD_AOT_PROGRESS.md's
2026-10-31 entry and the fork session that root-caused it.

The fix: ``DwWgradFoldSpec`` grew a ``dtype_d`` field so its final store casts
the fp32 reduction down to the real dtype
(:mod:`kernels.common.conv_direct_dw_wgrad`), and a new, separate, minimal
kernel -- ``DirectWgradCastSpec`` / ``build_direct_wgrad_cast``
(:mod:`kernels.common.conv_direct_wgrad_cast``) -- does the equivalent flat
narrowing cast for the two non-folded kernels, whose "D" offset formula
already matches dW's final layout exactly (no transpose needed, unlike the
fold).

This file exercises that fix for all 7 direct-wgrad variants with a REAL
fp16/bf16 dW buffer end to end: build -> compile -> launch (atomic / partial
producer) -> launch (fold or cast) -> compare against a float32
conv2d-weight-gradient reference, same tolerances as
test_direct_conv_correctness.py's established bf16/fp16 convention.

Requires a ROCm GPU (gfx950) and torch. Run:
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \
        rocke/library/tests/test_direct_wgrad_correctness.py -v
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest
from typing import Tuple

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None

if _HAS_TORCH:
    import torch

    # Claim the process HIP context for torch before rocke's runtime touches
    # it, same reasoning as test_direct_conv_correctness.py's module docstring.
    torch.cuda.is_available()

GPU_ARCH = get_device_arch(0)
_IS_MFMA = GPU_ARCH in ("gfx942", "gfx950")


def _skip_reason() -> str:
    if not GPU_ARCH:
        return "no ROCm GPU detected"
    if not _HAS_TORCH:
        return "torch not importable"
    if not _IS_MFMA:
        return f"unsupported arch {GPU_ARCH!r} (need gfx942 or gfx950)"
    return ""


_SKIP_REASON = _skip_reason()

# Same bf16/fp16 tolerance convention as test_direct_conv_correctness.py.
_TOL = 5e-2
_TOL_BF16 = 1e-1


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


class _ProblemAdapter:
    """Adapts a DirectConvProblem to conv_reference.wgrad_reference's interface."""

    is_3d = False

    def __init__(self, p):
        self.N = p.N
        self.C = p.total_c
        self.K = p.total_k
        self.Hi = p.H
        self.Wi = p.W
        self.Y = p.KH
        self.X = p.KW
        self.sH = p.stride
        self.sW = p.stride
        self.pH = p.PAD
        self.pW = p.PAD
        self.dH = p.dil_h
        self.dW = p.dil_w
        self.groups = p.groups


def _make_tensors(p, dtype: str):
    import torch

    td = torch.bfloat16 if dtype == "bf16" else torch.float16
    torch.manual_seed(1234)
    X_t = torch.empty(p.N, p.H, p.W, p.total_c, dtype=td).uniform_(-1.0, 1.0)
    dY_t = torch.empty(p.N, p.Ho, p.Wo, p.total_k, dtype=td).uniform_(-1.0, 1.0)
    return X_t, dY_t


def _check(dW_cpu, ref, dtype: str, label: str) -> Tuple[bool, str]:
    import torch

    out_f32 = dW_cpu.float()
    ref_f32 = ref.float().cpu()
    abs_diff = (out_f32 - ref_f32).abs()
    ref_scale = ref_f32.abs().max().clamp(min=1.0)
    rel_err = float((abs_diff.max() / ref_scale).nan_to_num(float("inf")))
    tol = _TOL_BF16 if dtype == "bf16" else _TOL
    passed = rel_err < tol
    reason = f"{label}: rel_err={rel_err:.3e} tol={tol:.1e}"
    return passed, reason


def _run_tiled_or_bare(variant: str, p, dtype: str) -> Tuple[bool, str]:
    """direct_wgrad / direct_wgrad_tiled: atomic-add fp32 into a scratch the
    same shape as dW, then a flat narrowing cast into the real dW dtype."""
    import torch

    from builders.common.conv_reference import wgrad_reference
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_wgrad_cast import (
        DirectWgradCastSpec,
        build_direct_wgrad_cast,
        direct_wgrad_cast_signature,
    )
    from rocke import compile_kernel
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    if variant == "direct_wgrad":
        from kernels.common.conv_direct_grouped import (
            DirectConvWgradSpec,
            build_direct_conv_wgrad,
            is_valid_wgrad_spec,
        )

        spec = DirectConvWgradSpec(problem=p, name="test_direct_wgrad")
        ok, why = is_valid_wgrad_spec(spec, arch=GPU_ARCH)
        build = build_direct_conv_wgrad
    else:
        from kernels.common.conv_direct_wgrad_tiled import (
            DirectWgradTiledSpec,
            build_direct_wgrad_tiled,
            is_valid_wgrad_tiled_spec,
        )

        spec = DirectWgradTiledSpec(problem=p, name="test_direct_wgrad_tiled")
        ok, why = is_valid_wgrad_tiled_spec(spec, arch=GPU_ARCH)
        build = build_direct_wgrad_tiled
    if not ok:
        return False, f"skip {why}"

    artifact = compile_kernel(build(spec, arch=GPU_ARCH), arch=GPU_ARCH)
    cast_spec = DirectWgradCastSpec(dtype_d=dtype)
    cast_artifact = compile_kernel(build_direct_wgrad_cast(cast_spec, arch=GPU_ARCH), arch=GPU_ARCH)

    X_t, dY_t = _make_tensors(p, dtype)
    ref = wgrad_reference(X_t, dY_t, _ProblemAdapter(p))

    dw_elems = p.total_k * p.KH * p.KW * p.cpg
    scratch_t = torch.zeros(dw_elems, dtype=torch.float32)
    td = torch.bfloat16 if dtype == "bf16" else torch.float16
    dW_t = torch.empty(dw_elems, dtype=td)

    rt = Runtime()
    X_dev = rt.alloc(X_t.nbytes)
    dY_dev = rt.alloc(dY_t.nbytes)
    scratch_dev = rt.alloc(scratch_t.nbytes)
    dW_dev = rt.alloc(dW_t.nbytes)
    rt.memcpy_h2d(X_dev, _u8(X_t), X_t.nbytes)
    rt.memcpy_h2d(dY_dev, _u8(dY_t), dY_t.nbytes)
    rt.memset(scratch_dev, 0, scratch_t.nbytes)  # atomic contract: caller zero-inits

    try:
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=conv_direct_args_signature(dtype, direction="wgrad"),
        )
        cast_launcher = KernelLauncher(
            hsaco=cast_artifact.hsaco,
            kernel_name=cast_artifact.kernel_name,
            signature=direct_wgrad_cast_signature(cast_spec),
        )
    except HipError as e:
        rt.free(X_dev)
        rt.free(dY_dev)
        rt.free(scratch_dev)
        rt.free(dW_dev)
        return False, f"skip kernel load failed: {e}"

    if variant == "direct_wgrad":
        from kernels.common.conv_direct_grouped import direct_launch_geometry

        grid, block = direct_launch_geometry(spec)
    else:
        grid, block = spec.launch_geometry()
    args = ConvArgs.from_problem(p, direction="wgrad")
    values = args.to_launch_values(
        int(dY_dev), int(X_dev), int(scratch_dev), dY_t.nbytes, X_t.nbytes, scratch_t.nbytes
    )
    launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))

    cast_grid, cast_block = cast_spec.launch_geometry(dw_elems)
    cast_values = {
        "A": int(scratch_dev),
        "D": int(dW_dev),
        "A_bytes": scratch_t.nbytes,
        "D_bytes": dW_t.nbytes,
        "count": dw_elems,
    }
    cast_launcher(cast_values, config=LaunchConfig(grid=cast_grid, block=cast_block, fence=True))

    dW_cpu = torch.empty_like(dW_t)
    rt.memcpy_d2h(_u8(dW_cpu), dW_dev, dW_t.nbytes)
    rt.free(X_dev)
    rt.free(dY_dev)
    rt.free(scratch_dev)
    rt.free(dW_dev)
    synchronize_and_release(0)

    return _check(dW_cpu.view(p.total_k, p.KH, p.KW, p.cpg), ref, dtype, variant)


def _run_folded(variant: str, p, dtype: str, *, build_fn, spec, taps: int, channels: int) -> Tuple[bool, str]:
    """A folded variant: writes a tap-major partial-slot scratch (fp32), then
    direct_wgrad_fold (at the real dW dtype) reduces it into dW."""
    import torch

    from builders.common.conv_reference import wgrad_reference
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_dw_wgrad import (
        DwWgradFoldSpec,
        build_dw_wgrad_fold,
        dw_wgrad_fold_signature,
    )
    from rocke import compile_kernel
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    artifact = compile_kernel(build_fn(spec, arch=GPU_ARCH), arch=GPU_ARCH)
    fold_spec = DwWgradFoldSpec(taps=taps, dtype_d=dtype)
    fold_artifact = compile_kernel(build_dw_wgrad_fold(fold_spec, arch=GPU_ARCH), arch=GPU_ARCH)

    X_t, dY_t = _make_tensors(p, dtype)
    ref = wgrad_reference(X_t, dY_t, _ProblemAdapter(p))

    n_partitions = spec.n_partitions()
    ws_bytes = n_partitions * taps * channels * 4
    dw_elems = p.total_k * p.KH * p.KW * p.cpg
    td = torch.bfloat16 if dtype == "bf16" else torch.float16
    dW_t = torch.empty(dw_elems, dtype=td)

    rt = Runtime()
    X_dev = rt.alloc(X_t.nbytes)
    dY_dev = rt.alloc(dY_t.nbytes)
    ws_dev = rt.alloc(ws_bytes)
    dW_dev = rt.alloc(dW_t.nbytes)
    rt.memcpy_h2d(X_dev, _u8(X_t), X_t.nbytes)
    rt.memcpy_h2d(dY_dev, _u8(dY_t), dY_t.nbytes)
    rt.memset(ws_dev, 0, ws_bytes)

    try:
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=conv_direct_args_signature(dtype, direction="wgrad"),
        )
        fold_launcher = KernelLauncher(
            hsaco=fold_artifact.hsaco,
            kernel_name=fold_artifact.kernel_name,
            signature=dw_wgrad_fold_signature(fold_spec),
        )
    except HipError as e:
        rt.free(X_dev)
        rt.free(dY_dev)
        rt.free(ws_dev)
        rt.free(dW_dev)
        return False, f"skip kernel load failed: {e}"

    grid, block = spec.launch_geometry()
    args = ConvArgs.from_problem(p, direction="wgrad")
    values = args.to_launch_values(
        int(dY_dev), int(X_dev), int(ws_dev), dY_t.nbytes, X_t.nbytes, ws_bytes
    )
    launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))

    fold_grid, fold_block = fold_spec.launch_geometry(channels)
    fold_values = {
        "A": int(ws_dev),
        "D": int(dW_dev),
        "A_bytes": ws_bytes,
        "D_bytes": dW_t.nbytes,
        "parts": n_partitions,
        "channels": channels,
    }
    fold_launcher(fold_values, config=LaunchConfig(grid=fold_grid, block=fold_block, fence=True))

    dW_cpu = torch.empty_like(dW_t)
    rt.memcpy_d2h(_u8(dW_cpu), dW_dev, dW_t.nbytes)
    rt.free(X_dev)
    rt.free(dY_dev)
    rt.free(ws_dev)
    rt.free(dW_dev)
    synchronize_and_release(0)

    return _check(dW_cpu.view(p.total_k, p.KH, p.KW, p.cpg), ref, dtype, variant)


def _dw_lds(p, dtype):
    from kernels.common.conv_direct_dw_wgrad import DwWgradLdsSpec, build_dw_wgrad_lds

    spec = DwWgradLdsSpec(problem=p)
    return _run_folded(
        "direct_wgrad_dw", p, dtype, build_fn=build_dw_wgrad_lds, spec=spec,
        taps=p.KH * p.KW, channels=p.groups,
    )  # fmt: skip


def _dw_cols(p, dtype):
    from kernels.common.conv_direct_dw_wgrad import DwWgradColsSpec, build_dw_wgrad_cols

    spec = DwWgradColsSpec(problem=p)
    return _run_folded(
        "direct_wgrad_dw_cols", p, dtype, build_fn=build_dw_wgrad_cols, spec=spec,
        taps=p.KH * p.KW, channels=p.groups,
    )  # fmt: skip


def _dw_mfma(p, dtype):
    from kernels.common.conv_direct_dw_wgrad_mfma import DwWgradMfmaSpec, build_dw_wgrad_mfma

    spec = DwWgradMfmaSpec(problem=p)
    return _run_folded(
        "direct_wgrad_dw_mfma", p, dtype, build_fn=build_dw_wgrad_mfma, spec=spec,
        taps=p.KH * p.KW, channels=p.groups,
    )  # fmt: skip


def _grouped_mfma(p, dtype):
    from kernels.common.conv_direct_grouped_wgrad_mfma import (
        GroupedWgradMfmaSpec,
        build_grouped_wgrad_mfma,
    )

    spec = GroupedWgradMfmaSpec(problem=p)
    return _run_folded(
        "direct_wgrad_grouped_mfma", p, dtype, build_fn=build_grouped_wgrad_mfma, spec=spec,
        taps=spec.fold_taps(), channels=spec.fold_channels(),
    )  # fmt: skip


def _gather(p, dtype):
    from kernels.common.conv_direct_wgrad_gather import WgradGatherSpec, build_wgrad_gather

    spec = WgradGatherSpec(problem=p)
    return _run_folded(
        "direct_wgrad_gather", p, dtype, build_fn=build_wgrad_gather, spec=spec,
        taps=spec.fold_taps(), channels=spec.fold_channels(),
    )  # fmt: skip


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestDirectWgradRealDtypeCorrectness(unittest.TestCase):
    """All 7 direct-wgrad variants, end to end, with a real fp16/bf16 dW
    buffer -- the scenario the hipDNN integration actually dispatches, and
    the one every earlier (fp32-dW) test in this tree did not cover."""

    def _check_variant(self, variant: str, p, dtype: str, runner) -> None:
        passed, reason = runner(p, dtype)
        if reason.startswith("skip"):
            self.skipTest(reason)
        self.assertTrue(passed, f"FAIL {variant} {dtype} on {GPU_ARCH}: {reason}")

    def _problem(self, **over):
        from kernels.common.conv_direct_grouped import DirectConvProblem

        kw = dict(N=2, H=14, W=14, groups=32, cpg=32, kpg=32, KH=3, KW=3, PAD=1, stride=1)
        kw.update(over)
        return DirectConvProblem(**kw)

    def test_direct_wgrad_bare(self):
        for dtype in ("fp16", "bf16"):
            for groups, cpg, kpg in ((1, 64, 64), (32, 32, 32)):
                with self.subTest(dtype=dtype, groups=groups):
                    p = self._problem(groups=groups, cpg=cpg, kpg=kpg, dtype=dtype)
                    self._check_variant(
                        "direct_wgrad", p, dtype,
                        lambda p, dtype: _run_tiled_or_bare("direct_wgrad", p, dtype),
                    )  # fmt: skip

    def test_direct_wgrad_tiled(self):
        for dtype in ("fp16", "bf16"):
            for groups, cpg, kpg in ((1, 64, 64), (32, 32, 32)):
                with self.subTest(dtype=dtype, groups=groups):
                    p = self._problem(groups=groups, cpg=cpg, kpg=kpg, dtype=dtype)
                    self._check_variant(
                        "direct_wgrad_tiled", p, dtype,
                        lambda p, dtype: _run_tiled_or_bare("direct_wgrad_tiled", p, dtype),
                    )  # fmt: skip

    def test_direct_wgrad_dw(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                p = self._problem(groups=64, cpg=1, kpg=1, H=16, W=32, dtype=dtype)
                self._check_variant("direct_wgrad_dw", p, dtype, _dw_lds)

    def test_direct_wgrad_dw_cols(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                p = self._problem(groups=3, cpg=1, kpg=1, KH=7, KW=7, PAD=3, H=16, W=32, dtype=dtype)
                self._check_variant("direct_wgrad_dw_cols", p, dtype, _dw_cols)

    def test_direct_wgrad_dw_mfma(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                p = self._problem(groups=64, cpg=1, kpg=1, H=16, W=32, dtype=dtype)
                self._check_variant("direct_wgrad_dw_mfma", p, dtype, _dw_mfma)

    def test_direct_wgrad_grouped_mfma(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                p = self._problem(groups=16, cpg=4, kpg=4, H=16, W=32, dtype=dtype)
                self._check_variant("direct_wgrad_grouped_mfma", p, dtype, _grouped_mfma)

    def test_direct_wgrad_gather(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                p = self._problem(groups=1, cpg=3, kpg=24, H=16, W=32, stride=2, dtype=dtype)
                self._check_variant("direct_wgrad_gather", p, dtype, _gather)


if __name__ == "__main__":
    unittest.main(verbosity=2)
