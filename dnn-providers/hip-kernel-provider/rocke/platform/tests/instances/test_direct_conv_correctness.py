# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Correctness tests for direct grouped convolution across cpg variants.

Covers all four grouped variants (cpg = 4, 8, 16, 32) and the depthwise
variant (cpg = 1).  Each test builds a kernel, compiles it, launches it on
GPU, and compares the output against a float32 reference produced by
torch.nn.functional.conv2d.

The test shapes are kept intentionally small (fast compile + run) while
hitting the branch points that differ across implementations:
  - grouped DirectConvSpec (mfma_f32_16x16x16_f16) across representative cpg values
  - depthwise scalar-FMA path (cpg=1)

Requires a ROCm GPU (gfx942 or gfx950) and torch.  Run:
    PYTHONPATH=rocke/platform/python <torch-python> -m pytest \
        rocke/platform/tests/instances/test_direct_conv_correctness.py -v
"""

from __future__ import annotations

import ctypes
import importlib.util
import math
import unittest
from dataclasses import dataclass
from typing import List, Tuple

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
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

_TOL = 5e-2


# ---------------------------------------------------------------------------
# Test shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Shape:
    """One test problem for direct conv.

    ``cpg`` must equal ``kpg`` and must be 1 (depthwise) or a positive
    multiple of 4.  ``stride`` must be 1 for depthwise.
    """

    id: str
    N: int
    H: int
    W: int
    groups: int
    cpg: int  # channels-per-group (= kpg)
    KH: int = 3
    KW: int = 3
    PAD: int = 1
    stride: int = 1


# One representative shape per cpg variant.  groups is chosen to be a
# multiple of the default block_groups for each spec so that the kernel
# actually launches on any valid machine rather than being skipped by the
# "groups not divisible by block_groups" validator:
#   cpg=4  → DirectConv4cSpec   default block_groups=16, DirectConvSpec default=8
#   cpg=8  → DirectConv8cSpec   default block_groups=8
#   cpg=16 → DirectConv16cSpec  default block_groups=8
#   cpg=32 → DirectConv32cSpec  default block_groups=4, DirectConvSpec default=8 → lcm=8
#   cpg=1  → DirectDepthwiseSpec default block_ch=block_waves*wave=64
_SHAPES: List[_Shape] = [
    # cpg=4 — DirectConv4cSpec (mfma_f32_4x4x4_f16); groups=16 satisfies block_groups=16
    _Shape("4c_N2H14W14_g16", N=2, H=14, W=14, groups=16, cpg=4),
    # cpg=8 — DirectConv8cSpec (mfma_f32_16x16x16_f16, fold two K=8 slices)
    _Shape("8c_N2H14W14_g8", N=2, H=14, W=14, groups=8, cpg=8),
    # cpg=16 — DirectConv16cSpec (mfma_f32_16x16x16_f16 or 16x16x32)
    _Shape("16c_N2H14W14_g8", N=2, H=14, W=14, groups=8, cpg=16),
    # cpg=32 — DirectConv32cSpec (mfma_f32_32x32x8_f16); groups=8 satisfies both
    # DirectConv32cSpec default block_groups=4 and DirectConvSpec default block_groups=8
    _Shape("32c_N2H8W8_g8", N=2, H=8, W=8, groups=8, cpg=32),
    # cpg=1 — depthwise (DirectDepthwiseSpec, scalar FMA); groups=64 satisfies block_ch=64
    _Shape("dw_N2H14W14_g64", N=2, H=14, W=14, groups=64, cpg=1),
    # 1×1 pointwise for cpg=16 (PAD=0, KH=KW=1)
    _Shape("16c_1x1_N2H16W16_g8", N=2, H=16, W=16, groups=8, cpg=16, KH=1, KW=1, PAD=0),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _u8(t):
    import torch  # noqa: F401

    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _conv_ref_grouped(A_t, B_t, p) -> "torch.Tensor":
    """Return NHWK float32 reference on CUDA via torch.nn.functional.conv2d."""
    import torch
    import torch.nn.functional as F

    # A: (N, H, W, C) → (N, C, H, W);  B: (K, KH, KW, cpg) → (K, cpg, KH, KW)
    A_nchw = A_t.permute(0, 3, 1, 2).float()
    B_nchw = B_t.permute(0, 3, 1, 2).float()
    out_nchw = F.conv2d(A_nchw, B_nchw, padding=p.PAD, stride=p.stride, groups=p.groups)
    return out_nchw.permute(0, 2, 3, 1).contiguous().cuda()


def _run_grouped_one(arch: str, shape: _Shape) -> Tuple[bool, str]:
    """Build, compile, launch, and verify one grouped direct-conv kernel.

    Uses the generic ``DirectConvSpec`` dispatcher which selects the right
    cpg-specialised kernel (4c / 8c / 16c / 32c) automatically.

    Returns ``(passed, reason)``.  ``reason`` starts with ``"skip "`` when
    the combination is architecturally unsupported.
    """
    import torch

    from rocke import compile_kernel
    from rocke.helpers.manifest import conv_args_signature
    from rocke.instances.common.conv_direct_grouped import (
        DirectConvProblem,
        DirectConvSpec,
        build_direct_conv,
        is_valid_spec,
    )
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    p = DirectConvProblem(
        N=shape.N,
        H=shape.H,
        W=shape.W,
        groups=shape.groups,
        cpg=shape.cpg,
        kpg=shape.cpg,
        KH=shape.KH,
        KW=shape.KW,
        PAD=shape.PAD,
        stride=shape.stride,
    )

    spec = DirectConvSpec(
        problem=p,
        name=f"test_direct_{shape.id}",
    )

    ok, reason = is_valid_spec(spec, arch=arch)
    if not ok:
        return False, f"invalid spec (shapes should be pre-validated): {reason}"

    try:
        kernel = build_direct_conv(spec, arch=arch)
    except ValueError as e:
        return False, f"build failed (shapes should be pre-validated): {e}"

    try:
        artifact = compile_kernel(kernel, arch=arch)
    except Exception as e:
        return False, f"compile failed: {e}"

    torch.manual_seed(0)
    total_c = shape.groups * shape.cpg
    total_k = shape.groups * shape.cpg
    A_t = torch.empty(p.N, p.H, p.W, total_c, dtype=torch.float16).uniform_(-1.0, 1.0)
    B_t = torch.empty(total_k, p.KH, p.KW, shape.cpg, dtype=torch.float16).uniform_(
        -1.0, 1.0
    )
    D_t = torch.empty(p.N, p.H, p.W, total_k, dtype=torch.float16)

    ref = _conv_ref_grouped(A_t, B_t, p)

    rt = Runtime()
    A_dev = rt.alloc(A_t.nbytes)
    B_dev = rt.alloc(B_t.nbytes)
    D_dev = rt.alloc(D_t.nbytes)
    rt.memcpy_h2d(A_dev, _u8(A_t), A_t.nbytes)
    rt.memcpy_h2d(B_dev, _u8(B_t), B_t.nbytes)
    rt.memset(D_dev, 0, D_t.nbytes)

    sig = conv_args_signature("fp16")
    try:
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=sig,
        )
    except HipError as e:
        rt.free(A_dev)
        rt.free(B_dev)
        rt.free(D_dev)
        return False, f"kernel load failed: {e}"

    q_tiles = (p.W + spec.block_q - 1) // spec.block_q
    g_tiles = p.groups // spec.block_groups
    grid = (q_tiles, g_tiles, p.N)
    block = (spec.threads_per_block, 1, 1)

    values = {
        "A": A_dev,
        "B": B_dev,
        "D": D_dev,
        "A_bytes": A_t.nbytes,
        "B_bytes": B_t.nbytes,
        "D_bytes": D_t.nbytes,
    }
    launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))

    D_cpu = torch.empty_like(D_t)
    rt.memcpy_d2h(_u8(D_cpu), D_dev, D_t.nbytes)
    rt.free(A_dev)
    rt.free(B_dev)
    rt.free(D_dev)
    synchronize_and_release(0)

    out_f32 = D_cpu.float()
    ref_f32 = ref.float().cpu()
    abs_diff = (out_f32 - ref_f32).abs()
    ref_scale = ref_f32.abs().max().clamp(min=1.0)
    rel_err = float(abs_diff.max() / ref_scale)
    passed = rel_err < _TOL
    if not passed:
        return False, f"rel_err={rel_err:.3e} > tol={_TOL:.1e}"
    print(
        f"  PASS  {shape.id}  {arch}  rel_err={rel_err:.2e}",
        flush=True,
    )
    return True, ""


def _run_depthwise_one(arch: str, shape: _Shape) -> Tuple[bool, str]:
    """Build, compile, launch, and verify one depthwise direct-conv kernel.

    Uses ``DirectDepthwiseSpec`` (cpg = kpg = 1).  Stride must be 1.

    Returns ``(passed, reason)``.
    """
    import torch

    from rocke import compile_kernel
    from rocke.helpers.manifest import conv_args_signature
    from rocke.instances.common.conv_direct_grouped import (
        DirectConvProblem,
        DirectDepthwiseSpec,
        build_direct_depthwise,
        is_valid_depthwise_spec,
    )
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    assert shape.cpg == 1, "depthwise path requires cpg=1"

    p = DirectConvProblem(
        N=shape.N,
        H=shape.H,
        W=shape.W,
        groups=shape.groups,
        cpg=1,
        kpg=1,
        KH=shape.KH,
        KW=shape.KW,
        PAD=shape.PAD,
        stride=shape.stride,
    )

    spec = DirectDepthwiseSpec(
        problem=p,
        name=f"test_direct_dw_{shape.id}",
    )

    ok, reason = is_valid_depthwise_spec(spec, arch=arch)
    if not ok:
        return False, f"invalid spec (shapes should be pre-validated): {reason}"

    try:
        kernel = build_direct_depthwise(spec, arch=arch)
    except ValueError as e:
        return False, f"build failed (shapes should be pre-validated): {e}"

    try:
        artifact = compile_kernel(kernel, arch=arch)
    except Exception as e:
        return False, f"compile failed: {e}"

    torch.manual_seed(0)
    total_c = shape.groups
    total_k = shape.groups
    A_t = torch.empty(p.N, p.H, p.W, total_c, dtype=torch.float16).uniform_(-1.0, 1.0)
    B_t = torch.empty(total_k, p.KH, p.KW, 1, dtype=torch.float16).uniform_(-1.0, 1.0)
    D_t = torch.empty(p.N, p.H, p.W, total_k, dtype=torch.float16)

    ref = _conv_ref_grouped(A_t, B_t, p)

    rt = Runtime()
    A_dev = rt.alloc(A_t.nbytes)
    B_dev = rt.alloc(B_t.nbytes)
    D_dev = rt.alloc(D_t.nbytes)
    rt.memcpy_h2d(A_dev, _u8(A_t), A_t.nbytes)
    rt.memcpy_h2d(B_dev, _u8(B_t), B_t.nbytes)
    rt.memset(D_dev, 0, D_t.nbytes)

    sig = conv_args_signature("fp16")
    try:
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=sig,
        )
    except HipError as e:
        rt.free(A_dev)
        rt.free(B_dev)
        rt.free(D_dev)
        return False, f"kernel load failed: {e}"

    q_tiles = math.ceil(p.W / spec.block_w)
    g_tiles = math.ceil(p.groups / spec.block_ch)
    grid = (q_tiles, g_tiles, p.N)
    block = (spec.threads_per_block, 1, 1)

    values = {
        "A": A_dev,
        "B": B_dev,
        "D": D_dev,
        "A_bytes": A_t.nbytes,
        "B_bytes": B_t.nbytes,
        "D_bytes": D_t.nbytes,
    }
    launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))

    D_cpu = torch.empty_like(D_t)
    rt.memcpy_d2h(_u8(D_cpu), D_dev, D_t.nbytes)
    rt.free(A_dev)
    rt.free(B_dev)
    rt.free(D_dev)
    synchronize_and_release(0)

    out_f32 = D_cpu.float()
    ref_f32 = ref.float().cpu()
    abs_diff = (out_f32 - ref_f32).abs()
    ref_scale = ref_f32.abs().max().clamp(min=1.0)
    rel_err = float(abs_diff.max() / ref_scale)
    passed = rel_err < _TOL
    if not passed:
        return False, f"rel_err={rel_err:.3e} > tol={_TOL:.1e}"
    print(
        f"  PASS  {shape.id}  {arch}  rel_err={rel_err:.2e}",
        flush=True,
    )
    return True, ""


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestDirectConvCorrectness(unittest.TestCase):
    """Correctness sweep for all direct-conv cpg variants on the detected arch."""

    def _run_grouped(self, shape: _Shape) -> None:
        passed, reason = _run_grouped_one(GPU_ARCH, shape)
        if reason.startswith("skip"):
            self.skipTest(reason)
        self.assertTrue(
            passed,
            f"FAIL {shape.id} on {GPU_ARCH}: {reason}",
        )

    def _run_depthwise(self, shape: _Shape) -> None:
        passed, reason = _run_depthwise_one(GPU_ARCH, shape)
        if reason.startswith("skip"):
            self.skipTest(reason)
        self.assertTrue(
            passed,
            f"FAIL {shape.id} on {GPU_ARCH}: {reason}",
        )

    def test_cpg4(self):
        for s in _SHAPES:
            if s.cpg == 4:
                with self.subTest(shape=s.id):
                    self._run_grouped(s)

    def test_cpg8(self):
        for s in _SHAPES:
            if s.cpg == 8:
                with self.subTest(shape=s.id):
                    self._run_grouped(s)

    def test_cpg16(self):
        for s in _SHAPES:
            if s.cpg == 16:
                with self.subTest(shape=s.id):
                    self._run_grouped(s)

    def test_cpg32(self):
        for s in _SHAPES:
            if s.cpg == 32:
                with self.subTest(shape=s.id):
                    self._run_grouped(s)

    def test_depthwise(self):
        for s in _SHAPES:
            if s.cpg == 1:
                with self.subTest(shape=s.id):
                    self._run_depthwise(s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
