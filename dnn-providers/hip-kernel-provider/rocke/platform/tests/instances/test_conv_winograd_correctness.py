# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Correctness tests for the Winograd convolution kernels.

Tests input-to-output correctness: given an NHWC input and a KYXC filter,
the Winograd pipeline must produce the same NHWK output as torch.nn.functional.conv2d.

The Winograd pipeline is:
  GPU  data_transform  : B^T × input_patch × B    per (n, tile_h, tile_w, c)
  GPU  filter_transform: G × filter × G^T          per (k, c)   [once]
  CPU  GEMM            : DataWs @ FilterWs^T        per (xh, xw) [batched matmul]
  GPU  output_transform: A^T × gemm_result × A     per (n, tile_h, tile_w, k)

The GEMM step runs on CPU (torch.matmul) so only the transform kernels are
tested on GPU. This is the standard approach for validating transform-only
kernel families before integrating the full GPU GEMM backend.

Shapes: 3×3 filter, stride=1, dilation=1, NHWC layout.

Requires a ROCm GPU and torch. Run:
    PYTHONPATH=rocke/platform/python <torch-python> -m pytest \\
        rocke/platform/tests/instances/test_conv_winograd_correctness.py -v
"""

from __future__ import annotations

import ctypes
import importlib.util
import math
import sys
import unittest
from dataclasses import dataclass
from typing import List, Tuple

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
GPU_ARCH = get_device_arch(0)
_SUPPORTED = GPU_ARCH in (
    "gfx942",
    "gfx950",
    "gfx1100",
    "gfx1101",
    "gfx1151",
    "gfx1200",
    "gfx1201",
    "gfx1250",
)


def _skip_reason() -> str:
    if not GPU_ARCH:
        return "no ROCm GPU detected"
    if not _HAS_TORCH:
        return "torch not importable"
    if not _SUPPORTED:
        return f"unsupported arch {GPU_ARCH} for Winograd"
    return ""


_SKIP_REASON = _skip_reason()
_TOL = 5e-2  # peak-normalised relative error threshold (fp16 + transform rounding)

# ---------------------------------------------------------------------------
# Test shapes — stride=1, dilation=1, 3×3 filter only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Shape:
    id: str
    N: int
    Hi: int
    Wi: int
    C: int
    K: int
    pH: int = 1
    pW: int = 1


_SHAPES: List[_Shape] = [
    _Shape("N2H8W8C16K16", N=2, Hi=8, Wi=8, C=16, K=16),
    _Shape("N1H7W7C32K32", N=1, Hi=7, Wi=7, C=32, K=32),
    _Shape("N4H14W14C32K32", N=4, Hi=14, Wi=14, C=32, K=32),
    _Shape("N2H14W14C64K64", N=2, Hi=14, Wi=14, C=64, K=64),
    _Shape("N2H8W8C16K16_nopad", N=2, Hi=8, Wi=8, C=16, K=16, pH=0, pW=0),
    _Shape("N2H7W14C16K16_asym", N=2, Hi=7, Wi=14, C=16, K=16),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _u8(t):
    import torch  # noqa: F401

    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _winograd_sig(kern):
    """Build a KernelLauncher signature from kernel params."""

    def _size(type_name: str) -> int:
        return 8 if type_name.startswith("ptr") else 4

    return [
        {"name": p.name, "type": p.type.name, "size_bytes": _size(p.type.name)}
        for p in kern.params
    ]


def _run_winograd_conv(
    arch: str,
    shape: _Shape,
    out_tile: int,
) -> Tuple[bool, str]:
    """Run the full Winograd pipeline and compare against torch F.conv2d.

    Input:  (N, Hi, Wi, C)  fp16 NHWC
    Filter: (K, 3, 3, C)    fp16 KYXC
    Output: (N, Ho, Wo, K)  fp16 NHWK  compared against torch reference

    Returns (passed, message).
    """
    import torch
    import torch.nn.functional as F

    from rocke import compile_kernel
    from rocke.instances.common.conv_winograd import (
        WinogradConvSpec,
        WinogradProblem,
        build_winograd_data_transform,
        build_winograd_filter_transform,
        build_winograd_output_transform,
        is_valid_spec,
        winograd_data_transform_grid,
        winograd_filter_transform_grid,
        winograd_output_transform_grid,
    )
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    # Problem object
    problem = WinogradProblem(
        N=shape.N,
        Hi=shape.Hi,
        Wi=shape.Wi,
        C=shape.C,
        K=shape.K,
        pH=shape.pH,
        pW=shape.pW,
    )

    block_c = min(32, shape.C)
    block_k = min(32, shape.K)
    block_nhw = 4

    spec = WinogradConvSpec(
        problem=problem,
        name=f"test_winograd_{shape.id}_f{out_tile}x3",
        out_tile=out_tile,
        block_c=block_c,
        block_k=block_k,
        block_nhw=block_nhw,
    )

    ok, reason = is_valid_spec(spec, arch)
    if not ok:
        return True, f"skip (invalid spec): {reason}"

    # Build all three transform kernels
    try:
        kd = build_winograd_data_transform(spec, arch=arch)
        kf = build_winograd_filter_transform(spec, arch=arch)
        ko = build_winograd_output_transform(spec, arch=arch)
    except (ValueError, Exception) as e:
        return False, f"IR build failed: {e}"

    try:
        art_d = compile_kernel(kd, arch=arch)
        art_f = compile_kernel(kf, arch=arch)
        art_o = compile_kernel(ko, arch=arch)
    except Exception as e:
        return False, f"compile failed: {e}"

    # Input tensors
    torch.manual_seed(42)
    A_t = torch.empty(
        shape.N, shape.Hi, shape.Wi, shape.C, dtype=torch.float16
    ).uniform_(-1, 1)
    W_t = torch.empty(shape.K, 3, 3, shape.C, dtype=torch.float16).uniform_(-1, 1)
    D_t = torch.zeros(shape.N, problem.Ho, problem.Wo, shape.K, dtype=torch.float16)

    # Reference: torch F.conv2d (NCHW in, NCHW out → NHWK)
    A_nchw = A_t.float().permute(0, 3, 1, 2).contiguous()
    W_kcyx = W_t.float().permute(0, 3, 1, 2).contiguous()
    ref_nchw = F.conv2d(A_nchw, W_kcyx, padding=shape.pH)
    ref_nhwk = ref_nchw.permute(0, 2, 3, 1).contiguous().half()

    # Workspace sizes (f32)
    xs = spec.xform_size
    ntotal = shape.N * spec.num_tiles
    dws_bytes = xs * xs * ntotal * shape.C * 4
    fws_bytes = xs * xs * shape.K * shape.C * 4
    gws_bytes = xs * xs * ntotal * shape.K * 4

    if max(dws_bytes, fws_bytes, gws_bytes) > 2**31 - 1:
        return True, "skip (workspace exceeds i32 range)"

    # GPU allocations
    rt = Runtime()
    A_dev = rt.alloc(A_t.nbytes)
    W_dev = rt.alloc(W_t.nbytes)
    DataWs_dev = rt.alloc(dws_bytes)
    FilterWs_dev = rt.alloc(fws_bytes)
    GemmWs_dev = rt.alloc(gws_bytes)
    D_dev = rt.alloc(D_t.nbytes)

    rt.memcpy_h2d(A_dev, _u8(A_t), A_t.nbytes)
    rt.memcpy_h2d(W_dev, _u8(W_t), W_t.nbytes)
    rt.memset(DataWs_dev, 0, dws_bytes)
    rt.memset(FilterWs_dev, 0, fws_bytes)
    rt.memset(GemmWs_dev, 0, gws_bytes)
    rt.memset(D_dev, 0, D_t.nbytes)

    def _free_all():
        for dev in (A_dev, W_dev, DataWs_dev, FilterWs_dev, GemmWs_dev, D_dev):
            rt.free(dev)

    # Launchers
    try:
        launch_d = KernelLauncher(
            hsaco=art_d.hsaco,
            kernel_name=art_d.kernel_name,
            signature=_winograd_sig(kd),
        )
        launch_f = KernelLauncher(
            hsaco=art_f.hsaco,
            kernel_name=art_f.kernel_name,
            signature=_winograd_sig(kf),
        )
        launch_o = KernelLauncher(
            hsaco=art_o.hsaco,
            kernel_name=art_o.kernel_name,
            signature=_winograd_sig(ko),
        )
    except HipError as e:
        _free_all()
        return False, f"kernel load failed: {e}"

    # ----- GPU Step 1: data transform -----
    launch_d(
        {
            "A": A_dev,
            "A_bytes": A_t.nbytes,
            "DataWs": DataWs_dev,
            "DataWs_bytes": dws_bytes,
        },
        config=LaunchConfig(
            grid=winograd_data_transform_grid(spec),
            block=(block_nhw * block_c, 1, 1),
            fence=True,
        ),
    )

    # ----- GPU Step 2: filter transform -----
    launch_f(
        {
            "W": W_dev,
            "W_bytes": W_t.nbytes,
            "FilterWs": FilterWs_dev,
            "FilterWs_bytes": fws_bytes,
        },
        config=LaunchConfig(
            grid=winograd_filter_transform_grid(spec),
            block=(block_k * block_c, 1, 1),
            fence=True,
        ),
    )

    # ----- CPU Step 3: GEMM in xform domain -----
    DataWs_cpu = torch.zeros(xs * xs * ntotal * shape.C, dtype=torch.float32)
    FilterWs_cpu = torch.zeros(xs * xs * shape.K * shape.C, dtype=torch.float32)
    rt.memcpy_d2h(_u8(DataWs_cpu), DataWs_dev, dws_bytes)
    rt.memcpy_d2h(_u8(FilterWs_cpu), FilterWs_dev, fws_bytes)

    # Reshape to (xs, xs, ntotal, C) and (xs, xs, K, C)
    DataWs_4d = DataWs_cpu.view(xs, xs, ntotal, shape.C)
    FilterWs_4d = FilterWs_cpu.view(xs, xs, shape.K, shape.C)

    # Batched matmul: (xs, xs, ntotal, C) × (xs, xs, C, K) → (xs, xs, ntotal, K)
    GemmWs_4d = torch.bmm(
        DataWs_4d.view(xs * xs, ntotal, shape.C),
        FilterWs_4d.view(xs * xs, shape.K, shape.C).transpose(1, 2),
    ).view(xs, xs, ntotal, shape.K)

    GemmWs_flat = GemmWs_4d.reshape(-1).contiguous()
    rt.memcpy_h2d(GemmWs_dev, _u8(GemmWs_flat), gws_bytes)

    # ----- GPU Step 4: output transform -----
    launch_o(
        {
            "GemmWs": GemmWs_dev,
            "GemmWs_bytes": gws_bytes,
            "D": D_dev,
            "D_bytes": D_t.nbytes,
        },
        config=LaunchConfig(
            grid=winograd_output_transform_grid(spec),
            block=(block_nhw * block_k, 1, 1),
            fence=True,
        ),
    )

    D_cpu = torch.zeros_like(D_t)
    rt.memcpy_d2h(_u8(D_cpu), D_dev, D_t.nbytes)
    _free_all()

    # ----- Compare -----
    out_f32 = D_cpu.float()
    ref_f32 = ref_nhwk.float()
    abs_diff = (out_f32 - ref_f32).abs()
    ref_scale = ref_f32.abs().max().clamp(min=1.0)
    rel_err = float(abs_diff.max() / ref_scale)
    passed = rel_err < _TOL
    msg = f"rel_err={rel_err:.2e} tol={_TOL:.1e}"
    if not passed:
        msg += (
            f"  max_abs_diff={float(abs_diff.max()):.4f}"
            f"  ref_max={float(ref_f32.abs().max()):.4f}"
        )
    return passed, msg


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestWinogradConvCorrectness(unittest.TestCase):
    """End-to-end conv correctness: input NHWC → Winograd pipeline → output NHWK.

    Compares against torch.nn.functional.conv2d for each (shape, out_tile) pair.
    The GEMM step runs on CPU (torch.bmm); only the transform kernels run on GPU.
    """

    def _check(self, shape: _Shape, out_tile: int) -> None:
        arch = GPU_ARCH
        passed, msg = _run_winograd_conv(arch, shape, out_tile)
        if msg.startswith("skip"):
            self.skipTest(msg)
        self.assertTrue(
            passed,
            f"FAIL {shape.id} f{out_tile}x3 on {arch}: {msg}",
        )
        print(f"  PASS {shape.id} f{out_tile}x3  {msg}", flush=True)

    # One test method per out_tile so failures are clearly attributed.

    def test_f2x3(self):
        """F(2,3) — 4×4 transform domain."""
        for shape in _SHAPES:
            with self.subTest(shape=shape.id):
                self._check(shape, out_tile=2)

    def test_f4x3(self):
        """F(4,3) — 6×6 transform domain."""
        for shape in _SHAPES:
            with self.subTest(shape=shape.id):
                self._check(shape, out_tile=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
