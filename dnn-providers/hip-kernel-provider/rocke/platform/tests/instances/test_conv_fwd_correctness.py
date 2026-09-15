# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Correctness tests for implicit-GEMM forward convolution across pipelines.

Runs a small sweep of conv shapes and pipeline/epilogue combinations, verifies
each kernel output against a float32 reference, and asserts no failures.

Coverage:
  - Pipelines: mem, compv3, compv4, basic (MFMA arches); mem, wavelet (WMMA/gfx1250)
  - Epilogues: default, cshuffle
  - Shapes: regular 3x3, pointwise 1x1, strided 2x2, dilated, padded, large channel
  - Dtypes: fp16, bf16
  - Corner cases: single-element output, groups=2, groups=4 (cpg not div-by-8), asymmetric HW

Requires a ROCm GPU and torch. Run:
    PYTHONPATH=rocke/platform/python <torch-python> -m pytest \\
        rocke/platform/tests/instances/test_conv_fwd_correctness.py
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest
from dataclasses import dataclass
from typing import List, Tuple

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
GPU_ARCH = get_device_arch(0)
_IS_WMMA = GPU_ARCH == "gfx1250"  # wave32 / WMMA target
_IS_MFMA = GPU_ARCH in ("gfx942", "gfx950")  # wave64 / MFMA targets


def _skip_reason() -> str:
    if not GPU_ARCH:
        return "no ROCm GPU detected"
    if not _HAS_TORCH:
        return "torch not importable"
    if not (_IS_WMMA or _IS_MFMA):
        return f"unsupported arch {GPU_ARCH} (need gfx942/gfx950/gfx1250)"
    return ""


_SKIP_REASON = _skip_reason()


# ---------------------------------------------------------------------------
# Small shape table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Shape:
    id: str
    N: int
    Hi: int
    Wi: int
    C: int
    K: int
    Y: int
    X: int
    sH: int = 1
    sW: int = 1
    pH: int = 0
    pW: int = 0
    dH: int = 1
    dW: int = 1
    groups: int = 1


# Shapes designed to stay small (fast compile + run) while exercising
# different code paths in the descriptor DAG.
_SHAPES: List[_Shape] = [
    # --- standard 3x3 conv with padding (the "bake-off" canonical shape, shrunk)
    _Shape("3x3_N2H14W14C32K32", N=2, Hi=14, Wi=14, C=32, K=32, Y=3, X=3, pH=1, pW=1),
    # --- pointwise 1x1 (no pad, no dilation — shortcut path in descriptor)
    _Shape("1x1_N2H14W14C32K32", N=2, Hi=14, Wi=14, C=32, K=32, Y=1, X=1),
    # --- stride-2 (output spatial halved)
    _Shape(
        "3x3_stride2_N2H8W8C16K16",
        N=2,
        Hi=8,
        Wi=8,
        C=16,
        K=16,
        Y=3,
        X=3,
        sH=2,
        sW=2,
        pH=1,
        pW=1,
    ),
    # --- dilation-2 (effective receptive field expands)
    _Shape(
        "3x3_dil2_N1H16W16C16K16",
        N=1,
        Hi=16,
        Wi=16,
        C=16,
        K=16,
        Y=3,
        X=3,
        dH=2,
        dW=2,
        pH=2,
        pW=2,
    ),
    # --- corner: output is 1×1 (minimum spatial extent)
    _Shape(
        "3x3_out1x1_N1H3W3C16K16", N=1, Hi=3, Wi=3, C=16, K=16, Y=3, X=3, pH=1, pW=1
    ),
    # --- larger channels to stress the K-loop (more K-tiles)
    _Shape("1x1_N1H8W8C64K64", N=1, Hi=8, Wi=8, C=64, K=64, Y=1, X=1),
    # --- asymmetric H≠W
    _Shape(
        "3x3_asym_N2H7W14C16K16", N=2, Hi=7, Wi=14, C=16, K=16, Y=3, X=3, pH=1, pW=1
    ),
    # --- grouped conv (groups=2): exercises the grid-per-group index and the
    # group-aware weight/output slicing; the reference tensor uses C//groups
    # per-group channels (kept here because the reference fix is what made
    # this shape correct to test — removing it hides grouped-conv regressions)
    _Shape(
        "3x3_g2_N2H8W8C32K32",
        N=2,
        Hi=8,
        Wi=8,
        C=32,
        K=32,
        Y=3,
        X=3,
        pH=1,
        pW=1,
        groups=2,
    ),
    # --- grouped conv where cpg/kpg is NOT a multiple of 8 even though C/K are.
    # C=16/32, K=16/32, groups=4 → cpg=kpg=4/8: default_vector_sizes must use cpg/kpg
    # (→ vec=4), not C/K (→ vec=8, which straddles group boundaries and corrupts
    # the load/store addressing). Regression guard for the cpg/kpg fix.
    _Shape(
        "3x3_g4_N2H8W8C16K16",
        N=2,
        Hi=8,
        Wi=8,
        C=32,
        K=16,
        Y=3,
        X=3,
        pH=1,
        pW=1,
        groups=4,
    ),
    _Shape(
        "3x3_g4_N2H8W8C16K16",
        N=2,
        Hi=8,
        Wi=8,
        C=16,
        K=32,
        Y=3,
        X=3,
        pH=1,
        pW=1,
        groups=4,
    ),
]

# Tile config: (tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_mn)
# Kept intentionally small to compile quickly.
_MFMA_TILE = (32, 32, 32, 2, 2, 16)  # warp_tile_k chosen by atom selection
_WMMA_TILE = (32, 32, 32, 1, 1, 16)  # wave32 — smaller warp grid

# Pipelines valid per arch
_MFMA_PIPELINES = ("mem", "compv3", "compv4", "basic")
_WMMA_PIPELINES = ("mem", "wavelet")

_EPILOGUES = ("default", "cshuffle")
_DTYPES = ("fp16", "bf16")

_TOL = {"fp16": 5e-2, "bf16": 5e-2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _u8(t):
    import torch  # noqa: F401 — only called when torch is available

    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _select_warp_tile_k(arch: str, dtype: str, warp_tile_mn: int, tile_k: int) -> int:
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    family = "wmma" if target.wave_size == 32 else "mma"
    atom = target.mma.select_largest_k(
        family=family,
        a_dtype=dtype,
        b_dtype=dtype,
        c_dtype="fp32",
        m=warp_tile_mn,
        n=warp_tile_mn,
        k_max=tile_k,
    )
    if atom is None:
        raise ValueError(
            f"No {family} atom for dtype={dtype} m=n={warp_tile_mn} tile_k={tile_k} on {arch}"
        )
    return atom.k


def _run_one(
    arch: str,
    shape: _Shape,
    dtype: str,
    pipeline: str,
    epilogue: str,
) -> Tuple[bool, str]:
    """Build, compile, launch, and verify one conv kernel.

    Returns ``(passed, reason)`` where ``reason`` is non-empty on skip or failure.
    """
    import torch

    from rocke import compile_kernel
    from rocke.benchmark.conv_reference import conv_reference, conv_reference_gfx1250
    from rocke.core.arch import ArchTarget
    from rocke.helpers.manifest import conv_args_signature
    from rocke.instances.common.conv_implicit_gemm import (
        ConvDataSpec,
        ConvProblem,
        ImplicitGemmConvSpec,
        build_implicit_gemm_conv,
        is_valid_spec_for_problem,
    )
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    target = ArchTarget.from_gfx(arch)
    wave_size = target.wave_size

    tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_mn = (
        _WMMA_TILE if arch == "gfx1250" else _MFMA_TILE
    )

    try:
        warp_tile_k = _select_warp_tile_k(arch, dtype, warp_tile_mn, tile_k)
    except ValueError as e:
        return True, f"skip (no atom): {e}"

    problem = ConvProblem(
        N=shape.N,
        Hi=shape.Hi,
        Wi=shape.Wi,
        C=shape.C,
        K=shape.K,
        Y=shape.Y,
        X=shape.X,
        sH=shape.sH,
        sW=shape.sW,
        pH=shape.pH,
        pW=shape.pW,
        dH=shape.dH,
        dW=shape.dW,
        groups=shape.groups,
    )

    # For wavelet on the small test tile (warp_m=warp_n=1, n_math_warps=1),
    # num_load_waves must be <= n_math_warps to avoid the over-provisioning warning.
    _num_load_waves = 1 if pipeline == "wavelet" else 4
    spec = ImplicitGemmConvSpec(
        problem=problem,
        name=f"test_conv_fwd_{shape.id}_{dtype}_{pipeline}_{epilogue}",
        data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        tile_m=tile_m,
        tile_n=tile_n,
        # Force scalar stores for the default epilogue so the vec_c constraint
        # never blocks small-K shapes. cshuffle handles its own vectorization.
        vector_size_c=1 if epilogue == "default" else None,
        tile_k=tile_k,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_tile_m=warp_tile_mn,
        warp_tile_n=warp_tile_mn,
        warp_tile_k=warp_tile_k,
        wave_size=wave_size,
        pipeline=pipeline,
        epilogue=epilogue,
        groups=shape.groups,
        num_load_waves=_num_load_waves,
    )

    ok, reason = is_valid_spec_for_problem(spec, problem, arch)
    if not ok:
        return True, f"skip (invalid spec): {reason}"

    try:
        kernel = build_implicit_gemm_conv(spec, arch=arch)
    except ValueError as e:
        return True, f"skip (build error): {e}"

    try:
        artifact = compile_kernel(kernel, arch=arch)
    except Exception as e:
        return False, f"compile failed: {e}"

    _torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    torch.manual_seed(0)
    A_t = (
        torch.empty(problem.N, problem.Hi, problem.Wi, problem.C)
        .uniform_(-1.0, 1.0)
        .to(_torch_dtype)
    )
    B_t = (
        torch.empty(problem.K, problem.Y, problem.X, problem.C // problem.groups)
        .uniform_(-1.0, 1.0)
        .to(_torch_dtype)
    )
    D_t = torch.empty(problem.N, problem.Ho, problem.Wo, problem.K, dtype=_torch_dtype)

    # Reference
    if arch == "gfx1250":
        ref = conv_reference_gfx1250(A_t, B_t, problem, out_dtype=_torch_dtype)
    else:
        ref = conv_reference(A_t, B_t, problem, out_dtype=_torch_dtype)

    rt = Runtime()
    A_dev = rt.alloc(A_t.nbytes)
    B_dev = rt.alloc(B_t.nbytes)
    D_dev = rt.alloc(D_t.nbytes)
    rt.memcpy_h2d(A_dev, _u8(A_t), A_t.nbytes)
    rt.memcpy_h2d(B_dev, _u8(B_t), B_t.nbytes)
    rt.memset(D_dev, 0, D_t.nbytes)

    sig = conv_args_signature(dtype)
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

    gx = (problem.N_gemm + tile_n - 1) // tile_n
    gy = (problem.M + tile_m - 1) // tile_m
    grid = (gx, gy, problem.groups)
    block = (spec.launch_block_size, 1, 1)

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

    out_f32 = D_cpu.float()
    ref_f32 = ref.float().cpu()
    abs_diff = (out_f32 - ref_f32).abs()
    ref_scale = ref_f32.abs().max().clamp(min=1.0)
    rel_err = float(abs_diff.max() / ref_scale)
    tol = _TOL[dtype]
    passed = rel_err < tol
    if not passed:
        return False, f"rel_err={rel_err:.3e} > tol={tol:.1e}"
    print(
        f"  PASS  {shape.id}  {dtype}  {pipeline}/{epilogue}  rel_err={rel_err:.2e}",
        flush=True,
    )
    return True, ""


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestConvFwdCorrectness(unittest.TestCase):
    """Forward conv correctness: each pipeline × epilogue × shape × dtype."""

    def _check(self, shape: _Shape, dtype: str, pipeline: str, epilogue: str) -> None:
        arch = GPU_ARCH
        passed, reason = _run_one(arch, shape, dtype, pipeline, epilogue)
        if reason.startswith("skip"):
            self.skipTest(reason)
        self.assertTrue(
            passed,
            f"FAIL {shape.id} {dtype} {pipeline}/{epilogue} on {arch}: {reason}",
        )

    def _pipelines(self):
        return _WMMA_PIPELINES if _IS_WMMA else _MFMA_PIPELINES

    # ------------------------------------------------------------------
    # One test method per pipeline so failures are clearly attributed.
    # ------------------------------------------------------------------

    def _sweep_pipeline(self, pipeline: str) -> None:
        for dtype in _DTYPES:
            for shape in _SHAPES:
                for epilogue in _EPILOGUES:
                    with self.subTest(shape=shape.id, dtype=dtype, epilogue=epilogue):
                        self._check(shape, dtype, pipeline, epilogue)

    def test_pipeline_mem(self):
        self._sweep_pipeline("mem")

    def test_pipeline_basic(self):
        if _IS_MFMA:
            self._sweep_pipeline("basic")

    def test_pipeline_compv3(self):
        if _IS_WMMA:
            self.skipTest("compv3 is MFMA-only (not valid on gfx1250/WMMA)")
        self._sweep_pipeline("compv3")

    def test_pipeline_compv4(self):
        if _IS_WMMA:
            self.skipTest("compv4 is MFMA-only (not valid on gfx1250/WMMA)")
        self._sweep_pipeline("compv4")

    def test_pipeline_wavelet(self):
        if _IS_MFMA:
            self.skipTest("wavelet is WMMA/gfx1250 only")
        self._sweep_pipeline("wavelet")


if __name__ == "__main__":
    unittest.main(verbosity=2)
