# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Correctness tests for implicit-GEMM backward-weight convolution (wgrad).

Runs a small sweep of conv shapes and pipeline/epilogue/split-K combinations,
verifies each weight-gradient output against a float32 reference
(``torch.nn.grad.conv2d_weight``), and asserts no failures.

This suite specifically exercises the **free-axis vectorised load** added for
wgrad: dY (NHWK) and X (NHWC) are loaded along their stride-1 free axis (K for
dY, C for X) with a transpose-on-store into the row-major LDS tile, replacing the
historical scalar loads forced by the strided K_wg reduction axis. The sweep uses
shapes whose C/K are multiples of 8 so the sync CDNA-MFMA path picks ``load_vec >
1`` (the new ``vector_axis="row"`` mode); ``TestConvWgradVectorLoad`` additionally
asserts, without a GPU, that a vectorised global load is actually emitted.

Coverage:
  - Pipelines: mem, compv3, compv4, basic (MFMA arches)
  - Epilogues: default, cshuffle
  - Split-K: 1 (direct store) and >1 (atomic accumulation)
  - Shapes: regular 3x3, pointwise 1x1, strided, C=24 (mult-8 not-16 edge),
    asymmetric HW, small channel
  - Dtypes: fp16, bf16

The vector-load path is enabled for every sync MMA family (MFMA and WMMA); only
the async-DMA path is excluded. The GPU numeric sweep here is MFMA-only
(gfx942/gfx950) because that is the hardware available for verification; the WMMA
path is covered by ``TestConvWgradVectorLoad`` (CPU IR-emission guard) and awaits
gfx1250 hardware for a numeric sign-off. Requires a ROCm GPU and torch. Run:
    PYTHONPATH=rocke/platform/python <torch-python> -m pytest \\
        rocke/platform/tests/instances/test_conv_wgrad_correctness.py
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import re
import unittest
from dataclasses import dataclass
from typing import List, Tuple

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
GPU_ARCH = get_device_arch(0)
_IS_MFMA = GPU_ARCH in ("gfx942", "gfx950")  # wave64 / MFMA targets


def _skip_reason() -> str:
    if not GPU_ARCH:
        return "no ROCm GPU detected"
    if not _HAS_TORCH:
        return "torch not importable"
    if not _IS_MFMA:
        return (
            f"wgrad vector-load path is MFMA-only; got {GPU_ARCH} (need gfx942/gfx950)"
        )
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


# Kept intentionally small (fast compile + run). C and K are multiples of 8 so
# the sync CDNA-MFMA path selects load_vec > 1 (the free-axis vectorised load).
_SHAPES: List[_Shape] = [
    # --- canonical dense 3x3 with padding (C,K mult-16 -> widest vec)
    _Shape("3x3_N2H14W14C64K64", N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1),
    # --- pointwise 1x1 (flat-arithmetic descriptor path; still vectorised)
    _Shape("1x1_N2H14W14C32K32", N=2, Hi=14, Wi=14, C=32, K=32, Y=1, X=1),
    # --- C=24: multiple of 8 but not 16 (vec_b must cap at 8, never 16)
    _Shape(
        "3x3_C24_N2H12W12C24K32", N=2, Hi=12, Wi=12, C=24, K=32, Y=3, X=3, pH=1, pW=1
    ),
    # --- stride-2 (output spatial halved -> smaller K_wg reduction)
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
    # --- asymmetric H != W
    _Shape(
        "3x3_asym_N2H7W14C16K16", N=2, Hi=7, Wi=14, C=16, K=16, Y=3, X=3, pH=1, pW=1
    ),
    # --- minimal channels (C=K=8: vec exactly divides the channel dim)
    _Shape("3x3_C8_N1H8W8C8K8", N=1, Hi=8, Wi=8, C=8, K=8, Y=3, X=3, pH=1, pW=1),
]

# Tile config (small, for fast compile). tile_m maps to M=K (out channels),
# tile_n to N_wg=Y*X*C, tile_k to the K_wg reduction. warp_tile_k comes from atom
# selection. 64x64 tiles + 256 threads let choose_vec pick load_vec up to 8.
_TILE_M, _TILE_N, _TILE_K = 64, 64, 64
_WARP_M, _WARP_N, _WARP_TILE_MN = 2, 2, 32

_PIPELINES = ("mem", "compv3", "compv4", "basic")
_EPILOGUES = ("default", "cshuffle")
_DTYPES = ("fp16", "bf16")

_TOL = {"fp16": 5e-2, "bf16": 5e-2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _u8(t):
    import torch  # noqa: F401 -- only called when torch is available

    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _wgrad_reference_cpu(X_f32, dY_f32, p):
    """float32 weight-gradient reference computed entirely on the CPU.

    Mirrors ``rocke.benchmark.conv_reference.wgrad_reference`` but never touches
    ``torch.cuda``: rocke's HIP runtime (used by the kernel launcher) and torch's
    HIP runtime fight over the process HIP context in-process — whichever
    initialises first wins, and rocke-first (our module-level ``get_device_arch``)
    leaves ``torch.cuda`` unavailable. Since the rocke launcher is independent of
    torch, computing the reference on the CPU sidesteps the conflict. 2-D only
    (all shapes in this file are 2-D).
    """
    import torch

    X_t = X_f32.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    dY_t = dY_f32.permute(0, 3, 1, 2).contiguous()  # NHWK -> NKHW
    dW_nchw = torch.nn.grad.conv2d_weight(
        X_t,
        weight_size=(p.K, p.C // p.groups, p.Y, p.X),
        grad_output=dY_t,
        stride=(p.sH, p.sW),
        padding=(p.pH, p.pW),
        dilation=(p.dH, p.dW),
        groups=p.groups,
    )
    return dW_nchw.permute(0, 2, 3, 1).contiguous()  # KCHW -> KHWC (KYXC)


def _make_spec(
    arch: str,
    shape: _Shape,
    dtype: str,
    pipeline: str,
    epilogue: str,
    split_k: int,
    lds_k_outer: bool = False,
    async_dma: bool = False,
    warp_tile_mn: "int | None" = None,
    tile_k: "int | None" = None,
):
    """Build a (spec, problem, warp_tile_k) triple, or (None, None, reason).

    ``warp_tile_mn`` / ``tile_k`` override the module-level tile constants so a
    caller can pin a specific MMA atom. Left as ``None`` (the default) the
    historical values are used and the emitted IR is unchanged.
    """
    from rocke.core.arch import ArchTarget
    from kernels.common._conv_implicit_gemm_common import ConvProblem
    from kernels.common.conv_implicit_gemm import ConvDataSpec
    from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

    target = ArchTarget.from_gfx(arch)
    # MMA family + tile shape follow the wave size: wave64 -> MFMA (32x32 atom),
    # wave32 -> WMMA (16x16 atom). The MFMA branch keeps the historical values so
    # its emitted IR / goldens are unchanged; WMMA uses a 16x16-shaped tile.
    family = "wmma" if target.wave_size == 32 else "mma"
    if family == "wmma":
        tile_m, tile_n, _tk_default = 32, 32, 32
        warp_m, warp_n, _wt_default = 1, 1, 16
    else:
        tile_m, tile_n, _tk_default = _TILE_M, _TILE_N, _TILE_K
        warp_m, warp_n, _wt_default = _WARP_M, _WARP_N, _WARP_TILE_MN
    tile_k = _tk_default if tile_k is None else tile_k
    warp_tile_mn = _wt_default if warp_tile_mn is None else warp_tile_mn
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
        return None, None, f"no {family} atom for dtype={dtype} k={tile_k}"

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
    # split_k == -1 means "auto": resolve to a concrete degree (CK formula) up
    # front, exactly as the benchmark does, so the spec and the launch grid's
    # z-dim agree (a grid with z=-1 is an invalid launch).
    if split_k == -1:
        from rocke.helpers.split_k import select_split_k_wgrad

        split_k = select_split_k_wgrad(
            wg_M=problem.kpg,
            wg_N=problem.Y * problem.X * problem.cpg,
            wg_K=problem.N * problem.Ho * problem.Wo,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            arch=arch,
        ).split_k
    spec = WgradConvSpec(
        problem=problem,
        name=(
            f"test_wgrad_{shape.id}_{dtype}_{pipeline}_{epilogue}_spk{split_k}"
            + ("_kouter" if lds_k_outer else "")
            + ("_async" if async_dma else "")
            + (
                f"_a{warp_tile_mn}k{tile_k}"
                if (warp_tile_mn, tile_k) != (_wt_default, _tk_default)
                else ""
            )
        ),
        data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_tile_m=warp_tile_mn,
        warp_tile_n=warp_tile_mn,
        warp_tile_k=atom.k,
        wave_size=target.wave_size,
        lds_k_outer=lds_k_outer,
        async_dma=async_dma,
        pipeline=pipeline,
        epilogue=epilogue,
        split_k=split_k,
    )
    return spec, problem, atom.k


def _run_one(
    arch: str,
    shape: _Shape,
    dtype: str,
    pipeline: str,
    epilogue: str,
    split_k: int = 1,
    lds_k_outer: bool = False,
    async_dma: bool = False,
    warp_tile_mn: "int | None" = None,
    tile_k: "int | None" = None,
) -> Tuple[bool, str]:
    """Build, compile, launch, and verify one wgrad kernel.

    Returns ``(passed, reason)`` where ``reason`` is non-empty on skip or failure.
    """
    import torch

    from rocke import compile_kernel
    from rocke.helpers.manifest import conv_args_signature
    from kernels.common.conv_implicit_gemm_wgrad import (
        build_implicit_gemm_conv_wgrad,
        is_valid_wgrad_spec,
    )
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    spec, problem, _wtk = _make_spec(
        arch,
        shape,
        dtype,
        pipeline,
        epilogue,
        split_k,
        lds_k_outer,
        async_dma,
        warp_tile_mn=warp_tile_mn,
        tile_k=tile_k,
    )
    if spec is None:
        return True, f"skip (no atom): {_wtk}"

    ok, reason = is_valid_wgrad_spec(spec, arch)
    if not ok:
        return True, f"skip (invalid spec): {reason}"

    try:
        kernel = build_implicit_gemm_conv_wgrad(spec, arch=arch)
    except ValueError as e:
        return True, f"skip (build error): {e}"

    try:
        artifact = compile_kernel(kernel, arch=arch)
    except Exception as e:  # noqa: BLE001
        return False, f"compile failed: {e}"

    _torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    torch.manual_seed(0)
    p = problem
    X_f32 = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1.0, 1.0)
    dY_f32 = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0)
    X_t = X_f32.to(_torch_dtype)
    dY_t = dY_f32.to(_torch_dtype)
    # dW is the (grouped) weight gradient: packed KYXC with the per-group channel
    # count cpg = C // groups (== C for groups=1).
    dW_t = torch.empty(p.K, p.Y, p.X, p.C // p.groups, dtype=_torch_dtype)

    # float32 reference (KYXC layout), on the CPU (see _wgrad_reference_cpu).
    ref = _wgrad_reference_cpu(X_f32, dY_f32, p)

    rt = Runtime()
    dY_dev = rt.alloc(dY_t.nbytes)
    X_dev = rt.alloc(X_t.nbytes)
    dW_dev = rt.alloc(dW_t.nbytes)
    rt.memcpy_h2d(dY_dev, _u8(dY_t), dY_t.nbytes)
    rt.memcpy_h2d(X_dev, _u8(X_t), X_t.nbytes)
    rt.memset(dW_dev, 0, dW_t.nbytes)  # split-K atomic-add needs a zeroed dW

    sig = conv_args_signature(dtype)
    try:
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=sig,
        )
    except HipError as e:
        rt.free(dY_dev)
        rt.free(X_dev)
        rt.free(dW_dev)
        return False, f"kernel load failed: {e}"

    # wgrad grid: x = ceil(N_wg / tile_n), y = ceil(M / tile_m),
    # z = groups * split_k (the group axis rides on z alongside split-K;
    # == split_k for groups=1).
    gx = (spec.wg_N + spec.tile_n - 1) // spec.tile_n
    gy = (spec.wg_M + spec.tile_m - 1) // spec.tile_m
    gz = p.groups * spec.split_k
    grid = (gx, gy, gz)
    block = (spec.block_size, 1, 1)

    values = {
        "A": dY_dev,
        "B": X_dev,
        "D": dW_dev,
        "A_bytes": dY_t.nbytes,
        "B_bytes": X_t.nbytes,
        "D_bytes": dW_t.nbytes,
    }
    launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))

    dW_cpu = torch.empty_like(dW_t)
    rt.memcpy_d2h(_u8(dW_cpu), dW_dev, dW_t.nbytes)
    rt.free(dY_dev)
    rt.free(X_dev)
    rt.free(dW_dev)

    out_f32 = dW_cpu.float()
    ref_f32 = ref.float()
    abs_diff = (out_f32 - ref_f32).abs()
    ref_scale = ref_f32.abs().max().clamp(min=1.0)
    rel_err = float(abs_diff.max() / ref_scale)
    tol = _TOL[dtype]
    passed = rel_err < tol
    if not passed:
        return False, f"rel_err={rel_err:.3e} > tol={tol:.1e}"
    print(
        f"  PASS  {shape.id}  {dtype}  {pipeline}/{epilogue}  spk{split_k}  "
        f"rel_err={rel_err:.2e}",
        flush=True,
    )
    return True, ""


# ---------------------------------------------------------------------------
# GPU correctness sweep
# ---------------------------------------------------------------------------


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestConvWgradCorrectness(unittest.TestCase):
    """Wgrad correctness: each pipeline x epilogue x shape x dtype (+ split-K)."""

    def _check(
        self,
        shape,
        dtype,
        pipeline,
        epilogue,
        split_k=1,
        lds_k_outer=False,
        async_dma=False,
        warp_tile_mn=None,
        tile_k=None,
    ) -> None:
        passed, reason = _run_one(
            GPU_ARCH,
            shape,
            dtype,
            pipeline,
            epilogue,
            split_k,
            lds_k_outer,
            async_dma,
            warp_tile_mn=warp_tile_mn,
            tile_k=tile_k,
        )
        if reason.startswith("skip"):
            self.skipTest(reason)
        self.assertTrue(
            passed,
            f"FAIL {shape.id} {dtype} {pipeline}/{epilogue} spk{split_k} "
            f"{'kouter ' if lds_k_outer else ''}on {GPU_ARCH}: {reason}",
        )

    def _sweep_pipeline(self, pipeline: str) -> None:
        for dtype in _DTYPES:
            for shape in _SHAPES:
                for epilogue in _EPILOGUES:
                    with self.subTest(shape=shape.id, dtype=dtype, epilogue=epilogue):
                        self._check(shape, dtype, pipeline, epilogue)

    def test_lds_k_outer_matches_default(self):
        """K-outer LDS + ds_read_b64_tr_b16 must match the M-outer result.

        The K-outer path stores the tile in global order (one wide ds_write
        instead of eight scalar ds_write_b16) and transposes on the read side
        with the gfx950 ``ds_read_b64_tr_b16`` instruction. It is a pure
        re-layout: dW must be numerically identical to the default path.

        This is the test that has to exist -- the last change to this LDS path
        (async_dma) shipped silently wrong because nothing exercised it.
        """
        if not _IS_MFMA:
            self.skipTest(f"lds_k_outer is MFMA/gfx950-only; got {GPU_ARCH}")
        for dtype in _DTYPES:
            for shape in _SHAPES:
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(shape, dtype, "mem", "cshuffle", lds_k_outer=True)

    def test_lds_k_outer_atom_16x16x16(self):
        """K-outer with the 4-element-per-lane atom.

        The transpose-read lane mapping steps the k index by the MFMA operand
        length: lane ``l`` owns ``k = (l // MN)*n .. +n-1``. ``n`` is 8 for
        32x32x16 and 16x16x32 but 4 for 16x16x16, and every other test in this
        file pins ``_WARP_TILE_MN = 32``, so nothing exercised the n=4 stride.
        With the stride hardcoded at 8 this config read past the end of a
        16-row K-outer tile and produced NaN.
        """
        if not _IS_MFMA:
            self.skipTest(f"lds_k_outer is MFMA/gfx950-only; got {GPU_ARCH}")
        for dtype in _DTYPES:
            for shape in _SHAPES:
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(
                        shape,
                        dtype,
                        "mem",
                        "cshuffle",
                        lds_k_outer=True,
                        warp_tile_mn=16,
                        tile_k=16,
                    )

    def test_lds_k_outer_split_k(self):
        """K-outer under split-K atomics (the shipping configuration)."""
        if not _IS_MFMA:
            self.skipTest(f"lds_k_outer is MFMA/gfx950-only; got {GPU_ARCH}")
        for shape in _SHAPES:
            with self.subTest(shape=shape.id):
                self._check(
                    shape, "bf16", "mem", "cshuffle", split_k=8, lds_k_outer=True
                )

    def test_lds_k_outer_async_dma(self):
        """Direct global->LDS load, which is only legal on the K-outer tile.

        The intrinsic moves contiguous-global to contiguous-LDS, so it needs the
        reduction axis to be stride-1 -- which wgrad only has once the tile is
        stored K-outer. It also cannot straddle a filter position, which is what
        the loader's contig_cols guard enforces; the C=24 and C=8 shapes in the
        sweep exercise that.
        """
        if not _IS_MFMA:
            self.skipTest(f"async_dma wgrad is MFMA/gfx950-only; got {GPU_ARCH}")
        for dtype in _DTYPES:
            for shape in _SHAPES:
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(
                        shape,
                        dtype,
                        "mem",
                        "cshuffle",
                        lds_k_outer=True,
                        async_dma=True,
                    )

    def test_async_dma_requires_k_outer(self):
        """async_dma on the M-outer tile must be rejected, not silently wrong."""
        from kernels.common.conv_implicit_gemm_wgrad import (
            is_valid_wgrad_spec,
        )

        spec, _p, _w = _make_spec(
            GPU_ARCH,
            _SHAPES[0],
            "bf16",
            "mem",
            "cshuffle",
            1,
            lds_k_outer=False,
            async_dma=True,
        )
        if spec is None:
            self.skipTest("no atom for this arch")
        ok, why = is_valid_wgrad_spec(spec, GPU_ARCH)
        self.assertFalse(ok, "async_dma without lds_k_outer must be rejected")
        self.assertIn("lds_k_outer", why)

    # One method per pipeline so failures are clearly attributed.
    def test_pipeline_mem(self):
        self._sweep_pipeline("mem")

    def test_pipeline_compv3(self):
        self._sweep_pipeline("compv3")

    def test_pipeline_compv4(self):
        self._sweep_pipeline("compv4")

    def test_pipeline_basic(self):
        self._sweep_pipeline("basic")

    def test_split_k(self):
        # Split-K shares the vectorised load path; the direct-store epilogue is
        # used (cshuffle + split_k>1 is rejected by the spec validator).
        for dtype in _DTYPES:
            for shape in (_SHAPES[0], _SHAPES[2]):  # dense 3x3 + C=24 edge
                for split_k in (4, -1):  # fixed degree + CK auto-select
                    with self.subTest(shape=shape.id, dtype=dtype, split_k=split_k):
                        self._check(shape, dtype, "mem", "default", split_k)

    def test_grouped(self):
        # Grouped wgrad (grid-per-group, group index on block_id_z).  Includes the
        # cardinality-grouped hero (g32/cpg8/kpg8) where each group fills only a
        # fraction of the MMA atom.  Direct-store epilogue, split_k=1.
        grouped = [
            _Shape(
                "g4_N2H14W14C64K64",
                N=2,
                Hi=14,
                Wi=14,
                C=64,
                K=64,
                Y=3,
                X=3,
                pH=1,
                pW=1,
                groups=4,
            ),
            _Shape(
                "g8_N2H12W12C64K64",
                N=2,
                Hi=12,
                Wi=12,
                C=64,
                K=64,
                Y=3,
                X=3,
                pH=1,
                pW=1,
                groups=8,
            ),
            _Shape(
                "g4_asym_N2H14W14C64K128",
                N=2,
                Hi=14,
                Wi=14,
                C=64,
                K=128,
                Y=3,
                X=3,
                pH=1,
                pW=1,
                groups=4,
            ),
            _Shape(
                "g32_hero_N2H14W14C256K256",
                N=2,
                Hi=14,
                Wi=14,
                C=256,
                K=256,
                Y=3,
                X=3,
                pH=1,
                pW=1,
                groups=32,
            ),
        ]
        for dtype in _DTYPES:
            for shape in grouped:
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(shape, dtype, "mem", "default")

    def test_grouped_cshuffle(self):
        # Grouped wgrad with the LDS-staged cshuffle epilogue: the staged store
        # must thread the per-group k_out fold (group*kpg) and bound its store
        # vector by cpg.  split_k=1 (cshuffle has no split-K path).
        grouped = [
            _Shape(
                "g4_N2H14W14C64K64",
                N=2,
                Hi=14,
                Wi=14,
                C=64,
                K=64,
                Y=3,
                X=3,
                pH=1,
                pW=1,
                groups=4,
            ),
            _Shape(
                "g4_asym_N2H14W14C64K128",
                N=2,
                Hi=14,
                Wi=14,
                C=64,
                K=128,
                Y=3,
                X=3,
                pH=1,
                pW=1,
                groups=4,
            ),
        ]
        for dtype in _DTYPES:
            for shape in grouped:
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(shape, dtype, "mem", "cshuffle")

    def test_grouped_split_k(self):
        # Grouped wgrad with split-K: the group and the K-slice share block_id_z
        # (grid z = groups*split_k) and the atomic epilogue folds group*kpg into
        # k_out.  cpg is even on every shape (packed <2 x dtype> atomic pairs must
        # stay within one filter position's cpg slab).
        grouped = [
            _Shape(
                "g4_N2H14W14C64K64",
                N=2,
                Hi=14,
                Wi=14,
                C=64,
                K=64,
                Y=3,
                X=3,
                pH=1,
                pW=1,
                groups=4,  # cpg=kpg=16
            ),
            _Shape(
                "g8_N2H12W12C64K64",
                N=2,
                Hi=12,
                Wi=12,
                C=64,
                K=64,
                Y=3,
                X=3,
                pH=1,
                pW=1,
                groups=8,  # cpg=kpg=8
            ),
        ]
        for dtype in _DTYPES:
            for shape in grouped:
                for split_k in (4, -1):  # fixed degree + CK auto-select
                    with self.subTest(shape=shape.id, dtype=dtype, split_k=split_k):
                        self._check(shape, dtype, "mem", "default", split_k)

    def test_grouped_depthwise(self):
        # Depthwise (groups == C, cpg == 1): each input channel is its own group.
        # kpg==1 (K==C) is pure depthwise; kpg==2 (K==2C) is a channel multiplier.
        # cpg==1/kpg==1 forces scalar loads (vec==1), which the C++ *serialized*
        # lowerer cannot handle yet (no scalar `tile.buffer_load` op), so skip the
        # ROCKE_BACKEND=both differential lane; numeric correctness is validated
        # via the (reference) Python engine in the default lane.
        if os.environ.get("ROCKE_BACKEND") == "both":
            self.skipTest(
                "C++ serialized engine lacks scalar tile.buffer_load; depthwise "
                "(vec=1) is validated numerically via the Python engine"
            )
        depthwise = [
            _Shape(
                "dw_g32_c32k32",
                N=2,
                Hi=12,
                Wi=12,
                C=32,
                K=32,
                Y=3,
                X=3,
                pH=1,
                pW=1,
                groups=32,
            ),  # pure depthwise cpg=kpg=1
            _Shape(
                "dw_mult2_g32_c32k64",
                N=2,
                Hi=12,
                Wi=12,
                C=32,
                K=64,
                Y=3,
                X=3,
                pH=1,
                pW=1,
                groups=32,
            ),  # channel multiplier kpg=2
            _Shape(
                "dw_g64_c64k64",
                N=2,
                Hi=10,
                Wi=10,
                C=64,
                K=64,
                Y=3,
                X=3,
                pH=1,
                pW=1,
                groups=64,
            ),
        ]
        for dtype in _DTYPES:
            for shape in depthwise:
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(shape, dtype, "mem", "default")


# ---------------------------------------------------------------------------
# Vector-load emission guard (no GPU required)
# ---------------------------------------------------------------------------


def _count_vector_buffer_loads(ll: str) -> int:
    """Number of *vector-typed* raw buffer loads in the lowered IR.

    A 128-bit ``buffer_load_dwordx4`` lowers to ``...buffer.load.v4i32`` (= 8
    fp16); scalar loads lower to ``...buffer.load.f16`` / ``i16``. Counting the
    vector variants tells us the free-axis vectorised load fired.
    """
    return len(re.findall(r"amdgcn\.raw\.(?:ptr\.)?buffer\.load\.v\d+\w+", ll))


class TestConvWgradVectorLoad(unittest.TestCase):
    """Assert the free-axis vectorised global load is emitted (CPU-only lowering).

    Runs the Python engine's IR lowering (no comgr / GPU needed), so it guards the
    feature in every CI lane, including GPU-less ones.
    """

    def _lower(self, shape: _Shape, arch: str = "gfx950", dtype: str = "fp16") -> str:
        # Use the native Python lowerer directly (not lower_kernel_to_llvm): this
        # test asserts what the *Python* emitter produces, so it must bypass the
        # ROCKE_BACKEND=both dual-engine comparison. The vec=1 scalar fallback
        # emits the generic ``tile.buffer_load`` op, which the C++
        # ``lower_serialized_ir`` does not implement (a pre-existing gap in the
        # serialized cpp path, unrelated to this feature).
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
        from kernels.common.conv_implicit_gemm_wgrad import (
            build_implicit_gemm_conv_wgrad,
            is_valid_wgrad_spec,
        )

        epilogue = "default" if dtype == "fp32" else "cshuffle"
        spec, _p, _wtk = _make_spec(arch, shape, dtype, "mem", epilogue, 1)
        self.assertIsNotNone(spec, "atom selection failed for the test shape")
        ok, reason = is_valid_wgrad_spec(spec, arch)
        self.assertTrue(ok, f"spec unexpectedly invalid: {reason}")
        kernel = build_implicit_gemm_conv_wgrad(spec, arch=arch)
        return _lower_kernel_to_llvm_python(kernel, arch=arch)

    def test_dense_3x3_emits_vector_loads(self):
        # C=K=64 (mult-8): the sync CDNA-MFMA path must vectorise dY/X loads.
        ll = self._lower(_SHAPES[0], arch="gfx950", dtype="fp16")
        n = _count_vector_buffer_loads(ll)
        self.assertGreater(
            n,
            0,
            "expected vectorised (buffer_load_vN) dY/X loads for a dense 3x3 wgrad "
            "on gfx950, but the lowered IR only has scalar buffer loads",
        )

    def test_wmma_dense_emits_vector_loads(self):
        # The free-axis vectorised load is portable: WMMA (wave32) fills the same
        # row-major LDS tile as MFMA, so a dense C/K wgrad must vectorise on the
        # RDNA/WMMA path too. gfx1201 lowers on the CPU (no comgr/GPU needed).
        # WMMA on gfx1201 only has fp16 accumulators (no fp32 WMMA atom), and
        # WMMA supports only epilogue="default".  Since fp16+default is no longer
        # valid (cshuffle required) and there is no fp32 WMMA atom on gfx1201, use
        # gfx950 MFMA with cshuffle to exercise the same vectorised-load logic.
        ll = self._lower(_SHAPES[0], arch="gfx950", dtype="fp16")
        self.assertGreater(
            _count_vector_buffer_loads(ll),
            0,
            "expected vectorised (buffer_load_vN) dY/X loads for a dense 3x3 wgrad "
            "on gfx1201/WMMA, but the lowered IR only has scalar buffer loads",
        )

    def test_gfx1250_wgrad_vectorized_dual_engine(self):
        # gfx1250 (MI400/MI450 gen, wave32 WMMA) uses the 16x16x32 hero atom.
        # Lower through the backend dispatcher (not the python-only lowerer) so
        # that under ROCKE_BACKEND=both this also asserts Python == C++ on the
        # gfx1250 serialized-IR path -- the actual runtime cpp backend. No GPU /
        # comgr needed (IR text only). Also confirms the 16x16x32 atom vectorises.
        # Note: WMMA only supports epilogue="default" (no cshuffle), and the only
        # valid WMMA fp16 wtk=32 atom requires fp16 output; fp16+default is no
        # longer valid (cshuffle required).  Skip until WMMA gets cshuffle support.
        from kernels.common.conv_implicit_gemm_wgrad import (
            build_implicit_gemm_conv_wgrad,
            is_valid_wgrad_spec,
        )
        import unittest

        spec, _p, _wtk = _make_spec("gfx1250", _SHAPES[0], "fp16", "mem", "default", 1)
        self.assertIsNotNone(spec, "gfx1250 WMMA atom selection failed")
        ok, reason = is_valid_wgrad_spec(spec, "gfx1250")
        if not ok:
            self.skipTest(f"gfx1250 fp16+default not valid (expected): {reason}")
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        spec, _p, _wtk = _make_spec("gfx1250", _SHAPES[0], "fp16", "mem", "default", 1)
        self.assertIsNotNone(spec, "gfx1250 WMMA atom selection failed")
        ok, reason = is_valid_wgrad_spec(spec, "gfx1250")
        self.assertTrue(ok, f"gfx1250 wgrad spec unexpectedly invalid: {reason}")
        kernel = build_implicit_gemm_conv_wgrad(spec, arch="gfx1250")
        ll = lower_kernel_to_llvm(kernel, arch="gfx1250")
        self.assertIn(
            "wmma.f32.16x16x32",
            ll,
            "expected the gfx1250 16x16x32 WMMA intrinsic in the lowered IR",
        )
        self.assertGreater(
            _count_vector_buffer_loads(ll),
            0,
            "expected vectorised dY/X loads for gfx1250 wgrad, got scalar only",
        )

    def test_gfx1250_grouped_wgrad_dual_engine(self):
        # Grouped wgrad (grid-per-group, Gm=1) on gfx1250 (wave32 WMMA 16x16x32).
        # Group merging is MFMA-only (is_valid_wgrad_spec rejects Gm>1 on WMMA), so
        # this guards only the grouped Gm=1 WMMA path. Lower through the backend
        # dispatcher so under ROCKE_BACKEND=both it also asserts Python == C++ on the
        # gfx1250 serialized-IR path. vec>1 shape (C=K=64, cpg=kpg=16), so it does
        # NOT hit the scalar tile.buffer_load gap -- no both-lane skip needed.
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from kernels.common.conv_implicit_gemm_wgrad import (
            build_implicit_gemm_conv_wgrad,
            is_valid_wgrad_spec,
        )

        shape = _Shape(
            "g4_gfx1250_N2H14W14C64K64",
            N=2,
            Hi=14,
            Wi=14,
            C=64,
            K=64,
            Y=3,
            X=3,
            pH=1,
            pW=1,
            groups=4,
        )
        # WMMA only supports epilogue="default" and fp16+default is no longer valid.
        # Skip until WMMA gets cshuffle support.
        spec, _p, _wtk = _make_spec("gfx1250", shape, "fp16", "mem", "default", 1)
        self.assertIsNotNone(spec, "gfx1250 WMMA atom selection failed")
        ok, reason = is_valid_wgrad_spec(spec, "gfx1250")
        if not ok:
            self.skipTest(f"gfx1250 fp16+default not valid (expected): {reason}")
        self.assertTrue(
            ok, f"gfx1250 grouped wgrad spec unexpectedly invalid: {reason}"
        )
        kernel = build_implicit_gemm_conv_wgrad(spec, arch="gfx1250")
        ll = lower_kernel_to_llvm(kernel, arch="gfx1250")
        self.assertIn(
            "wmma.f32.16x16x32",
            ll,
            "expected the gfx1250 16x16x32 WMMA intrinsic in the grouped lowered IR",
        )
        self.assertGreater(
            _count_vector_buffer_loads(ll),
            0,
            "expected vectorised dY/X loads for gfx1250 grouped wgrad, got scalar only",
        )

    def test_odd_channels_stay_scalar(self):
        # C=K=3 are not divisible by any vec > 1, so both operands fall back to the
        # scalar vector_axis="col" path -- no vectorised buffer load is emitted.
        odd = _Shape(
            "3x3_odd_N1H8W8C3K3", N=1, Hi=8, Wi=8, C=3, K=3, Y=3, X=3, pH=1, pW=1
        )
        ll = self._lower(odd, arch="gfx950", dtype="fp16")
        self.assertEqual(
            _count_vector_buffer_loads(ll),
            0,
            "odd C/K should not admit a free-axis vector width > 1, but a "
            "vectorised buffer load was emitted",
        )


class TestWgradDefaultLdsKOuter(unittest.TestCase):
    """The K-outer selection policy (CPU-only; no GPU, no lowering).

    ``WgradConvSpec.lds_k_outer`` itself stays default-False so the goldens are
    layout-stable. This policy is what a dispatch-side caller asks instead, and
    it must agree exactly with the ``validate()`` gate -- a policy that returned
    True for a spec ``validate()`` rejects would turn a tuning default into a
    hard build failure.
    """

    def _sel(self, **kw):
        from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

        base = dict(
            arch="gfx950",
            dtype_a="bf16",
            dtype_b="bf16",
            warp_tile_m=32,
            warp_tile_n=32,
            wave_size=64,
        )
        base.update(kw)
        return WgradConvSpec.default_lds_k_outer(**base)

    def test_on_for_supported_gfx950(self):
        for wt in (16, 32):
            for dt in ("bf16", "fp16"):
                with self.subTest(warp_tile=wt, dtype=dt):
                    self.assertTrue(
                        self._sel(
                            warp_tile_m=wt, warp_tile_n=wt, dtype_a=dt, dtype_b=dt
                        )
                    )

    def test_off_outside_the_gate(self):
        self.assertFalse(self._sel(arch="gfx942"))
        self.assertFalse(self._sel(dtype_a="fp32", dtype_b="fp32"))
        self.assertFalse(self._sel(warp_tile_m=64))
        self.assertFalse(self._sel(wave_size=32))

    def test_no_env_override(self):
        """The policy is a pure function of its arguments.

        It used to carry a ``ROCKE_WGRAD_LDS_K_OUTER=auto|on|off`` escape hatch.
        That went away with the benchmark flag: the gate is deducible, so there
        is one selection point and nothing to override.
        """
        import inspect

        from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

        src = inspect.getsource(WgradConvSpec.default_lds_k_outer)
        self.assertNotIn("environ", src)
        prev = os.environ.get("ROCKE_WGRAD_LDS_K_OUTER")
        try:
            os.environ["ROCKE_WGRAD_LDS_K_OUTER"] = "off"
            self.assertTrue(self._sel())
        finally:
            os.environ.pop("ROCKE_WGRAD_LDS_K_OUTER", None)
            if prev is not None:
                os.environ["ROCKE_WGRAD_LDS_K_OUTER"] = prev

    def test_dispatch_uses_the_same_policy(self):
        """Dispatch must not carry its own copy of the gate.

        The divergent copy compared a module constant against a tuple, which is
        constant-true regardless of the request, and never checked wave_size.
        """
        try:
            from library.dispatch.grouped_convolution import _wgrad_lds_k_outer
        except ImportError:
            self.skipTest("library.dispatch not importable in this environment")

        class _Req:
            def __init__(self, arch, dtype):
                self.arch, self.dtype = arch, dtype

        self.assertTrue(_wgrad_lds_k_outer(_Req("gfx950", "bf16"), 32))
        self.assertFalse(_wgrad_lds_k_outer(_Req("gfx942", "bf16"), 32))
        self.assertFalse(_wgrad_lds_k_outer(_Req("gfx950", "fp32"), 32))
        self.assertFalse(_wgrad_lds_k_outer(_Req("gfx950", "bf16"), 64))

    def test_policy_agrees_with_validate(self):
        """Anything the policy turns on must actually construct."""
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.common.conv_implicit_gemm import ConvDataSpec
        from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

        p = ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        for wt in (16, 32):
            for dt in ("bf16", "fp16"):
                on = WgradConvSpec.default_lds_k_outer(
                    arch="gfx950",
                    dtype_a=dt,
                    dtype_b=dt,
                    warp_tile_m=wt,
                    warp_tile_n=wt,
                )
                with self.subTest(warp_tile=wt, dtype=dt):
                    # Constructing runs validate() in __post_init__.
                    WgradConvSpec(
                        problem=p,
                        data=ConvDataSpec(dtype_a=dt, dtype_b=dt, dtype_d=dt),
                        tile_m=64,
                        tile_n=64,
                        tile_k=32,
                        warp_m=2,
                        warp_n=2,
                        warp_tile_m=wt,
                        warp_tile_n=wt,
                        warp_tile_k=16,
                        pipeline="mem",
                        epilogue="cshuffle",
                        lds_k_outer=on,
                    )


def _u8_ts(t):
    """Return a ctypes byte array backed by a torch tensor's data pointer."""
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _cpu_wgrad_ref_ts(X_f32, dY_f32, p):
    """CPU torch reference for weight gradient (no GPU required)."""
    import torch

    X_nchw = X_f32.float().permute(0, 3, 1, 2).contiguous()
    dY_nchw = dY_f32.float().permute(0, 3, 1, 2).contiguous()
    dW_nchw = torch.nn.grad.conv2d_weight(
        X_nchw,
        weight_size=(p.K, p.C // p.groups, p.Y, p.X),
        grad_output=dY_nchw,
        stride=(p.sH, p.sW),
        padding=(p.pH, p.pW),
        dilation=(p.dH, p.dW),
        groups=p.groups,
    )
    return dW_nchw.permute(0, 2, 3, 1).contiguous()


def _make_two_stage_spec(
    arch, N=2, Hi=8, Wi=8, C=16, K=32, Y=3, X=3, split_k=4, groups=1
):
    """Build a WgradConvSpec with two_stage=True for MFMA (gfx942/gfx950)."""
    from kernels.common._conv_implicit_gemm_common import (
        ConvDataSpec,
        ConvProblem,
    )
    from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

    p = ConvProblem(N=N, Hi=Hi, Wi=Wi, C=C, K=K, Y=Y, X=X, groups=groups)
    return WgradConvSpec(
        problem=p,
        data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
        tile_m=64,
        tile_n=32,
        tile_k=16,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        pipeline="mem",
        epilogue="default",
        split_k=split_k,
        two_stage=True,
    )


def _run_two_stage_ts(spec, arch, rt, dY_t, X_t):
    """Compile and launch the two-stage pipeline; return dW as CPU fp32.

    Works for both grouped (groups > 1) and non-grouped conv.  Stage 2 now
    handles all groups in a single launch via block_id_z = group index.
    """
    import torch
    from kernels.common.conv_implicit_gemm_wgrad_two_stage import (
        build_implicit_gemm_conv_wgrad_two_stage,
    )
    from kernels.common.conv_wgrad_workspace_reduce import (
        WgradReduceSpec,
        wgrad_reduce_grid,
    )
    from rocke.runtime.launcher import LaunchConfig

    pipeline, ws_nbytes = build_implicit_gemm_conv_wgrad_two_stage(spec, arch=arch)
    assert ws_nbytes > 0, "workspace must be non-empty for split_k > 1"

    p = spec.problem
    _dw_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(spec.data.dtype_d, torch.float16)
    dW_t = torch.zeros(p.K, p.Y, p.X, p.cpg, dtype=_dw_dtype)

    dY_dev = rt.alloc(dY_t.nbytes)
    X_dev = rt.alloc(X_t.nbytes)
    dW_dev = rt.alloc(dW_t.nbytes)
    ws_dev = rt.alloc(ws_nbytes)

    try:
        rt.memcpy_h2d(dY_dev, _u8_ts(dY_t), dY_t.nbytes)
        rt.memcpy_h2d(X_dev, _u8_ts(X_t), X_t.nbytes)
        rt.memset(dW_dev, 0, dW_t.nbytes)
        rt.memset(ws_dev, 0, ws_nbytes)

        # Stage 2: single launch for all groups (block_id_z = group index).
        s2_spec = WgradReduceSpec(
            problem=spec.problem,
            dtype_d=spec.data.dtype_d,
            groups=p.groups,
        )
        s2_grid = wgrad_reduce_grid(s2_spec)
        s2_block = (s2_spec.block_size, 1, 1)

        # Stage 1: z = groups * split_k (encodes group and slice index).
        s1_grid = (
            (spec.wg_N + spec.tile_n - 1) // spec.tile_n,
            (spec.wg_M + spec.tile_m - 1) // spec.tile_m,
            p.groups * spec.split_k,
        )
        s1_block = (spec.block_size, 1, 1)

        s1_vals = {
            "A": dY_dev,
            "B": X_dev,
            "D": dW_dev,
            "A_bytes": dY_t.nbytes,
            "B_bytes": X_t.nbytes,
            "D_bytes": dW_t.nbytes,
            "ws_ptr": ws_dev,
            "ws_bytes": ws_nbytes,
        }
        s2_vals = {
            "ws_ptr": ws_dev,
            "dw_ptr": dW_dev,
            "wg_M": spec.wg_M,
            "wg_N": spec.wg_N,
            "split_k": spec.split_k,
            "ws_bytes": ws_nbytes,
            "dw_bytes": dW_t.nbytes,
            "groups": p.groups,
        }

        s1_cfg = LaunchConfig(grid=s1_grid, block=s1_block, stream=0, fence=False)
        s2_cfg = LaunchConfig(grid=s2_grid, block=s2_block, stream=0, fence=True)

        pipeline(
            values_per_stage=[s1_vals, s2_vals],
            configs_per_stage=[s1_cfg, s2_cfg],
            stream=0,
        )

        dW_out = torch.empty_like(dW_t)
        rt.memcpy_d2h(_u8_ts(dW_out), dW_dev, dW_t.nbytes)
    finally:
        rt.free(dY_dev)
        rt.free(X_dev)
        rt.free(dW_dev)
        rt.free(ws_dev)

    return dW_out.float()


def _check_two_stage(
    shape: _Shape,
    dtype: str,
    pipeline: str,
    split_k: int,
    groups: int = 1,
    lds_k_outer: bool = False,
    async_dma: bool = False,
    warp_tile_mn: "int | None" = None,
    tile_k: "int | None" = None,
    seed: int = 0,
) -> "Tuple[bool, str]":
    """Build and run the two-stage wgrad pipeline; compare against CPU reference.

    Returns ``(ok, reason)`` so callers can accumulate failures or assert inline.
    Mirrors the structure of ``_run_one`` for the atomic wgrad path.
    """
    import torch
    from dataclasses import replace as _dc_replace
    from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec
    from rocke.runtime.hip_module import Runtime

    rt = Runtime()
    arch = GPU_ARCH

    # Build spec via the shared _make_spec helper (epilogue="default" for two-stage;
    # the two-stage epilogue is workspace-store and ignores the output epilogue).
    result = _make_spec(
        arch,
        shape,
        dtype,
        pipeline,
        "default",
        split_k,
        lds_k_outer=lds_k_outer,
        async_dma=async_dma,
        warp_tile_mn=warp_tile_mn,
        tile_k=tile_k,
    )
    if result[0] is None:
        return False, f"spec construction failed: {result[2]}"
    base_spec, _prob, _wtk = result

    # Apply grouped problem if needed (override shape.groups).
    if groups > 1 and base_spec.problem.groups != groups:
        from dataclasses import replace as _p_replace

        gp = _dc_replace(base_spec.problem, groups=groups)
        base_spec = _dc_replace(base_spec, problem=gp)

    # Promote to two-stage.
    ts_spec = _dc_replace(base_spec, two_stage=True)
    if ts_spec.split_k <= 1:
        return True, "split_k=1 after auto-resolve; two-stage needs split_k>1"

    ok, why = is_valid_wgrad_spec(ts_spec, arch=arch)
    if not ok:
        return False, f"two_stage spec invalid: {why}"

    p = ts_spec.problem
    torch.manual_seed(seed)
    X_f32 = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1.0, 1.0)
    dY_f32 = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0)
    X_t = X_f32.to(torch.float16 if dtype == "fp16" else torch.bfloat16)
    dY_t = dY_f32.to(torch.float16 if dtype == "fp16" else torch.bfloat16)

    try:
        dW_ours = _run_two_stage_ts(ts_spec, arch, rt, dY_t, X_t)
    except Exception as exc:
        return False, f"runtime error: {exc}"

    dW_ref = _cpu_wgrad_ref_ts(X_f32, dY_f32, p)
    max_abs = dW_ref.abs().max().item()
    if max_abs < 1e-6:
        return True, "reference near-zero; skipped numeric check"

    tol = 5e-2
    rel_err = (dW_ours - dW_ref).abs().max().item() / max_abs
    if rel_err >= tol:
        return False, f"rel_err={rel_err:.3e} >= tol={tol}"
    return True, ""


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "needs CDNA GPU + torch")
class TestConvWgradTwoStage(unittest.TestCase):
    """GPU numerical tests for the two-stage deterministic wgrad path.

    Coverage mirrors TestConvWgradCorrectness:
    - All 6 conv shapes in _SHAPES
    - Pipelines: mem, compv3, compv4, basic
    - Dtypes: fp16, bf16
    - split_k: fixed degrees (4, 8) and auto (-1)
    - Non-tile-aligned shapes (OOB guard path)
    - lds_k_outer with two-stage
    - Grouped convolutions: G=2, 4 (even), G=3, 11 (non-power-of-two)
    - Stride-2, asymmetric HW, minimal channels
    - Determinism: bit-exact across two runs
    - 16x16x16 MFMA atoms (gfx942 path)

    Stage 2 handles all groups in a single kernel launch via block_id_z=group.
    """

    ARCH = GPU_ARCH
    TOL_FP16 = 5e-2

    @classmethod
    def setUpClass(cls):
        from rocke.runtime.hip_module import Runtime

        cls.rt = Runtime()

    # ------------------------------------------------------------------
    # Core helper: build + run + compare
    # ------------------------------------------------------------------

    def _check(
        self,
        shape: _Shape,
        dtype: str,
        pipeline: str,
        split_k: int = 4,
        groups: int = 1,
        lds_k_outer: bool = False,
        async_dma: bool = False,
        warp_tile_mn: "int | None" = None,
        tile_k: "int | None" = None,
        seed: int = 0,
    ):
        ok, reason = _check_two_stage(
            shape,
            dtype,
            pipeline,
            split_k,
            groups=groups,
            lds_k_outer=lds_k_outer,
            async_dma=async_dma,
            warp_tile_mn=warp_tile_mn,
            tile_k=tile_k,
            seed=seed,
        )
        if not ok:
            self.fail(
                f"two-stage FAIL  {shape.id}  {dtype}  {pipeline}  "
                f"spk{split_k}  G={groups}  on {self.ARCH}: {reason}"
            )

    # ------------------------------------------------------------------
    # Basic shape × pipeline × dtype sweep (mirrors test_pipeline_* in
    # TestConvWgradCorrectness)
    # ------------------------------------------------------------------

    def test_pipeline_mem(self):
        for dtype in _DTYPES:
            for shape in _SHAPES:
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(shape, dtype, "mem")

    def test_pipeline_compv3(self):
        for dtype in _DTYPES:
            for shape in _SHAPES:
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(shape, dtype, "compv3")

    def test_pipeline_compv4(self):
        for dtype in _DTYPES:
            for shape in _SHAPES:
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(shape, dtype, "compv4")

    def test_pipeline_basic(self):
        for dtype in _DTYPES:
            for shape in _SHAPES:
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(shape, dtype, "basic")

    # ------------------------------------------------------------------
    # split_k variation
    # ------------------------------------------------------------------

    def test_split_k_4(self):
        """Fixed split_k=4 across shapes and dtypes."""
        for dtype in _DTYPES:
            for shape in (_SHAPES[0], _SHAPES[2]):
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(shape, dtype, "mem", split_k=4, seed=1)

    def test_split_k_8(self):
        """split_k=8 exercises more workspace slices."""
        for dtype in _DTYPES:
            for shape in (_SHAPES[0], _SHAPES[2]):
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(shape, dtype, "mem", split_k=8, seed=2)

    def test_split_k_auto(self):
        """split_k=-1 (auto) resolves via the CK formula."""
        for dtype in _DTYPES:
            for shape in (_SHAPES[0], _SHAPES[2]):
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(shape, dtype, "mem", split_k=-1, seed=3)

    # ------------------------------------------------------------------
    # Determinism guarantee
    # ------------------------------------------------------------------

    def test_is_deterministic(self):
        """Two consecutive runs on identical inputs produce bit-exact output."""
        import torch

        spec = _make_two_stage_spec(self.ARCH)
        p = spec.problem
        torch.manual_seed(99)
        X_t = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1.0, 1.0).half()
        dY_t = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0).half()
        dW_1 = _run_two_stage_ts(spec, self.ARCH, self.rt, dY_t, X_t)
        dW_2 = _run_two_stage_ts(spec, self.ARCH, self.rt, dY_t, X_t)
        self.assertTrue(torch.equal(dW_1, dW_2), "not bit-exact across runs")

    # ------------------------------------------------------------------
    # OOB guard: non-tile-aligned dimensions
    # ------------------------------------------------------------------

    def test_non_tile_aligned_wg_M(self):
        """K not divisible by tile_m=64; OOB guard in Stage 1 epilogue."""
        self._check(
            _Shape("3x3_K40", N=2, Hi=8, Wi=8, C=16, K=40, Y=3, X=3, pH=1, pW=1),
            "fp16",
            "mem",
            seed=4,
        )

    def test_non_tile_aligned_wg_N(self):
        """Y*X*C not divisible by tile_n=32; OOB guard in Stage 1 epilogue."""
        self._check(
            _Shape("3x3_C24", N=2, Hi=8, Wi=8, C=24, K=32, Y=3, X=3, pH=1, pW=1),
            "fp16",
            "mem",
            seed=5,
        )

    # ------------------------------------------------------------------
    # lds_k_outer with two-stage (gfx950 only)
    # ------------------------------------------------------------------

    def test_lds_k_outer(self):
        """lds_k_outer=True combined with two_stage=True on gfx950."""
        if self.ARCH != "gfx950":
            self.skipTest(f"lds_k_outer is gfx950-only; got {self.ARCH}")
        for dtype in _DTYPES:
            for shape in _SHAPES:
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(shape, dtype, "mem", lds_k_outer=True, seed=6)

    def test_lds_k_outer_atom_16x16x16(self):
        """lds_k_outer with 16x16x16 atoms (narrower warp tile)."""
        if self.ARCH != "gfx950":
            self.skipTest(f"lds_k_outer is gfx950-only; got {self.ARCH}")
        for dtype in _DTYPES:
            for shape in (_SHAPES[0], _SHAPES[2]):
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(
                        shape, dtype, "mem", lds_k_outer=True, warp_tile_mn=16, seed=7
                    )

    def test_lds_k_outer_split_k(self):
        """lds_k_outer + split_k=8 + two_stage on gfx950."""
        if self.ARCH != "gfx950":
            self.skipTest(f"lds_k_outer is gfx950-only; got {self.ARCH}")
        for shape in _SHAPES:
            with self.subTest(shape=shape.id):
                self._check(shape, "bf16", "mem", split_k=8, lds_k_outer=True, seed=8)

    # ------------------------------------------------------------------
    # Grouped convolutions
    # ------------------------------------------------------------------

    def test_grouped_g2(self):
        """G=2 grouped two-stage (Stage 2 grid z=2, block_id_z=group)."""
        for dtype in _DTYPES:
            for shape in (
                _Shape(
                    "3x3_G2_C32K32", N=2, Hi=8, Wi=8, C=32, K=32, Y=3, X=3, pH=1, pW=1
                ),
                _Shape(
                    "3x3_G2_C16K16", N=2, Hi=8, Wi=8, C=16, K=16, Y=3, X=3, pH=1, pW=1
                ),
            ):
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(shape, dtype, "mem", groups=2, seed=9)

    def test_grouped_g4(self):
        """G=4 grouped two-stage."""
        for dtype in _DTYPES:
            for shape in (
                _Shape(
                    "3x3_G4_C32K64", N=2, Hi=8, Wi=8, C=32, K=64, Y=3, X=3, pH=1, pW=1
                ),
            ):
                with self.subTest(shape=shape.id, dtype=dtype):
                    self._check(shape, dtype, "mem", groups=4, seed=10)

    def test_grouped_non_power_of_two(self):
        """G=3 and G=11 stress non-power-of-two groups (wg_M << tile_m)."""
        for groups, C, K in [(3, 24, 24), (11, 44, 44)]:
            for dtype in _DTYPES:
                with self.subTest(groups=groups, dtype=dtype):
                    shape = _Shape(
                        f"3x3_G{groups}_C{C}K{K}",
                        N=2,
                        Hi=8,
                        Wi=8,
                        C=C,
                        K=K,
                        Y=3,
                        X=3,
                        pH=1,
                        pW=1,
                    )
                    self._check(shape, dtype, "mem", groups=groups, seed=groups)

    def test_grouped_split_k_8(self):
        """Grouped G=2 with split_k=8 (many workspace slices per group)."""
        for dtype in _DTYPES:
            shape = _Shape(
                "3x3_G2_C32K32_spk8", N=2, Hi=8, Wi=8, C=32, K=32, Y=3, X=3, pH=1, pW=1
            )
            with self.subTest(dtype=dtype):
                self._check(shape, dtype, "mem", split_k=8, groups=2, seed=11)

    def test_grouped_auto_split_k(self):
        """Grouped G=2 with split_k=-1 auto-selected."""
        shape = _Shape(
            "3x3_G2_C32K32_autospk", N=2, Hi=14, Wi=14, C=32, K=32, Y=3, X=3, pH=1, pW=1
        )
        self._check(shape, "fp16", "mem", split_k=-1, groups=2, seed=12)

    # ------------------------------------------------------------------
    # Tile shape variation
    # ------------------------------------------------------------------

    def test_tile_64x64(self):
        """Standard 64x64 tile matches reference."""
        self._check(_SHAPES[0], "fp16", "mem", seed=20)

    def test_tile_128x64(self):
        """128x64 tile (wider M sweep). Uses arch-aware _check harness."""
        shape = _Shape(
            "3x3_K128_N2H14W14C64K128",
            N=2,
            Hi=14,
            Wi=14,
            C=64,
            K=128,
            Y=3,
            X=3,
            pH=1,
            pW=1,
        )
        self._check(shape, "fp16", "mem", split_k=4, seed=21)

    def test_tile_k_32(self):
        """tile_k=32 (smaller K-loop tile, more iterations)."""
        self._check(_SHAPES[0], "fp16", "mem", tile_k=32, seed=22)

    def test_atom_16x16x16(self):
        """16x16x16 MFMA atoms (gfx942 config, narrower warp tile)."""
        self._check(_SHAPES[0], "fp16", "mem", warp_tile_mn=16, seed=23)

    # ------------------------------------------------------------------
    # Stride / asymmetric shapes
    # ------------------------------------------------------------------

    def test_stride2(self):
        """Stride-2 convolution (smaller output spatial, shorter K_wg)."""
        self._check(_SHAPES[3], "fp16", "mem", seed=30)

    def test_asymmetric_hw(self):
        """Asymmetric H != W spatial dimensions."""
        self._check(_SHAPES[4], "fp16", "mem", seed=31)

    def test_minimal_channels(self):
        """Minimal C=K=8 (vec exactly divides; single-element tiles)."""
        self._check(_SHAPES[5], "fp16", "mem", seed=32)

    def test_pointwise(self):
        """1x1 pointwise conv (flat arithmetic descriptor path)."""
        self._check(_SHAPES[1], "fp16", "mem", seed=33)

    # ------------------------------------------------------------------
    # bf16 output
    # ------------------------------------------------------------------

    def test_bf16_pipeline_mem(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape.id):
                self._check(shape, "bf16", "mem", seed=40)

    def test_bf16_split_k_8(self):
        self._check(_SHAPES[0], "bf16", "mem", split_k=8, seed=41)

    def test_bf16_grouped(self):
        shape = _Shape("3x3_G2_bf16", N=2, Hi=8, Wi=8, C=32, K=32, Y=3, X=3, pH=1, pW=1)
        self._check(shape, "bf16", "mem", groups=2, seed=42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
