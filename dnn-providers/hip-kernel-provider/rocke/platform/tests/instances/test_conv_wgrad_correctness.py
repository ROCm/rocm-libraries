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
    arch: str, shape: _Shape, dtype: str, pipeline: str, epilogue: str, split_k: int
):
    """Build a (spec, problem, warp_tile_k) triple, or (None, None, reason)."""
    from rocke.core.arch import ArchTarget
    from rocke.instances.common._conv_implicit_gemm_common import ConvProblem
    from rocke.instances.common.conv_implicit_gemm import ConvDataSpec
    from rocke.instances.common.conv_implicit_gemm_wgrad import WgradConvSpec

    target = ArchTarget.from_gfx(arch)
    # MMA family + tile shape follow the wave size: wave64 -> MFMA (32x32 atom),
    # wave32 -> WMMA (16x16 atom). The MFMA branch keeps the historical values so
    # its emitted IR / goldens are unchanged; WMMA uses a 16x16-shaped tile.
    family = "wmma" if target.wave_size == 32 else "mma"
    if family == "wmma":
        tile_m, tile_n, tile_k = 32, 32, 32
        warp_m, warp_n, warp_tile_mn = 1, 1, 16
    else:
        tile_m, tile_n, tile_k = _TILE_M, _TILE_N, _TILE_K
        warp_m, warp_n, warp_tile_mn = _WARP_M, _WARP_N, _WARP_TILE_MN
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
        name=f"test_wgrad_{shape.id}_{dtype}_{pipeline}_{epilogue}_spk{split_k}",
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
) -> Tuple[bool, str]:
    """Build, compile, launch, and verify one wgrad kernel.

    Returns ``(passed, reason)`` where ``reason`` is non-empty on skip or failure.
    """
    import torch

    from rocke import compile_kernel
    from rocke.helpers.manifest import conv_args_signature
    from rocke.instances.common.conv_implicit_gemm_wgrad import (
        build_implicit_gemm_conv_wgrad,
        is_valid_wgrad_spec,
    )
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    spec, problem, _wtk = _make_spec(arch, shape, dtype, pipeline, epilogue, split_k)
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

    def _check(self, shape, dtype, pipeline, epilogue, split_k=1) -> None:
        passed, reason = _run_one(GPU_ARCH, shape, dtype, pipeline, epilogue, split_k)
        if reason.startswith("skip"):
            self.skipTest(reason)
        self.assertTrue(
            passed,
            f"FAIL {shape.id} {dtype} {pipeline}/{epilogue} spk{split_k} "
            f"on {GPU_ARCH}: {reason}",
        )

    def _sweep_pipeline(self, pipeline: str) -> None:
        for dtype in _DTYPES:
            for shape in _SHAPES:
                for epilogue in _EPILOGUES:
                    with self.subTest(shape=shape.id, dtype=dtype, epilogue=epilogue):
                        self._check(shape, dtype, pipeline, epilogue)

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
        from rocke.instances.common.conv_implicit_gemm_wgrad import (
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
        from rocke.instances.common.conv_implicit_gemm_wgrad import (
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
        from rocke.instances.common.conv_implicit_gemm_wgrad import (
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
