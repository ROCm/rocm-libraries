# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Correctness tests for direct grouped convolution across cpg variants.

Covers all four grouped variants (cpg = 4, 8, 16, 32), the depthwise
variant (cpg = 1), and the non-grouped (groups == 1) ``DirectNongroupedConvSpec``.  Each test builds a kernel, compiles it, launches it on
GPU, and compares the output against a float32 reference produced by
torch.nn.functional.conv2d.

The test shapes are kept intentionally small (fast compile + run) while
hitting the branch points that differ across implementations:
  - grouped DirectConvSpec (mfma_f32_16x16x16_f16) across representative cpg values
  - depthwise scalar-FMA path (cpg=1)

Requires a ROCm GPU (gfx942 or gfx950) and torch.  Run:
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \
        rocke/library/tests/test_direct_conv_correctness.py -v
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest
from dataclasses import dataclass, replace
from typing import List, Tuple

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None

if _HAS_TORCH:
    # Claim the process HIP context for torch before rocke's runtime touches it.
    # rocke's HIP runtime and torch's fight over the context and whichever
    # initialises first wins; rocke-first leaves torch with "No HIP GPUs are
    # available" for the rest of the process, breaking the .cuda() reference
    # below. See _wgrad_reference_cpu in test_conv_wgrad_correctness.py.
    import torch

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
# The depthwise forward kernels are scalar FMA with one lane per channel (or
# per channel and column), so they also run on the wave32 gfx1250.
_IS_WAVE32_DW = GPU_ARCH == "gfx1250"


def _wave_size(arch: str) -> int:
    """The target's wave size: what a direct depthwise spec must be built with."""
    from rocke.core.arch import ArchTarget

    return ArchTarget.from_gfx(arch).wave_size


_TOL = 5e-2
_TOL_BF16 = 1e-1  # bf16 has 3 fewer mantissa bits than fp16 (~8x coarser precision)
# The tiled depthwise kernel accumulates in fp32 and rounds once on store, so a
# correct result is within about one output ulp; the loose tolerances above
# would still pass it with a filter tap dropped.
_TILED_TOL = {"fp16": 5e-3, "bf16": 1.5e-2}


# ---------------------------------------------------------------------------
# Test shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Shape:
    """One test problem for direct conv.

    ``cpg`` must equal ``kpg`` and must be 1 (depthwise) or a positive
    multiple of 4.  ``stride`` may exceed 1 for every variant here except
    ``DirectDepthwiseSpec``, whose preloaded-weight kernel the ``_SHAPES``
    entries keep at 1.

    ``dtype``, ``block_h``, ``block_w`` and ``block_waves`` are honoured by the
    column-streamed depthwise runner only; the other three runners are
    fp16-only and take those knobs from their spec defaults, so the fields are
    inert for them.

    For fprop ``cpg`` must equal ``kpg`` (both symmetric). For dgrad they may
    differ; ``kpg=0`` (the default) means kpg == cpg (symmetric).
    """

    id: str
    N: int
    H: int
    W: int
    groups: int
    cpg: int  # input channels-per-group
    KH: int = 3
    KW: int = 3
    PAD: int = 1
    stride: int = 1
    dtype: str = "fp16"
    block_w: int = 1
    block_waves: int = 1
    block_h: int = 16  # col runner: output rows per block
    kpg: int = 0  # output channels-per-group; 0 means same as cpg
    block_groups: int = 0  # dgrad block_groups override; 0 means use spec default


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
    # stride=2 cases — output H=6, W=6 for H=W=14, PAD=1, KH=KW=3
    # groups=8 satisfies DirectConv16cSpec/DirectConvSpec default block_groups=8
    _Shape("4c_N2H14W14_g8_s2", N=2, H=14, W=14, groups=8, cpg=4, stride=2),
    _Shape("16c_N2H14W14_g8_s2", N=2, H=14, W=14, groups=8, cpg=16, stride=2),
    # Depthwise 7x7 "same": the FMA loop runs (H + KH - 1) // KH row groups and
    # the load-free flush loop the rest up to (H + PAD + KH - 1) // KH, so the
    # 3x3 case above (5 and 5) never enters the flush. H=14 gives 2 and 3, the
    # non-multiple H=13 2 and 3, the small H=5 1 and 2 -- each shifts the
    # accumulator slots the flush drains.
    _Shape("dw_k7_N2H14W14_g64", N=2, H=14, W=14, groups=64, cpg=1, KH=7, KW=7, PAD=3),
    _Shape(
        "dw_k7s2_N2H14W14_g64",
        N=2,
        H=14,
        W=14,
        groups=64,
        cpg=1,
        KH=7,
        KW=7,
        PAD=3,
        stride=2,
    ),
    _Shape(
        "dw_k7s2_N2H13W11_g64",
        N=2,
        H=13,
        W=11,
        groups=64,
        cpg=1,
        KH=7,
        KW=7,
        PAD=3,
        stride=2,
    ),
    _Shape(
        "dw_k7s2_N1H5W9_g64",
        N=1,
        H=5,
        W=9,
        groups=64,
        cpg=1,
        KH=7,
        KW=7,
        PAD=3,
        stride=2,
    ),
]


# Shapes for DirectDepthwiseSpatialSpec (groups <= wave_size=64, cpg=kpg=1).
# groups=3 is intentionally not a power-of-two to cover the non-divisor path;
# groups=64 exercises full-wave utilisation; stride=2 validates Ho/Wo output.
# Output-stationary depthwise (DirectDepthwiseTiledSpec): (shape, knobs). Each
# pins one branch: a 31x31 filter, the unrolled filter-row loop, stride-2
# phases on ragged H/W tiles (rolled and unrolled), block_h 1 (no carried
# window), non-"same" padding, and a partial channel tile over two waves.
_TILED_CASES: List[Tuple[_Shape, dict]] = [
    (
        _Shape(
            "dwt_k31_N1H28W28_g64",
            N=1,
            H=28,
            W=28,
            groups=64,
            cpg=1,
            KH=31,
            KW=31,
            PAD=15,
        ),
        dict(block_w=8, block_h=2),
    ),
    (
        _Shape(
            "dwt_k7_N2H14W14_g64", N=2, H=14, W=14, groups=64, cpg=1, KH=7, KW=7, PAD=3
        ),
        dict(block_w=14, block_h=7, unroll_rows=True),
    ),
    (
        _Shape(
            "dwt_k5s2_N2H13W11_g64",
            N=2,
            H=13,
            W=11,
            groups=64,
            cpg=1,
            KH=5,
            KW=5,
            PAD=2,
            stride=2,
        ),
        dict(block_w=4, block_h=4),
    ),
    (
        _Shape(
            "dwt_k5s2u_N2H13W11_g64",
            N=2,
            H=13,
            W=11,
            groups=64,
            cpg=1,
            KH=5,
            KW=5,
            PAD=2,
            stride=2,
        ),
        dict(block_w=4, block_h=4, unroll_rows=True),
    ),
    (
        _Shape("dwt_k3_N2H14W14_g64", N=2, H=14, W=14, groups=64, cpg=1),
        dict(block_w=16, block_h=1),
    ),
    (
        _Shape("dwt_k3p0_N2H14W14_g64", N=2, H=14, W=14, groups=64, cpg=1, PAD=0),
        dict(block_w=4, block_h=2),
    ),
    (
        _Shape(
            "dwt_k13_N2H17W19_g96",
            N=2,
            H=17,
            W=19,
            groups=96,
            cpg=1,
            KH=13,
            KW=13,
            PAD=6,
        ),
        dict(block_w=7, block_h=4, block_waves=2),
    ),
]

_SPATIAL_SHAPES: List[_Shape] = [
    _Shape("sp_dw_N2H14W14_g3", N=2, H=14, W=14, groups=3, cpg=1),
    _Shape("sp_dw_N2H14W14_g64", N=2, H=14, W=14, groups=64, cpg=1),
    _Shape("sp_dw_N2H14W14_g3_s2", N=2, H=14, W=14, groups=3, cpg=1, stride=2),
    # 7x7: the kernel prefetches PF = min(KH, ceil(16 / KW)) = 3 rows, fewer
    # than KH, so the row window rotates within a filter's rows -- the 3x3
    # cases above (PF = KH) never do.
    _Shape("sp_dw_k7_N2H14W14_g9", N=2, H=14, W=14, groups=9, cpg=1, KH=7, KW=7, PAD=3),
    _Shape(
        "sp_dw_k7s2_N2H13W11_g3",
        N=2,
        H=13,
        W=11,
        groups=3,
        cpg=1,
        KH=7,
        KW=7,
        PAD=3,
        stride=2,
    ),
]


# Shapes for DirectDepthwiseColSpec (cpg=kpg=1).  Geometry sweep, fp16 only --
# the dtype axis is swept separately by _COL_DTYPE_SHAPES so a dtype failure
# does not masquerade as a stride or tail failure.
#
# The col kernel decides at *emission* time which (y, r) taps are live, via
# ``(y - r) % stride == 0 and 0 <= (y - r) // stride < block_h``.  A wrong
# formula therefore produces a kernel that is silently missing or
# double-counting rows rather than one that crashes, so the axes below are
# chosen to make each static-pruning decision observable:
#
#   stride       1 / 2 / 3       -- the pruning predicate itself
#   Wo % block_w -- the runtime ``out_q < Wo`` tail guard on the W axis
#   groups % block_ch -- the runtime ``ch < groups`` channel tail
#   Ho % block_h, N > 1 -- the row-tile decode of block_id_z and the
#                          ``out_h < Ho`` tail guard
#   PAD=0, PAD>(KH-1)/2 -- ``n_iters = (block_h-1)*stride + KH`` row coverage
#   KH != KW, KW=31 -- the KW-independence that is the kernel's whole point
#   block_waves=2 -- block_ch=128, the multi-wave channel mapping
_COL_SHAPES: List[_Shape] = [
    # --- stride sweep, exact W tiling ---------------------------------------
    _Shape("col_s1_g64_bw1", N=2, H=14, W=14, groups=64, cpg=1),
    _Shape("col_s2_g64_bw2", N=2, H=28, W=28, groups=64, cpg=1, stride=2, block_w=2),
    _Shape("col_s3_g64_bw1", N=1, H=28, W=28, groups=64, cpg=1, stride=3),
    # --- W tail: Wo=14 is not a multiple of block_w=4 ------------------------
    _Shape("col_s1_wtail_bw4", N=2, H=14, W=14, groups=64, cpg=1, block_w=4),
    _Shape("col_s2_wtail_bw4", N=1, H=28, W=28, groups=64, cpg=1, stride=2, block_w=4),
    # Wo=10, block_w=3 -> tail of 1, with a 5x5 filter at stride 3 so the tap
    # grid is ragged on both axes at once.
    _Shape(
        "col_s3_k5_bw3",
        N=1,
        H=28,
        W=28,
        groups=64,
        cpg=1,
        KH=5,
        KW=5,
        PAD=2,
        stride=3,
        block_w=3,
    ),
    # --- channel tail and non-power-of-two groups ----------------------------
    # 100 % 64 = 36 lanes masked off in the last channel tile.
    _Shape("col_chtail_g100", N=1, H=14, W=14, groups=100, cpg=1, block_w=2),
    # groups < block_ch: a single, mostly-masked tile.
    _Shape("col_g3", N=2, H=14, W=14, groups=3, cpg=1),
    _Shape("col_g12_s2", N=2, H=16, W=16, groups=12, cpg=1, stride=2, block_w=2),
    # --- padding -------------------------------------------------------------
    _Shape("col_pad0", N=1, H=10, W=10, groups=64, cpg=1, PAD=0, block_w=2),
    _Shape(
        "col_pad0_s2", N=1, H=17, W=17, groups=64, cpg=1, PAD=0, stride=2, block_w=2
    ),
    # PAD=2 > (KH-1)/2=1: the padded input is taller than the input, which is
    # what motivated ``n_iters = (Ho-1)*stride + KH``.  Only reachable at
    # stride>1 -- at stride 1 this same overhang makes Ho > H, which the
    # validator rejects outright.
    _Shape("col_pad2_s2", N=1, H=12, W=12, groups=64, cpg=1, PAD=2, stride=2),
    # --- filter geometry -----------------------------------------------------
    _Shape("col_k3x7", N=1, H=16, W=16, groups=64, cpg=1, KW=7, block_w=2),
    _Shape("col_k1x1", N=2, H=12, W=12, groups=64, cpg=1, KH=1, KW=1, PAD=0, block_w=4),
    _Shape(
        "col_k1x1_s2",
        N=1,
        H=12,
        W=12,
        groups=64,
        cpg=1,
        KH=1,
        KW=1,
        PAD=0,
        stride=2,
        block_w=4,
    ),
    # KW=31: the regime where the preload variant cannot be built at all.
    _Shape("col_k3x31", N=1, H=16, W=40, groups=64, cpg=1, KW=31),
    _Shape("col_k3x31_s2", N=1, H=32, W=40, groups=64, cpg=1, KW=31, stride=2),
    # --- row tiles: several per image, the last one partial -----------------
    # Ho=14, block_h=4: tiles of 4,4,4,2 per image, two images along z.
    _Shape("col_htail_bh4", N=2, H=14, W=14, groups=64, cpg=1, block_h=4),
    _Shape("col_bh1", N=2, H=10, W=10, groups=64, cpg=1, block_h=1, block_w=2),
    # Ho=14 at stride 2: a tile's receptive field starts mid-image.
    _Shape(
        "col_s2_htail_bh3",
        N=2,
        H=28,
        W=28,
        groups=64,
        cpg=1,
        stride=2,
        block_h=3,
        block_w=2,
    ),
    # Ho=10, ragged on rows, columns and taps at once.
    _Shape(
        "col_s3_k5_bh4",
        N=1,
        H=28,
        W=28,
        groups=64,
        cpg=1,
        KH=5,
        KW=5,
        PAD=2,
        stride=3,
        block_h=4,
        block_w=3,
    ),
    # Ho=7: the padding overhang lands on the first and last tile.
    _Shape(
        "col_pad2_s2_bh2", N=1, H=12, W=12, groups=64, cpg=1, PAD=2, stride=2, block_h=2
    ),
    _Shape("col_k3x31_bh5", N=1, H=16, W=40, groups=64, cpg=1, KW=31, block_h=5),
    # --- multi-wave blocks (block_ch = 128) ----------------------------------
    _Shape(
        "col_2wv_g128", N=1, H=14, W=14, groups=128, cpg=1, block_w=2, block_waves=2
    ),
    _Shape(
        "col_2wv_chtail_g200",
        N=1,
        H=14,
        W=14,
        groups=200,
        cpg=1,
        stride=2,
        block_w=2,
        block_waves=2,
    ),
]


# Dtype sweep.  The first two are the same geometry at both dtypes, so a failure
# isolates to the element type; the rest pair a dtype with a stride/tail/wide-KW
# case, since the load/store width and the tap pruning are independent code paths
# that both have to be right at once.
_COL_DTYPE_SHAPES: List[_Shape] = [
    _Shape("coldt_fp16", N=2, H=14, W=14, groups=64, cpg=1, block_w=2, dtype="fp16"),
    _Shape("coldt_bf16", N=2, H=14, W=14, groups=64, cpg=1, block_w=2, dtype="bf16"),
    _Shape(
        "coldt_bf16_s2_k5",
        N=1,
        H=28,
        W=28,
        groups=64,
        cpg=1,
        KH=5,
        KW=5,
        PAD=2,
        stride=2,
        block_w=3,
        dtype="bf16",
    ),
    _Shape(
        "coldt_bf16_s3_chtail",
        N=1,
        H=28,
        W=28,
        groups=100,
        cpg=1,
        stride=3,
        block_w=4,
        dtype="bf16",
    ),
    _Shape("coldt_fp16_k3x31", N=1, H=16, W=40, groups=64, cpg=1, KW=31, dtype="fp16"),
    _Shape(
        "coldt_bf16_htail_bh4",
        N=2,
        H=14,
        W=14,
        groups=100,
        cpg=1,
        block_h=4,
        block_w=4,
        dtype="bf16",
    ),
]


# One binary, many images: the col kernel is AOT, so a kernel built for one
# problem must give the right answer on every problem with the same filter,
# PAD and stride. Each entry is (caps, shapes); the first shape is the one
# the binary is built for, and every shape is then launched on that binary.
_COL_AOT_SHAPES: List[Tuple[dict, List[Tuple[int, int, int, int]]]] = [
    # (N, H, W, groups): batch, both spatial extents, the group count and with
    # it the channel tail all vary.
    (
        dict(KH=3, KW=3, PAD=1, stride=1),
        [(1, 16, 16, 64), (2, 14, 14, 64), (3, 29, 37, 100), (1, 7, 64, 3)],
    ),
    (
        dict(KH=5, KW=5, PAD=2, stride=2),
        [(1, 32, 32, 64), (2, 28, 20, 128), (1, 17, 45, 70)],
    ),
]


# fp16 and bf16 round the *output* to 10/7 mantissa bits, so the meaningful
# bound is on the ref_scale-normalised max-abs error.  bf16 carries 3 fewer
# mantissa bits than fp16, so it gets the same looser bound the rest of this
# suite already uses for it (_TOL_BF16) rather than borrowing fp16's.
_COL_TOL = {"fp16": _TOL, "bf16": _TOL_BF16}


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
    out_nchw = F.conv2d(
        A_nchw,
        B_nchw,
        padding=p.PAD,
        stride=p.stride,
        dilation=(p.dil_h, p.dil_w),
        groups=p.groups,
    )
    return out_nchw.permute(0, 2, 3, 1).contiguous().cuda()


def _run_grouped_one(arch: str, shape: _Shape, dtype: str = "fp16") -> Tuple[bool, str]:
    """Build, compile, launch, and verify one grouped direct-conv kernel.

    Uses the generic ``DirectConvSpec`` dispatcher which selects the right
    cpg-specialised kernel (4c / 8c / 16c / 32c) automatically.

    Returns ``(passed, reason)``.  ``reason`` starts with ``"skip "`` when
    the combination is architecturally unsupported.
    """
    import torch

    from rocke import compile_kernel
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_direct_grouped import (
        direct_launch_geometry,
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
        dtype=dtype,
    )

    spec = DirectConvSpec(
        problem=p,
        name=f"test_direct_{shape.id}",
    )

    ok, reason = is_valid_spec(spec, arch=arch)
    if not ok:
        return False, f"skip {reason}"

    try:
        kernel = build_direct_conv(spec, arch=arch)
    except ValueError as e:
        return False, f"skip build failed: {e}"

    try:
        artifact = compile_kernel(kernel, arch=arch)
    except Exception as e:
        return False, f"compile failed: {e}"

    torch.manual_seed(0)
    total_c = shape.groups * shape.cpg
    total_k = shape.groups * shape.cpg
    _td = torch.bfloat16 if dtype == "bf16" else torch.float16
    A_t = torch.empty(p.N, p.H, p.W, total_c, dtype=_td).uniform_(-1.0, 1.0)
    B_t = torch.empty(total_k, p.KH, p.KW, shape.cpg, dtype=_td).uniform_(-1.0, 1.0)
    D_t = torch.empty(p.N, p.Ho, p.Wo, total_k, dtype=_td)

    ref = _conv_ref_grouped(A_t, B_t, p)

    rt = Runtime()
    A_dev = rt.alloc(A_t.nbytes)
    B_dev = rt.alloc(B_t.nbytes)
    D_dev = rt.alloc(D_t.nbytes)
    rt.memcpy_h2d(A_dev, _u8(A_t), A_t.nbytes)
    rt.memcpy_h2d(B_dev, _u8(B_t), B_t.nbytes)
    rt.memset(D_dev, 0, D_t.nbytes)

    sig = conv_direct_args_signature(dtype)
    _direct_args = ConvArgs.from_problem(p)
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

    grid, block = direct_launch_geometry(spec)

    values = _direct_args.to_launch_values(
        int(A_dev),
        int(B_dev),
        int(D_dev),
        A_t.nbytes,
        B_t.nbytes,
        D_t.nbytes,
    )
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
    tol = _TOL_BF16 if dtype == "bf16" else _TOL
    passed = rel_err < tol
    if not passed:
        return False, f"rel_err={rel_err:.3e} > tol={tol:.1e}"
    print(
        f"  PASS  {shape.id}  {arch}  {dtype}  rel_err={rel_err:.2e}",
        flush=True,
    )
    return True, ""


def _run_depthwise_one(
    arch: str,
    shape: _Shape,
    dtype: str = "fp16",
    spatial: bool = False,
    tiled: "dict | None" = None,
    wave_size: "int | None" = None,
) -> Tuple[bool, str]:
    """Build, compile, launch, and verify one depthwise forward kernel.

    ``DirectDepthwiseSpec`` (one channel per lane) by default,
    ``DirectDepthwiseSpatialSpec`` (lanes split over channels and output
    columns, groups <= wave_size) with ``spatial``, and the output-stationary
    ``DirectDepthwiseTiledSpec`` with the knobs ``tiled``. cpg = kpg = 1.
    ``wave_size`` defaults to the target's.

    Returns ``(passed, reason)``.  ``reason`` starts with ``"skip "`` when a
    spatial shape is architecturally unsupported.
    """
    import torch

    from rocke import compile_kernel
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_direct_grouped import (
        direct_launch_geometry,
        DirectConvProblem,
        DirectDepthwiseSpatialSpec,
        DirectDepthwiseSpec,
        DirectDepthwiseTiledSpec,
        build_direct_depthwise,
        build_direct_depthwise_spatial,
        build_direct_depthwise_tiled,
        is_valid_depthwise_spatial_spec,
        is_valid_depthwise_spec,
        is_valid_depthwise_tiled_spec,
    )
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    assert shape.cpg == 1, "depthwise path requires cpg=1"
    if wave_size is None:
        wave_size = _wave_size(arch)

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
        dtype=dtype,
    )

    if tiled is not None:
        spec = DirectDepthwiseTiledSpec(
            problem=p,
            name=f"test_direct_dwt_{shape.id}",
            wave_size=wave_size,
            **tiled,
        )
        ok, reason = is_valid_depthwise_tiled_spec(spec, arch=arch)
        if not ok:
            return False, f"invalid spec (cases should be pre-validated): {reason}"
        build = build_direct_depthwise_tiled
    elif spatial:
        spec = DirectDepthwiseSpatialSpec(
            problem=p, name=f"test_direct_sp_dw_{shape.id}", wave_size=wave_size
        )
        ok, reason = is_valid_depthwise_spatial_spec(spec, arch=arch)
        if not ok:
            return False, f"skip {reason}"
        build = build_direct_depthwise_spatial
    else:
        spec = DirectDepthwiseSpec(
            problem=p, name=f"test_direct_dw_{shape.id}", wave_size=wave_size
        )
        ok, reason = is_valid_depthwise_spec(spec, arch=arch)
        if not ok:
            return False, f"invalid spec (shapes should be pre-validated): {reason}"
        build = build_direct_depthwise

    try:
        kernel = build(spec, arch=arch)
    except ValueError as e:
        return False, f"build failed: {e}"

    try:
        artifact = compile_kernel(kernel, arch=arch)
    except Exception as e:
        return False, f"compile failed: {e}"

    torch.manual_seed(0)
    total_c = shape.groups
    total_k = shape.groups
    _td = torch.bfloat16 if dtype == "bf16" else torch.float16
    A_t = torch.empty(p.N, p.H, p.W, total_c, dtype=_td).uniform_(-1.0, 1.0)
    B_t = torch.empty(total_k, p.KH, p.KW, 1, dtype=_td).uniform_(-1.0, 1.0)
    D_t = torch.empty(p.N, p.Ho, p.Wo, total_k, dtype=_td)

    ref = _conv_ref_grouped(A_t, B_t, p)

    rt = Runtime()
    A_dev = rt.alloc(A_t.nbytes)
    B_dev = rt.alloc(B_t.nbytes)
    D_dev = rt.alloc(D_t.nbytes)
    rt.memcpy_h2d(A_dev, _u8(A_t), A_t.nbytes)
    rt.memcpy_h2d(B_dev, _u8(B_t), B_t.nbytes)
    rt.memset(D_dev, 0, D_t.nbytes)

    sig = conv_direct_args_signature(dtype)
    _direct_args = ConvArgs.from_problem(p)
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

    grid, block = direct_launch_geometry(spec)

    values = _direct_args.to_launch_values(
        int(A_dev),
        int(B_dev),
        int(D_dev),
        A_t.nbytes,
        B_t.nbytes,
        D_t.nbytes,
    )
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
    if tiled is not None:
        tol = _TILED_TOL[dtype]
    else:
        tol = _TOL_BF16 if dtype == "bf16" else _TOL
    if not rel_err < tol:
        return False, f"rel_err={rel_err:.3e} > tol={tol:.1e}"
    print(
        f"  PASS  {shape.id}  {arch}  {dtype}  rel_err={rel_err:.2e}",
        flush=True,
    )
    return True, ""


def _torch_dtype(name: str):
    import torch

    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _run_depthwise_device(
    artifact,
    p,
    *,
    dtype: str,
    grid: Tuple[int, int, int],
    block: Tuple[int, int, int],
    tol: float,
    label: str,
    arch: str,
) -> Tuple[bool, str]:
    """Allocate, launch and verify one depthwise (``cpg = kpg = 1``) kernel.

    The tail every depthwise runner shares: NHWC ``A``, ``(K, KH, KW, 1)`` ``B``,
    ``(N, Ho, Wo, K)`` ``D``, torch reference in f32, and the
    ``ref_scale``-normalised max-abs comparison.

    Used by the column-streamed runner only.  The three older runners predate it
    and are left as they are; converting them would be a rewrite of passing code,
    not part of what this covers.

    Returns ``(passed, reason)``.
    """
    import torch

    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    td = _torch_dtype(dtype)
    torch.manual_seed(0)
    total_c = p.groups
    total_k = p.groups
    A_t = torch.empty(p.N, p.H, p.W, total_c, dtype=td).uniform_(-1.0, 1.0)
    B_t = torch.empty(total_k, p.KH, p.KW, 1, dtype=td).uniform_(-1.0, 1.0)
    D_t = torch.empty(p.N, p.Ho, p.Wo, total_k, dtype=td)

    ref = _conv_ref_grouped(A_t, B_t, p)

    rt = Runtime()
    A_dev = rt.alloc(A_t.nbytes)
    B_dev = rt.alloc(B_t.nbytes)
    D_dev = rt.alloc(D_t.nbytes)
    rt.memcpy_h2d(A_dev, _u8(A_t), A_t.nbytes)
    rt.memcpy_h2d(B_dev, _u8(B_t), B_t.nbytes)
    # Zero D rather than leaving it uninitialised: the tail guards are supposed
    # to leave the out-of-range lanes alone, and a garbage-filled D is the only
    # way a guard that writes where it must not shows up as a mismatch.
    rt.memset(D_dev, 0, D_t.nbytes)

    try:
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=conv_direct_args_signature(dtype),
        )
    except HipError as e:
        rt.free(A_dev)
        rt.free(B_dev)
        rt.free(D_dev)
        return False, f"kernel load failed: {e}"

    # Direct conv is AOT: the whole shape travels as kernargs.
    values = ConvArgs.from_problem(p).to_launch_values(
        int(A_dev),
        int(B_dev),
        int(D_dev),
        A_t.nbytes,
        B_t.nbytes,
        D_t.nbytes,
    )
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
    if not rel_err < tol:
        return False, f"rel_err={rel_err:.3e} > tol={tol:.1e}"
    print(f"  PASS  {label}  {arch}  rel_err={rel_err:.2e}", flush=True)
    return True, ""


def _run_depthwise_col_one(arch: str, shape: _Shape) -> Tuple[bool, str]:
    """Build, compile, launch, and verify one column-streamed depthwise kernel.

    Uses ``DirectDepthwiseColSpec`` (cpg = kpg = 1), which supports stride >= 1
    and fp16/bf16.

    Modelled on ``_run_depthwise_spatial_one``, not on ``_run_depthwise_one``:
    the latter allocates ``D`` as ``(N, H, W, K)`` and grids on ``ceil(W/block_w)``,
    which only works because it is pinned to stride 1.

    Returns ``(passed, reason)``.
    """
    from rocke import compile_kernel
    from kernels.common.conv_direct_grouped import (
        DirectConvProblem,
        DirectDepthwiseColSpec,
        build_direct_depthwise_col,
        direct_launch_geometry,
        is_valid_depthwise_col_spec,
    )

    assert shape.cpg == 1, "column-streamed depthwise path requires cpg=1"

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

    spec = DirectDepthwiseColSpec(
        problem=p,
        name=f"test_direct_dw_col_{shape.id}",
        block_h=shape.block_h,
        block_w=shape.block_w,
        block_waves=shape.block_waves,
        dtype=shape.dtype,
        wave_size=_wave_size(arch),
    )

    ok, reason = is_valid_depthwise_col_spec(spec, arch=arch)
    if not ok:
        # Every _COL_SHAPES entry is meant to be a supported configuration, so a
        # rejection here is a bug in the shape table or the validator -- not a
        # reason to quietly skip.
        return False, f"invalid spec (shapes should be pre-validated): {reason}"

    try:
        kernel = build_direct_depthwise_col(spec, arch=arch)
    except ValueError as e:
        return False, f"build failed (shapes should be pre-validated): {e}"

    try:
        artifact = compile_kernel(kernel, arch=arch)
    except Exception as e:
        return False, f"compile failed: {e}"

    grid, block = direct_launch_geometry(spec)
    return _run_depthwise_device(
        artifact,
        p,
        dtype=shape.dtype,
        grid=grid,
        block=block,
        tol=_COL_TOL[shape.dtype],
        label=shape.id,
        arch=arch,
    )


def _run_depthwise_col_aot(
    arch: str, caps: dict, shapes: List[Tuple[int, int, int, int]]
) -> List[Tuple[str, bool, str]]:
    """Build one col kernel for ``shapes[0]`` and launch it on every shape.

    Only the launch grid and the kernargs follow the runtime problem; the
    binary is the one built for the first shape. Returns one
    ``(label, passed, reason)`` per shape.
    """
    from dataclasses import replace

    from rocke import compile_kernel
    from kernels.common.conv_direct_grouped import (
        DirectConvProblem,
        DirectDepthwiseColSpec,
        build_direct_depthwise_col,
        direct_launch_geometry,
        is_valid_depthwise_col_spec,
    )

    def problem(N, H, W, groups):
        return DirectConvProblem(N=N, H=H, W=W, groups=groups, cpg=1, kpg=1, **caps)

    spec = DirectDepthwiseColSpec(
        problem=problem(*shapes[0]),
        name="test_direct_dw_col_aot",
        block_h=4,
        block_w=2,
        wave_size=_wave_size(arch),
    )
    artifact = compile_kernel(build_direct_depthwise_col(spec, arch=arch), arch=arch)
    out = []
    for shape in shapes:
        run_spec = replace(spec, problem=problem(*shape))
        label = (
            f"aot_r{caps['KH']}x{caps['KW']}_p{caps['PAD']}_s{caps['stride']}_"
            f"{run_spec.problem.short()}"
        )
        ok, why = is_valid_depthwise_col_spec(run_spec, arch=arch)
        if not ok:
            out.append((label, False, f"shape table entry is invalid: {why}"))
            continue
        grid, block = direct_launch_geometry(run_spec)
        passed, reason = _run_depthwise_device(
            artifact,
            run_spec.problem,
            dtype=spec.dtype,
            grid=grid,
            block=block,
            tol=_COL_TOL[spec.dtype],
            label=label,
            arch=arch,
        )
        out.append((label, passed, reason))
    return out


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

    def _run_depthwise_spatial(self, shape: _Shape) -> None:
        passed, reason = _run_depthwise_one(GPU_ARCH, shape, spatial=True)
        if reason.startswith("skip"):
            self.skipTest(reason)
        self.assertTrue(
            passed,
            f"FAIL {shape.id} on {GPU_ARCH}: {reason}",
        )

    def test_depthwise_spatial(self):
        for s in _SPATIAL_SHAPES:
            with self.subTest(shape=s.id):
                self._run_depthwise_spatial(s)

    def test_depthwise_tiled(self):
        for s, knobs in _TILED_CASES:
            with self.subTest(shape=s.id):
                passed, reason = _run_depthwise_one(GPU_ARCH, s, tiled=knobs)
                self.assertTrue(passed, f"FAIL {s.id} on {GPU_ARCH}: {reason}")

    def _assert_ran(self, ran: int, what: str) -> None:
        """A sweep whose every subTest skipped still reports *passed*.

        ``_run_depthwise_col_one`` turns a rejected spec or a failed build into
        a failure rather than a skip on purpose, so no skip is reachable inside
        the col path today.  This backstop is what keeps that property true if
        a skip is ever added, and it catches the other way a sweep can report a
        green without testing anything: a shape table that has been emptied or
        filtered down to nothing.
        """
        self.assertGreater(
            ran,
            0,
            f"{what}: no config ran on {GPU_ARCH}, so the column-streamed "
            f"depthwise path was never executed -- this is a false green, not "
            f"a pass. Check the shape table and is_valid_depthwise_col_spec.",
        )

    def _run_depthwise_col(self, shape: _Shape) -> None:
        passed, reason = _run_depthwise_col_one(GPU_ARCH, shape)
        if reason.startswith("skip"):
            self.skipTest(reason)
        self.assertTrue(
            passed,
            f"FAIL {shape.id} on {GPU_ARCH}: {reason}",
        )

    def test_depthwise_col(self):
        """Geometry sweep at fp16: stride, W tail, channel tail, padding, filter."""
        ran = 0
        for s in _COL_SHAPES:
            with self.subTest(shape=s.id):
                self._run_depthwise_col(s)
                ran += 1
        self._assert_ran(ran, "test_depthwise_col")

    def test_depthwise_col_dtypes(self):
        """Element-type sweep, separate from the geometry sweep on purpose.

        Split from ``test_depthwise_col`` so the report distinguishes "bf16
        stores are wrong" from "stride-3 tap pruning is wrong" instead of
        reporting one failing blob.
        """
        ran = 0
        for s in _COL_DTYPE_SHAPES:
            with self.subTest(shape=s.id, dtype=s.dtype):
                self._run_depthwise_col(s)
                ran += 1
        self._assert_ran(ran, "test_depthwise_col_dtypes")

    def test_depthwise_col_aot_reuse(self):
        """One compiled binary serves every image with its filter geometry."""
        ran = 0
        for caps, shapes in _COL_AOT_SHAPES:
            for label, passed, reason in _run_depthwise_col_aot(GPU_ARCH, caps, shapes):
                with self.subTest(shape=label):
                    self.assertTrue(passed, f"FAIL {label} on {GPU_ARCH}: {reason}")
                    ran += 1
        self._assert_ran(ran, "test_depthwise_col_aot_reuse")


@unittest.skipUnless(
    _HAS_TORCH and _IS_WAVE32_DW, "needs torch and a wave32 (gfx1250) GPU"
)
class TestDirectDepthwiseFwdWave32(unittest.TestCase):
    """The depthwise forward sweeps above, at wave_size 32 on gfx1250.

    The runners build every spec at the target's wave size, so these are the
    same cases: one lane per channel (preloading, tiled, column-streamed) or a
    lane split over channels and columns (spatial, groups below the wave).
    """

    _run_depthwise = TestDirectConvCorrectness._run_depthwise
    _run_depthwise_spatial = TestDirectConvCorrectness._run_depthwise_spatial
    _run_depthwise_col = TestDirectConvCorrectness._run_depthwise_col
    _assert_ran = TestDirectConvCorrectness._assert_ran

    test_depthwise = TestDirectConvCorrectness.test_depthwise
    test_depthwise_spatial = TestDirectConvCorrectness.test_depthwise_spatial
    test_depthwise_tiled = TestDirectConvCorrectness.test_depthwise_tiled
    test_depthwise_col = TestDirectConvCorrectness.test_depthwise_col
    test_depthwise_col_dtypes = TestDirectConvCorrectness.test_depthwise_col_dtypes
    test_depthwise_col_aot_reuse = (
        TestDirectConvCorrectness.test_depthwise_col_aot_reuse
    )

    def test_depthwise_bf16(self):
        for s in _SHAPES:
            if s.cpg == 1:
                with self.subTest(shape=s.id):
                    passed, reason = _run_depthwise_one(GPU_ARCH, s, dtype="bf16")
                    self.assertTrue(passed, f"FAIL {s.id} bf16 on {GPU_ARCH}: {reason}")

    def test_logical_wave64(self):
        # The preloading and spatial kernels use no cross-lane operation, so a
        # wave_size-64 spec is a logical wave the wave32 target runs as two
        # hardware waves. groups=48 is spatial-only at wave64 (it needs
        # groups < wave_size).
        shapes = [s for s in _SHAPES if s.cpg == 1] + [
            _Shape("sp_g48", N=2, H=14, W=14, groups=48, cpg=1),
            _Shape("sp_g48_s2", N=2, H=15, W=17, groups=48, cpg=1, stride=2),
        ]
        ran = 0
        for s in shapes:
            for spatial in (False, True) if s.groups < 64 else (False,):
                with self.subTest(shape=s.id, spatial=spatial):
                    passed, reason = _run_depthwise_one(
                        GPU_ARCH, s, spatial=spatial, wave_size=64
                    )
                    self.assertTrue(passed, f"FAIL {s.id} wave64: {reason}")
                    ran += 1
        self.assertGreater(ran, 0)


# ---------------------------------------------------------------------------
# Dgrad shapes
# ---------------------------------------------------------------------------

_DGRAD_SHAPES: List[_Shape] = [
    _Shape("dg_16c_N2H8W8_g8", N=2, H=8, W=8, groups=8, cpg=16),
    _Shape("dg_32c_N2H8W8_g8", N=2, H=8, W=8, groups=8, cpg=32),
    # Asymmetric grouped: cpg != kpg — exercises the independent cpg/kpg path.
    # block_groups=4 to satisfy groups % block_groups == 0 with groups=4.
    _Shape(
        "dg_asym_cpg16_kpg32_g4",
        N=2,
        H=8,
        W=8,
        groups=4,
        cpg=16,
        kpg=32,
        block_groups=4,
    ),
    # Grouped stride-2: non-unit stride grouped dgrad.
    _Shape("dg_16c_N2H8W8_g8_s2", N=2, H=8, W=8, groups=8, cpg=16, stride=2),
    # Padding other than "same": the dgrad kernels bound their rows by p_Ho,
    # so the AOT cache offers them for every PAD in [0, KH-1].
    _Shape("dg_8c_N2H9W9_g8_p0_s2", N=2, H=9, W=9, groups=8, cpg=8, PAD=0, stride=2),
    _Shape("dg_8c_N2H8W8_g8_p2", N=2, H=8, W=8, groups=8, cpg=8, PAD=2),
]

# Depthwise dgrad shapes (cpg=kpg=1).  Stride-2 exercises the divisibility
# checks and the ho/wo output-size path.
_DW_DGRAD_SHAPES: List[_Shape] = [
    _Shape("dw_dgrad_s1_N2H14W14_g64", N=2, H=14, W=14, groups=64, cpg=1, stride=1),
    _Shape("dw_dgrad_s2_N2H14W14_g64", N=2, H=14, W=14, groups=64, cpg=1, stride=2),
    # Padding other than "same" and a larger filter (see _DGRAD_SHAPES).
    _Shape("dw_dgrad_k3p0_N2H14W14_g64", N=2, H=14, W=14, groups=64, cpg=1, PAD=0),
    _Shape(
        "dw_dgrad_k5p4_s2_N2H14W14_g64",
        N=2,
        H=14,
        W=14,
        groups=64,
        cpg=1,
        KH=5,
        KW=5,
        PAD=4,
        stride=2,
    ),
]

# Ho-streaming depthwise dgrad (DirectDepthwiseDgradStreamSpec). Each loop
# iteration consumes KH dY rows with the next KH prefetched, and a dX row is
# flushed from its circular slot once its last (ho, r) pair is in, so the
# cases cover: stride 1 and 2, a filter taller than the remaining dY rows
# (7x7 on Ho=7), a height that is not a multiple of KH, non-"same" padding,
# and a multi-wave block.
_DW_DGRAD_STREAM_SHAPES: List[_Shape] = [
    _Shape("dw_dgrad_stream_s1_N2H14W14_g64", N=2, H=14, W=14, groups=64, cpg=1),
    _Shape(
        "dw_dgrad_stream_s2_N2H14W14_g64", N=2, H=14, W=14, groups=64, cpg=1, stride=2
    ),
    _Shape(
        "dw_dgrad_stream_k7s2_N2H13W11_g64",
        N=2,
        H=13,
        W=11,
        groups=64,
        cpg=1,
        KH=7,
        KW=7,
        PAD=3,
        stride=2,
    ),
    _Shape(
        "dw_dgrad_stream_k5p4_s2_N2H14W14_g64",
        N=2,
        H=14,
        W=14,
        groups=64,
        cpg=1,
        KH=5,
        KW=5,
        PAD=4,
        stride=2,
    ),
    _Shape("dw_dgrad_stream_k3p0_N2H9W9_g96", N=2, H=9, W=9, groups=96, cpg=1, PAD=0),
]


def _run_dgrad_one(arch: str, shape: _Shape, dtype: str = "fp16") -> Tuple[bool, str]:
    """Build, compile, launch, and verify the direct dgrad kernel.

    Returns ``(passed, reason)``.
    """
    import torch

    from rocke import compile_kernel
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_direct_grouped import (
        direct_launch_geometry,
        DirectConvDgradSpec,
        DirectConvProblem,
        build_direct_conv_dgrad,
        is_valid_dgrad_spec,
    )
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    kpg = shape.kpg if shape.kpg > 0 else shape.cpg
    p = DirectConvProblem(
        N=shape.N,
        H=shape.H,
        W=shape.W,
        groups=shape.groups,
        cpg=shape.cpg,
        kpg=kpg,
        KH=shape.KH,
        KW=shape.KW,
        PAD=shape.PAD,
        stride=shape.stride,
        dtype=dtype,
    )
    spec_kwargs = {"problem": p, "name": f"test_dgrad_{shape.id}"}
    if shape.block_groups > 0:
        spec_kwargs["block_groups"] = shape.block_groups
    spec = DirectConvDgradSpec(**spec_kwargs)

    ok, reason = is_valid_dgrad_spec(spec, arch=arch)
    if not ok:
        return False, f"skip invalid spec: {reason}"

    try:
        kernel = build_direct_conv_dgrad(spec, arch=arch)
    except ValueError as e:
        return False, f"build failed: {e}"

    try:
        artifact = compile_kernel(kernel, arch=arch)
    except Exception as e:
        return False, f"compile failed: {e}"

    torch.manual_seed(42)
    total_c = shape.groups * shape.cpg
    total_k = shape.groups * kpg
    _td = torch.bfloat16 if dtype == "bf16" else torch.float16

    # dY: output gradient [N, Ho, Wo, K]
    dY = torch.empty(p.N, p.Ho, p.Wo, total_k, dtype=_td).uniform_(-0.5, 0.5)
    # W:  weights         [K, KH, KW, cpg]
    W = torch.empty(total_k, p.KH, p.KW, shape.cpg, dtype=_td).uniform_(-0.5, 0.5)
    dX = torch.zeros(p.N, p.H, p.W, total_c, dtype=_td)

    # Reference: dX = conv_transpose2d(dY, W)
    # output_padding recovers the exact input H, W (matters when stride > 1).
    dY_nchw = dY.permute(0, 3, 1, 2).float()
    W_nchw = W.permute(0, 3, 1, 2).float()  # [K, cpg, KH, KW]
    h_base = (p.Ho - 1) * p.stride - 2 * p.PAD + p.KH
    w_base = (p.Wo - 1) * p.stride - 2 * p.PAD + p.KW
    ref_nchw = torch.nn.functional.conv_transpose2d(
        dY_nchw,
        W_nchw,
        padding=p.PAD,
        stride=p.stride,
        groups=p.groups,
        output_padding=(p.H - h_base, p.W - w_base),
    )
    ref = ref_nchw.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

    rt = Runtime()
    dY_dev = rt.alloc(dY.nbytes)
    W_dev = rt.alloc(W.nbytes)
    dX_dev = rt.alloc(dX.nbytes)
    rt.memcpy_h2d(dY_dev, _u8(dY), dY.nbytes)
    rt.memcpy_h2d(W_dev, _u8(W), W.nbytes)
    rt.memset(dX_dev, 0, dX.nbytes)

    sig = conv_direct_args_signature(dtype, direction="dgrad")
    _direct_args = ConvArgs.from_problem(p, direction="dgrad")
    try:
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=sig,
        )
    except HipError as e:
        rt.free(dY_dev)
        rt.free(W_dev)
        rt.free(dX_dev)
        return False, f"kernel load failed: {e}"

    grid, block = direct_launch_geometry(spec)

    values = _direct_args.to_launch_values(
        int(dY_dev),
        int(W_dev),
        int(dX_dev),
        dY.nbytes,
        W.nbytes,
        dX.nbytes,
    )
    launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))

    dX_cpu = torch.empty_like(dX)
    rt.memcpy_d2h(_u8(dX_cpu), dX_dev, dX.nbytes)
    rt.free(dY_dev)
    rt.free(W_dev)
    rt.free(dX_dev)
    synchronize_and_release(0)

    out_f32 = dX_cpu.float()
    ref_f32 = ref.float().cpu()
    abs_diff = (out_f32 - ref_f32).abs()
    ref_scale = ref_f32.abs().max().clamp(min=1.0)
    rel_err = float(abs_diff.max() / ref_scale)
    tol = _TOL_BF16 if dtype == "bf16" else _TOL
    passed = rel_err < tol
    if not passed:
        return False, f"rel_err={rel_err:.3e} > tol={tol:.1e}"
    print(f"  PASS  {shape.id}  {arch}  {dtype}  rel_err={rel_err:.2e}", flush=True)
    return True, ""


def _run_dw_dgrad_one(
    arch: str,
    shape: _Shape,
    dtype: str = "fp16",
    streaming: bool = False,
    block_waves: int = 1,
) -> Tuple[bool, str]:
    """Build, compile, launch, and verify a direct depthwise dgrad kernel.

    ``DirectDepthwiseDgradSpec`` by default; with ``streaming`` the ho-streaming
    ``DirectDepthwiseDgradStreamSpec`` the AOT cache and the benchmark sweep
    build, whose prefetched dY row window and circular dX accumulator slots
    are only exercised by running it.
    """
    import torch

    from rocke import compile_kernel
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_direct_grouped import (
        direct_launch_geometry,
        DirectConvProblem,
        DirectDepthwiseDgradSpec,
        DirectDepthwiseDgradStreamSpec,
        build_direct_depthwise_dgrad,
        build_direct_depthwise_dgrad_streaming,
    )
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

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
        dtype=dtype,
    )
    if streaming:
        spec = DirectDepthwiseDgradStreamSpec(
            problem=p,
            name=f"test_dw_dgrad_stream_{shape.id}",
            block_waves=block_waves,
        )
        build = build_direct_depthwise_dgrad_streaming
    else:
        spec = DirectDepthwiseDgradSpec(
            problem=p, name=f"test_dw_dgrad_{shape.id}", block_waves=block_waves
        )
        build = build_direct_depthwise_dgrad

    try:
        kernel = build(spec, arch=arch)
    except ValueError as e:
        return False, f"build failed: {e}"

    try:
        artifact = compile_kernel(kernel, arch=arch)
    except Exception as e:
        return False, f"compile failed: {e}"

    torch.manual_seed(42)
    total_c = shape.groups  # cpg=kpg=1
    _td = torch.bfloat16 if dtype == "bf16" else torch.float16

    dY = torch.empty(p.N, p.Ho, p.Wo, total_c, dtype=_td).uniform_(-0.5, 0.5)
    W = torch.empty(total_c, p.KH, p.KW, 1, dtype=_td).uniform_(-0.5, 0.5)
    dX = torch.zeros(p.N, p.H, p.W, total_c, dtype=_td)

    dY_nchw = dY.permute(0, 3, 1, 2).float()
    W_nchw = W.permute(0, 3, 1, 2).float()
    h_base = (p.Ho - 1) * p.stride - 2 * p.PAD + p.KH
    w_base = (p.Wo - 1) * p.stride - 2 * p.PAD + p.KW
    ref_nchw = torch.nn.functional.conv_transpose2d(
        dY_nchw,
        W_nchw,
        padding=p.PAD,
        stride=p.stride,
        groups=p.groups,
        output_padding=(p.H - h_base, p.W - w_base),
    )
    ref = ref_nchw.permute(0, 2, 3, 1).contiguous()

    rt = Runtime()
    dY_dev = rt.alloc(dY.nbytes)
    W_dev = rt.alloc(W.nbytes)
    dX_dev = rt.alloc(dX.nbytes)
    rt.memcpy_h2d(dY_dev, _u8(dY), dY.nbytes)
    rt.memcpy_h2d(W_dev, _u8(W), W.nbytes)
    rt.memset(dX_dev, 0, dX.nbytes)

    sig = conv_direct_args_signature(dtype, direction="dgrad")
    _direct_args = ConvArgs.from_problem(p, direction="dgrad")
    try:
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=sig,
        )
    except HipError as e:
        rt.free(dY_dev)
        rt.free(W_dev)
        rt.free(dX_dev)
        return False, f"kernel load failed: {e}"

    grid, block = direct_launch_geometry(spec)

    values = _direct_args.to_launch_values(
        int(dY_dev),
        int(W_dev),
        int(dX_dev),
        dY.nbytes,
        W.nbytes,
        dX.nbytes,
    )
    launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))

    out_host = torch.empty_like(dX)
    rt.memcpy_d2h(_u8(out_host), dX_dev, dX.nbytes)
    rt.free(dY_dev)
    rt.free(W_dev)
    rt.free(dX_dev)
    synchronize_and_release(0)

    ref_f32 = ref.float().cpu()
    out_f32 = out_host.float()
    abs_diff = (out_f32 - ref_f32).abs()
    ref_scale = ref_f32.abs().max().clamp(min=1.0)
    rel_err = float(abs_diff.max() / ref_scale)
    tol = _TOL_BF16 if dtype == "bf16" else _TOL
    if not (rel_err < tol):
        return False, f"rel_err={rel_err:.3e} > tol={tol:.1e}"
    print(f"  PASS  {shape.id}  {arch}  {dtype}  rel_err={rel_err:.2e}", flush=True)
    return True, ""


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestDirectConvDgradWgradCorrectness(unittest.TestCase):
    """Correctness tests for direct conv backward pass (dgrad only)."""

    def _run_dgrad(self, shape: _Shape) -> None:
        passed, reason = _run_dgrad_one(GPU_ARCH, shape)
        if reason.startswith("skip"):
            self.skipTest(reason)
        self.assertTrue(passed, f"FAIL dgrad {shape.id} on {GPU_ARCH}: {reason}")

    def test_dgrad(self):
        for s in _DGRAD_SHAPES:
            with self.subTest(shape=s.id):
                self._run_dgrad(s)

    def _run_dw_dgrad(self, shape: _Shape) -> None:
        passed, reason = _run_dw_dgrad_one(GPU_ARCH, shape)
        if reason.startswith("skip"):
            self.skipTest(reason)
        self.assertTrue(passed, f"FAIL dw_dgrad {shape.id} on {GPU_ARCH}: {reason}")

    def test_dw_dgrad(self):
        for s in _DW_DGRAD_SHAPES:
            with self.subTest(shape=s.id):
                self._run_dw_dgrad(s)

    def test_dw_dgrad_streaming(self):
        for s in _DW_DGRAD_STREAM_SHAPES:
            for waves in (1, 2):
                with self.subTest(shape=s.id, block_waves=waves):
                    passed, reason = _run_dw_dgrad_one(
                        GPU_ARCH, s, streaming=True, block_waves=waves
                    )
                    self.assertTrue(
                        passed,
                        f"FAIL dw_dgrad_stream {s.id} bw{waves}wv on {GPU_ARCH}: {reason}",
                    )


# ---------------------------------------------------------------------------
# bf16 correctness tests
# bf16 is supported by cpg=8, cpg=16, cpg=32 (not cpg=4 — no 4x4x4 bf16 atom)
# and by the scalar dgrad path. gfx950 is required for the 16x16x32 fold_k32
# atom; 16x16x16 bf16 (non-fold path) works on both gfx942 and gfx950.
# ---------------------------------------------------------------------------

# Subset of _SHAPES with cpg values that support bf16.
_BF16_FWD_SHAPES: List[_Shape] = [s for s in _SHAPES if s.cpg in (8, 16, 32)]

# Dgrad shapes that support bf16 (scalar FMA dgrad handles all cpg/kpg).
_BF16_DGRAD_SHAPES: List[_Shape] = list(_DGRAD_SHAPES)


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestDirectConvBf16Correctness(unittest.TestCase):
    """Correctness tests for direct conv with bf16 I/O tensors.

    Uses the same harness as ``TestDirectConvCorrectness`` but with
    ``dtype="bf16"`` and a looser tolerance (``_TOL_BF16``).  cpg=4 is
    excluded because there is no ``mfma_f32_4x4x4_bf16`` atom on CDNA.
    """

    def _run_fwd(self, shape: _Shape) -> None:
        passed, reason = _run_grouped_one(GPU_ARCH, shape, dtype="bf16")
        if reason.startswith("skip"):
            self.skipTest(reason)
        self.assertTrue(
            passed,
            f"FAIL bf16 fwd {shape.id} on {GPU_ARCH}: {reason}",
        )

    def test_bf16_cpg8(self):
        for s in _BF16_FWD_SHAPES:
            if s.cpg == 8:
                with self.subTest(shape=s.id):
                    self._run_fwd(s)

    def test_bf16_cpg16(self):
        for s in _BF16_FWD_SHAPES:
            if s.cpg == 16:
                with self.subTest(shape=s.id):
                    self._run_fwd(s)

    def test_bf16_cpg32(self):
        for s in _BF16_FWD_SHAPES:
            if s.cpg == 32:
                with self.subTest(shape=s.id):
                    self._run_fwd(s)

    def _run_dgrad(self, shape: _Shape) -> None:
        passed, reason = _run_dgrad_one(GPU_ARCH, shape, dtype="bf16")
        if reason.startswith("skip"):
            self.skipTest(reason)
        self.assertTrue(
            passed,
            f"FAIL bf16 dgrad {shape.id} on {GPU_ARCH}: {reason}",
        )

    def test_bf16_dgrad(self):
        for s in _BF16_DGRAD_SHAPES:
            with self.subTest(shape=s.id):
                self._run_dgrad(s)

    def test_bf16_depthwise(self):
        for s in _SHAPES:
            if s.cpg == 1:
                with self.subTest(shape=s.id):
                    passed, reason = _run_depthwise_one(GPU_ARCH, s, dtype="bf16")
                    self.assertTrue(
                        passed, f"FAIL bf16 depthwise {s.id} on {GPU_ARCH}: {reason}"
                    )

    def test_bf16_depthwise_tiled(self):
        for s, knobs in _TILED_CASES:
            with self.subTest(shape=s.id):
                passed, reason = _run_depthwise_one(
                    GPU_ARCH, s, dtype="bf16", tiled=knobs
                )
                self.assertTrue(
                    passed, f"FAIL bf16 depthwise tiled {s.id} on {GPU_ARCH}: {reason}"
                )

    def test_bf16_depthwise_spatial(self):
        for s in _SPATIAL_SHAPES:
            with self.subTest(shape=s.id):
                passed, reason = _run_depthwise_one(
                    GPU_ARCH, s, dtype="bf16", spatial=True
                )
                if reason.startswith("skip"):
                    self.skipTest(reason)
                self.assertTrue(
                    passed,
                    f"FAIL bf16 depthwise spatial {s.id} on {GPU_ARCH}: {reason}",
                )

    def test_bf16_dw_dgrad_streaming(self):
        for s in _DW_DGRAD_STREAM_SHAPES:
            with self.subTest(shape=s.id):
                passed, reason = _run_dw_dgrad_one(
                    GPU_ARCH, s, dtype="bf16", streaming=True
                )
                self.assertTrue(
                    passed, f"FAIL bf16 dw_dgrad_stream {s.id} on {GPU_ARCH}: {reason}"
                )


# ---------------------------------------------------------------------------
# Non-grouped (groups == 1) direct conv
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _NgCase:
    """One non-grouped correctness case: shape plus the geometry to build."""

    id: str
    N: int
    H: int
    W: int
    C: int
    K: int
    KH: int = 3
    KW: int = 3
    PAD: int = 1
    stride: int = 1
    dtype: str = "bf16"
    dil_h: int = 1
    dil_w: int = 1
    tile_h: int = 8
    tile_w: int = 32
    tile_k: int = 64
    ck: int = 32
    waves_m: int = 2
    waves_n: int = 2
    atom: str = "32x32x16"
    # Schedule / pipeline knobs (DirectNongroupedConvSpec defaults).
    double_buffer: bool = False
    iglp: "int | None" = None
    waves_per_eu: "int | None" = None
    chiplet_swizzle: bool = True
    swizzle_wgm: int = 8


# Each case pins down one branch of the addressing: exact vs partial output
# tiles, K not covering tile_k, the 16-wide atom used for widths that are not a
# multiple of 32, stride 2, a 1x1 filter, and the fp16 operand path. The
# second group pins the pipeline/schedule knobs: ping-pong LDS, iglp, the
# occupancy hint, the grid decode with and without the chiplet swizzle, and a
# grid large enough (> num_xcds * chiplet_chunk = 512 WGs) that the chiplet
# remap is not the identity, with a partial tail block past the last full one.
_NG_CASES: List[_NgCase] = [
    _NgCase("ng_exact", N=1, H=16, W=32, C=64, K=64),
    # Dilation: tap rows / columns dil apart in a taller / wider staged halo,
    # with the padding the dilated workloads use (not "same" for the filter).
    _NgCase("ng_dil3x1", N=2, H=14, W=14, C=64, K=64, dil_h=3),
    _NgCase("ng_dil1x4_p4", N=1, H=32, W=32, C=64, K=64, PAD=4, dil_w=4),
    _NgCase("ng_dil3x1_s2", N=1, H=29, W=33, C=64, K=96, stride=2, dil_h=3),
    _NgCase(
        "ng_dil2x2_fp16_db",
        N=1,
        H=20,
        W=40,
        C=64,
        K=64,
        PAD=2,
        dtype="fp16",
        dil_h=2,
        dil_w=2,
        ck=16,
        double_buffer=True,
    ),
    _NgCase("ng_partial_w", N=2, H=16, W=40, C=64, K=64),
    _NgCase("ng_partial_h", N=1, H=20, W=32, C=64, K=64, tile_h=16, waves_n=4),
    _NgCase("ng_partial_k", N=1, H=16, W=32, C=64, K=96),
    _NgCase("ng_multi_colblock", N=1, H=16, W=64, C=64, K=64, tile_w=64),
    _NgCase("ng_stride2", N=1, H=32, W=64, C=64, K=64, stride=2),
    _NgCase("ng_pointwise", N=1, H=16, W=32, C=64, K=64, KH=1, KW=1, PAD=0),
    _NgCase("ng_fp16", N=1, H=16, W=32, C=64, K=64, dtype="fp16"),
    _NgCase(
        "ng_atom16",
        N=2,
        H=20,
        W=40,
        C=64,
        K=64,
        tile_w=48,
        tile_k=32,
        waves_m=1,
        atom="16x16x32",
    ),
    _NgCase(
        "ng_atom16x16",
        N=1,
        H=16,
        W=32,
        C=64,
        K=64,
        tile_w=16,
        tile_k=32,
        ck=16,
        waves_m=1,
        atom="16x16x16",
    ),
    _NgCase("ng_32x32x8", N=1, H=16, W=32, C=64, K=64, ck=16, atom="32x32x8"),
    _NgCase("ng_5x5", N=1, H=16, W=32, C=64, K=96, KH=5, KW=5, PAD=2),
    _NgCase("ng_double_buffer", N=2, H=16, W=40, C=128, K=64, double_buffer=True),
    _NgCase(
        "ng_double_buffer_s2_fp16",
        N=1,
        H=32,
        W=64,
        C=64,
        K=64,
        stride=2,
        ck=16,
        dtype="fp16",
        double_buffer=True,
        iglp=1,
    ),
    _NgCase("ng_iglp0_we3", N=1, H=16, W=32, C=128, K=64, iglp=0, waves_per_eu=3),
    _NgCase("ng_no_swizzle", N=2, H=16, W=64, C=64, K=128, chiplet_swizzle=False),
    _NgCase("ng_wgm1", N=2, H=16, W=64, C=64, K=192, swizzle_wgm=1),
    _NgCase(
        "ng_chiplet_remap",
        N=9,
        H=64,
        W=64,
        C=32,
        K=256,
        iglp=0,
    ),
]


# The cases are written for gfx950. gfx942 has neither wide-K atom for
# fp16/bf16 and a 64 KiB LDS, so there each case falls back to the narrower-K
# atom of the same tile size and then shrinks ck (and, failing that, tile_h)
# until the staged tiles fit. The addressing branch a case pins down depends on
# its shape, stride and filter -- not on ck -- so it still gets a numeric check
# instead of a skip. On gfx950 every case is used as written.
_NG_ATOM_FALLBACK = {"32x32x16": "32x32x8", "16x16x32": "16x16x16"}


def _fit_nongrouped_to_arch(spec, arch: str):
    """``spec`` with its atom / ck / tile_h narrowed until ``arch`` can run it."""
    from kernels.common.conv_direct_nongrouped import _ATOMS, _X_LOAD_VEC
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    ab = "bf16" if spec.problem.dtype == "bf16" else "f16"
    t, k, _ = _ATOMS[spec.atom]
    if not target.mma.has_shape(a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=t, n=t, k=k):
        spec = replace(spec, atom=_NG_ATOM_FALLBACK.get(spec.atom, spec.atom))
    min_ck = max(spec.atom_k, _X_LOAD_VEC)
    while not target.fits_lds(spec.lds_bytes) and spec.ck // 2 >= min_ck:
        spec = replace(spec, ck=spec.ck // 2)
    while (
        not target.fits_lds(spec.lds_bytes)
        and spec.tile_h > spec.waves_n
        and (spec.tile_h // 2) % spec.waves_n == 0
    ):
        spec = replace(spec, tile_h=spec.tile_h // 2)
    return spec


def _run_nongrouped_one(arch: str, case: _NgCase) -> Tuple[bool, str]:
    """Build, compile, launch, and verify one non-grouped direct-conv kernel."""
    from kernels.common.conv_direct_grouped import DirectConvProblem
    from kernels.common.conv_direct_nongrouped import DirectNongroupedConvSpec

    p = DirectConvProblem(
        N=case.N,
        H=case.H,
        W=case.W,
        groups=1,
        cpg=case.C,
        kpg=case.K,
        KH=case.KH,
        KW=case.KW,
        PAD=case.PAD,
        stride=case.stride,
        dtype=case.dtype,
        dil_h=case.dil_h,
        dil_w=case.dil_w,
    )
    spec = DirectNongroupedConvSpec(
        problem=p,
        tile_h=case.tile_h,
        tile_w=case.tile_w,
        tile_k=case.tile_k,
        ck=case.ck,
        waves_m=case.waves_m,
        waves_n=case.waves_n,
        atom=case.atom,
        double_buffer=case.double_buffer,
        iglp=case.iglp,
        waves_per_eu=case.waves_per_eu,
        chiplet_swizzle=case.chiplet_swizzle,
        swizzle_wgm=case.swizzle_wgm,
    )
    return _run_nongrouped_spec(arch, _fit_nongrouped_to_arch(spec, arch), case.id)


def _run_nongrouped_spec(
    arch: str, spec, case_id: str, artifact=None
) -> Tuple[bool, str]:
    """Compile, launch and verify one ``DirectNongroupedConvSpec`` against conv2d.

    ``artifact`` reuses a binary compiled for another spec with the same
    capabilities and tile; the shape of ``spec.problem`` then only reaches the
    kernel through its kernargs.
    """
    import torch

    from rocke import compile_kernel
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_nongrouped import (
        DirectSplitReduceSpec,
        build_direct_conv_nongrouped,
        build_direct_conv_split_reduce,
        is_valid_nongrouped_spec,
    )
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    p = spec.problem
    ok, why = is_valid_nongrouped_spec(spec, arch=arch)
    if not ok:
        return False, f"skip {why}"

    if artifact is None:
        try:
            artifact = compile_kernel(
                build_direct_conv_nongrouped(spec, arch=arch), arch=arch
            )
        except Exception as e:  # noqa: BLE001
            return False, f"build/compile failed: {e}"

    td = torch.bfloat16 if p.dtype == "bf16" else torch.float16
    torch.manual_seed(0)
    A_t = torch.empty(p.N, p.H, p.W, p.total_c, dtype=td).uniform_(-1.0, 1.0)
    B_t = torch.empty(p.total_k, p.KH, p.KW, p.cpg, dtype=td).uniform_(-1.0, 1.0)
    D_t = torch.empty(p.N, p.Ho, p.Wo, p.total_k, dtype=td)
    ref = _conv_ref_grouped(A_t, B_t, p)

    rt = Runtime()
    A_dev = rt.alloc(A_t.nbytes)
    B_dev = rt.alloc(B_t.nbytes)
    D_dev = rt.alloc(D_t.nbytes)
    rt.memcpy_h2d(A_dev, _u8(A_t), A_t.nbytes)
    rt.memcpy_h2d(B_dev, _u8(B_t), B_t.nbytes)
    rt.memset(D_dev, 0, D_t.nbytes)
    # A split-C kernel writes fp32 partial sums into a workspace that a second
    # pass folds into D. The workspace is poisoned (NaN bytes) so a slab the
    # conv pass fails to write shows up in the output.
    split = spec.split_c > 1
    WS_dev = rt.alloc(spec.workspace_bytes) if split else None
    if split:
        rt.memset(WS_dev, 0xFF, spec.workspace_bytes)
    bufs = [A_dev, B_dev, D_dev] + ([WS_dev] if split else [])

    sig = conv_direct_args_signature(p.dtype)
    try:
        launcher = KernelLauncher(
            hsaco=artifact.hsaco, kernel_name=artifact.kernel_name, signature=sig
        )
        if split:
            red_spec = DirectSplitReduceSpec(dtype=p.dtype, split_c=spec.split_c)
            red = compile_kernel(build_direct_conv_split_reduce(red_spec), arch=arch)
            reducer = KernelLauncher(
                hsaco=red.hsaco, kernel_name=red.kernel_name, signature=sig
            )
    except HipError as e:
        for buf in bufs:
            rt.free(buf)
        return False, f"kernel load failed: {e}"

    args = ConvArgs.from_problem(p)
    conv_out, conv_out_bytes = (
        (WS_dev, spec.workspace_bytes) if split else (D_dev, D_t.nbytes)
    )
    launcher(
        args.to_launch_values(
            int(A_dev),
            int(B_dev),
            int(conv_out),
            A_t.nbytes,
            B_t.nbytes,
            conv_out_bytes,
        ),
        config=LaunchConfig(
            grid=spec.grid(), block=(spec.threads_per_block, 1, 1), fence=True
        ),
    )
    if split:
        reducer(
            args.to_launch_values(
                int(WS_dev),
                int(B_dev),
                int(D_dev),
                spec.workspace_bytes,
                B_t.nbytes,
                D_t.nbytes,
            ),
            config=LaunchConfig(
                grid=red_spec.grid(p), block=(red_spec.threads, 1, 1), fence=True
            ),
        )

    D_cpu = torch.empty_like(D_t)
    rt.memcpy_d2h(_u8(D_cpu), D_dev, D_t.nbytes)
    for buf in bufs:
        rt.free(buf)
    synchronize_and_release(0)

    diff = (D_cpu.float() - ref.float().cpu()).abs()
    rel_err = float(diff.max() / ref.abs().max().clamp(min=1.0))
    tol = _TOL_BF16 if p.dtype == "bf16" else _TOL
    if rel_err >= tol:
        return False, f"rel_err={rel_err:.3e} > tol={tol:.1e}"
    print(f"  PASS  {case_id}  {arch}  rel_err={rel_err:.2e}", flush=True)
    return True, ""


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestDirectConvNongroupedCorrectness(unittest.TestCase):
    """Correctness sweep for the non-grouped (groups == 1) direct-conv family."""

    def test_nongrouped(self):
        for case in _NG_CASES:
            with self.subTest(case=case.id):
                passed, reason = _run_nongrouped_one(GPU_ARCH, case)
                if reason.startswith("skip"):
                    self.skipTest(reason)
                self.assertTrue(passed, f"FAIL {case.id} on {GPU_ARCH}: {reason}")

    def test_sweep_candidates(self):
        """An evenly spaced sample of what ``nongrouped_specs`` hands the
        benchmark. The winner of a sweep is whichever config is fastest, so
        every candidate the sweep can pick has to be correct -- not just the
        hand-picked geometries above. Wo=48 exercises both the exact-width
        16-wide atom and the padded 32-wide one."""
        from kernels.common.conv_direct_grouped import DirectConvProblem
        from kernels.common.conv_direct_nongrouped import nongrouped_specs

        p = DirectConvProblem(N=2, H=16, W=48, groups=1, cpg=64, kpg=128, dtype="bf16")
        specs = nongrouped_specs(p, arch=GPU_ARCH)
        self.assertGreater(len(specs), 0)
        for spec in specs[:: max(1, len(specs) // 8)]:
            name = spec.kernel_name()
            with self.subTest(spec=name):
                passed, reason = _run_nongrouped_spec(GPU_ARCH, spec, name)
                self.assertTrue(passed, f"FAIL {name} on {GPU_ARCH}: {reason}")

    def test_one_binary_serves_every_shape(self):
        """AOT: a kernel compiled once runs any shape with its capabilities.

        The binary is built against a probe problem, then launched on shapes
        that differ in batch, both spatial extents (exact and partial tiles,
        so the runtime tile counts and bounds checks change) and both channel
        counts (so the runtime channel-loop trip count and the K mask change).
        """
        from rocke import compile_kernel
        from kernels.common.conv_direct_grouped import DirectConvProblem
        from kernels.common.conv_direct_nongrouped import (
            DirectNongroupedConvSpec,
            build_direct_conv_nongrouped,
        )

        def _p(N, H, W, C, K):
            return DirectConvProblem(
                N=N, H=H, W=W, groups=1, cpg=C, kpg=K, dtype="bf16"
            )

        spec = _fit_nongrouped_to_arch(
            DirectNongroupedConvSpec(
                problem=_p(1, 16, 32, 64, 64),
                tile_h=8,
                tile_w=32,
                tile_k=64,
                ck=32,
                waves_m=2,
                waves_n=2,
                iglp=0,
            ),
            GPU_ARCH,
        )
        artifact = compile_kernel(
            build_direct_conv_nongrouped(spec, arch=GPU_ARCH), arch=GPU_ARCH
        )
        for shape in [
            (1, 16, 32, 64, 64),
            (3, 20, 40, 128, 96),
            (2, 33, 17, 256, 192),
            (9, 64, 64, 32, 256),
        ]:
            case = replace(spec, problem=_p(*shape))
            with self.subTest(shape=shape):
                passed, reason = _run_nongrouped_spec(
                    GPU_ARCH, case, f"aot_{shape}", artifact=artifact
                )
                self.assertTrue(passed, f"FAIL {shape} on {GPU_ARCH}: {reason}")


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestDirectConvSplitGroupedCorrectness(unittest.TestCase):
    """The split-C (fp32 workspace + reduce pass) and grouped / group-merged
    (block-diagonal weights) modes of the LDS-halo kernel."""

    def _check(self, cases):
        from kernels.common.conv_direct_grouped import DirectConvProblem
        from kernels.common.conv_direct_nongrouped import DirectNongroupedConvSpec

        for name, prob, knobs in cases:
            spec = _fit_nongrouped_to_arch(
                DirectNongroupedConvSpec(
                    problem=DirectConvProblem(**prob), iglp=0, **knobs
                ),
                GPU_ARCH,
            )
            with self.subTest(case=name):
                passed, reason = _run_nongrouped_spec(GPU_ARCH, spec, name)
                self.assertTrue(passed, f"FAIL {name} on {GPU_ARCH}: {reason}")

    def test_split_c(self):
        dense = dict(groups=1, KH=3, KW=3, PAD=1)
        self._check(
            [
                # 7x7 map, one chunk per split.
                (
                    "sc8_7x7",
                    dict(
                        N=1, H=7, W=7, cpg=256, kpg=64, stride=1, dtype="bf16", **dense
                    ),
                    dict(
                        tile_h=8,
                        tile_w=16,
                        tile_k=32,
                        ck=32,
                        waves_m=1,
                        waves_n=4,
                        atom="16x16x32",
                        split_c=8,
                    ),
                ),
                # Stride 2, a 12-row tile, partial K tile.
                (
                    "sc4_s2_th12",
                    dict(
                        N=2,
                        H=24,
                        W=24,
                        cpg=128,
                        kpg=96,
                        stride=2,
                        dtype="fp16",
                        **dense,
                    ),
                    dict(
                        tile_h=12,
                        tile_w=16,
                        tile_k=64,
                        ck=32,
                        waves_m=1,
                        waves_n=4,
                        atom="16x16x32",
                        split_c=4,
                    ),
                ),
                # 10 chunks over 4 splits (2/3/2/3), double-buffered, partial
                # spatial tiles, no swizzle.
                (
                    "sc4_uneven_db",
                    dict(
                        N=3, H=9, W=13, cpg=320, kpg=64, stride=1, dtype="bf16", **dense
                    ),
                    dict(
                        tile_h=8,
                        tile_w=16,
                        tile_k=32,
                        ck=32,
                        waves_m=1,
                        waves_n=4,
                        atom="16x16x32",
                        split_c=4,
                        double_buffer=True,
                        chiplet_swizzle=False,
                    ),
                ),
            ]
        )

    def test_grouped(self):
        g = dict(KH=3, KW=3, PAD=1)
        self._check(
            [
                (
                    "g32_c4_gm8",
                    dict(
                        N=2,
                        H=20,
                        W=24,
                        groups=32,
                        cpg=4,
                        kpg=4,
                        stride=1,
                        dtype="bf16",
                        **g,
                    ),
                    dict(
                        tile_h=8,
                        tile_w=32,
                        tile_k=32,
                        ck=32,
                        waves_m=1,
                        waves_n=4,
                        atom="16x16x32",
                        grouped=True,
                        group_merge=8,
                    ),
                ),
                (
                    "g16_c8_gm4_s2",
                    dict(
                        N=2,
                        H=30,
                        W=34,
                        groups=16,
                        cpg=8,
                        kpg=8,
                        stride=2,
                        dtype="fp16",
                        **g,
                    ),
                    dict(
                        tile_h=4,
                        tile_w=32,
                        tile_k=32,
                        ck=16,
                        waves_m=1,
                        waves_n=2,
                        atom="32x32x16",
                        grouped=True,
                        group_merge=4,
                    ),
                ),
                (
                    "g4_c32_gm1",
                    dict(
                        N=2,
                        H=17,
                        W=40,
                        groups=4,
                        cpg=32,
                        kpg=32,
                        stride=1,
                        dtype="fp16",
                        **g,
                    ),
                    dict(
                        tile_h=8,
                        tile_w=32,
                        tile_k=32,
                        ck=32,
                        waves_m=1,
                        waves_n=4,
                        atom="16x16x32",
                        grouped=True,
                        group_merge=1,
                    ),
                ),
                # cpg 12: fragments straddle groups at offsets that are not a
                # multiple of 8.
                (
                    "g8_c12_gm4",
                    dict(
                        N=1,
                        H=12,
                        W=16,
                        groups=8,
                        cpg=12,
                        kpg=12,
                        stride=1,
                        dtype="bf16",
                        **g,
                    ),
                    dict(
                        tile_h=8,
                        tile_w=16,
                        tile_k=48,
                        ck=16,
                        waves_m=1,
                        waves_n=4,
                        atom="16x16x16",
                        grouped=True,
                        group_merge=4,
                    ),
                ),
                # Grouped and split together, double-buffered.
                (
                    "g8_c64_gm2_sc2",
                    dict(
                        N=1,
                        H=10,
                        W=12,
                        groups=8,
                        cpg=64,
                        kpg=32,
                        stride=1,
                        dtype="bf16",
                        **g,
                    ),
                    dict(
                        tile_h=8,
                        tile_w=16,
                        tile_k=64,
                        ck=32,
                        waves_m=1,
                        waves_n=4,
                        atom="16x16x32",
                        grouped=True,
                        group_merge=2,
                        split_c=2,
                        double_buffer=True,
                    ),
                ),
            ]
        )


class TestDirectConvNongroupedValidation(unittest.TestCase):
    """Non-grouped validation that does not require a GPU."""

    @staticmethod
    def _problem(**kw):
        from kernels.common.conv_direct_grouped import DirectConvProblem

        base = dict(N=1, H=32, W=32, groups=1, cpg=64, kpg=64, dtype="bf16")
        base.update(kw)
        return DirectConvProblem(**base)

    def test_grouped_problem_rejected(self):
        from kernels.common.conv_direct_nongrouped import DirectNongroupedConvSpec

        p = self._problem(groups=4, cpg=16, kpg=16)
        with self.assertRaises(ValueError):
            DirectNongroupedConvSpec(problem=p).validate()

    def test_grouped_without_groups_rejected(self):
        """grouped=True needs a positive group count: 0 and negative multiples
        of group_merge pass the divisibility check alone."""
        from kernels.common.conv_direct_nongrouped import DirectNongroupedConvSpec

        for groups, group_merge in ((0, 1), (0, 2), (-4, 2)):
            p = self._problem(groups=groups, cpg=8, kpg=8)
            with self.subTest(groups=groups, group_merge=group_merge):
                with self.assertRaisesRegex(ValueError, "groups >= 1"):
                    DirectNongroupedConvSpec(
                        problem=p, grouped=True, group_merge=group_merge
                    ).validate()

    def test_oversized_accumulator_rejected(self):
        """A tile needing more than 256 accumulator registers must not build.

        Left unguarded this is not just slow: a 2048-register tile hangs the
        backend scheduler rather than failing, which stalls a whole sweep.
        """
        from kernels.common.conv_direct_nongrouped import DirectNongroupedConvSpec

        spec = DirectNongroupedConvSpec(
            problem=self._problem(),
            tile_h=16,
            tile_w=64,
            tile_k=128,
            ck=16,
            waves_m=1,
            waves_n=1,
        )
        self.assertGreater(spec.acc_vgprs, 256)
        with self.assertRaises(ValueError):
            spec.validate()

    def test_flattened_grid_overflow_rejected(self):
        """The 1-D grid (and the kernel's i32 workgroup count) must fit i32."""
        from kernels.common.conv_direct_nongrouped import DirectNongroupedConvSpec

        # 1 x 4 spatial tiles x 16 channel tiles per image, 2**25 images.
        spec = DirectNongroupedConvSpec(
            problem=self._problem(N=1 << 25, cpg=64, kpg=512),
            tile_h=8,
            tile_w=32,
            tile_k=32,
            ck=32,
            waves_m=1,
            waves_n=2,
        )
        self.assertGreater(spec.grid()[0], (1 << 31) - 1)
        with self.assertRaisesRegex(ValueError, "flattened grid"):
            spec.validate()

    def test_is_valid_never_raises(self):
        """``is_valid_nongrouped_spec`` reports, it does not raise -- including
        when the arch-table queries past ``validate()`` fail."""
        from unittest import mock

        from kernels.common.conv_direct_nongrouped import (
            DirectNongroupedConvSpec,
            is_valid_nongrouped_spec,
        )
        from rocke.core.arch import ArchTarget

        spec = DirectNongroupedConvSpec(
            problem=self._problem(), tile_h=8, tile_k=64, ck=32, waves_n=2
        )
        self.assertTrue(is_valid_nongrouped_spec(spec)[0])

        real = ArchTarget.from_gfx("gfx950")
        broken = mock.MagicMock(wraps=real)
        broken.mma.has_shape.side_effect = KeyError("no mma table")
        with mock.patch.object(ArchTarget, "from_gfx", return_value=broken):
            ok, why = is_valid_nongrouped_spec(spec)
        self.assertFalse(ok)
        self.assertIn("no mma table", why)

        ok, why = is_valid_nongrouped_spec(spec, arch="gfx_nonexistent")
        self.assertFalse(ok)

    def test_tile_w_candidates_cover_exactly(self):
        """Candidate widths must tile Wo without wasted columns when possible."""
        from kernels.common.conv_direct_nongrouped import tile_w_candidates

        for wo, at in ((96, 32), (160, 32), (112, 16), (208, 16)):
            for tw in tile_w_candidates(wo, at):
                self.assertEqual(wo % tw, 0, f"Wo={wo} atom={at} tw={tw}")
        # 40 is not a multiple of 32, so the 32-wide atom falls back to its
        # least-wasteful widths (32 and 64 both compute 64 columns) rather than
        # returning nothing.
        self.assertEqual(tile_w_candidates(40, 32), [32, 64])

    def test_specs_are_unique_and_valid(self):
        from kernels.common.conv_direct_nongrouped import (
            is_valid_nongrouped_spec,
            nongrouped_specs,
        )

        specs = nongrouped_specs(self._problem(H=64, W=64, cpg=256, kpg=256))
        self.assertGreater(len(specs), 0)
        names = [s.kernel_name() for s in specs]
        self.assertEqual(len(names), len(set(names)))
        for s in specs:
            self.assertTrue(is_valid_nongrouped_spec(s)[0])


class TestDirectConvValidation(unittest.TestCase):
    """Validation-only tests that do not require a GPU."""

    def test_cpg4_bf16_accepted(self):
        """cpg=4 + bf16 validates: the 4c kernel takes mfma_f32_4x4x4_bf16 (the
        CDNA2+ ``_1k`` atom), and the kernel name carries the dtype. Any other
        dtype is still refused."""
        from kernels.common.conv_direct_grouped import (
            DirectConv4cSpec,
            DirectConvProblem,
            is_valid_spec_4c,
        )

        p = DirectConvProblem(N=1, H=8, W=8, groups=16, cpg=4, kpg=4, dtype="bf16")
        spec = DirectConv4cSpec(problem=p)
        spec.validate()
        for arch in ("gfx942", "gfx950"):
            self.assertTrue(is_valid_spec_4c(spec, arch=arch)[0])
        self.assertTrue(spec.kernel_name().endswith("_bf16"))
        bad = DirectConv4cSpec(
            problem=DirectConvProblem(
                N=1, H=8, W=8, groups=16, cpg=4, kpg=4, dtype="fp32"
            )
        )
        with self.assertRaises(ValueError):
            bad.validate()
        self.assertFalse(is_valid_spec_4c(bad)[0])

    def test_depthwise_filter_limit(self):
        """Both depthwise forward variants accept filters up to 32x32 and reject
        larger ones, as the C++ validators do: its builders keep the weights in
        tables of ``ROCKE_DCONV_DW_MAX_K{H,W}`` entries. The preloading variant
        is additionally capped at ``_DW_MAX_PRELOAD_TAPS`` filter taps."""
        from kernels.common.conv_direct_grouped import (
            _DW_MAX_PRELOAD_TAPS,
            DirectConvProblem,
            DirectDepthwiseSpatialSpec,
            DirectDepthwiseSpec,
            is_valid_depthwise_spatial_spec,
            is_valid_depthwise_spec,
        )

        cases = (
            (DirectDepthwiseSpec, is_valid_depthwise_spec, 64),
            (DirectDepthwiseSpatialSpec, is_valid_depthwise_spatial_spec, 3),
        )
        for spec_cls, is_valid, groups in cases:
            # 31 is the largest odd extent under the cap (forward needs odd).
            for kh, kw in ((31, 31), (33, 33), (33, 3), (3, 33)):
                p = DirectConvProblem(
                    N=1,
                    H=40,
                    W=40,
                    groups=groups,
                    cpg=1,
                    kpg=1,
                    KH=kh,
                    KW=kw,
                    PAD=(kh - 1) // 2,
                )
                with self.subTest(spec=spec_cls.__name__, KH=kh, KW=kw):
                    ok, why = is_valid(spec_cls(problem=p), "gfx950")
                    in_table = kh <= 32 and kw <= 32
                    preload_ok = (
                        spec_cls is not DirectDepthwiseSpec
                        or kh * kw <= _DW_MAX_PRELOAD_TAPS
                    )
                    self.assertEqual(ok, in_table and preload_ok, why)
                    if not in_table:
                        self.assertIn("must be in 1..32", why)
                    elif not preload_ok:
                        self.assertIn("DirectDepthwiseColSpec", why)


# ---------------------------------------------------------------------------
# Wgrad shapes
# ---------------------------------------------------------------------------


# wgrad needs ds_read_tr16_b64 for its LDS transpose staging, which is gfx950+.
# It therefore gets its own class gate rather than reusing _SKIP_REASON: that one
# is empty on gfx942, so every case would come back as a per-subtest validator
# rejection and the suite would report green without ever compiling or launching
# a kernel.
def _wgrad_skip_reason() -> str:
    if _SKIP_REASON:
        return _SKIP_REASON
    if GPU_ARCH != "gfx950":
        return f"wgrad needs gfx950 (ds_read_tr16_b64), got {GPU_ARCH!r}"
    return ""


_WGRAD_SKIP_REASON = _wgrad_skip_reason()


_WGRAD_SHAPES: List[_Shape] = [
    # Grouped, symmetric channels -- the baseline case.
    _Shape("wg_16c_N2H8W8_g8", N=2, H=8, W=8, groups=8, cpg=16),
    # groups=1 with cpg != kpg: the shape the direct wgrad dispatch actually
    # sees, and the only entry spanning several k AND c tiles at once.
    _Shape("wg_g1_c48k192", N=2, H=8, W=8, groups=1, cpg=48, kpg=192),
    # W is not a multiple of the MFMA spatial block, so the last wo tile is
    # partially masked and the S strip runs off the right edge.
    _Shape("wg_oddW37", N=2, H=8, W=37, groups=1, cpg=32, kpg=32),
    # 1x1: no halo at all -- STRIP_COLS collapses onto the MFMA block.
    _Shape("wg_1x1", N=2, H=8, W=8, groups=1, cpg=16, kpg=32, KH=1, KW=1, PAD=0),
    # 5x5: a halo wider than one tap on each side.
    _Shape("wg_5x5", N=2, H=8, W=8, groups=1, cpg=16, kpg=32, KH=5, KW=5, PAD=2),
]

# (waves_k, waves_c, waves_q, mfma_k, ho_per_block).
#
# The wave spread is what the single-wave default never reaches: waves_k > 1
# splits the S-strip loader across waves (so one wave reads LDS that another
# wave wrote), waves_c > 1 gives each c-wave its own strip partition while the
# dY tile stays shared, and mfma_k=16 takes the narrow atom with one transpose
# read per fragment instead of two.
_WGRAD_CONFIGS = [
    (1, 1, 1, 32, 4),  # default
    (2, 1, 1, 32, 4),  # K split -> STRIP_GROUPS=2, cross-wave strip
    (1, 2, 1, 32, 4),  # C split -> per-c strip partitions, shared dY
    (2, 2, 1, 32, 3),  # both, with ho_per_block not dividing H
    (1, 1, 2, 32, 4),  # spatial split
    (1, 1, 1, 16, 2),  # narrow MFMA atom
]


# Runtime-channel wgrad: per-group channel counts the AOT cache must serve
# from one binary per knob set.
_WGRAD_RTC_SHAPES: List[_Shape] = [
    _Shape("wg_rtc_c24k24", N=2, H=12, W=40, groups=1, cpg=24, kpg=24),
    _Shape("wg_rtc_c48k192", N=2, H=10, W=20, groups=1, cpg=48, kpg=192),
    _Shape("wg_rtc_c64k256", N=2, H=8, W=12, groups=1, cpg=64, kpg=256),
    _Shape("wg_rtc_g4_c64", N=2, H=7, W=14, groups=4, cpg=64, kpg=64),
    _Shape("wg_rtc_g32_c16", N=2, H=7, W=7, groups=32, cpg=16, kpg=16),
    _Shape("wg_rtc_c40k24_pad0", N=1, H=9, W=33, groups=2, cpg=40, kpg=24, PAD=0),
]

# (waves_k, waves_c, waves_q, mfma_k, ho_per_block): single wave, a K/C/Q
# split with a three-tile C block, and the widest C block a cache entry has.
_WGRAD_RTC_CONFIGS = [
    (1, 1, 1, 32, 4),
    (2, 3, 2, 32, 3),
    (1, 4, 1, 32, 30),
]


def _run_wgrad_one(
    arch: str,
    shape: _Shape,
    dtype: str = "fp16",
    cfg: Tuple[int, int, int, int, int] = (1, 1, 1, 32, 4),
    runtime_channels: bool = False,
) -> Tuple[bool, str]:
    """Build, compile, launch, and verify the direct wgrad kernel.

    ``cfg`` is (waves_k, waves_c, waves_q, mfma_k, ho_per_block).

    With ``runtime_channels`` the kernel is built the way the AOT cache builds
    it -- on a probe problem with other channel counts, group count and image
    size -- and launched on ``shape``, so the binary has to take every channel
    count it uses from the kernargs.

    Returns ``(passed, reason)``.
    """
    from dataclasses import replace
    import torch

    from rocke import compile_kernel
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_direct_grouped import (
        direct_launch_geometry,
        DirectConvWgradSpec,
        DirectConvProblem,
        build_direct_conv_wgrad,
        is_valid_wgrad_spec,
    )
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    waves_k, waves_c, waves_q, mfma_k, hpb = cfg
    kpg = shape.kpg or shape.cpg
    p = DirectConvProblem(
        N=shape.N,
        H=shape.H,
        W=shape.W,
        groups=shape.groups,
        cpg=shape.cpg,
        kpg=kpg,
        KH=shape.KH,
        KW=shape.KW,
        PAD=shape.PAD,
        stride=shape.stride,
        dtype=dtype,
    )
    # Every knob lands in kernel_name() (bk/bc/hpb/mk plus the wq and bf16
    # flags), so each case here compiles to a distinct symbol.
    spec = DirectConvWgradSpec(
        problem=p,
        name=f"test_wgrad_{shape.id}",
        waves_k=waves_k,
        waves_c=waves_c,
        waves_q=waves_q,
        mfma_k=mfma_k,
        ho_per_block=hpb,
        runtime_channels=runtime_channels,
    )

    ok, reason = is_valid_wgrad_spec(spec, arch=arch)
    if not ok:
        return False, f"skip invalid spec: {reason}"

    build_spec = spec
    if runtime_channels:
        build_spec = replace(
            spec,
            problem=replace(p, N=1, H=64, W=64, groups=1, cpg=256, kpg=256),
        )
    try:
        kernel = build_direct_conv_wgrad(build_spec, arch=arch)
    except ValueError as e:
        return False, f"build failed: {e}"

    try:
        artifact = compile_kernel(kernel, arch=arch)
    except Exception as e:
        return False, f"compile failed: {e}"

    torch.manual_seed(42)
    total_c = shape.groups * shape.cpg
    total_k = shape.groups * kpg

    _td = torch.bfloat16 if dtype == "bf16" else torch.float16
    # X:  input          [N, H, W, C]
    X = torch.empty(p.N, p.H, p.W, total_c, dtype=_td).uniform_(-0.5, 0.5)
    # dY: output gradient [N, Ho, Wo, K]
    dY = torch.empty(p.N, p.Ho, p.Wo, total_k, dtype=_td).uniform_(-0.5, 0.5)
    # dW: weight gradient [K, KH, KW, cpg] fp32 — zeroed before launch
    dW = torch.zeros(total_k, p.KH, p.KW, shape.cpg, dtype=torch.float32)

    # Reference: dW = conv2d wgrad via torch autograd
    X_t = X.float().cuda().requires_grad_(False)
    W_ref = torch.zeros(
        total_k, shape.cpg, p.KH, p.KW, dtype=torch.float32, device="cuda"
    )
    W_ref.requires_grad_(True)
    X_nchw = X_t.permute(0, 3, 1, 2)
    out_ref = torch.nn.functional.conv2d(
        X_nchw, W_ref, padding=p.PAD, stride=p.stride, groups=p.groups
    )
    dY_nchw = dY.float().cuda().permute(0, 3, 1, 2)
    out_ref.backward(dY_nchw)
    ref_dw = W_ref.grad  # [K, cpg, KH, KW]
    # Convert to [K, KH, KW, cpg] layout to match dW
    ref_dw_krsc = ref_dw.permute(0, 2, 3, 1).contiguous().cpu()

    rt = Runtime()
    X_dev = rt.alloc(X.nbytes)
    dY_dev = rt.alloc(dY.nbytes)
    dW_dev = rt.alloc(dW.nbytes)
    rt.memcpy_h2d(X_dev, _u8(X), X.nbytes)
    rt.memcpy_h2d(dY_dev, _u8(dY), dY.nbytes)
    rt.memset(dW_dev, 0, dW.nbytes)  # caller must zero dW

    # Direct conv is AOT: the whole shape travels as kernargs. D is the fp32
    # dW accumulator rather than an io-typed tensor.
    sig_wg = conv_direct_args_signature(dtype, direction="wgrad")
    try:
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=sig_wg,
        )
    except HipError as e:
        rt.free(X_dev)
        rt.free(dY_dev)
        rt.free(dW_dev)
        return False, f"kernel load failed: {e}"

    grid, block = direct_launch_geometry(spec)

    values = ConvArgs.from_problem(p, direction="wgrad").to_launch_values(
        int(dY_dev),
        int(X_dev),
        int(dW_dev),
        dY.nbytes,
        X.nbytes,
        dW.nbytes,
    )
    launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))

    dW_cpu = torch.empty_like(dW)
    rt.memcpy_d2h(_u8(dW_cpu), dW_dev, dW.nbytes)
    rt.free(X_dev)
    rt.free(dY_dev)
    rt.free(dW_dev)
    synchronize_and_release(0)

    out_f32 = dW_cpu.float()
    ref_f32 = ref_dw_krsc.float()
    abs_diff = (out_f32 - ref_f32).abs()
    ref_scale = ref_f32.abs().max().clamp(min=1.0)
    rel_err = float(abs_diff.max() / ref_scale)
    tol = _TOL_BF16 if dtype == "bf16" else _TOL
    passed = rel_err < tol
    if not passed:
        return False, f"rel_err={rel_err:.3e} > tol={tol:.1e}"
    print(
        f"  PASS  {shape.id}  {arch}  {dtype}  "
        f"wk={waves_k} wc={waves_c} wq={waves_q} mk={mfma_k} hpb={hpb}  "
        f"rtc={int(runtime_channels)}  rel_err={rel_err:.2e}",
        flush=True,
    )
    return True, ""


@unittest.skipUnless(not _WGRAD_SKIP_REASON, _WGRAD_SKIP_REASON or "no GPU")
class TestDirectConvWgradCorrectness(unittest.TestCase):
    """Correctness tests for direct conv backward weights (wgrad)."""

    def setUp(self) -> None:
        # Cases that got past the validator and really compiled and launched.
        # Every test method asserts this ended non-zero, so a spec rejection can
        # never quietly stand in for a pass the way the arch gate once let it.
        self._ran = 0

    def _run_wgrad(
        self, shape: _Shape, dtype: str = "fp16", cfg=None, runtime_channels=False
    ) -> None:
        kwargs = {"dtype": dtype, "runtime_channels": runtime_channels}
        if cfg is not None:
            kwargs["cfg"] = cfg
        passed, reason = _run_wgrad_one(GPU_ARCH, shape, **kwargs)
        if reason.startswith("skip"):
            self.skipTest(reason)
        # Counted before the assert, so a real failure is reported as itself
        # rather than as a second "nothing ran" error.
        self._ran += 1
        self.assertTrue(
            passed,
            f"FAIL wgrad {shape.id} {dtype} cfg={cfg} on {GPU_ARCH}: {reason}",
        )

    def _assert_ran(self) -> None:
        self.assertGreater(
            self._ran,
            0,
            f"no wgrad case ran on {GPU_ARCH} -- every spec was rejected",
        )

    def test_wgrad(self):
        """Every shape on the default single-wave spread."""
        for s in _WGRAD_SHAPES:
            with self.subTest(shape=s.id):
                self._run_wgrad(s)
        self._assert_ran()

    def test_wgrad_bf16(self):
        """Same shapes on the bf16 MFMA atom and bf16 LDS staging."""
        for s in _WGRAD_SHAPES:
            with self.subTest(shape=s.id):
                self._run_wgrad(s, dtype="bf16")
        self._assert_ran()

    def test_wgrad_wave_configs(self):
        """The wave spread, on the shapes with channels enough to split.

        This is the coverage the default-only tests miss: the cross-wave S-strip
        split, the per-c strip partitions, the spatial split and the narrow MFMA
        atom. Run in BOTH dtypes -- otherwise bf16 is only ever seen at the
        single-wave default, so bf16 x mfma_k=16 (one ds_read_tr per fragment)
        and bf16 x multi-wave never execute at all. ``wg_16c_N2H8W8_g8`` is in
        the list for groups > 1, so the group term of the ``bx`` decode is
        exercised under the spread rather than only at groups=1.

        A spread the shape cannot afford is rejected by the validator and
        skipped, not failed.
        """
        shapes = [
            s
            for s in _WGRAD_SHAPES
            if s.id in ("wg_16c_N2H8W8_g8", "wg_g1_c48k192", "wg_oddW37")
        ]
        for dtype in ("fp16", "bf16"):
            for s in shapes:
                for cfg in _WGRAD_CONFIGS:
                    with self.subTest(shape=s.id, dtype=dtype, cfg=cfg):
                        self._run_wgrad(s, dtype=dtype, cfg=cfg)
        self._assert_ran()

    def test_wgrad_runtime_channels(self):
        """Runtime channel counts: a binary built on a cpg=kpg=256 probe, as the
        AOT cache builds it, launched on other cpg/kpg and group counts --
        ragged channel tiles, cpg != kpg and grouped problems included."""
        for dtype in ("fp16", "bf16"):
            for s in _WGRAD_RTC_SHAPES:
                for cfg in _WGRAD_RTC_CONFIGS:
                    with self.subTest(shape=s.id, dtype=dtype, cfg=cfg):
                        self._run_wgrad(s, dtype=dtype, cfg=cfg, runtime_channels=True)
        self._assert_ran()


# ---------------------------------------------------------------------------
# MFMA dgrad pipeline (transpose -> reorganize -> forward on the transposed
# problem), baked and with runtime channels.
#
# A runtime-channel binary is built once per knob set on the first problem of
# its list and launched, unchanged, on every problem: they share its K-atom and
# M-tile counts but differ in N, H, W, groups and the exact cpg/kpg (dense,
# grouped, partial K-atoms and M-tiles) -- what the AOT cache does.
# ---------------------------------------------------------------------------

# (label, fprop knobs, fold_k32, [(N, H, W, groups, cpg, kpg), ...])
_MFMA_DGRAD_RTC_CASES = [
    (
        "plain_loop",
        dict(block_q=32, block_groups=1, block_h=8),
        False,
        [(2, 14, 14, 1, 24, 24), (1, 9, 20, 3, 20, 28), (2, 11, 17, 2, 32, 20)],
    ),
    (
        "wk4_preload",
        dict(block_groups=1, block_h=8, waves_k=4),
        False,
        [(1, 12, 18, 1, 48, 192), (2, 7, 9, 1, 36, 180)],
    ),
    (
        "wk2_k32",
        dict(block_groups=1, block_h=16, waves_k=2),
        True,
        [(1, 12, 18, 1, 48, 192), (2, 9, 13, 2, 44, 192)],
    ),
    (
        "wq2_rk",
        dict(block_q=32, block_groups=1, waves_q=2, runtime_k_loop=True),
        False,
        [(2, 10, 21, 1, 20, 48), (2, 10, 21, 2, 20, 40), (1, 13, 34, 4, 32, 36)],
    ),
    (
        "bg2_plain_loop",
        dict(block_groups=2, block_h=8),
        False,
        [(2, 14, 14, 4, 16, 16), (1, 9, 9, 8, 12, 12)],
    ),
]


def _launch_cfg(spec):
    from kernels.common.conv_direct_grouped import direct_launch_geometry
    from rocke.runtime.launcher import LaunchConfig

    grid, block = direct_launch_geometry(spec)
    return LaunchConfig(grid=grid, block=block, fence=True)


def _run_mfma_dgrad_pipeline(arch, problem, fprop_spec, kernels, dtype):
    """Run (transpose[, reorganize], fprop) on ``problem``; returns rel_err."""
    import torch

    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_grouped import (
        DIRECT_WEIGHT_TRANSFORM_RTC_ARGS,
        DirectReorganizeWeightsSpec,
        DirectTransposeWeightsDgradSpec,
        direct_dgrad_coalesced_workspace_bytes,
        direct_dgrad_workspace_bytes,
    )
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher

    p = problem
    rtc = fprop_spec.runtime_channels
    coalesced = fprop_spec.waves_k > 1 or fprop_spec.runtime_k_loop
    elem = "bf16" if dtype == "bf16" else "f16"
    xform_sig = [
        {"name": "A", "type": f"ptr<{elem}, global>", "size_bytes": 8},
        {"name": "D", "type": f"ptr<{elem}, global>", "size_bytes": 8},
        {"name": "A_bytes", "type": "i32", "size_bytes": 4},
        {"name": "D_bytes", "type": "i32", "size_bytes": 4},
    ]
    if rtc:
        xform_sig += [
            {"name": n, "type": "i32", "size_bytes": 4}
            for n in DIRECT_WEIGHT_TRANSFORM_RTC_ARGS
        ]
    scalars = {"p_cpg": p.cpg, "p_kpg": p.kpg} if rtc else {}

    torch.manual_seed(7)
    _td = _torch_dtype(dtype)
    dY = torch.empty(p.N, p.Ho, p.Wo, p.total_k, dtype=_td).uniform_(-0.5, 0.5)
    W = torch.empty(p.total_k, p.KH, p.KW, p.cpg, dtype=_td).uniform_(-0.5, 0.5)
    dX = torch.zeros(p.N, p.H, p.W, p.total_c, dtype=_td)
    ref = (
        torch.nn.functional.conv_transpose2d(
            dY.permute(0, 3, 1, 2).float(),
            W.permute(0, 3, 1, 2).float(),
            padding=p.PAD,
            groups=p.groups,
        )
        .permute(0, 2, 3, 1)
        .contiguous()
    )

    rt = Runtime()
    ws_t_bytes = direct_dgrad_workspace_bytes(p)
    ws_coa_bytes = direct_dgrad_coalesced_workspace_bytes(
        p, fold_k32=fprop_spec.fold_k32
    )
    bufs = {
        "dy": rt.alloc(dY.nbytes),
        "w": rt.alloc(W.nbytes),
        "dx": rt.alloc(dX.nbytes),
        "ws_t": rt.alloc(ws_t_bytes),
        "ws_coa": rt.alloc(ws_coa_bytes),
    }
    rt.memcpy_h2d(bufs["dy"], _u8(dY), dY.nbytes)
    rt.memcpy_h2d(bufs["w"], _u8(W), W.nbytes)
    # NaN in dX (0xFFFF is a NaN in fp16 and bf16): every output must be
    # written. Garbage in the workspaces: a correct pipeline overwrites what
    # it reads.
    rt.memset(bufs["dx"], 0xFF, dX.nbytes)
    rt.memset(bufs["ws_t"], 0x7F, ws_t_bytes)
    rt.memset(bufs["ws_coa"], 0x7F, ws_coa_bytes)

    t_art, r_art, f_art = kernels
    t_spec = DirectTransposeWeightsDgradSpec(problem=p, runtime_channels=rtc)
    KernelLauncher(
        hsaco=t_art.hsaco, kernel_name=t_art.kernel_name, signature=xform_sig
    )(
        {
            "A": bufs["w"],
            "D": bufs["ws_t"],
            "A_bytes": W.nbytes,
            "D_bytes": ws_t_bytes,
            **scalars,
        },
        config=_launch_cfg(t_spec),
    )
    weights = "ws_t"
    if coalesced:
        r_spec = DirectReorganizeWeightsSpec(
            problem=p, fold_k32=fprop_spec.fold_k32, runtime_channels=rtc
        )
        KernelLauncher(
            hsaco=r_art.hsaco, kernel_name=r_art.kernel_name, signature=xform_sig
        )(
            {
                "A": bufs["ws_t"],
                "D": bufs["ws_coa"],
                "A_bytes": ws_t_bytes,
                "D_bytes": ws_coa_bytes,
                **scalars,
            },
            config=_launch_cfg(r_spec),
        )
        weights = "ws_coa"
    sizes = {
        "dy": dY.nbytes,
        "dx": dX.nbytes,
        "ws_t": ws_t_bytes,
        "ws_coa": ws_coa_bytes,
    }
    values = ConvArgs.from_problem(fprop_spec.problem).to_launch_values(
        bufs["dy"], bufs[weights], bufs["dx"], sizes["dy"], sizes[weights], sizes["dx"]
    )
    KernelLauncher(
        hsaco=f_art.hsaco,
        kernel_name=f_art.kernel_name,
        signature=conv_direct_args_signature(dtype),
    )(values, config=_launch_cfg(fprop_spec))

    out = torch.empty_like(dX)
    rt.memcpy_d2h(_u8(out), bufs["dx"], dX.nbytes)
    for ptr in bufs.values():
        rt.free(ptr)
    synchronize_and_release(0)
    diff = (out.float() - ref.float()).abs().max()
    return float(diff / ref.float().abs().max().clamp(min=1.0))


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestDirectConvMfmaDgradCorrectness(unittest.TestCase):
    """The MFMA dgrad pipeline end to end, against conv_transpose2d."""

    def _check(self, label, knobs, fold_k32, shapes, dtype, runtime_channels):
        from rocke import compile_kernel
        from kernels.common.conv_direct_grouped import (
            DirectConvProblem,
            DirectReorganizeWeightsSpec,
            DirectTransposeWeightsDgradSpec,
            build_direct_conv,
            build_direct_reorganize_weights,
            build_direct_transpose_weights_dgrad,
            make_dgrad_fprop_spec,
        )

        if fold_k32 and GPU_ARCH != "gfx950":
            self.skipTest("fold_k32 needs the 16x16x32 atom")

        def fprop_spec_for(problem):
            spec = replace(make_dgrad_fprop_spec(problem, **knobs), fold_k32=fold_k32)
            if runtime_channels:
                spec = replace(
                    spec,
                    runtime_channels=True,
                    k_atoms=spec.problem_k_atoms(),
                    m_tiles=spec.problem_m_tiles(),
                )
            return spec

        def build(problem):
            spec = fprop_spec_for(problem)
            return (
                compile_kernel(
                    build_direct_transpose_weights_dgrad(
                        DirectTransposeWeightsDgradSpec(
                            problem=problem, runtime_channels=runtime_channels
                        ),
                        arch=GPU_ARCH,
                    ),
                    arch=GPU_ARCH,
                ),
                compile_kernel(
                    build_direct_reorganize_weights(
                        DirectReorganizeWeightsSpec(
                            problem=problem,
                            fold_k32=fold_k32,
                            runtime_channels=runtime_channels,
                        ),
                        arch=GPU_ARCH,
                    ),
                    arch=GPU_ARCH,
                ),
                compile_kernel(build_direct_conv(spec, arch=GPU_ARCH), arch=GPU_ARCH),
            )

        kernels = None
        tol = _TOL_BF16 if dtype == "bf16" else _TOL
        for N, H, W, groups, cpg, kpg in shapes:
            problem = DirectConvProblem(
                N=N, H=H, W=W, groups=groups, cpg=cpg, kpg=kpg, dtype=dtype
            )
            spec = fprop_spec_for(problem)
            if kernels is None or not runtime_channels:
                kernels = build(problem)
            with self.subTest(case=label, dtype=dtype, shape=problem.short()):
                rel_err = _run_mfma_dgrad_pipeline(
                    GPU_ARCH, problem, spec, kernels, dtype
                )
                self.assertLess(rel_err, tol, f"{label} {problem.short()} {dtype}")

    def test_mfma_dgrad_baked(self):
        # One baked binary per shape, on every shape of every case: grouped
        # problems and partial K-atoms included. The coalesced cases
        # (waves_k > 1, runtime_k_loop) run the baked reorganize pass, which
        # must read W_T's per-group column and zero-fill out-of-range lanes.
        for dtype in ("fp16", "bf16"):
            for label, knobs, fold_k32, shapes in _MFMA_DGRAD_RTC_CASES:
                self._check(label, knobs, fold_k32, shapes, dtype, False)

    def test_mfma_dgrad_runtime_channels(self):
        for dtype in ("fp16", "bf16"):
            for label, knobs, fold_k32, shapes in _MFMA_DGRAD_RTC_CASES:
                self._check(label, knobs, fold_k32, shapes, dtype, True)


# ---------------------------------------------------------------------------
# The AOT sweep's direct wgrad kernels: conv_direct_wgrad_tiled (fp32 atomics),
# conv_direct_dw_wgrad (depthwise, partial slots) and conv_direct_wgrad_gather
# (dense small C, partial slots).
# ---------------------------------------------------------------------------
#
# Each case is built the way the kernel cache builds it -- on the probe problem
# of its capability row -- and launched on the problem listed here, so every
# extent the binary takes as a kernarg (and the host/kernel agreement on the
# partition and split counts) is exercised. The partial-slot kernels run the
# fold after the pass, as the sweep's two-step plan does.

# (id, variant, DirectConvProblem fields, knobs)
_AOT_WGRAD_CASES = [
    (
        "tiled_c64",
        "direct_wgrad_tiled",
        dict(N=2, H=16, W=32, groups=1, cpg=64, kpg=64),
        dict(rows_per_seg=30),
    ),
    # Valid padding, a partial wave tile in K and C, grouped.
    (
        "tiled_g2_pad0",
        "direct_wgrad_tiled",
        dict(N=2, H=13, W=37, groups=2, cpg=48, kpg=40, PAD=0),
        dict(wave_k16=1, wave_c16=1, waves_k=2, waves_c=2, rows_per_seg=12),
    ),
    # Four images packed into one 32-column strip of a narrow map.
    (
        "tiled_unfold4",
        "direct_wgrad_tiled",
        dict(N=6, H=15, W=8, groups=1, cpg=32, kpg=32),
        dict(waves_k=1, waves_c=1, rows_per_seg=15, unfold_n=4),
    ),
    # Two wave groups, the second half a row behind (pingpong); segment counts
    # that leave the second group a dummy segment.
    (
        "tiled_qg2_pp",
        "direct_wgrad_tiled",
        dict(N=3, H=9, W=40, groups=1, cpg=32, kpg=64),
        dict(
            wave_k16=1,
            waves_k=2,
            rows_per_seg=12,
            prefetch=3,
            q_groups=2,
            pingpong=True,
            target_blocks=8,
        ),
    ),
    (
        "dw_c64",
        "direct_wgrad_dw",
        dict(N=2, H=17, W=33, groups=64, cpg=1, kpg=1),
        dict(lanes_c=8, lanes_w=8, prefetch=2, rows_per_seg=16, target_blocks=1024),
    ),
    # Stride 2 and a channel count that leaves a partial channel tile.
    (
        "dw_c40_s2",
        "direct_wgrad_dw",
        dict(N=2, H=20, W=24, groups=40, cpg=1, kpg=1, stride=2),
        dict(lanes_c=8, lanes_w=16, prefetch=1, rows_per_seg=16, target_blocks=2048),
    ),
    # The stem: C = 3, 7x7, stride 2.
    (
        "dw_cols_c3_7x7_s2",
        "direct_wgrad_dw_cols",
        dict(N=2, H=20, W=64, groups=3, cpg=1, kpg=1, KH=7, KW=7, PAD=3, stride=2),
        dict(channels=3, strip=8, threads=128, target_blocks=2048),
    ),
    # Valid padding (the capability rows the probe width makes buildable).
    (
        "dw_cols_c4_pad0",
        "direct_wgrad_dw_cols",
        dict(N=2, H=12, W=34, groups=4, cpg=1, kpg=1, PAD=0),
        dict(channels=4, strip=16, threads=256, target_blocks=4096),
    ),
    # Gathered X operand (S * C = 6).
    (
        "gather_c3_s2",
        "direct_wgrad_gather",
        dict(N=2, H=24, W=64, groups=1, cpg=3, kpg=24, stride=2),
        dict(wave_k16=2, waves_k=1, prefetch=2, rows_per_seg=8, target_blocks=256),
    ),
    # Transpose-read X operand.
    (
        "gather_c24_s2",
        "direct_wgrad_gather",
        dict(N=2, H=30, W=40, groups=1, cpg=24, kpg=96, stride=2),
        dict(wave_k16=1, waves_k=6, prefetch=2, rows_per_seg=8, target_blocks=256),
    ),
    # Stride 1 with a K that leaves a partial k tile.
    (
        "gather_c4_s1",
        "direct_wgrad_gather",
        dict(N=1, H=16, W=32, groups=1, cpg=4, kpg=40, stride=1),
        dict(wave_k16=2, waves_k=1, prefetch=1, rows_per_seg=16, target_blocks=1024),
    ),
    # Valid padding, three k-waves of one atom each over two k tiles.
    (
        "gather_c8_pad0_w1x3",
        "direct_wgrad_gather",
        dict(N=2, H=20, W=32, groups=1, cpg=8, kpg=96, PAD=0, stride=1),
        dict(wave_k16=1, waves_k=3, prefetch=2, rows_per_seg=8, target_blocks=256),
    ),
    # Valid padding at stride 2, three k-waves of two atoms, a partial k tile.
    (
        "gather_c3_pad0_w2x3_s2",
        "direct_wgrad_gather",
        dict(N=2, H=21, W=64, groups=1, cpg=3, kpg=104, PAD=0, stride=2),
        dict(wave_k16=2, waves_k=3, prefetch=1, rows_per_seg=16, target_blocks=1024),
    ),
    # The cases above give every block one segment. Two blocks walking several
    # segments each (ten, two): accumulation across segments and ring reuse
    # after the end-of-segment barrier.
    (
        "dw_c64_tb2",
        "direct_wgrad_dw",
        dict(N=2, H=40, W=40, groups=64, cpg=1, kpg=1),
        dict(lanes_c=8, lanes_w=8, prefetch=2, rows_per_seg=16, target_blocks=2),
    ),
    (
        "gather_c3_s2_tb2",
        "direct_wgrad_gather",
        dict(N=2, H=24, W=64, groups=1, cpg=3, kpg=24, stride=2),
        dict(wave_k16=2, waves_k=1, prefetch=2, rows_per_seg=8, target_blocks=2),
    ),
    # 128 partitions: the fold loops over its 64-partition stride twice.
    (
        "dw_c64_parts128",
        "direct_wgrad_dw",
        dict(N=4, H=64, W=64, groups=64, cpg=1, kpg=1),
        dict(lanes_c=8, lanes_w=8, prefetch=2, rows_per_seg=16, target_blocks=1024),
    ),
    # KW * C = 72 outputs over 64 threads (a second epilogue pass), one block
    # per filter row walking two units.
    (
        "dw_cols_c8_9x9_t64",
        "direct_wgrad_dw_cols",
        dict(N=2, H=16, W=32, groups=8, cpg=1, kpg=1, KH=9, KW=9, PAD=4),
        dict(channels=8, strip=8, threads=64, target_blocks=9),
    ),
    # Depthwise 4x4x4 MFMA, 7x7 in row chunks, a partial channel tile.
    (
        "dw_mfma_7x7_chunks",
        "direct_wgrad_dw_mfma",
        dict(N=2, H=17, W=23, groups=72, cpg=1, kpg=1, KH=7, KW=7, PAD=3),
        dict(q_tiles=4, rows_per_chunk=7, prefetch=2, target_blocks=1024, unfold_n=1),
    ),
    # Whole images per unit (the PAD-row prologue), one staging buffer.
    (
        "dw_mfma_7x7_whole",
        "direct_wgrad_dw_mfma",
        dict(N=3, H=9, W=30, groups=64, cpg=1, kpg=1, KH=7, KW=7, PAD=3),
        dict(q_tiles=8, rows_per_chunk=14, prefetch=1, target_blocks=1024, unfold_n=1),
    ),
    # Three 7x7 images packed in one tile, a batch that leaves one short.
    (
        "dw_mfma_unfold3",
        "direct_wgrad_dw_mfma",
        dict(N=5, H=7, W=7, groups=128, cpg=1, kpg=1, KH=7, KW=7, PAD=3),
        dict(q_tiles=8, rows_per_chunk=14, prefetch=2, target_blocks=1024, unfold_n=3),
    ),
    # 3x3 valid padding: no preshift, both prologue bodies differ.
    (
        "dw_mfma_3x3_pad0",
        "direct_wgrad_dw_mfma",
        dict(N=2, H=12, W=18, groups=64, cpg=1, kpg=1, PAD=0),
        dict(q_tiles=4, rows_per_chunk=7, prefetch=2, target_blocks=2048, unfold_n=1),
    ),
    # Grouped 4x4x4 MFMA, 4 channels per group.
    (
        "grouped_mfma_c4",
        "direct_wgrad_grouped_mfma",
        dict(N=2, H=13, W=21, groups=32, cpg=4, kpg=4),
        dict(waves_q=4, wave_q4=2, rows_per_chunk=28, prefetch=2, target_blocks=1024),
    ),
    # 8 channels per group, a partial channel tile.
    (
        "grouped_mfma_c8",
        "direct_wgrad_grouped_mfma",
        dict(N=2, H=13, W=21, groups=9, cpg=8, kpg=8),
        dict(waves_q=2, wave_q4=1, rows_per_chunk=28, prefetch=3, target_blocks=1024),
    ),
]


def _run_aot_wgrad_one(arch: str, variant: str, problem, knobs: dict) -> float:
    """Build ``variant`` on its cache probe, launch it on ``problem``, return
    the max error relative to the largest reference dW entry."""
    import torch
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_dw_wgrad import (
        DW_WGRAD_FOLD_ARGS,
        dw_wgrad_fold_signature,
    )
    from kernels.conv.direct_aot_jobs import DirectCaps, probe_problem
    from kernels.conv.direct_variants import (
        DIRECT_WGRAD_FOLD,
        DIRECT_WGRAD_FOLDED,
        direct_wgrad_fold_knobs,
        make_direct_spec,
        select_direct_kernel_builder,
        validate_direct_spec,
    )
    from rocke import compile_kernel
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    p = problem
    folded = variant in DIRECT_WGRAD_FOLDED
    if variant == "direct_wgrad_tiled":
        caps = DirectCaps(KH=p.KH, PAD=p.PAD, stride=p.stride, cpg=0, kpg=0)
    elif variant == "direct_wgrad_gather":
        caps = DirectCaps(KH=p.KH, PAD=p.PAD, stride=p.stride, cpg=p.cpg, kpg=0)
    elif variant == "direct_wgrad_grouped_mfma":
        caps = DirectCaps(KH=p.KH, PAD=p.PAD, stride=p.stride, cpg=p.cpg, kpg=p.kpg)
    else:
        caps = DirectCaps(KH=p.KH, PAD=p.PAD, stride=p.stride, cpg=1, kpg=1)
    probe = probe_problem(caps, p.dtype, variant, knobs)
    build_spec = make_direct_spec(variant, probe, knobs)
    spec = make_direct_spec(variant, p, knobs)
    for s, what in ((build_spec, "probe"), (spec, "problem")):
        ok, why = validate_direct_spec(variant, s, arch)
        assert ok, f"{variant} rejects its {what}: {why}"
    assert build_spec.kernel_name() == spec.kernel_name()
    art = compile_kernel(
        select_direct_kernel_builder(variant, build_spec, arch), arch=arch
    )

    torch.manual_seed(7)
    td = torch.bfloat16 if p.dtype == "bf16" else torch.float16
    total_c, total_k = p.groups * p.cpg, p.groups * p.kpg
    if folded and p.cpg == 1 and p.kpg == 1:
        total_k = total_c = p.groups  # depthwise
    X = torch.empty(p.N, p.H, p.W, total_c, dtype=td).uniform_(-0.5, 0.5)
    dY = torch.empty(p.N, p.Ho, p.Wo, total_k, dtype=td).uniform_(-0.5, 0.5)
    dw_elems = total_k * p.KH * p.KW * p.cpg

    W_ref = torch.zeros(
        total_k, p.cpg, p.KH, p.KW, dtype=torch.float32, device="cuda"
    ).requires_grad_(True)
    out = torch.nn.functional.conv2d(
        X.float().cuda().permute(0, 3, 1, 2),
        W_ref,
        padding=p.PAD,
        stride=p.stride,
        groups=p.groups,
    )
    out.backward(dY.float().cuda().permute(0, 3, 1, 2))
    ref = W_ref.grad.permute(0, 2, 3, 1).contiguous().cpu()  # [K, KH, KW, cpg]

    rt = Runtime()
    bufs = {
        "x": (rt.alloc(X.nbytes), X.nbytes),
        "dy": (rt.alloc(dY.nbytes), dY.nbytes),
        "dw": (rt.alloc(dw_elems * 4), dw_elems * 4),
    }
    if folded:
        ws_bytes = spec.workspace_bytes()
        bufs["ws"] = (rt.alloc(ws_bytes), ws_bytes)
    try:
        rt.memcpy_h2d(bufs["x"][0], _u8(X), X.nbytes)
        rt.memcpy_h2d(bufs["dy"][0], _u8(dY), dY.nbytes)
        # The atomic kernel adds into a zeroed dW; the fold writes dW whole,
        # so poison it to catch an element the fold misses.
        rt.memset(bufs["dw"][0], 0 if not folded else 0xFF, bufs["dw"][1])
        out_buf = "ws" if folded else "dw"
        grid, block = spec.launch_geometry()
        launcher = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=conv_direct_args_signature(p.dtype, direction="wgrad"),
        )
        values = ConvArgs.from_problem(p, direction="wgrad").to_launch_values(
            int(bufs["dy"][0]),
            int(bufs["x"][0]),
            int(bufs[out_buf][0]),
            bufs["dy"][1],
            bufs["x"][1],
            bufs[out_buf][1],
        )
        launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))
        if folded:
            fold_knobs = direct_wgrad_fold_knobs(variant, p.KH, p.KW, p.cpg)
            fold_spec = make_direct_spec(DIRECT_WGRAD_FOLD, p, fold_knobs)
            fold_art = compile_kernel(
                select_direct_kernel_builder(DIRECT_WGRAD_FOLD, fold_spec, arch),
                arch=arch,
            )
            channels = (
                spec.fold_channels() if hasattr(spec, "fold_channels") else p.groups
            )
            fgrid, fblock = fold_spec.launch_geometry(channels)
            fold = KernelLauncher(
                hsaco=fold_art.hsaco,
                kernel_name=fold_art.kernel_name,
                signature=dw_wgrad_fold_signature(fold_spec),
            )
            fvalues = {
                "A": int(bufs["ws"][0]),
                "D": int(bufs["dw"][0]),
                "A_bytes": bufs["ws"][1],
                "D_bytes": bufs["dw"][1],
                **dict(zip(DW_WGRAD_FOLD_ARGS, (spec.n_partitions(), channels))),
            }
            fold(fvalues, config=LaunchConfig(grid=fgrid, block=fblock, fence=True))
        dW = torch.empty(dw_elems, dtype=torch.float32)
        rt.memcpy_d2h(_u8(dW), bufs["dw"][0], bufs["dw"][1])
    finally:
        for ptr, _ in bufs.values():
            rt.free(ptr)
        synchronize_and_release(0)

    got = dW.view(total_k, p.KH, p.KW, p.cpg)
    scale = ref.abs().max().clamp(min=1.0)
    return float(((got - ref).abs().max() / scale).nan_to_num(float("inf")))


@unittest.skipUnless(not _WGRAD_SKIP_REASON, _WGRAD_SKIP_REASON or "no GPU")
class TestDirectWgradAotKernelsCorrectness(unittest.TestCase):
    """The tiled, depthwise and gather direct wgrad kernels of the AOT sweep."""

    def _check(self, dtype: str) -> None:
        from kernels.common.conv_direct_grouped import DirectConvProblem

        tol = _TOL_BF16 if dtype == "bf16" else _TOL
        for case_id, variant, shape, knobs in _AOT_WGRAD_CASES:
            problem = DirectConvProblem(dtype=dtype, **shape)
            with self.subTest(case=case_id, dtype=dtype):
                rel_err = _run_aot_wgrad_one(GPU_ARCH, variant, problem, knobs)
                print(f"  {case_id:24s} {dtype}  rel_err={rel_err:.2e}", flush=True)
                self.assertLess(rel_err, tol, f"{case_id} {problem.short()} {dtype}")

    def test_aot_wgrad_fp16(self):
        self._check("fp16")

    def test_aot_wgrad_bf16(self):
        self._check("bf16")


if __name__ == "__main__":
    unittest.main(verbosity=2)
