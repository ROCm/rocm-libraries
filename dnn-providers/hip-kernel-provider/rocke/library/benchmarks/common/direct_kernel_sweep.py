# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The AOT cache for the direct convolution kernels.

Direct conv splits its parameters differently from implicit GEMM. Batch,
spatial extents and the group count are kernargs, so one binary serves any
image; the filter geometry (``KH``/``KW``), ``stride``, ``PAD`` and the
per-group channel counts (``cpg``/``kpg``) shape the unrolled MFMA chain and
the LDS row layout, so they are *capabilities* baked into the binary. A cache
therefore holds one binary per (capability, tuning knob) pair, and
``--run-from-cache`` offers a binary only to problems with exactly its
capabilities.

Which capabilities to build is a product decision. :data:`DIRECT_CAPABILITIES`
follows the set the reference direct-conv library covers on CDNA and drops
what rocke's kernels cannot compute (see the table's comments).

The non-grouped (``groups == 1``) forward kernel bakes less: its channel counts
are kernargs as well, so its capabilities record ``cpg = kpg = 0`` (runtime)
and one binary serves every ``C``/``K`` its tile divides. The wgrad kernel is
built the same way (``runtime_channels``): one binary per filter, padding and
tuning knob serves every per-group channel count its knobs fit. So is the
runtime-channel MFMA dgrad pipeline (``direct_dgrad_mfma_rtc``), which serves
dense as well as grouped problems: its compute pass keeps the weights of its
K-atoms in registers, so what it bakes is the *number* of K-atoms and M-tiles
(knobs ``k_atoms`` / ``m_tiles``), and the exact channel counts are kernargs.

The run side is split in two. :func:`direct_plans` is pure CPU: it picks the
cached kernels that fit a problem, re-validates each spec against the real
shape, and returns a launch plan -- binaries, signatures, grids and the
buffer each kernarg slot binds to. The benchmark only allocates memory and
launches the plan.
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
import struct
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from benchmarks.common.kernel_cache import KernelCache, KernelIdentity
from benchmarks.common.kernel_sweep import BuildJob, compile_jobs

# Operand dtypes the direct kernels build for. The cache is filled for one at a
# time (--compile-all --dtype); each binary records its dtype and is only
# offered to problems of that dtype. Variants without a kernel for a dtype
# (the 4c atom is fp16-only) are dropped by their spec validators.
DIRECT_DTYPES = ("fp16", "bf16")
DIRECT_DTYPE = "fp16"  # the default dtype
# Convolution directions the direct cache holds kernels for.
DIRECT_DIRECTIONS = ("fwd", "dgrad", "wgrad")

# ---------------------------------------------------------------------------
# Tuning grids -- shared with benchmark_direct_conv's JIT sweep so the cache
# and the sweep explore the same configurations.
# ---------------------------------------------------------------------------

# Grouped forward (DirectConvSpec) and the MFMA dgrad's fprop pass.
BLOCK_Q = (16, 32)
BLOCK_GROUPS = (1, 2, 4, 8, 16)
DOUBLE_BUFFER = (True, False)
# Depthwise (cpg == 1) forward and dgrad.
# block_w=32 is omitted from both directions: across the depthwise corpus, at
# both dtypes, it never won a single geometry -- every shape whose best config
# used a wide block_w landed on 8 or 16 -- while still costing a quarter of the
# candidate builds and tuning launches.
#
# The surviving values differ by direction, because the two directions do not
# have the same fallback.  Forward is overwhelmingly a block_w=4 story: the
# column-streamed variant (its own grid below) covers the wide-block cases, so
# the preloading kernel rarely needs to go wide, and the few shapes that do go
# wide land on 16 rather than 8.  Dropping 8 from the forward grid is therefore
# free, while dropping 16 is not.  Dgrad has no second variant to fall back on,
# so its tail keeps both 8 and 16 -- dropping either regresses small-spatial /
# large-filter shapes, 8 the more severely of the two.  Dgrad also carries 7:
# it tiles the 7- and 14-wide maps of the 7x7 depthwise stages without a
# ragged edge, and on those it beats every power of two (14 never did).
# Neither tuple is reducible further without giving up a shape's best config.
DW_BLOCK_W_FWD = (4, 16)
DW_BLOCK_W_DGRAD = (4, 7, 8, 16)
DW_BLOCK_WAVES = (1, 2, 4)
# Output-stationary depthwise forward. Widths and heights of 7 and 14 tile the
# 7/14/28/56-wide feature maps of the common CNN stages without a ragged edge;
# the spec validator drops the tiles that overflow the register budget and the
# unrolled loops that would take long to compile.
DW_TILED_BLOCK_W = (4, 7, 8, 14, 16)
DW_TILED_BLOCK_H = (1, 2, 4, 7)
DW_TILED_BLOCK_WAVES = (1, 2)
DW_TILED_UNROLL = (False, True)

# Column-streamed depthwise forward. It keeps only ``block_h*block_w + KH`` f32
# live per lane instead of ``KH*KW + KH*block_w``, so its sweet spot sits at far
# smaller block_w than the weight-preloading kernel's. block_h is the output-row
# tile: the row loop is unrolled at build time, so the tile height is a
# capability while the image height stays a kernarg. Pairs whose band does not
# fit the arch's live-f32 budget are dropped by the spec validator.
#
# Measured on gfx950 over 15 depthwise shapes (3x3..11x11, stride 1 and 2,
# 7x7..112x112 images, large enough batches to clear launch latency): block_w
# 8/16/32 never won a shape, and this 3x3x3 grid matches the best of the full
# 3x6x3 grid on every one. All three block_h values are needed -- dropping 32
# costs up to 16% on a shape, dropping 8 up to 74%.
DW_COL_BLOCK_W = (1, 2, 4)
DW_COL_BLOCK_H = (8, 16, 32)
# Grouped scalar-FMA dgrad.
DGRAD_BLOCK_Q = (4, 8, 16, 32)
# MFMA dgrad fprop pass: output rows per block, then
# (waves_q, waves_k, runtime_k_loop, persistent_grid, fold_k32).
DGRAD_BLOCK_H = (8, 16)
DGRAD_WAVES = (
    (1, 1, False, False, False),
    (1, 4, False, False, False),
    # A 2-way split of the 16-channel atoms: even atom counts 4 does not
    # divide (kpg 96, 160, 224, ...) get a preloading config too.
    (1, 2, False, False, False),
    # runtime_k_loop: one K-atom at a time -> low VGPR use, more blocks/CU.
    (1, 4, True, False, False),
    (1, 2, False, False, True),
    (1, 6, False, False, True),
)
# Direct wgrad (DirectConvWgradSpec): (waves_k, waves_c, waves_q). The JIT
# sweep tries waves_c in {1, 2, ceil(cpg/16)}; with runtime channels the cache
# cannot know cpg, so it carries the ceil(cpg/16) values of the common channel
# counts and plan_for drops the ones wider than a problem's channel tiles.
WGRAD_WAVES_FIXED = (
    (1, 1, 1),
    (2, 1, 1),
    (4, 1, 1),
    (6, 1, 1),
    (8, 1, 1),
    (2, 1, 2),
    (4, 1, 2),
    (1, 2, 1),
    (2, 2, 1),
    (4, 2, 1),
)
WGRAD_WAVES_C_WIDE = (3, 4, 6, 8, 12, 16)
WGRAD_WAVES_K_WIDE = (1, 2, 4)
WGRAD_HPB = (30, 60, 120)
WGRAD_MFMA_K = (32,)

# Runtime-channel MFMA dgrad (direct_dgrad_mfma_rtc). Its compute pass is the
# forward kernel on the transposed problem, whose K reduction runs over the
# dgrad's kpg (dY channels) and whose M-tiles cover its cpg (dX channels).
# A binary bakes the tile counts, so the cache holds one per count:
#   m_tiles: 16-channel tiles of cpg -- dX channels per group up to 64.
DGRAD_RTC_M_TILES = (1, 2, 3, 4)
#   k_atoms: K-atoms of kpg, up to this many dY channels per group, in steps
#   of waves_k atoms (each wave preloads an equal slice).
DGRAD_RTC_MAX_KPG = 512
#   The plain-weight loop (waves_k == 1) re-reads its weights every row and
#   only wins for a few K-atoms; past that the preloading configs take over.
#   (Odd atom counts above it -- kpg 80, 112, ... -- have no binary: no split
#   divides them, and longer plain loops with wide block_groups overflow the
#   backend's LDS limit.)
DGRAD_RTC_LOOP_MAX_ATOMS = 4
#   Share of a wave's register budget the weights and accumulators of a
#   binary may fill (dgrad_rtc_vgprs; the budget shrinks as more waves of a
#   block share a SIMD). The estimate leaves out addresses and loop state;
#   calibrated on gfx950 builds, configs past it spill to scratch, so they
#   are not cached.
DGRAD_RTC_VGPR_FILL = 0.875


def wgrad_waves() -> Tuple[Tuple[int, int, int], ...]:
    """Every (waves_k, waves_c, waves_q) the wgrad cache builds."""
    out = set(WGRAD_WAVES_FIXED)
    out.update((wk, wc, 1) for wk in WGRAD_WAVES_K_WIDE for wc in WGRAD_WAVES_C_WIDE)
    return tuple(
        sorted(w for w in out if w[0] * w[1] <= 16 and w[0] * w[1] * w[2] * 64 <= 1024)
    )


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectCaps:
    """What one binary is baked for; everything else about a shape is runtime.

    ``cpg = kpg = 0`` means the channel counts are runtime too (the non-grouped
    kernel).
    """

    KH: int
    PAD: int
    stride: int
    cpg: int
    kpg: int
    # Filter dilation per axis; only the non-grouped forward kernel has rows
    # with dilation.
    dil_h: int = 1
    dil_w: int = 1

    @property
    def KW(self) -> int:
        # Every capability below is a square filter.
        return self.KH


def _grouped_channels() -> Tuple[int, ...]:
    # The reference library covers cpg == kpg in {4, 8, 16, 32}.
    return (4, 8, 16, 32)


# (cpg, kpg) of the dense (groups == 1) 3x3 convolutions whose dgrad the MFMA
# direct pipeline serves. Implicit GEMM maps dX channels to its N tile, so a
# narrow C (4, 24, 48) leaves most of every MFMA idle there, while the direct
# kernel reduces over K instead: on these shapes it runs up to twice as fast.
_DENSE_MFMA_DGRAD_CHANNELS = ((4, 320), (24, 24), (48, 192))


_DEPTHWISE_FILTERS = (3, 5, 7, 9, 11)
# Large-kernel depthwise forward (RepLKNet/ConvNeXt-style 13x13 and 31x31):
# stride 1, "same" padding only -- the layers that use them -- so they add a
# handful of capabilities rather than a full padding/stride range each.
_DEPTHWISE_FILTERS_LARGE = (13, 31)
# The output-stationary depthwise kernel keeps one filter row in registers, so
# large filters (RepLKNet-style 13x13 .. 31x31) build quickly; it covers
# every odd filter up to the depthwise limit.
_DEPTHWISE_TILED_FILTERS = tuple(range(3, 32, 2))
_STRIDES = (1, 2)
# The non-grouped kernel exists for 3x3: its LDS halo reuse is what lets it beat
# implicit GEMM there. A 1x1 has no halo to share and is a plain GEMM.
_NONGROUPED_FILTERS = (3,)
# Wgrad filters the cache carries. The kernel takes any KH, KW <= 8; a 1x1
# wgrad is a plain GEMM with no halo to reuse, so it is left to implicit GEMM.
_WGRAD_FILTERS = (3, 5, 7)
# Dilated 3x3 rows of the non-grouped kernel: (PAD, stride, dil_h, dil_w). The
# kernel bounds its rows and columns by the runtime output extents, so any
# padding works; the rows bake the "same" padding of the symmetric dilations
# and the asymmetric ones the reference workloads use (DeepLab-style row/column
# dilation with the padding of the undilated axis).
_NONGROUPED_DILATED = (
    (2, 1, 2, 2),
    (3, 1, 3, 3),
    (4, 1, 4, 4),
    (1, 1, 3, 1),
    (1, 2, 3, 1),
    (2, 1, 1, 2),
    (4, 1, 1, 4),
)


def _same_pad(kh: int) -> int:
    return (kh - 1) // 2


def _large_depthwise_fwd_caps(max_kh: int = 31) -> Tuple["DirectCaps", ...]:
    return tuple(
        DirectCaps(KH=k, PAD=_same_pad(k), stride=1, cpg=1, kpg=1)
        for k in _DEPTHWISE_FILTERS_LARGE
        if k <= max_kh
    )


def _all_pads(kh: int) -> range:
    # The reference library takes the padding as a runtime value in [0, KH-1].
    return range(kh)


def _col_pads(kh: int) -> Tuple[int, ...]:
    # The column-streamed kernel can compute any PAD in [0, KH-1] (bar stride-1
    # over-padding, which its validator rejects), but every padding is its own
    # binary. The cache carries the two networks actually use -- "valid"
    # (PAD=0) and "same" -- rather than the whole range: the intermediate ones
    # would be two thirds of the variant's entries for shapes that are rare in
    # practice. Other paddings still build on demand (the JIT sweep).
    return tuple(sorted({0, _same_pad(kh)}))


# variant -> (direction, capability list). Forward rows only list "same"
# padding: every forward direct kernel streams input rows and flushes output
# rows by input-row index, which is only right for PAD == (KH-1)/2 (see
# kernels.common.conv_direct_grouped.forward_padding_reason). Dgrad kernels
# bound their rows by p_Ho and are padding-generic, so they get the full
# [0, KH-1] range like the reference.
DIRECT_CAPABILITIES: Dict[str, Tuple[str, Tuple[DirectCaps, ...]]] = {
    # Grouped forward, 3x3. The reference also serves stride 2; rocke's
    # grouped forward kernels are stride-1 only, so that row is absent.
    "direct_grouped": (
        "fwd",
        tuple(
            DirectCaps(KH=3, PAD=1, stride=1, cpg=c, kpg=c) for c in _grouped_channels()
        ),
    ),
    # Non-grouped forward (groups == 1), C and K runtime. The kernel bounds
    # rows and columns against the runtime output extents, so it is not tied
    # to "same" padding the way the row-streaming kernels are; the cache bakes
    # the "same" padding the forward rows here share.
    "direct_nongrouped": (
        "fwd",
        tuple(
            DirectCaps(KH=k, PAD=_same_pad(k), stride=s, cpg=0, kpg=0)
            for k, s in itertools.product(_NONGROUPED_FILTERS, _STRIDES)
        )
        + tuple(
            DirectCaps(KH=3, PAD=pad, stride=s, cpg=0, kpg=0, dil_h=dh, dil_w=dw)
            for pad, s, dh, dw in _NONGROUPED_DILATED
        ),
    ),
    # Grouped forward through the non-grouped kernel's LDS-halo pipeline:
    # groups and per-group channels runtime, small cpg merged into one MFMA
    # problem with block-diagonal weights (knob group_merge).
    "direct_grouped_halo": (
        "fwd",
        tuple(
            DirectCaps(KH=k, PAD=_same_pad(k), stride=s, cpg=0, kpg=0)
            for k, s in itertools.product(_NONGROUPED_FILTERS, _STRIDES)
        ),
    ),
    # Depthwise forward: square odd filters 3..11, stride 1 and 2, plus the
    # large stride-1 filters.
    # The weight-preloading kernel holds KH*KW taps per lane, so 31x31 does
    # not fit it; the column-streamed and spatial kernels cover that one.
    "direct_depthwise": (
        "fwd",
        tuple(
            DirectCaps(KH=k, PAD=_same_pad(k), stride=s, cpg=1, kpg=1)
            for k, s in itertools.product(_DEPTHWISE_FILTERS, _STRIDES)
        )
        + _large_depthwise_fwd_caps(max_kh=13),
    ),
    # Output-stationary depthwise forward: every odd filter up to 31x31.
    "direct_depthwise_tiled": (
        "fwd",
        tuple(
            DirectCaps(KH=k, PAD=_same_pad(k), stride=s, cpg=1, kpg=1)
            for k, s in itertools.product(_DEPTHWISE_TILED_FILTERS, _STRIDES)
        ),
    ),
    # Depthwise forward for groups < wave_size: lanes split over channels and
    # output columns. Same capabilities; the spec validator offers it only to
    # problems with fewer groups than a wave.
    "direct_depthwise_spatial": (
        "fwd",
        tuple(
            DirectCaps(KH=k, PAD=_same_pad(k), stride=s, cpg=1, kpg=1)
            for k, s in itertools.product(_DEPTHWISE_FILTERS, _STRIDES)
        )
        + _large_depthwise_fwd_caps(),
    ),
    # Column-streamed depthwise forward: same filters and strides. It bounds
    # its output rows against p_Ho rather than streaming them by input row, so
    # unlike the other forward kernels it also serves "valid" padding.
    "direct_depthwise_col": (
        "fwd",
        tuple(
            DirectCaps(KH=k, PAD=pad, stride=s, cpg=1, kpg=1)
            for k, s in itertools.product(_DEPTHWISE_FILTERS, _STRIDES)
            for pad in _col_pads(k)
        )
        + _large_depthwise_fwd_caps(),
    ),
    # Depthwise dgrad (row-streaming kernel): same filters and strides.
    "direct_depthwise_dgrad": (
        "dgrad",
        tuple(
            DirectCaps(KH=k, PAD=pad, stride=s, cpg=1, kpg=1)
            for k, s in itertools.product(_DEPTHWISE_FILTERS, _STRIDES)
            for pad in _all_pads(k)
        ),
    ),
    # Grouped dgrad, scalar FMA: any stride and padding the reference covers.
    "direct_grouped_dgrad": (
        "dgrad",
        tuple(
            DirectCaps(KH=3, PAD=pad, stride=s, cpg=c, kpg=c)
            for c, s, pad in itertools.product(
                _grouped_channels(), _STRIDES, _all_pads(3)
            )
        ),
    ),
    # Grouped dgrad, MFMA pipeline (weight transpose -> reorganize -> fprop).
    # Its last pass is a forward kernel on the transposed problem, so it
    # inherits the forward limits: stride 1 and "same" padding. Besides the
    # grouped channel counts it carries the dense ones it wins on.
    "direct_grouped_dgrad_mfma": (
        "dgrad",
        tuple(
            DirectCaps(KH=3, PAD=1, stride=1, cpg=c, kpg=c) for c in _grouped_channels()
        )
        + tuple(
            DirectCaps(KH=3, PAD=1, stride=1, cpg=c, kpg=k)
            for c, k in _DENSE_MFMA_DGRAD_CHANNELS
        ),
    ),
    # The same pipeline with runtime channels: dense (groups == 1) and any
    # cpg/kpg whose tile counts a cached binary has (knobs k_atoms/m_tiles).
    "direct_dgrad_mfma_rtc": (
        "dgrad",
        (DirectCaps(KH=3, PAD=1, stride=1, cpg=0, kpg=0),),
    ),
    # Wgrad (grouped and non-grouped), channel counts runtime. The kernel is a
    # stride-1 algorithm; it bounds its rows and columns against the runtime
    # extents, so it serves "valid" padding as well as "same".
    "direct_wgrad": (
        "wgrad",
        tuple(
            DirectCaps(KH=k, PAD=pad, stride=1, cpg=0, kpg=0)
            for k in _WGRAD_FILTERS
            for pad in _col_pads(k)
        ),
    ),
}

# Weight-transform kernels of the MFMA dgrad pipeline, cached as their own
# entries and found by exact identity. The _RTC pair takes the channel counts
# as kernargs (one binary per filter and fold_k32).
_TRANSPOSE = "direct_dgrad_transpose"
_REORGANIZE = "direct_dgrad_reorganize"
_TRANSPOSE_RTC = "direct_dgrad_transpose_rtc"
_REORGANIZE_RTC = "direct_dgrad_reorganize_rtc"
_MFMA_DGRAD = ("direct_grouped_dgrad_mfma", "direct_dgrad_mfma_rtc")
# Variants built by conv_direct_nongrouped (dense, and grouped via merging).
_HALO_VARIANTS = ("direct_nongrouped", "direct_grouped_halo")
# Second pass of a split-C non-grouped forward kernel: folds the fp32 partial
# sums into the output. Shape-independent (one binary per split_c and dtype),
# cached under its own direction and found by exact identity.
_SPLIT_REDUCE = "direct_split_reduce"
_FWD_HELPER = "direct_fwd_helper"
# Capabilities of the split reduction: it bakes no filter or channel count.
_NO_CAPS = DirectCaps(KH=0, PAD=0, stride=0, cpg=0, kpg=0)
# variant -> (transpose helper, reorganize helper)
_MFMA_HELPERS = {
    "direct_grouped_dgrad_mfma": (_TRANSPOSE, _REORGANIZE),
    "direct_dgrad_mfma_rtc": (_TRANSPOSE_RTC, _REORGANIZE_RTC),
}


def dgrad_rtc_k_atoms(waves_k: int, fold_k32: bool) -> Tuple[int, ...]:
    """The K-atom counts the runtime-channel dgrad cache builds for a split.

    Preloading splits (``waves_k > 1``) take every multiple of ``waves_k`` up
    to :data:`DGRAD_RTC_MAX_KPG` channels; the plain-weight loop only the
    first :data:`DGRAD_RTC_LOOP_MAX_ATOMS`.
    """
    atom = 32 if fold_k32 else 16
    if waves_k == 1:
        return tuple(range(1, DGRAD_RTC_LOOP_MAX_ATOMS + 1))
    return tuple(range(waves_k, DGRAD_RTC_MAX_KPG // atom + 1, waves_k))


def dgrad_rtc_spills(knobs: dict, KH: int, KW: int, arch: str) -> bool:
    """Would this runtime-channel dgrad binary spill (see
    :data:`DGRAD_RTC_VGPR_FILL`)?"""
    from rocke.core.arch import ArchTarget

    limits = ArchTarget.from_gfx(arch).limits
    waves = knobs["block_groups"] * knobs["waves_q"] * knobs["waves_k"]
    budget = limits.vgprs // -(-waves // 4)  # waves of a block share 4 SIMDs
    return dgrad_rtc_vgprs(knobs, KH, KW) > DGRAD_RTC_VGPR_FILL * budget


def dgrad_rtc_vgprs(knobs: dict, KH: int, KW: int) -> int:
    """Registers per lane the compute pass of a runtime-channel dgrad binary
    holds in its weights (preloading splits only) and accumulators."""
    load_vec = 8 if knobs["fold_k32"] else 4
    acc = (knobs["block_q"] // knobs["waves_q"] // 16) * knobs["m_tiles"] * KH * 4
    if knobs["waves_k"] == 1:
        return acc
    per_wave = knobs["k_atoms"] // knobs["waves_k"]
    return acc + KH * KW * per_wave * knobs["m_tiles"] * (load_vec // 2)


def _knob_grid(
    variant: str, arch: str, dtype: str, caps: Optional[DirectCaps] = None
) -> Iterator[dict]:
    if variant == "direct_nongrouped":
        from kernels.common.conv_direct_nongrouped import (
            nongrouped_knobs,
            nongrouped_split_knobs,
        )

        # Every width (no Wo to pick from): plan_for keeps those that fit.
        yield from nongrouped_knobs(arch, dtype)
        # The split-C rows serve small images with many channels; the dilated
        # rows (DeepLab-style, large maps) never need them.
        if caps is None or (caps.dil_h, caps.dil_w) == (1, 1):
            yield from nongrouped_split_knobs(arch, dtype)
    elif variant == "direct_grouped_halo":
        from kernels.common.conv_direct_nongrouped import grouped_knobs

        yield from grouped_knobs(arch, dtype)
    elif variant == "direct_grouped":
        for bq, bg, db in itertools.product(BLOCK_Q, BLOCK_GROUPS, DOUBLE_BUFFER):
            yield dict(block_q=bq, block_groups=bg, double_buffer=db)
    elif variant == "direct_depthwise":
        for bw, waves in itertools.product(DW_BLOCK_W_FWD, DW_BLOCK_WAVES):
            yield dict(block_w=bw, block_waves=waves)
    elif variant == "direct_depthwise_spatial":
        for waves in DW_BLOCK_WAVES:
            yield dict(block_waves=waves)
    elif variant == "direct_depthwise_tiled":
        for bw, bh, waves, unroll in itertools.product(
            DW_TILED_BLOCK_W, DW_TILED_BLOCK_H, DW_TILED_BLOCK_WAVES, DW_TILED_UNROLL
        ):
            yield dict(block_w=bw, block_h=bh, block_waves=waves, unroll_rows=unroll)
    elif variant == "direct_depthwise_dgrad":
        for bw, waves in itertools.product(DW_BLOCK_W_DGRAD, DW_BLOCK_WAVES):
            yield dict(block_w=bw, block_waves=waves)
    elif variant == "direct_depthwise_col":
        for bh, bw, waves in itertools.product(
            DW_COL_BLOCK_H, DW_COL_BLOCK_W, DW_BLOCK_WAVES
        ):
            yield dict(block_h=bh, block_w=bw, block_waves=waves)
    elif variant == "direct_grouped_dgrad":
        for bq, bg in itertools.product(DGRAD_BLOCK_Q, BLOCK_GROUPS):
            yield dict(block_q=bq, block_groups=bg)
    elif variant == "direct_grouped_dgrad_mfma":
        for bq, bg, bh, (wq, wk, rk, pg, k32) in itertools.product(
            BLOCK_Q, BLOCK_GROUPS, DGRAD_BLOCK_H, DGRAD_WAVES
        ):
            yield dict(
                block_q=bq,
                block_groups=bg,
                block_h=bh,
                waves_q=wq,
                waves_k=wk,
                runtime_k_loop=rk,
                persistent_grid=pg,
                fold_k32=k32,
            )
    elif variant == "direct_dgrad_mfma_rtc":
        for bq, bg, bh, (wq, wk, rk, pg, k32) in itertools.product(
            BLOCK_Q, BLOCK_GROUPS, DGRAD_BLOCK_H, DGRAD_WAVES
        ):
            if wk > 1 and rk:
                # A K split preloads its weights, which takes precedence over
                # the runtime K loop: the binary would duplicate rk=False.
                continue
            if wk > 1 and bg > 1:
                # Rejected by the spec (the LDS reduction is per group).
                continue
            for ka, mt in itertools.product(
                dgrad_rtc_k_atoms(wk, k32), DGRAD_RTC_M_TILES
            ):
                yield dict(
                    block_q=bq,
                    block_groups=bg,
                    block_h=bh,
                    waves_q=wq,
                    waves_k=wk,
                    runtime_k_loop=rk,
                    persistent_grid=pg,
                    fold_k32=k32,
                    k_atoms=ka,
                    m_tiles=mt,
                )
    elif variant == "direct_wgrad":
        for (wk, wc, wq), hpb, mk in itertools.product(
            wgrad_waves(), WGRAD_HPB, WGRAD_MFMA_K
        ):
            yield dict(waves_k=wk, waves_c=wc, waves_q=wq, ho_per_block=hpb, mfma_k=mk)
    else:
        raise ValueError(f"unknown direct conv variant {variant!r}")


# ---------------------------------------------------------------------------
# Spec construction -- the single place a (variant, problem, knobs) triple
# turns into a kernel spec, used by both the build and the run side.
# ---------------------------------------------------------------------------


def make_spec(variant: str, problem, knobs: dict):
    """The spec ``variant`` builds for ``problem``.

    For the MFMA dgrad pipeline this is the spec of its compute pass (a
    forward kernel on the transposed problem); the weight transforms are
    derived from the original problem separately.
    """
    from kernels.common import conv_direct_grouped as dc
    from kernels.common import conv_direct_nongrouped as dn

    if variant == "direct_grouped":
        return dc.DirectConvSpec(problem=problem, name="rocke_direct_conv", **knobs)
    if variant == "direct_nongrouped":
        return dn.DirectNongroupedConvSpec(
            problem=problem, name="rocke_direct_conv_nongrouped", **knobs
        )
    if variant == "direct_grouped_halo":
        return dn.DirectNongroupedConvSpec(
            problem=problem, name="rocke_direct_conv_grouped_halo", **knobs
        )
    if variant == "direct_depthwise":
        return dc.DirectDepthwiseSpec(
            problem=problem, name="rocke_direct_depthwise", **knobs
        )
    if variant == "direct_depthwise_spatial":
        return dc.DirectDepthwiseSpatialSpec(
            problem=problem, name="rocke_direct_depthwise_spatial", **knobs
        )
    if variant == "direct_depthwise_tiled":
        return dc.DirectDepthwiseTiledSpec(
            problem=problem, name="rocke_direct_depthwise_tiled", **knobs
        )
    if variant == "direct_depthwise_col":
        # The col spec carries its own element type; a binary serves only the
        # dtype it was built for, so it follows the problem's.
        return dc.DirectDepthwiseColSpec(
            problem=problem,
            name="rocke_direct_depthwise_col",
            dtype=problem.dtype,
            **knobs,
        )
    if variant == "direct_depthwise_dgrad":
        return dc.DirectDepthwiseDgradStreamSpec(
            problem=problem, name="rocke_direct_dw_dgrad", **knobs
        )
    if variant == "direct_grouped_dgrad":
        return dc.DirectConvDgradSpec(
            problem=problem, name="rocke_direct_dgrad", **knobs
        )
    if variant == "direct_grouped_dgrad_mfma":
        fold_k32 = knobs["fold_k32"]
        spec = dc.make_dgrad_fprop_spec(
            problem, **{k: v for k, v in knobs.items() if k != "fold_k32"}
        )
        return replace(spec, fold_k32=fold_k32) if fold_k32 else spec
    if variant == "direct_dgrad_mfma_rtc":
        fold_k32 = knobs["fold_k32"]
        spec = dc.make_dgrad_fprop_spec(
            problem,
            runtime_channels=True,
            **{k: v for k, v in knobs.items() if k != "fold_k32"},
        )
        return replace(spec, fold_k32=fold_k32) if fold_k32 else spec
    if variant == "direct_wgrad":
        return dc.DirectConvWgradSpec(
            problem=problem,
            name="rocke_direct_conv_wgrad",
            runtime_channels=True,
            **knobs,
        )
    if variant == _SPLIT_REDUCE:
        return dn.DirectSplitReduceSpec(dtype=problem.dtype, **knobs)
    if variant == _TRANSPOSE:
        return dc.DirectTransposeWeightsDgradSpec(problem=problem)
    if variant == _REORGANIZE:
        return dc.DirectReorganizeWeightsSpec(problem=problem, **knobs)
    if variant == _TRANSPOSE_RTC:
        return dc.DirectTransposeWeightsDgradSpec(
            problem=problem, runtime_channels=True
        )
    if variant == _REORGANIZE_RTC:
        return dc.DirectReorganizeWeightsSpec(
            problem=problem, runtime_channels=True, **knobs
        )
    raise ValueError(f"unknown direct conv variant {variant!r}")


def _mfma_knobs_reason(problem, knobs: dict) -> Optional[str]:
    """Constraints of the MFMA dgrad pipeline that its spec cannot see.

    The fprop spec validates K-atoms as ``ceil(cpg/16)``; with ``fold_k32``
    the pass actually runs ``kpg/32`` atoms of 32, and each wave along K
    takes an equal slice, so the split has to divide that count instead.

    With ``waves_k > 1`` the fprop pass preloads its whole weight slice into
    registers and keeps ``KH`` accumulator slots per (Q-subtile, M-tile) on
    top. Past :data:`_MFMA_PRELOAD_VGPR_BUDGET` the kernel sits at the
    512-VGPR ceiling and spills, and comgr's VGPR-to-AGPR spilling there has
    produced a kernel that silently computes the wrong dX (correct at -O1
    and with ``-amdgpu-spill-vgpr-to-agpr=0``). No spec under the budget
    spills, so the cut costs no kernel that runs out of registers cleanly.
    """
    if knobs["fold_k32"]:
        if problem.kpg % 32:
            return f"fold_k32 needs kpg % 32 == 0 (got kpg={problem.kpg})"
        n_atoms = problem.kpg // 32
    else:
        n_atoms = (problem.kpg + 15) // 16
    if n_atoms == 0 or n_atoms % knobs["waves_k"]:
        return f"{n_atoms} K-atoms do not split over waves_k={knobs['waves_k']}"
    if knobs["waves_k"] > 1:
        # The fprop pass runs on the transposed problem: it reduces over this
        # kpg and its M-tiles cover this cpg.
        n_m_tiles = (problem.cpg + 15) // 16
        frag_vgprs = 4 if knobs["fold_k32"] else 2
        weights = (
            problem.KH
            * problem.KW
            * (n_atoms // knobs["waves_k"])
            * n_m_tiles
            * frag_vgprs
        )
        q_subtiles = knobs["block_q"] // knobs.get("waves_q", 1) // 16
        accumulators = q_subtiles * n_m_tiles * problem.KH * 4
        if weights + accumulators > _MFMA_PRELOAD_VGPR_BUDGET:
            return (
                f"preloaded weights ({weights}) + accumulators ({accumulators}) "
                f"VGPRs > {_MFMA_PRELOAD_VGPR_BUDGET}: the kernel would spill"
            )
    return None


# Register budget for the MFMA dgrad fprop pass's preloaded weights and
# accumulators (see _mfma_knobs_reason): three quarters of the 512-VGPR file,
# leaving the rest for the LDS fragments and addressing.
_MFMA_PRELOAD_VGPR_BUDGET = 384


def _uses_coalesced_weights(knobs: dict) -> bool:
    """Does the MFMA dgrad's fprop pass read the reorganized weight layout?

    Only when K is split over waves or walked by the runtime K loop; otherwise
    it reads the plain transposed weights and the reorganize pass is skipped.
    """
    return knobs["waves_k"] > 1 or knobs["runtime_k_loop"]


def _nongrouped_spills(spec, arch: str) -> bool:
    """Would this non-grouped kernel spill registers? Such a binary pays scratch
    traffic every channel chunk and is the slowest to compile, so it is not
    cached.
    Its register footprint is fixed by the baked capabilities and the tile, not
    by the probe shape, so one answer holds for every problem it would serve."""
    from kernels.common.conv_direct_nongrouped import nongrouped_register_reason

    return nongrouped_register_reason(spec, arch) is not None


def _depthwise_stream_spills(spec) -> bool:
    """Would this row-streaming depthwise kernel spill registers? Such a binary
    takes minutes to compile and runs far slower than the output-stationary
    kernel the grid also builds for the shape, so it is not cached."""
    from kernels.common.conv_direct_grouped import depthwise_stream_register_reason

    return depthwise_stream_register_reason(spec) is not None


def validate_spec(variant: str, spec, arch: str) -> Tuple[bool, str]:
    """``(ok, reason)`` for ``spec`` on ``arch``, from the kernel's own validator."""
    from kernels.common import conv_direct_grouped as dc
    from kernels.common import conv_direct_nongrouped as dn

    try:
        if variant in _HALO_VARIANTS:
            return dn.is_valid_nongrouped_spec(spec, arch=arch)
        if variant in ("direct_grouped",) + _MFMA_DGRAD:
            spec.validate()
            return dc.is_valid_spec(spec, arch=arch)
        if variant == "direct_depthwise":
            return dc.is_valid_depthwise_spec(spec, arch=arch)
        if variant == "direct_depthwise_spatial":
            return dc.is_valid_depthwise_spatial_spec(spec, arch=arch)
        if variant == "direct_depthwise_tiled":
            return dc.is_valid_depthwise_tiled_spec(spec, arch=arch)
        if variant == "direct_depthwise_col":
            spec.validate()
            return dc.is_valid_depthwise_col_spec(spec, arch=arch)
        if variant == "direct_depthwise_dgrad":
            return dc.is_valid_depthwise_dgrad_stream_spec(spec, arch=arch)
        if variant == "direct_grouped_dgrad":
            return dc.is_valid_dgrad_spec(spec, arch=arch)
        if variant == "direct_wgrad":
            spec.validate()
            return dc.is_valid_wgrad_spec(spec, arch=arch)
        if variant in (_TRANSPOSE, _REORGANIZE, _TRANSPOSE_RTC, _REORGANIZE_RTC):
            return True, "ok"
        if variant == _SPLIT_REDUCE:
            spec.validate()
            return True, "ok"
    except ValueError as e:
        return False, str(e)
    raise ValueError(f"unknown direct conv variant {variant!r}")


def _build_kernel(variant: str, spec, arch: str):
    from kernels.common import conv_direct_grouped as dc
    from kernels.common import conv_direct_nongrouped as dn

    builders = {
        "direct_grouped": dc.build_direct_conv,
        "direct_nongrouped": dn.build_direct_conv_nongrouped,
        "direct_grouped_halo": dn.build_direct_conv_nongrouped,
        "direct_grouped_dgrad_mfma": dc.build_direct_conv,
        "direct_dgrad_mfma_rtc": dc.build_direct_conv,
        "direct_depthwise": dc.build_direct_depthwise,
        "direct_depthwise_spatial": dc.build_direct_depthwise_spatial,
        "direct_depthwise_tiled": dc.build_direct_depthwise_tiled,
        "direct_depthwise_col": dc.build_direct_depthwise_col,
        "direct_depthwise_dgrad": dc.build_direct_depthwise_dgrad_streaming,
        "direct_grouped_dgrad": dc.build_direct_conv_dgrad,
        "direct_wgrad": dc.build_direct_conv_wgrad,
        _TRANSPOSE: dc.build_direct_transpose_weights_dgrad,
        _REORGANIZE: dc.build_direct_reorganize_weights,
        _TRANSPOSE_RTC: dc.build_direct_transpose_weights_dgrad,
        _REORGANIZE_RTC: dc.build_direct_reorganize_weights,
        _SPLIT_REDUCE: dn.build_direct_conv_split_reduce,
    }
    return builders[variant](spec, arch=arch)


# ---------------------------------------------------------------------------
# Identities and jobs
# ---------------------------------------------------------------------------


def _identity(
    arch: str,
    wave_size: int,
    variant: str,
    direction: str,
    caps: DirectCaps,
    knobs: dict,
    dtype: str = DIRECT_DTYPE,
) -> KernelIdentity:
    return KernelIdentity(
        arch=arch,
        direction=direction,
        algorithm=variant,
        dtype_a=dtype,
        dtype_b=dtype,
        dtype_d=dtype,
        # Direct conv is not a GEMM: no tile, warp or vector fields.
        tile_m=0,
        tile_n=0,
        tile_k=0,
        warp_m=0,
        warp_n=0,
        warp_tile_m=0,
        warp_tile_n=0,
        warp_tile_k=0,
        pipeline="direct",
        epilogue="direct",
        wave_size=wave_size,
        vector_size_a=0,
        vector_size_b=0,
        vector_size_c=0,
        filter_h=caps.KH,
        filter_w=caps.KW,
        stride_h=caps.stride,
        stride_w=caps.stride,
        dilation_h=caps.dil_h,
        dilation_w=caps.dil_w,
        pad_h=caps.PAD,
        pad_w=caps.PAD,
        cpg=caps.cpg,
        kpg=caps.kpg,
        knobs=json.dumps(knobs, sort_keys=True),
    )


def _helper_identity(
    arch: str,
    wave_size: int,
    variant: str,
    caps: DirectCaps,
    knobs: dict,
    dtype: str = DIRECT_DTYPE,
) -> KernelIdentity:
    # The weight transforms depend on the filter and channel counts only;
    # stride and padding are zeroed so every pipeline entry that shares a
    # filter shares the helper binary.
    return _identity(
        arch,
        wave_size,
        variant,
        "direct_dgrad_helper",
        replace(caps, PAD=0, stride=0),
        knobs,
        dtype,
    )


def _split_reduce_identity(
    arch: str, wave_size: int, split_c: int, dtype: str = DIRECT_DTYPE
) -> KernelIdentity:
    return _identity(
        arch,
        wave_size,
        _SPLIT_REDUCE,
        _FWD_HELPER,
        _NO_CAPS,
        {"split_c": split_c},
        dtype,
    )


def _split_reduce_job(
    arch: str, wave_size: int, split_c: int, dtype: str = DIRECT_DTYPE
) -> BuildJob:
    return _job(
        _split_reduce_identity(arch, wave_size, split_c, dtype),
        _NO_CAPS,
        {"split_c": split_c},
    )


def _caps_of(identity: KernelIdentity) -> DirectCaps:
    return DirectCaps(
        KH=identity.filter_h,
        PAD=identity.pad_h,
        stride=identity.stride_h,
        cpg=identity.cpg,
        kpg=identity.kpg,
        dil_h=identity.dilation_h,
        dil_w=identity.dilation_w,
    )


def probe_problem(
    caps: DirectCaps,
    dtype: str = DIRECT_DTYPE,
    variant: Optional[str] = None,
    knobs: Optional[dict] = None,
):
    """A problem with exactly ``caps`` that every knob in the grid fits.

    The emitted IR does not depend on N, H, W or groups (the shape-invariance
    cases in test_conv_abi.py assert it), so the probe only has to satisfy the
    spec validators: 64 groups divide every block_groups, 256 depthwise
    channels fill the widest depthwise block (the spatial depthwise kernel
    takes fewer groups than a wave, so it gets 8; the dense MFMA dgrad channel
    counts serve groups == 1, so they get 1), and 64x64 leaves every
    stride-2 output wider than the widest W tile. Runtime channel counts
    (``cpg == 0``: the non-grouped forward kernel and wgrad) take one group of
    256: a multiple of every channel chunk and of every channel tile, and as
    many 16-channel tiles as the widest wgrad ``waves_c`` -- 1024 for the
    non-grouped kernel, whose split-C rows need a chunk per split.

    The runtime-channel MFMA dgrad bakes its channel *tile counts* instead,
    so its probe takes them from ``knobs``: ``k_atoms`` full K-atoms of dY
    channels and ``m_tiles`` full 16-channel tiles of dX channels, over 16
    groups (a multiple of every block_groups).
    """
    from kernels.common.conv_direct_grouped import DirectConvProblem

    if variant == "direct_dgrad_mfma_rtc":
        atom = 32 if knobs["fold_k32"] else 16
        return DirectConvProblem(
            N=1,
            H=64,
            W=64,
            groups=16,
            cpg=knobs["m_tiles"] * 16,
            kpg=knobs["k_atoms"] * atom,
            KH=caps.KH,
            KW=caps.KW,
            PAD=caps.PAD,
            stride=caps.stride,
            dtype=dtype,
        )
    if variant == "direct_grouped_halo":
        # Runtime groups and channels: 64 channels per group are a multiple of
        # every chunk and channel tile at every merge degree, and 16 groups a
        # multiple of every degree.
        return DirectConvProblem(
            N=1,
            H=64,
            W=64,
            groups=16,
            cpg=64,
            kpg=64,
            KH=caps.KH,
            KW=caps.KW,
            PAD=caps.PAD,
            stride=caps.stride,
            dtype=dtype,
        )
    if caps.cpg == 0:
        # The non-grouped kernel's split-C rows need a whole chunk per split:
        # 1024 channels cover the deepest split (16 x ck 64).
        channels = 1024 if variant == "direct_nongrouped" else 256
        return DirectConvProblem(
            N=1,
            H=64,
            W=64,
            groups=1,
            cpg=channels,
            kpg=channels,
            KH=caps.KH,
            KW=caps.KW,
            PAD=caps.PAD,
            stride=caps.stride,
            dtype=dtype,
            dil_h=caps.dil_h,
            dil_w=caps.dil_w,
        )
    if variant == "direct_depthwise_spatial":
        groups = 8
    elif (caps.cpg, caps.kpg) in _DENSE_MFMA_DGRAD_CHANNELS:
        # Dense channel counts serve groups == 1: one group per block. Wider
        # blocks would only build kernels no problem with these counts can
        # launch (and some of them fail to compile).
        groups = 1
    else:
        groups = 256 if caps.cpg == 1 else 64
    return DirectConvProblem(
        N=1,
        H=64,
        W=64,
        groups=groups,
        cpg=caps.cpg,
        kpg=caps.kpg,
        KH=caps.KH,
        KW=caps.KW,
        PAD=caps.PAD,
        stride=caps.stride,
        dtype=dtype,
        dil_h=caps.dil_h,
        dil_w=caps.dil_w,
    )


def _job(identity: KernelIdentity, caps: DirectCaps, knobs: dict) -> BuildJob:
    return BuildJob(
        identity=identity,
        direction=identity.direction,
        spec_kwargs=knobs,
        caps=dict(
            KH=caps.KH,
            PAD=caps.PAD,
            stride=caps.stride,
            cpg=caps.cpg,
            kpg=caps.kpg,
            dil_h=caps.dil_h,
            dil_w=caps.dil_w,
        ),
    )


def _caps_jobs(
    arch: str,
    wave_size: int,
    variant: str,
    direction: str,
    caps: DirectCaps,
    dtype: str,
) -> Tuple[int, List[Tuple[str, BuildJob]]]:
    """Walk one (variant, capability) cell of the grid.

    Returns the raw candidate count and the valid ``(key, job)`` pairs in
    generator order; duplicates are resolved by the caller's ordered merge.
    The MFMA dgrad entries pull in their weight-transform helpers.
    """
    out: List[Tuple[str, BuildJob]] = []
    n_raw = 0
    rtc_dgrad = variant == "direct_dgrad_mfma_rtc"
    # The runtime-channel dgrad probe follows each knob's tile counts.
    probe = None if rtc_dgrad else probe_problem(caps, dtype, variant)
    for knobs in _knob_grid(variant, arch, dtype, caps):
        n_raw += 1
        if rtc_dgrad:
            probe = probe_problem(caps, dtype, variant, knobs)
            if dgrad_rtc_spills(knobs, caps.KH, caps.KW, arch):
                continue
        if variant in _MFMA_DGRAD and _mfma_knobs_reason(probe, knobs):
            continue
        spec = make_spec(variant, probe, knobs)
        if not validate_spec(variant, spec, arch)[0]:
            continue
        if variant in _HALO_VARIANTS and _nongrouped_spills(spec, arch):
            continue
        if variant == "direct_depthwise" and _depthwise_stream_spills(spec):
            continue
        cell = [
            _job(
                _identity(
                    arch, wave_size, variant, f"direct_{direction}", caps, knobs, dtype
                ),
                caps,
                knobs,
            )
        ]
        if variant == "direct_nongrouped" and knobs.get("split_c", 1) > 1:
            cell.append(_split_reduce_job(arch, wave_size, knobs["split_c"], dtype))
        if variant in _MFMA_DGRAD:
            transpose, reorganize = _MFMA_HELPERS[variant]
            cell.append(
                _job(
                    _helper_identity(arch, wave_size, transpose, caps, {}, dtype),
                    caps,
                    {},
                )
            )
            if _uses_coalesced_weights(knobs):
                reorg = {"fold_k32": knobs["fold_k32"]}
                cell.append(
                    _job(
                        _helper_identity(
                            arch, wave_size, reorganize, caps, reorg, dtype
                        ),
                        caps,
                        reorg,
                    )
                )
        out.extend((job.identity.stable_hash(), job) for job in cell)
    return n_raw, out


def _caps_worker(payload):
    return _caps_jobs(*payload)


def count_direct_jobs(
    directions: Sequence[str], arch: str, dtype: str = DIRECT_DTYPE
) -> int:
    """Raw candidate count :func:`enumerate_direct_jobs` walks (the progress
    denominator): capabilities times tuning knobs, before validation."""
    return sum(
        sum(1 for _ in _knob_grid(variant, arch, dtype, caps))
        for variant, (direction, caps_list) in DIRECT_CAPABILITIES.items()
        if direction in directions
        for caps in caps_list
    )


def enumerate_direct_jobs(
    *,
    arch: str,
    target,
    directions: Sequence[str],
    dtype: str = DIRECT_DTYPE,
    jobs: int = 1,
    log=None,
    log_every_s: float = 5.0,
) -> List[BuildJob]:
    """Every direct kernel worth caching for ``arch``, pre-validated.

    ``directions`` takes the benchmark's names, ``fwd``, ``dgrad`` and
    ``wgrad``. The MFMA dgrad entries pull in their weight-transform helpers.

    The grid is walked one (variant, capability) cell at a time; with
    ``jobs > 1`` the cells run in a process pool. Cells are merged in order,
    so the result (including its order) does not depend on ``jobs``. With
    ``log`` set a progress line is emitted at most every ``log_every_s``
    seconds, as for the implicit-GEMM enumeration.
    """
    for d in directions:
        if d not in DIRECT_DIRECTIONS:
            raise ValueError(
                f"direct conv has no {d!r} kernel; expected one of {DIRECT_DIRECTIONS}"
            )
    if dtype not in DIRECT_DTYPES:
        raise ValueError(f"direct conv builds {DIRECT_DTYPES}, not {dtype!r}")
    wave_size = target.wave_size
    jobs = max(1, int(jobs))
    payloads = [
        (arch, wave_size, variant, direction, caps, dtype)
        for variant, (direction, caps_list) in DIRECT_CAPABILITIES.items()
        if direction in directions
        for caps in caps_list
    ]

    total = 0
    if log is not None:
        total = count_direct_jobs(directions, arch, dtype)
        log(f"  {total} candidates to check with {jobs} process(es)")

    n_raw = n_valid = 0
    started = last_log = time.perf_counter()
    results: List[Optional[List[Tuple[str, BuildJob]]]] = [None] * len(payloads)

    def _done(idx: int, cell_raw: int, cell_out) -> None:
        nonlocal n_raw, n_valid, last_log
        results[idx] = cell_out
        n_raw += cell_raw
        n_valid += len(cell_out)
        if log is None:
            return
        now = time.perf_counter()
        if now - last_log >= log_every_s:
            last_log = now
            pct = 100.0 * n_raw / total if total else 0.0
            log(
                f"  enumerating: {n_raw}/{total} candidates checked "
                f"({pct:.1f}%), ~{n_valid} valid ({now - started:.0f}s)"
            )

    if jobs <= 1:
        for idx, payload in enumerate(payloads):
            _done(idx, *_caps_jobs(*payload))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(_caps_worker, payload): idx
                for idx, payload in enumerate(payloads)
            }
            for fut in as_completed(futures):
                _done(futures[fut], *fut.result())

    seen = set()
    per_direction: Dict[str, List[BuildJob]] = {d: [] for d in directions}
    for payload, cell_out in zip(payloads, results):
        for key, job in cell_out:
            if key not in seen:
                seen.add(key)
                per_direction[payload[3]].append(job)
    # Round-robin across directions, as the implicit-GEMM enumeration does, so
    # a --limit smoke build covers every requested direction.
    out: List[BuildJob] = []
    streams = [iter(per_direction[d]) for d in directions]
    while streams:
        still_running = []
        for stream in streams:
            job = next(stream, None)
            if job is not None:
                out.append(job)
                still_running.append(stream)
        streams = still_running
    if log is not None:
        log(
            f"  enumerated {n_raw} candidates -> {len(out)} unique valid "
            f"({time.perf_counter() - started:.0f}s)"
        )
    return out


@dataclass(frozen=True)
class _DtypeOnly:
    """The only problem field a shape-independent helper's spec reads."""

    dtype: str


def build_direct_job(job: BuildJob, arch: str, dtype: str):
    """Build one direct job's kernel IR. Returns ``(kernel, meta)``.

    Compilation, deduplication and incremental rebuilds are
    :func:`compile_jobs`'s; this only decides what the kernel is.
    """
    if dtype not in DIRECT_DTYPES:
        raise ValueError(f"direct conv builds {DIRECT_DTYPES}, not {dtype!r}")
    caps = DirectCaps(**job.caps)
    variant = job.identity.algorithm
    if variant == _SPLIT_REDUCE:
        spec = make_spec(variant, _DtypeOnly(dtype), job.spec_kwargs)
        return _build_kernel(variant, spec, arch), {}
    probe = probe_problem(caps, dtype, variant, job.spec_kwargs)
    spec = make_spec(variant, probe, job.spec_kwargs)
    ok, why = validate_spec(variant, spec, arch)
    if not ok:
        raise ValueError(why)
    return _build_kernel(variant, spec, arch), {}


def direct_job_for_identity(identity: KernelIdentity) -> BuildJob:
    """The job ``--compile-all`` builds the direct identity ``identity`` from.

    A direct identity carries its capabilities and knobs verbatim, so the job
    is read straight off it -- and then checked: the identity those caps and
    knobs produce has to equal ``identity``, or the grid cannot build it (a
    non-square filter, say) and ``LookupError`` is raised.
    """
    if not identity.is_direct:
        raise LookupError(f"not a direct conv identity: {identity.algorithm}")
    if identity.dtype_a not in DIRECT_DTYPES:
        raise LookupError(
            f"direct conv builds {DIRECT_DTYPES}, not {identity.dtype_a!r}"
        )
    caps = _caps_of(identity)
    knobs = json.loads(identity.knobs) if identity.knobs else {}
    if identity.direction == _FWD_HELPER:
        split_c = knobs.get("split_c")
        rebuilt = (
            _split_reduce_identity(
                identity.arch, identity.wave_size, split_c, identity.dtype_a
            )
            if identity.algorithm == _SPLIT_REDUCE and isinstance(split_c, int)
            else None
        )
        if (
            rebuilt is None
            or replace(rebuilt, llvm_flavor=identity.llvm_flavor) != identity
        ):
            raise LookupError(
                f"the direct grid cannot build this identity: {identity.label()}"
            )
        return _job(identity, _NO_CAPS, {"split_c": split_c})
    if identity.direction == "direct_dgrad_helper":
        rebuilt = _helper_identity(
            identity.arch,
            identity.wave_size,
            identity.algorithm,
            caps,
            knobs,
            identity.dtype_a,
        )
    else:
        rebuilt = _identity(
            identity.arch,
            identity.wave_size,
            identity.algorithm,
            identity.direction,
            caps,
            knobs,
            identity.dtype_a,
        )
    rebuilt = replace(rebuilt, llvm_flavor=identity.llvm_flavor)
    if rebuilt != identity:
        want, got = identity.to_dict(), rebuilt.to_dict()
        diff = {k: want[k] for k in want if want[k] != got[k]}
        raise LookupError(f"the direct grid cannot build this identity: {diff}")
    if identity.direction != "direct_dgrad_helper":
        # The fields can be consistent and still name a kernel no variant
        # computes (a dilated grouped kernel, say): ask the kernel's validator.
        variant = identity.algorithm
        probe = probe_problem(caps, identity.dtype_a, variant, knobs)
        spec = make_spec(variant, probe, knobs)
        ok, why = validate_spec(variant, spec, identity.arch)
        if not ok:
            raise LookupError(f"the direct grid cannot build this identity: {why}")
    if identity.direction == "direct_dgrad_helper":
        # A helper's identity zeroes the padding and stride (its code does
        # not read them), but the grid builds it with the capabilities of the
        # MFMA dgrad row that pulled it in -- the one row, baked or runtime
        # channels (cpg == kpg == 0), with its filter and channel counts.
        caps = next(
            (
                row
                for variant in _MFMA_DGRAD
                for row in DIRECT_CAPABILITIES[variant][1]
                if (row.KH, row.cpg, row.kpg) == (caps.KH, caps.cpg, caps.kpg)
            ),
            caps,
        )
    return _job(identity, caps, knobs)


def compile_all_direct(
    *,
    cache: KernelCache,
    arch: str,
    target,
    directions: Sequence[str],
    jobs: int = 1,
    limit: Optional[int] = None,
    log=print,
    dtype: str = DIRECT_DTYPE,
) -> int:
    """Populate ``cache`` with every direct kernel in :data:`DIRECT_CAPABILITIES`
    for operand dtype ``dtype``."""
    log(
        f"AOT compile-all: enumerating direct variants for {arch}/{dtype} "
        f"({', '.join(directions)})"
    )
    all_jobs = enumerate_direct_jobs(
        arch=arch,
        target=target,
        directions=directions,
        dtype=dtype,
        jobs=jobs,
        log=log,
    )
    rc = compile_jobs(
        cache=cache,
        all_jobs=all_jobs,
        build=build_direct_job,
        arch=arch,
        dtype=dtype,
        directions=[f"direct_{d}" for d in directions],
        jobs=jobs,
        limit=limit,
        log=log,
    )
    if "fwd" in directions:
        prune_halo_entries(cache, all_jobs, log=log)
    return rc


def _elf_code_facts(hsaco: bytes) -> Optional[Tuple[bytes, int]]:
    """``(code digest, scratch bytes)`` of an ELF64 code object, or None for
    anything else.

    The code is the ``.text`` section followed by every kernel descriptor
    (``<kernel>.kd``): what the GPU executes and how it is launched, so two
    binaries with equal digests run identically whatever their symbol names.
    Only its SHA-256 is kept, so a sweep's worth of facts stays small.
    ``scratch`` is the largest descriptor's private segment size -- non-zero
    when the kernel spills.
    """
    if hsaco[:5] != b"\x7fELF\x02" or len(hsaco) < 64:
        return None
    (shoff,) = struct.unpack_from("<Q", hsaco, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", hsaco, 0x3A)
    secs = [
        struct.unpack_from("<IIQQQQII", hsaco, shoff + i * shentsize)
        for i in range(shnum)
    ]
    # (name, type, flags, addr, offset, size, link, info)
    strtab_off = secs[shstrndx][4]

    def cstr(off: int) -> str:
        return hsaco[off : hsaco.index(b"\0", off)].decode()

    text = b""
    kds = []
    for name_off, sh_type, _, _, off, size, link, _ in secs:
        if cstr(strtab_off + name_off) == ".text":
            text = hsaco[off : off + size]
        if sh_type != 2:  # SHT_SYMTAB
            continue
        names = secs[link][4]
        for sym in range(off, off + size, 24):
            st_name, _, _, shndx, value, st_size = struct.unpack_from(
                "<IBBHQQ", hsaco, sym
            )
            if not cstr(names + st_name).endswith(".kd") or shndx >= shnum:
                continue
            _, _, _, addr, sec_off, _, _, _ = secs[shndx]
            at = sec_off + value - addr
            kd = hsaco[at : at + st_size]
            # Bytes 16..24 locate the entry point relative to the descriptor:
            # file layout, not behaviour, so they are left out.
            kds.append(kd[:16] + kd[24:])
    if not kds:
        return None
    scratch = max(struct.unpack_from("<I", kd, 4)[0] for kd in kds)
    return hashlib.sha256(text + b"".join(sorted(kds))).digest(), scratch


def prune_halo_entries(cache: KernelCache, jobs: Sequence[BuildJob], log=print) -> int:
    """Drop the cached grouped-halo and split-C forward entries no sweep should
    be offered (the dense unsplit grid is left as it is).

    Two kinds, both only visible in the compiled code: binaries that spill to
    scratch (never the best of their shape), and ``waves_per_eu=3`` binaries
    whose code equals their ``waves_per_eu=None`` twin's -- the hint changed
    nothing, so the twin already is that kernel. The binaries stay in the blob
    store: a later ``--compile-all`` re-links those entries without compiling
    and this drops them again. Returns the number of entries dropped.
    """
    wanted = {
        j.identity
        for j in jobs
        if j.identity.direction == "direct_fwd"
        and (
            j.identity.algorithm == "direct_grouped_halo"
            or (
                j.identity.algorithm == "direct_nongrouped"
                and json.loads(j.identity.knobs).get("split_c", 1) > 1
            )
        )
    }
    present = {
        ident: path for ident, path in cache.list_all("direct_fwd") if ident in wanted
    }
    facts = {
        ident: _elf_code_facts(path.read_bytes()) for ident, path in present.items()
    }
    drop: Dict[KernelIdentity, str] = {}
    twins: Dict[KernelIdentity, Dict[object, KernelIdentity]] = {}
    for ident, fact in facts.items():
        if fact is not None and fact[1] > 0:
            drop[ident] = "spills"
        knobs = json.loads(ident.knobs)
        we = knobs.pop("waves_per_eu", None)
        key = replace(ident, knobs=json.dumps(knobs, sort_keys=True))
        twins.setdefault(key, {})[we] = ident
    for pair in twins.values():
        hinted, plain = pair.get(3), pair.get(None)
        if hinted is None or plain is None or hinted in drop:
            continue
        a, b = facts[hinted], facts[plain]
        if a is not None and b is not None and a[0] == b[0]:
            drop[hinted] = "same code as waves_per_eu=None"
    for ident in drop:
        cache.discard(ident)
    if drop:
        reasons = collections.Counter(drop.values())
        log(
            f"AOT compile-all: dropped {len(drop)} of {len(present)} grouped-halo "
            "and split-C forward entries ("
            + ", ".join(f"{n} {why}" for why, n in sorted(reasons.items()))
            + ")"
        )
    return len(drop)


# ---------------------------------------------------------------------------
# Running a problem out of the cache (CPU half)
# ---------------------------------------------------------------------------


def _transform_signature(dtype: str, runtime_channels: bool = False) -> list:
    """Launch signature of the weight-transform kernels: source, destination
    and their byte sizes, then -- for the runtime-channel ones -- the
    per-group channel counts."""
    from kernels.common.conv_direct_grouped import DIRECT_WEIGHT_TRANSFORM_RTC_ARGS

    elem = {"fp16": "f16", "bf16": "bf16"}[dtype]
    sig = [
        {"name": "A", "type": f"ptr<{elem}, global>", "size_bytes": 8},
        {"name": "D", "type": f"ptr<{elem}, global>", "size_bytes": 8},
        {"name": "A_bytes", "type": "i32", "size_bytes": 4},
        {"name": "D_bytes", "type": "i32", "size_bytes": 4},
    ]
    if runtime_channels:
        sig += [
            {"name": n, "type": "i32", "size_bytes": 4}
            for n in DIRECT_WEIGHT_TRANSFORM_RTC_ARGS
        ]
    return sig


@dataclass(frozen=True)
class DirectStep:
    """One kernel launch of a plan.

    ``buffers`` names the buffers bound to the A, B and D kernarg slots (B is
    ``None`` for the two-operand weight transforms). ``conv_args`` is ``None``
    for those too; a conv kernel takes its extents from it. ``scalars`` are
    the transforms' remaining kernargs (name, value), if any.
    """

    kernel_name: str
    hsaco_path: Path
    signature: list
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    buffers: Tuple[str, Optional[str], str]
    conv_args: object = None
    scalars: Tuple[Tuple[str, int], ...] = ()
    # Per A/B/D slot, ``(byte offset, byte size)`` of the window of its
    # buffer the step binds, or None for the whole buffer: a batch-split
    # launch sees only its images (see _batch_chunks).
    views: Tuple[Optional[Tuple[int, int]], ...] = ()


@dataclass(frozen=True)
class DirectPlan:
    """Everything needed to run one cached kernel (or pipeline) on a problem.

    Buffer names: ``x``/``w``/``y`` for forward, ``dy``/``w``/``dx`` for
    dgrad, ``dy``/``x``/``dw`` (fp32) for wgrad, plus the scratch buffers in
    ``workspaces`` (name -> bytes).
    """

    identity: KernelIdentity
    steps: Tuple[DirectStep, ...]
    workspaces: Dict[str, int] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.identity.label()


def _conv_step(identity, hsaco_path, meta, spec, problem, direction, buffers):
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_grouped import direct_launch_geometry
    from kernels.common.conv_direct_nongrouped import DirectNongroupedConvSpec

    if isinstance(spec, DirectNongroupedConvSpec):
        grid, block = spec.grid(), (spec.threads_per_block, 1, 1)
    else:
        grid, block = direct_launch_geometry(spec)
    return DirectStep(
        kernel_name=meta["kernel_name"],
        hsaco_path=hsaco_path,
        signature=conv_direct_args_signature(identity.dtype_a, direction=direction),
        grid=grid,
        block=block,
        buffers=buffers,
        conv_args=ConvArgs.from_problem(problem, direction=direction),
    )


def _batch_chunks(problem) -> Optional[List[int]]:
    """Images per launch for a forward problem the LDS-halo kernels can only
    run in pieces, or None when one launch serves it.

    Those kernels address every operand through one 32-bit buffer resource and
    redirect masked accesses past ``_OOB_BASE``, so no operand may reach it
    (a 768x768x256 bf16 batch of 8 is 2.4 GB). The images are independent, so
    the batch is split into launches whose activation and output windows each
    fit; the chunks are balanced rather than greedy.
    """
    from kernels.common.conv_direct_nongrouped import _OOB_BASE

    in_img = 2 * problem.H * problem.W * problem.total_c
    out_img = 2 * problem.Ho * problem.Wo * problem.total_k
    per_launch = _OOB_BASE // max(in_img, out_img)
    if problem.N <= per_launch or per_launch < 1:
        return None
    n_steps = -(-problem.N // per_launch)
    base, extra = divmod(problem.N, n_steps)
    return [base + (i < extra) for i in range(n_steps)]


def plan_for(
    cache: KernelCache,
    identity: KernelIdentity,
    hsaco_path: Path,
    meta: dict,
    problem,
    arch: str,
) -> Tuple[Optional[DirectPlan], str]:
    """Launch plan for one cached kernel on ``problem``, or ``(None, why)``.

    The capability match (:meth:`KernelCache.supports_problem`) is necessary
    but not sufficient: tuning knobs have shape rules of their own (groups
    divisible by block_groups, a persistent grid's cell budget, ...). The
    spec is rebuilt on the real problem and run through the kernel's own
    validator, so a binary is only offered where a fresh build would accept
    the shape too.
    """
    from kernels.common import conv_direct_grouped as dc

    if not meta.get("kernel_name"):
        return None, "cache entry has no kernel_name (rebuild the cache)"
    variant = identity.algorithm
    knobs = json.loads(identity.knobs) if identity.knobs else {}
    if variant in _MFMA_DGRAD:
        why = _mfma_knobs_reason(problem, knobs)
        if why:
            return None, why
    chunks = None
    if variant in _HALO_VARIANTS and identity.direction == "direct_fwd":
        chunks = _batch_chunks(problem)
        if chunks is not None:
            if knobs.get("split_c", 1) > 1:
                return None, "a batch-split launch does not take split-C kernels"
            return _batch_split_plan(
                cache, identity, hsaco_path, meta, problem, arch, knobs, chunks
            )
    spec = make_spec(variant, problem, knobs)
    ok, why = validate_spec(variant, spec, arch)
    if not ok:
        return None, why
    if variant in _HALO_VARIANTS:
        # The cache holds every width; offer only those a sweep for this
        # shape would try, so no binary spends MFMA work past the image edge
        # when one that does not is cached.
        from kernels.common.conv_direct_nongrouped import tile_w_candidates

        widths = tile_w_candidates(problem.Wo, spec.atom_tile)
        if spec.tile_w not in widths:
            return None, (
                f"tile_w={spec.tile_w} does not fit Wo={problem.Wo} "
                f"(widths tried for it: {widths})"
            )
    if variant == "direct_grouped_halo" and spec.tile_k > spec.merged_k:
        return None, (
            f"tile_k={spec.tile_k} exceeds the {spec.merged_k} output channels "
            f"of a merged block"
        )

    if variant == "direct_wgrad":
        # As the JIT sweep: no C tile wider than the problem's channels, so no
        # wave stages and multiplies only channels the epilogue then drops.
        c_tiles = -(-problem.cpg // spec.wave_tile_c)
        if spec.waves_c > c_tiles:
            return None, (
                f"waves_c={spec.waves_c} exceeds the {c_tiles} channel tile(s) "
                f"of cpg={problem.cpg}"
            )

    if identity.direction == "direct_fwd" and getattr(spec, "split_c", 1) > 1:
        # Split-C: the conv pass writes fp32 partial sums into a workspace,
        # and the reduction folds them into y. Both take the forward ABI.
        from kernels.common.conv_args import ConvArgs

        reduce_id = _split_reduce_identity(
            identity.arch, identity.wave_size, spec.split_c, identity.dtype_a
        )
        entry = cache.get(reduce_id)
        if entry is None:
            return None, f"missing cached {_SPLIT_REDUCE} helper (rebuild the cache)"
        reduce_spec = make_spec(_SPLIT_REDUCE, problem, {"split_c": spec.split_c})
        conv = _conv_step(
            identity, hsaco_path, meta, spec, problem, "fwd", ("x", "w", "ws_split")
        )
        fold = DirectStep(
            kernel_name=entry[1]["kernel_name"],
            hsaco_path=cache.hsaco_path(reduce_id),
            signature=conv.signature,
            grid=reduce_spec.grid(problem),
            block=(reduce_spec.threads, 1, 1),
            buffers=("ws_split", "w", "y"),
            conv_args=ConvArgs.from_problem(problem, direction="fwd"),
        )
        return (
            DirectPlan(identity, (conv, fold), {"ws_split": spec.workspace_bytes}),
            "ok",
        )
    if identity.direction == "direct_fwd":
        step = _conv_step(
            identity, hsaco_path, meta, spec, problem, "fwd", ("x", "w", "y")
        )
        return DirectPlan(identity, (step,)), "ok"
    if identity.direction == "direct_wgrad":
        # dY rides A, X rides B, and D is the fp32 dW the kernel accumulates
        # into atomically (the caller zeroes it).
        step = _conv_step(
            identity, hsaco_path, meta, spec, problem, "wgrad", ("dy", "x", "dw")
        )
        return DirectPlan(identity, (step,)), "ok"
    if variant not in _MFMA_DGRAD:
        step = _conv_step(
            identity, hsaco_path, meta, spec, problem, "dgrad", ("dy", "w", "dx")
        )
        return DirectPlan(identity, (step,)), "ok"

    # MFMA pipeline: W -> W_T (transpose), optionally W_T -> W_coa
    # (reorganize), then a forward pass over the transposed problem.
    caps = _caps_of(identity)
    transpose, reorganize = _MFMA_HELPERS[variant]
    runtime_channels = variant == "direct_dgrad_mfma_rtc"
    helpers = [(transpose, {}, "w", "ws_t")]
    use_coalesced = _uses_coalesced_weights(knobs)
    if use_coalesced:
        helpers.append((reorganize, {"fold_k32": knobs["fold_k32"]}, "ws_t", "ws_coa"))
    # The runtime-channel transforms take the original problem's per-group
    # channel counts as kernargs.
    scalars = (
        tuple(
            zip(
                dc.DIRECT_WEIGHT_TRANSFORM_RTC_ARGS,
                (problem.cpg, problem.kpg),
            )
        )
        if runtime_channels
        else ()
    )
    steps = []
    for helper, helper_knobs, src, dst in helpers:
        helper_id = _helper_identity(
            identity.arch,
            identity.wave_size,
            helper,
            caps,
            helper_knobs,
            identity.dtype_a,
        )
        entry = cache.get(helper_id)
        if entry is None:
            return None, f"missing cached {helper} helper (rebuild the cache)"
        helper_path = cache.hsaco_path(helper_id)
        helper_spec = make_spec(helper, problem, helper_knobs)
        grid, block = dc.direct_launch_geometry(helper_spec)
        steps.append(
            DirectStep(
                kernel_name=entry[1]["kernel_name"],
                hsaco_path=helper_path,
                signature=_transform_signature(identity.dtype_a, runtime_channels),
                grid=grid,
                block=block,
                buffers=(src, None, dst),
                scalars=scalars,
            )
        )
    weights = "ws_coa" if use_coalesced else "ws_t"
    # The compute pass is a *forward* kernel: forward ABI, transposed extents.
    steps.append(
        _conv_step(
            identity, hsaco_path, meta, spec, spec.problem, "fwd", ("dy", weights, "dx")
        )
    )
    workspaces = {"ws_t": dc.direct_dgrad_workspace_bytes(problem)}
    if use_coalesced:
        workspaces["ws_coa"] = dc.direct_dgrad_coalesced_workspace_bytes(
            problem, fold_k32=knobs["fold_k32"]
        )
    return DirectPlan(identity, tuple(steps), workspaces), "ok"


def _batch_split_plan(
    cache, identity, hsaco_path, meta, problem, arch, knobs, chunks
) -> Tuple[Optional[DirectPlan], str]:
    """One forward launch per batch chunk (see :func:`_batch_chunks`)."""
    from kernels.common.conv_direct_nongrouped import tile_w_candidates

    widths = tile_w_candidates(
        problem.Wo, make_spec(identity.algorithm, problem, knobs).atom_tile
    )
    if knobs["tile_w"] not in widths:
        return None, f"tile_w={knobs['tile_w']} does not fit Wo={problem.Wo}"
    in_img = 2 * problem.H * problem.W * problem.total_c
    out_img = 2 * problem.Ho * problem.Wo * problem.total_k
    steps = []
    n0 = 0
    for n in chunks:
        sub = replace(problem, N=n)
        spec = make_spec(identity.algorithm, sub, knobs)
        ok, why = validate_spec(identity.algorithm, spec, arch)
        if not ok:
            return None, why
        step = _conv_step(identity, hsaco_path, meta, spec, sub, "fwd", ("x", "w", "y"))
        views = ((n0 * in_img, n * in_img), None, (n0 * out_img, n * out_img))
        steps.append(replace(step, views=views))
        n0 += n
    return DirectPlan(identity, tuple(steps)), "ok"


def direct_plans(
    cache: KernelCache, problem, direction: str, arch: str
) -> Tuple[List[DirectPlan], List[Tuple[KernelIdentity, str]]]:
    """Launch plans for every cached kernel that can run ``problem``.

    Returns ``(plans, rejected)``; ``rejected`` pairs each capability-matching
    kernel that failed the per-shape re-validation with the reason, so an
    empty plan list is diagnosable.
    """
    plans: List[DirectPlan] = []
    rejected: List[Tuple[KernelIdentity, str]] = []
    for identity, hsaco_path, meta in cache.compatible(
        problem, direction=f"direct_{direction}"
    ):
        # A binary only serves the operand dtype it was built for.
        if identity.dtype_a != getattr(problem, "dtype", DIRECT_DTYPE):
            continue
        # The non-grouped family only exists for one group; offering it to a
        # grouped problem would bury the grouped kernels' reasons in noise.
        if identity.algorithm == "direct_nongrouped" and problem.groups != 1:
            continue
        # ... and the grouped LDS-halo kernel only for more than one; on a
        # dense problem it is the non-grouped kernel with extra divides.
        if identity.algorithm == "direct_grouped_halo" and problem.groups == 1:
            continue
        plan, why = plan_for(cache, identity, hsaco_path, meta, problem, arch)
        if plan is None:
            rejected.append((identity, why))
        else:
            plans.append(plan)
    return plans, rejected


def launch_values(
    step: DirectStep, ptrs: Dict[str, int], sizes: Dict[str, int]
) -> dict:
    """Kernarg values of ``step`` with its buffers bound to ``ptrs``/``sizes``."""
    a, b, d = step.buffers

    def bind(slot: int, name: str) -> Tuple[int, int]:
        view = step.views[slot] if step.views else None
        if view is None:
            return ptrs[name], sizes[name]
        return ptrs[name] + view[0], view[1]

    (a_ptr, a_bytes), (d_ptr, d_bytes) = bind(0, a), bind(2, d)
    if step.conv_args is None:
        return {
            "A": a_ptr,
            "D": d_ptr,
            "A_bytes": a_bytes,
            "D_bytes": d_bytes,
            **dict(step.scalars),
        }
    b_ptr, b_bytes = bind(1, b)
    return step.conv_args.to_launch_values(
        a_ptr, b_ptr, d_ptr, a_bytes, b_bytes, d_bytes
    )
