# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The direct convolution AOT grid: which kernels the cache builds, and how one
of them is built.

:data:`DIRECT_CAPABILITIES` (what each binary is baked for) times the tuning
grids (what it is tuned with) is every direct kernel the cache holds;
:func:`direct_job_for_identity` reads the job back off one identity and
:func:`direct_kernel_for_job` turns it into kernel IR. Enumerating in parallel,
compiling and planning launches is :mod:`benchmarks.common.direct_kernel_sweep`;
this module sits below ``dispatch`` so the dispatcher can build a registered
instance itself.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, replace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from kernels.conv.aot_jobs import BuildJob
from kernels.conv.direct_variants import (
    DIRECT_MFMA_DGRAD as _MFMA_DGRAD,
    direct_dgrad_weight_transforms,
    direct_mfma_knobs_reason as _mfma_knobs_reason,
    make_direct_spec as make_spec,
    select_direct_kernel_builder as _build_kernel,
    validate_direct_spec as validate_spec,
)
from kernels.conv.kernel_identity import KernelIdentity


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


def _knob_grid(variant: str, arch: str, dtype: str) -> Iterator[dict]:
    if variant == "direct_nongrouped":
        from kernels.common.conv_direct_nongrouped import nongrouped_knobs

        # Every width (no Wo to pick from): plan_for keeps those that fit.
        yield from nongrouped_knobs(arch, dtype)
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
    many 16-channel tiles as the widest wgrad ``waves_c``.

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
    if caps.cpg == 0:
        return DirectConvProblem(
            N=1,
            H=64,
            W=64,
            groups=1,
            cpg=256,
            kpg=256,
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
    for knobs in _knob_grid(variant, arch, dtype):
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
        if variant == "direct_nongrouped" and _nongrouped_spills(spec, arch):
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
        if variant in _MFMA_DGRAD:
            for helper, helper_knobs, _, _ in direct_dgrad_weight_transforms(
                knobs, variant
            ):
                cell.append(
                    _job(
                        _helper_identity(
                            arch, wave_size, helper, caps, helper_knobs, dtype
                        ),
                        caps,
                        helper_knobs,
                    )
                )
        out.extend((job.identity.stable_hash(), job) for job in cell)
    return n_raw, out


def count_direct_jobs(
    directions: Sequence[str], arch: str, dtype: str = DIRECT_DTYPE
) -> int:
    """Raw candidate count ``direct_kernel_sweep.enumerate_direct_jobs`` walks
    (the progress denominator): capabilities times tuning knobs, before validation."""
    return sum(
        len(caps_list) * sum(1 for _ in _knob_grid(variant, arch, dtype))
        for variant, (direction, caps_list) in DIRECT_CAPABILITIES.items()
        if direction in directions
    )


def direct_kernel_for_job(job: BuildJob, arch: str, dtype: str):
    """Build one direct job's kernel IR. Returns ``(kernel, meta)``.

    Compilation, deduplication and incremental rebuilds are
    :func:`compile_jobs`'s; this only decides what the kernel is.
    """
    if dtype not in DIRECT_DTYPES:
        raise ValueError(f"direct conv builds {DIRECT_DTYPES}, not {dtype!r}")
    caps = DirectCaps(**job.caps)
    variant = job.identity.algorithm
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
