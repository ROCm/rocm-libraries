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
    DIRECT_DGRAD_WMMA as _DGRAD_WMMA,
    DIRECT_GROUPED_HALO as _GROUPED_HALO,
    DIRECT_HALO_VARIANTS as _HALO_VARIANTS,
    DIRECT_MFMA_DGRAD as _MFMA_DGRAD,
    DIRECT_SPLIT_REDUCE as _SPLIT_REDUCE,
    DIRECT_WAVE_PARAMETRIC,
    DIRECT_WGRAD_FOLD as _WGRAD_FOLD,
    DIRECT_WGRAD_FOLDED as _WGRAD_FOLDED,
    direct_wgrad_fold_knobs,
    dgrad_rtc_vgprs,
    direct_dgrad_weight_transforms,
    direct_mfma_knobs_reason as _mfma_knobs_reason,
    make_direct_spec as make_spec,
    select_direct_kernel_builder as _build_kernel,
    validate_direct_spec as validate_spec,
)
from kernels.conv.kernel_identity import KernelIdentity

# Direction name of the forward helper kernels (shape-independent passes like
# the split-C reduction) -- found by exact identity, not by capability match.
_FWD_HELPER = "direct_fwd_helper"


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
# Windowed depthwise dgrad (direct_depthwise_dgrad_win, stride 1). Its row
# loop is unrolled over a tile of block_h dX rows, so the tile is a knob while
# the image height stays a kernarg. The cache carries the image of the tile
# policy rather than a product: a tile that keeps a small feature map whole
# (block_w == block_h, the 7..16-wide maps of the late CNN stages), and
# narrower column tiles over row chunks for taller maps. block_waves covers
# small channel counts (1) and the 256-channel block (4 waves of single
# channels, 2 of channel pairs). Small filters (KW == 3) also get packed
# channel pairs; wider ones pair their taps on fdot2 where the arch has it.
DW_WIN_TILES = (
    (7, 7),
    (8, 8),
    (12, 12),
    (14, 14),
    (16, 16),
    (7, 14),
    (8, 14),
    (8, 16),
    (7, 28),
)
DW_WIN_WAVES = (1, 4)
DW_WIN_PAIR_WAVES = (2,)
# Statically unrolled FMAs per lane the cache builds at most (the validator
# allows more; past this the compile time grows without a matching win).
DW_WIN_UNROLL_BUDGET = 1 << 14
# Toeplitz MFMA form: "same"-padded filters, images folded 1/2/4 per tile,
# 4- and 8-wave blocks, and row tiles that keep up to 16-row maps whole or
# chunk taller ones.
DW_WIN_MFMA_FILTERS = (5, 7, 9)
DW_WIN_MFMA_FOLDS = (1, 2, 4)
DW_WIN_MFMA_WAVES = (4, 8)
DW_WIN_MFMA_BLOCK_H = (4, 7, 8, 12, 14, 16)
DW_WIN_MFMA_PREFETCH = 2
# 4c dgrad (direct_grouped_dgrad_4c, cpg = kpg = 4): the 4x4x4 MFMA kernel on
# the transposed problem with fused weights. The cache carries the image of
# the 4c dispatch policy -- the (4, 16) tile with gathered weights, LDS-staged
# weights (gfx950) and the row-staged form (gfx950) -- plus a small
# neighbourhood of wider column and group tiles. The row-staged form takes one
# q wave per 4 columns (waves_q = block_q / 4).
DGRAD_4C_BLOCK_Q = (4, 8, 16)
DGRAD_4C_BLOCK_GROUPS = (16, 32, 64)
# (dgrad_weights_lds, stage_rows); the validator drops the LDS forms on
# targets without transpose LDS reads.
DGRAD_4C_FORMS = ((False, False), (True, False), (True, True))
# Fused-weight generic dgrad (direct_grouped_dgrad_fused): the generic
# forward row stream on the transposed problem reading the original weights
# in its prologue (gathered, or LDS-staged transpose reads on gfx950). The
# cache carries the image of the generic direct dgrad policy per capability
# (see _dgrad_fused_policy_reason) rather than a product: 16-column strips
# (32 on H-tiled images with wide channel groups), block_groups at or below
# the policy's per-channel row (halved for 5x5/7x7), and the tile heights a
# tiled or whole-image pick maps to -- a pick that streamed the whole image
# takes the smallest tile at least as tall, and the row guard skips the MFMA
# work of the rows past it.
DGRAD_FUSED_FILTERS = (1, 3, 5, 7)
DGRAD_FUSED_CHANNELS = (4, 8, 12, 16, 20, 24, 28, 32)
DGRAD_FUSED_BLOCK_Q = (16, 32)
DGRAD_FUSED_BLOCK_GROUPS = (1, 2, 4)
DGRAD_FUSED_BLOCK_H = (4, 8, 16, 32)
# Tile heights of 5x5 / 7x7 filters: the policy always tiles them by 4 rows
# (8 covers a whole image of up to 8 rows).
DGRAD_FUSED_BLOCK_H_LARGE = (4, 8)
# Preloaded weight registers up to which the policy adds the waves_per_eu
# hint (16-wide atoms only).
DGRAD_FUSED_WPE = 4
DGRAD_FUSED_WPE_MAX_VGPRS = 36
# The policy's row-stream stack (square channel groups, 16-column strips,
# waves_m = 2 not valid): the two-row prefetch with the LDS-only row barrier
# and the XCD row-band order, plus (lds_pad, stage_out) -- the policy tries
# (pad, staged), (pad, unstaged), (0, unstaged) and keeps the first that
# does not add launch rounds, a shape-dependent choice, so all three are
# cached. ``pad`` is DGRAD_FUSED_LDS_PAD when a staged input column is an
# even number of 16-byte units, else 0. None is the spec without the stack.
DGRAD_FUSED_PREFETCH_ROWS = 2
# Tile heights the policy only reaches through whole-image picks (it tiles
# with 4 and 8 rows): those entries also leave the workgroup once the image
# is done (row_exit), so a short image skips the tile's dead rows.
DGRAD_FUSED_EXIT_BLOCK_H = (16, 32)
DGRAD_FUSED_LDS_PAD = 8
DGRAD_FUSED_STREAM = (None, (0, False), (0, True), (8, False), (8, True))
_DGRAD_FUSED_STREAM_KEYS = (
    "prefetch_rows",
    "lds_only_sync",
    "xcd_tiles",
    "lds_pad",
    "stage_out",
)
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
# Wave32 WMMA direct dgrad (direct_dgrad_wmma): (waves_c, waves_p, tiles_c,
# tiles_p, tile_w, tile_n, ck, stages, prefetch), prefetch builds with the
# iterative-ilp scheduler. The union of the five best configurations per
# shape of a tuning sweep over the dense and grouped (cpg >= 8) stride-1 3x3
# dgrad layers of the reference workloads; each binary serves any extent and
# channel count, and the per-problem tile-efficiency check
# (direct_spec_for_problem) drops the ones a shape would mostly pad.
DGRAD_WMMA_KNOBS = (
    (1, 2, 1, 4, 8, 1, 32, 3, True),
    (1, 2, 1, 4, 16, 1, 32, 3, True),
    (1, 2, 1, 4, 32, 1, 32, 3, True),
    (1, 2, 4, 4, 8, 1, 64, 3, True),
    (1, 2, 4, 4, 16, 1, 64, 2, True),
    (1, 2, 4, 4, 16, 1, 64, 3, True),
    (1, 4, 1, 4, 16, 1, 32, 2, True),
    (1, 4, 1, 4, 16, 2, 32, 3, True),
    (1, 4, 1, 4, 32, 1, 32, 3, True),
    (1, 4, 1, 8, 16, 2, 32, 2, False),
    (1, 4, 1, 8, 16, 2, 32, 2, True),
    (1, 4, 2, 4, 16, 1, 32, 2, False),
    (1, 4, 2, 8, 16, 2, 32, 2, False),
    (1, 4, 2, 8, 16, 2, 32, 2, True),
    (1, 4, 4, 4, 16, 1, 32, 2, True),
    (1, 4, 4, 4, 16, 1, 64, 2, True),
    (1, 4, 4, 4, 16, 2, 64, 2, True),
    (1, 4, 4, 4, 32, 1, 32, 2, True),
    (1, 4, 4, 4, 32, 2, 32, 2, True),
    (1, 4, 4, 6, 16, 1, 32, 2, False),
    (1, 4, 4, 6, 16, 1, 32, 2, True),
    (1, 4, 4, 6, 16, 1, 64, 2, True),
    (1, 4, 4, 6, 16, 2, 32, 2, True),
    (1, 4, 4, 6, 16, 2, 64, 2, True),
    (1, 4, 4, 6, 32, 1, 32, 2, False),
    (1, 4, 4, 6, 32, 1, 32, 2, True),
    (1, 4, 4, 6, 32, 1, 64, 2, True),
    (1, 4, 4, 6, 32, 2, 64, 2, True),
    (1, 4, 8, 4, 16, 1, 32, 2, True),
    (1, 4, 8, 4, 32, 1, 32, 2, True),
    (1, 4, 8, 4, 64, 1, 32, 2, True),
    (1, 8, 1, 2, 8, 4, 32, 2, False),
    (1, 8, 1, 2, 8, 4, 32, 2, True),
    (1, 8, 1, 2, 8, 4, 32, 3, False),
    (1, 8, 1, 2, 8, 4, 32, 3, True),
    (1, 8, 1, 4, 16, 2, 32, 2, False),
    (1, 8, 1, 4, 16, 2, 32, 2, True),
    (1, 8, 2, 4, 16, 2, 32, 2, False),
    (1, 8, 2, 4, 16, 2, 32, 2, True),
    (2, 2, 2, 4, 16, 1, 64, 2, True),
    (2, 2, 2, 4, 16, 1, 64, 3, True),
    (2, 2, 4, 4, 8, 2, 32, 3, True),
    (2, 2, 4, 4, 16, 1, 32, 3, True),
    (2, 2, 4, 6, 16, 1, 32, 2, True),
    (2, 2, 4, 6, 16, 1, 32, 3, True),
    (2, 2, 4, 6, 32, 1, 32, 2, True),
    (2, 2, 4, 8, 32, 1, 32, 2, True),
    (2, 2, 4, 8, 64, 1, 32, 2, True),
    (2, 4, 2, 4, 16, 1, 64, 2, True),
    (2, 4, 2, 4, 16, 2, 64, 2, True),
    (2, 4, 2, 8, 32, 1, 32, 2, True),
    (2, 4, 4, 6, 4, 4, 32, 2, True),
    (2, 4, 4, 6, 8, 2, 32, 2, False),
    (2, 4, 4, 6, 8, 2, 32, 2, True),
    (2, 4, 4, 6, 8, 4, 32, 2, True),
    (2, 4, 4, 6, 16, 1, 32, 2, True),
    (2, 4, 4, 6, 64, 1, 32, 2, True),
    (2, 4, 4, 8, 16, 1, 32, 2, True),
    (2, 4, 4, 8, 16, 2, 32, 2, True),
    (2, 4, 4, 8, 16, 4, 32, 2, True),
    (2, 4, 4, 8, 32, 1, 32, 2, True),
    (2, 4, 4, 8, 32, 2, 32, 2, True),
    (2, 4, 4, 8, 32, 4, 32, 2, True),
    (2, 4, 4, 8, 64, 1, 32, 2, True),
    (2, 4, 6, 4, 16, 1, 32, 2, True),
    (2, 4, 6, 4, 16, 2, 32, 2, True),
    (2, 4, 6, 4, 32, 1, 32, 2, True),
    (4, 2, 2, 8, 16, 1, 32, 2, True),
    (4, 2, 2, 8, 16, 1, 32, 3, False),
    (4, 2, 2, 8, 16, 1, 32, 3, True),
    (4, 2, 2, 8, 32, 1, 32, 2, True),
)
WGRAD_MFMA_K = (32,)

# Tiled direct wgrad (DirectWgradTiledSpec): (wave_k16, wave_c16, waves_k,
# waves_c) arrangements, (prefetch, row_unroll) pipelines, rows per segment,
# workgroups launched, images packed per strip. Measured on gfx950 bf16 over
# 3x3 C = K = 128 layers: 32x32 wave tiles in four-wave blocks (two per CU)
# lead; eight-wave blocks are kept for the shapes that fill few channel tiles;
# unfold_n = 4 is what a 40-wide image needs (Wo % 32 != 0).
WGRAD_TILED_WAVES = ((2, 2, 2, 2), (2, 2, 4, 2), (2, 1, 2, 2))
WGRAD_TILED_PIPE = ((2, 0), (5, 6))
WGRAD_TILED_RPS = (15, 30)
WGRAD_TILED_BLOCKS = (256, 512)
WGRAD_TILED_UNFOLD = (1, 4)
# Eight-wave blocks of two ping-ponged wave groups, on
# the plain two-row prefetch of four-wave 32x32 tiles.
WGRAD_TILED_PINGPONG = ((2, 2, 2, 2), (2, 0), (30,), (256,), (1,))
# Depthwise LDS-ring wgrad (DwWgradLdsSpec): lanes along columns, prefetch,
# output rows per segment, workgroups.
WGRAD_DW_LANES_W = (8, 16, 32)
WGRAD_DW_PREFETCH = (1, 2)
WGRAD_DW_RPS = (16, 32, 64)
WGRAD_DW_BLOCKS = (1024, 2048)
# Small-C depthwise (DwWgradColsSpec): channel counts it bakes, strip widths,
# threads, workgroups.
WGRAD_DW_COLS_CHANNELS = (1, 2, 3, 4, 8)
WGRAD_DW_COLS_STRIP = (8, 16)
WGRAD_DW_COLS_THREADS = (128, 256)
WGRAD_DW_COLS_BLOCKS = (2048, 4096)
# Dense small-C gather wgrad (WgradGatherSpec): (wave_k16, waves_k), prefetch,
# output rows per segment, workgroups.
WGRAD_GATHER_WAVES = ((2, 1), (1, 3), (2, 3), (1, 6), (2, 2))
WGRAD_GATHER_PREFETCH = (1, 2)
WGRAD_GATHER_RPS = (8, 16)
WGRAD_GATHER_BLOCKS = (256, 1024, 2048)

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
# Input channel counts the dense small-C gather wgrad bakes (KW * C <= 96).
_GATHER_CHANNELS = (3, 4, 8, 16, 24, 32)
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


def _dgrad_fused_weights_fit(kh: int, cpg: int, kpg: int) -> bool:
    """Do the fused dgrad's preloaded weights fit their register budget?

    The kernel runs the transposed problem: its K-atoms cover the dgrad's
    ``kpg`` and its 16-row M-tiles the dgrad's ``cpg``; each (tap, atom,
    M-tile) fragment holds 2 VGPRs per 16-wide atom (4 per 32-wide one, at
    half the atoms), so the count does not depend on fold_k32.
    """
    from kernels.common.conv_direct_grouped import PRELOAD_WEIGHT_VGPR_BUDGET

    return kh * kh * -(-kpg // 16) * -(-cpg // 16) * 2 <= PRELOAD_WEIGHT_VGPR_BUDGET


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
    _GROUPED_HALO: (
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
    # Depthwise dgrad, windowed kernel: stride 1, the same filters, every
    # padding (it bounds its rows by p_Ho and its columns by p_Wo).
    "direct_depthwise_dgrad_win": (
        "dgrad",
        tuple(
            DirectCaps(KH=k, PAD=pad, stride=1, cpg=1, kpg=1)
            for k in _DEPTHWISE_FILTERS
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
    # 4c dgrad: one fused-weight kernel, a forward 4c kernel on the
    # transposed problem, so stride 1 and "same" padding; 1x1 and 3x3.
    "direct_grouped_dgrad_4c": (
        "dgrad",
        tuple(
            DirectCaps(KH=k, PAD=_same_pad(k), stride=1, cpg=4, kpg=4) for k in (1, 3)
        ),
    ),
    # Generic dgrad with the weight transform fused into the kernel (one
    # forward-ABI kernel over the transposed problem, no pre-pass): stride 1,
    # "same"-padded square filters up to 7x7 and per-group channel counts in
    # 4-channel steps up to 32 on either side, including cpg != kpg.
    # (A 5x5 / 7x7 filter whose fused weights do not fit the preload register
    # budget at any knob has no row: the pre-pass pipeline serves it.)
    "direct_grouped_dgrad_fused": (
        "dgrad",
        tuple(
            DirectCaps(KH=k, PAD=_same_pad(k), stride=1, cpg=c, kpg=kk)
            for k in DGRAD_FUSED_FILTERS
            for c in DGRAD_FUSED_CHANNELS
            for kk in DGRAD_FUSED_CHANNELS
            if _dgrad_fused_weights_fit(k, c, kk)
        ),
    ),
    # The same pipeline with runtime channels: dense (groups == 1) and any
    # cpg/kpg whose tile counts a cached binary has (knobs k_atoms/m_tiles).
    "direct_dgrad_mfma_rtc": (
        "dgrad",
        (DirectCaps(KH=3, PAD=1, stride=1, cpg=0, kpg=0),),
    ),
    # Wave32 WMMA dgrad: stride 1, channels and groups runtime. It bounds rows
    # and columns against the runtime extents, so it serves "valid" padding as
    # well as "same".
    _DGRAD_WMMA: (
        "dgrad",
        tuple(
            DirectCaps(KH=3, PAD=pad, stride=1, cpg=0, kpg=0) for pad in _col_pads(3)
        ),
    ),
    # Tiled wgrad: dense and grouped, channel counts runtime, stride 1.
    "direct_wgrad_tiled": (
        "wgrad",
        tuple(
            DirectCaps(KH=k, PAD=pad, stride=1, cpg=0, kpg=0)
            for k in (3,)
            for pad in _col_pads(k)
        ),
    ),
    # Depthwise wgrad through LDS rings: 3x3 (8 channels x 9 taps of fp32
    # partial sums per lane), stride 1 and 2, C runtime (a multiple of 8).
    "direct_wgrad_dw": (
        "wgrad",
        tuple(
            DirectCaps(KH=3, PAD=pad, stride=s, cpg=1, kpg=1)
            for s in _STRIDES
            for pad in _col_pads(3)
        ),
    ),
    # Small-C depthwise wgrad (stem layers): every depthwise filter; the
    # channel count is a knob (``channels``).
    "direct_wgrad_dw_cols": (
        "wgrad",
        tuple(
            DirectCaps(KH=k, PAD=pad, stride=s, cpg=1, kpg=1)
            for k, s in itertools.product(_DEPTHWISE_FILTERS, _STRIDES)
            for pad in _col_pads(k)
        ),
    ),
    # Dense small-C wgrad, any stride: C is baked, K runtime.
    "direct_wgrad_gather": (
        "wgrad",
        tuple(
            DirectCaps(KH=3, PAD=pad, stride=s, cpg=c, kpg=0)
            for c in _GATHER_CHANNELS
            for s in _STRIDES
            for pad in _col_pads(3)
        ),
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
    elif variant == _GROUPED_HALO:
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
    elif variant == "direct_depthwise_dgrad_win":
        for (bw, bh), dot2, waves in itertools.product(
            DW_WIN_TILES, (False, True), DW_WIN_WAVES
        ):
            yield dict(
                block_w=bw, block_h=bh, block_waves=waves, ch_per_lane=1, dot2=dot2
            )
        for (bw, bh), waves in itertools.product(DW_WIN_TILES, DW_WIN_PAIR_WAVES):
            yield dict(
                block_w=bw, block_h=bh, block_waves=waves, ch_per_lane=2, dot2=False
            )
        for fold, waves, bh in itertools.product(
            DW_WIN_MFMA_FOLDS, DW_WIN_MFMA_WAVES, DW_WIN_MFMA_BLOCK_H
        ):
            yield dict(
                mfma=True,
                w_fold=fold,
                block_waves=waves,
                block_h=bh,
                prefetch_rows=DW_WIN_MFMA_PREFETCH,
            )
    elif variant == "direct_depthwise_col":
        for bh, bw, waves in itertools.product(
            DW_COL_BLOCK_H, DW_COL_BLOCK_W, DW_BLOCK_WAVES
        ):
            yield dict(block_h=bh, block_w=bw, block_waves=waves)
    elif variant == "direct_grouped_dgrad":
        for bq, bg in itertools.product(DGRAD_BLOCK_Q, BLOCK_GROUPS):
            yield dict(block_q=bq, block_groups=bg)
    elif variant == "direct_grouped_dgrad_4c":
        for bq, bg, (lds, sr) in itertools.product(
            DGRAD_4C_BLOCK_Q, DGRAD_4C_BLOCK_GROUPS, DGRAD_4C_FORMS
        ):
            yield {
                "block_q": bq,
                "block_groups": bg,
                "dgrad_weights_lds": lds,
                "stage_rows": sr,
            }
    elif variant == "direct_grouped_dgrad_fused":
        # Filtered per capability by _dgrad_fused_policy_reason.
        for bq, bg, bh, k32, lds, wpe, stream in itertools.product(
            DGRAD_FUSED_BLOCK_Q,
            DGRAD_FUSED_BLOCK_GROUPS,
            DGRAD_FUSED_BLOCK_H,
            (False, True),
            (False, True),
            (0, DGRAD_FUSED_WPE),
            DGRAD_FUSED_STREAM,
        ):
            knobs = {
                "block_q": bq,
                "block_groups": bg,
                "block_h": bh,
                "fold_k32": k32,
                "dgrad_weights_lds": lds,
                "waves_per_eu": wpe,
            }
            if bh in DGRAD_FUSED_EXIT_BLOCK_H:
                knobs["row_exit"] = True
            if stream is not None:
                knobs.update(
                    prefetch_rows=DGRAD_FUSED_PREFETCH_ROWS,
                    lds_only_sync=True,
                    xcd_tiles=True,
                    lds_pad=stream[0],
                    stage_out=stream[1],
                )
            yield knobs
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
    elif variant == _DGRAD_WMMA:
        from rocke.core.arch import ArchTarget

        if ArchTarget.from_gfx(arch).wave_size != 32:
            return  # a wave32 kernel: no tuning for this arch at all
        for wc, wp, tc, tp, tw, tn, ck, st, pf in DGRAD_WMMA_KNOBS:
            knobs = dict(
                waves_c=wc,
                waves_p=wp,
                tiles_c=tc,
                tiles_p=tp,
                tile_w=tw,
                tile_n=tn,
                ck=ck,
                stages=st,
            )
            if pf:
                knobs.update(prefetch=True, sched="iterative-ilp")
            yield knobs
    elif variant == "direct_wgrad_tiled":
        for (wk16, wc16, wk, wc), (pf, ru), rps, tb, un in itertools.product(
            WGRAD_TILED_WAVES,
            WGRAD_TILED_PIPE,
            WGRAD_TILED_RPS,
            WGRAD_TILED_BLOCKS,
            WGRAD_TILED_UNFOLD,
        ):
            yield dict(
                wave_k16=wk16,
                wave_c16=wc16,
                waves_k=wk,
                waves_c=wc,
                prefetch=pf,
                row_unroll=ru,
                rows_per_seg=rps,
                target_blocks=tb,
                unfold_n=un,
            )
        (wk16, wc16, wk, wc), (pf, ru), rpss, tbs, uns = WGRAD_TILED_PINGPONG
        for rps, tb, un in itertools.product(rpss, tbs, uns):
            yield dict(
                wave_k16=wk16,
                wave_c16=wc16,
                waves_k=wk,
                waves_c=wc,
                prefetch=pf,
                row_unroll=ru,
                rows_per_seg=rps,
                target_blocks=tb,
                unfold_n=un,
                q_groups=2,
                pingpong=True,
            )
    elif variant == "direct_wgrad_dw":
        for lw, pf, rps, tb in itertools.product(
            WGRAD_DW_LANES_W, WGRAD_DW_PREFETCH, WGRAD_DW_RPS, WGRAD_DW_BLOCKS
        ):
            yield dict(
                lanes_c=8, lanes_w=lw, prefetch=pf, rows_per_seg=rps, target_blocks=tb
            )
    elif variant == "direct_wgrad_dw_cols":
        for c, st, th, tb in itertools.product(
            WGRAD_DW_COLS_CHANNELS,
            WGRAD_DW_COLS_STRIP,
            WGRAD_DW_COLS_THREADS,
            WGRAD_DW_COLS_BLOCKS,
        ):
            yield dict(channels=c, strip=st, threads=th, target_blocks=tb)
    elif variant == "direct_wgrad_gather":
        for (wk16, wk), pf, rps, tb in itertools.product(
            WGRAD_GATHER_WAVES,
            WGRAD_GATHER_PREFETCH,
            WGRAD_GATHER_RPS,
            WGRAD_GATHER_BLOCKS,
        ):
            yield dict(
                wave_k16=wk16,
                waves_k=wk,
                prefetch=pf,
                rows_per_seg=rps,
                target_blocks=tb,
            )
    elif variant == "direct_wgrad":
        for (wk, wc, wq), hpb, mk in itertools.product(
            wgrad_waves(), WGRAD_HPB, WGRAD_MFMA_K
        ):
            yield dict(waves_k=wk, waves_c=wc, waves_q=wq, ho_per_block=hpb, mfma_k=mk)
    else:
        raise ValueError(f"unknown direct conv variant {variant!r}")


def _dw_win_policy_reason(caps: DirectCaps, knobs: dict, arch: str) -> Optional[str]:
    """Why the cache does not build this windowed depthwise dgrad knob for
    ``caps`` (``None`` when it does). These are cache-scope rules of the knob
    image (see :data:`DW_WIN_TILES`), not kernel limits: the validator still
    accepts what they skip, so a JIT sweep can try it."""
    from kernels.common.conv_direct_grouped import DW_DGRAD_WIN_DOT2_ARCHES

    if knobs.get("mfma"):
        if caps.KH not in DW_WIN_MFMA_FILTERS or caps.PAD != _same_pad(caps.KH):
            return "the MFMA form is cached for 'same'-padded 5x5..9x9 only"
        return None
    want_dot2 = caps.KW >= 5 and arch in DW_DGRAD_WIN_DOT2_ARCHES
    if knobs["dot2"] != want_dot2:
        return "dot2 is cached exactly where it pairs taps (KW >= 5, fdot2 arch)"
    if knobs["ch_per_lane"] > 1 and caps.KW != 3:
        return "packed channel pairs are cached for 3x3 only"
    fmas = (
        knobs["block_h"] * caps.KH * caps.KW * knobs["block_w"] * knobs["ch_per_lane"]
    )
    if fmas > DW_WIN_UNROLL_BUDGET:
        return f"{fmas} unrolled FMAs exceed the cache budget {DW_WIN_UNROLL_BUDGET}"
    return None


def _dgrad_fused_rule_block_groups(caps: DirectCaps) -> int:
    """block_groups of the generic direct dgrad policy's per-channel row:
    the wider channel count times block_groups stays near a 32..64-channel
    slice (1 from 32 channels, 2 from 16, else 4), halved for 5x5 / 7x7
    filters (their accumulator ring is KH deep)."""
    chans = max(caps.cpg, caps.kpg)
    bg = 1 if chans >= 32 else 2 if chans >= 16 else 4
    return max(1, bg // 2) if caps.KH >= 5 else bg


def _dgrad_fused_policy_reason(
    caps: DirectCaps, knobs: dict, arch: str, dtype: str = DIRECT_DTYPE
) -> str | None:
    """Why the cache does not build this fused generic dgrad knob for ``caps``
    (``None`` when it does). Cache-scope rules of the policy image (see
    :data:`DGRAD_FUSED_FILTERS`), not kernel limits: the validator still
    accepts what they skip, so a JIT sweep can try it."""
    from kernels.common import conv_direct_grouped as dc

    stream = "prefetch_rows" in knobs
    if stream:
        # Cheap rules of the row-stream stack first; the base knobs below are
        # decided without it, as the policy does.
        if caps.cpg != caps.kpg:
            return "the row-stream stack is cached for square channel groups only"
        if knobs["block_q"] != 16:
            return "the row-stream stack is cached on 16-column strips only"
        # The staged input column is block_groups * (dgrad kpg) channels.
        row_bytes = knobs["block_groups"] * caps.kpg * 2
        pad = DGRAD_FUSED_LDS_PAD if row_bytes % 32 == 0 else 0
        if (knobs["lds_pad"], knobs["stage_out"]) not in (
            (pad, True),
            (pad, False),
            (0, False),
        ):
            return f"the row-stream stack is cached with lds_pad {pad} or 0"
        knobs = {k: v for k, v in knobs.items() if k not in _DGRAD_FUSED_STREAM_KEYS}

    rule_bg = _dgrad_fused_rule_block_groups(caps)
    if knobs["block_groups"] > rule_bg:
        return f"block_groups above the policy's {rule_bg} for these channels"
    large = caps.KH >= 5
    if large and knobs["block_h"] not in DGRAD_FUSED_BLOCK_H_LARGE:
        return f"5x5/7x7 filters are cached at tiles {DGRAD_FUSED_BLOCK_H_LARGE}"
    if knobs["block_q"] != 16:
        chans = max(caps.cpg, caps.kpg)
        if knobs["block_h"] not in (4, 8):
            return "32-column strips are cached on 4- and 8-row tiles only"
        if not (chans >= 16 or (large and chans >= 8)):
            return "32-column strips are cached for wide channel groups only"
    if knobs["fold_k32"] and caps.kpg % 32:
        return "fold_k32 needs kpg % 32 == 0"
    probe = probe_problem(caps, dtype, "direct_grouped_dgrad_fused")
    if not knobs["fold_k32"] and caps.kpg % 32 == 0:
        # 16-wide atoms are the fallback where the 32-wide one is not valid
        # (no 16x16x32 atom on the target).
        k32 = make_spec(
            "direct_grouped_dgrad_fused", probe, {**knobs, "fold_k32": True}
        )
        if validate_spec("direct_grouped_dgrad_fused", k32, arch)[0]:
            return "16-wide atoms are cached where fold_k32 is invalid"
    want_wpe = 0
    if not knobs["fold_k32"]:
        base = make_spec(
            "direct_grouped_dgrad_fused", probe, {**knobs, "waves_per_eu": 0}
        )
        if dc.preload_weight_vgprs(base) <= DGRAD_FUSED_WPE_MAX_VGPRS:
            want_wpe = DGRAD_FUSED_WPE
    if knobs["waves_per_eu"] != want_wpe:
        return f"waves_per_eu is cached at the policy's {want_wpe} for these knobs"
    if not knobs["dgrad_weights_lds"]:
        # Gathered weights are the fallback where the LDS-staged form is not
        # valid (no transpose reads, or the staged slice does not fit).
        lds = make_spec(
            "direct_grouped_dgrad_fused", probe, {**knobs, "dgrad_weights_lds": True}
        )
        if validate_spec("direct_grouped_dgrad_fused", lds, arch)[0]:
            return "gathered weights are cached where the LDS-staged form is invalid"
    if stream:
        # The policy keeps the spec without the stack where waves_m = 2 is
        # valid (its split-M stack is opt-in, not cached).
        split_m = make_spec(
            "direct_grouped_dgrad_fused",
            probe,
            {
                **knobs,
                "prefetch_rows": DGRAD_FUSED_PREFETCH_ROWS,
                "lds_only_sync": True,
                "waves_m": 2,
            },
        )
        if validate_spec("direct_grouped_dgrad_fused", split_m, arch)[0]:
            return "the row-stream stack is not cached where waves_m = 2 is valid"
    return None


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
    # filter shares the helper binary. The wgrad fold depends on its knobs
    # alone (the dW extent it folds), so it zeroes the channel counts too.
    if variant == _WGRAD_FOLD:
        return _identity(
            arch,
            wave_size,
            variant,
            "direct_wgrad_helper",
            replace(caps, PAD=0, stride=0, cpg=0, kpg=0),
            knobs,
            dtype,
        )
    return _identity(
        arch,
        wave_size,
        variant,
        "direct_dgrad_helper",
        replace(caps, PAD=0, stride=0),
        knobs,
        dtype,
    )


# Capabilities of the split reduction: it bakes no filter or channel count.
_NO_CAPS = DirectCaps(KH=0, PAD=0, stride=0, cpg=0, kpg=0)


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


def _dw_cols_probe_width(caps: DirectCaps, channels: int, strip: int) -> int:
    """A width the small-C depthwise wgrad takes at ``caps``, or 64 if none.

    The kernel needs Wo to be a multiple of its strip and both W * C and
    Wo * C to be whole 16-byte vectors. W is a runtime extent, so any width
    that passes probes the same binary; a fixed 64 fails every one of them
    at PAD 0 (Wo = 62 at 3x3), which left those capability rows unbuilt.
    """
    for w in range(64, 64 + 16 * strip * caps.stride):
        wo = (w + 2 * caps.PAD - caps.KH) // caps.stride + 1
        if (w * channels) % 8 == 0 and (wo * channels) % 8 == 0 and wo % strip == 0:
            return w
    return 64


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

    if variant == "direct_wgrad_dw_cols":
        # The binary bakes C (knob ``channels``): one group per channel.
        return DirectConvProblem(
            N=1,
            H=64,
            W=_dw_cols_probe_width(caps, knobs["channels"], knobs["strip"]),
            groups=knobs["channels"],
            cpg=1,
            kpg=1,
            KH=caps.KH,
            KW=caps.KW,
            PAD=caps.PAD,
            stride=caps.stride,
            dtype=dtype,
        )
    if variant == "direct_wgrad_gather":
        # Dense, C baked, K runtime: a K wider than every block_k.
        return DirectConvProblem(
            N=1,
            H=64,
            W=64,
            groups=1,
            cpg=caps.cpg,
            kpg=96,
            KH=caps.KH,
            KW=caps.KW,
            PAD=caps.PAD,
            stride=caps.stride,
            dtype=dtype,
        )
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
    if variant == _GROUPED_HALO:
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
    n_images = 1
    if variant == "direct_depthwise_spatial":
        groups = 8
    elif variant == "direct_grouped_dgrad_fused":
        # Its rows share channel counts with the dense MFMA dgrad rows but
        # serve grouped problems: every block_groups must divide the probe,
        # and eight images give every tile height the row bands xcd_tiles
        # needs.
        groups = 64
        n_images = 8
    elif (caps.cpg, caps.kpg) in _DENSE_MFMA_DGRAD_CHANNELS:
        # Dense channel counts serve groups == 1: one group per block. Wider
        # blocks would only build kernels no problem with these counts can
        # launch (and some of them fail to compile).
        groups = 1
    else:
        groups = 256 if caps.cpg == 1 else 64
    return DirectConvProblem(
        N=n_images,
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
    # The runtime-channel dgrad probe follows each knob's tile counts, the
    # small-C depthwise wgrad's its baked channel count.
    knob_probe = rtc_dgrad or variant == "direct_wgrad_dw_cols"
    probe = None if knob_probe else probe_problem(caps, dtype, variant)
    for knobs in _knob_grid(variant, arch, dtype, caps):
        n_raw += 1
        if knob_probe:
            probe = probe_problem(caps, dtype, variant, knobs)
        if rtc_dgrad and dgrad_rtc_spills(knobs, caps.KH, caps.KW, arch):
            continue
        if variant in _MFMA_DGRAD and _mfma_knobs_reason(probe, knobs):
            continue
        if variant == "direct_depthwise_dgrad_win" and _dw_win_policy_reason(
            caps, knobs, arch
        ):
            continue
        if variant == "direct_grouped_dgrad_fused" and _dgrad_fused_policy_reason(
            caps, knobs, arch, dtype
        ):
            continue
        spec = make_spec(variant, probe, knobs, wave_size)
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
        if variant in _WGRAD_FOLDED:
            fold_knobs = direct_wgrad_fold_knobs(variant, caps.KH, caps.KW, caps.cpg)
            cell.append(
                _job(
                    _helper_identity(
                        arch, wave_size, _WGRAD_FOLD, caps, fold_knobs, dtype
                    ),
                    replace(caps, PAD=0, stride=0, cpg=0, kpg=0),
                    fold_knobs,
                )
            )
        if variant == "direct_nongrouped" and knobs.get("split_c", 1) > 1:
            cell.append(_split_reduce_job(arch, wave_size, knobs["split_c"], dtype))
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
        sum(1 for _ in _knob_grid(variant, arch, dtype, caps))
        for variant, (direction, caps_list) in DIRECT_CAPABILITIES.items()
        if direction in directions
        for caps in caps_list
    )


@dataclass(frozen=True)
class _DtypeOnly:
    """The only problem field a shape-independent helper's spec reads."""

    dtype: str


def direct_kernel_for_job(job: BuildJob, arch: str, dtype: str):
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
    spec = make_spec(variant, probe, job.spec_kwargs, job.identity.wave_size)
    ok, why = validate_spec(variant, spec, arch)
    if not ok:
        raise ValueError(why)
    # The wave size the binary is built at, which the run side checks against
    # the identity before it derives a launch geometry from it
    # (direct_entry_wave_reason).
    meta = (
        {"spec_wave_size": spec.wave_size} if variant in DIRECT_WAVE_PARAMETRIC else {}
    )
    return _build_kernel(variant, spec, arch), meta


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
    if identity.direction in ("direct_dgrad_helper", "direct_wgrad_helper"):
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
    if identity.direction not in ("direct_dgrad_helper", "direct_wgrad_helper"):
        # The fields can be consistent and still name a kernel no variant
        # computes (a dilated grouped kernel, say): ask the kernel's validator.
        variant = identity.algorithm
        probe = probe_problem(caps, identity.dtype_a, variant, knobs)
        spec = make_spec(variant, probe, knobs, identity.wave_size)
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
