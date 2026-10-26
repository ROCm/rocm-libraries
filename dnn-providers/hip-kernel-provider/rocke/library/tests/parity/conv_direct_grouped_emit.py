#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_grouped_emit.py -- Python reference emitter for the
# direct grouped convolution parity harness. Selects one of N sampled spec
# configs by argv[1], builds the DirectConv16cSpec / DirectConv4cSpec /
# DirectConv8cSpec / DirectConv32cSpec / DirectDepthwiseSpec /
# DirectDepthwiseColSpec /
# DirectConvDgradSpec / DirectDepthwiseDgradSpec / DirectConvWgradSpec (and,
# from index 42, the non-grouped DirectNongroupedConvSpec, from index 57 the
# output-stationary DirectDepthwiseTiledSpec, from index 63 the dilated
# non-grouped DirectNongroupedConvSpec, from index 66 specs both engines
# must reject, from index 71 the runtime-channel DirectConvWgradSpec, from
# index 75 the parametric DirectConvSpec and the MFMA dgrad weight transforms,
# baked and runtime-channel, from index 95 the split-C non-grouped
# DirectNongroupedConvSpec and its DirectSplitReduceSpec second pass, from
# index 103 its grouped / group-merge mode, and from index 114 the direct
# wgrad kernels: DirectWgradTiledSpec, DwWgradFoldSpec, DwWgradColsSpec,
# DwWgradLdsSpec, WgradGatherSpec, DwWgradMfmaSpec and GroupedWgradMfmaSpec,
# from index 163 the bf16 DirectConv4cSpec, and from index 166 the depthwise
# forward kernels on the wave32 gfx1250), builds the kernel
# via the matching build_direct_conv_* function (arch=<cfg arch>) and prints
# _native_lower(arch=<cfg arch>) to stdout so it can be byte-compared with
# the C emitter conv_direct_grouped_emit.c.
import sys

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    DirectConv16cSpec,
    DirectConv4cSpec,
    DirectConv8cSpec,
    DirectConv32cSpec,
    DirectConvWgradSpec,
    DirectDepthwiseSpec,
    DirectDepthwiseSpatialSpec,
    DirectDepthwiseColSpec,
    DirectDepthwiseTiledSpec,
    DirectConvDgradSpec,
    DirectConvSpec,
    DirectDepthwiseDgradSpec,
    DirectReorganizeWeightsSpec,
    DirectTransposeWeightsDgradSpec,
    build_direct_conv,
    build_direct_conv_16c,
    build_direct_conv_4c,
    build_direct_conv_8c,
    build_direct_conv_32c,
    build_direct_conv_wgrad,
    build_direct_depthwise,
    build_direct_depthwise_spatial,
    build_direct_depthwise_col,
    build_direct_depthwise_tiled,
    build_direct_conv_dgrad,
    build_direct_depthwise_dgrad,
    build_direct_reorganize_weights,
    build_direct_transpose_weights_dgrad,
)
from kernels.common.conv_direct_nongrouped import (
    DirectNongroupedConvSpec,
    DirectSplitReduceSpec,
    build_direct_conv_nongrouped,
    build_direct_conv_split_reduce,
)
from kernels.common.conv_direct_dw_wgrad import (
    DwWgradColsSpec,
    DwWgradFoldSpec,
    DwWgradLdsSpec,
    build_dw_wgrad_cols,
    build_dw_wgrad_fold,
    build_dw_wgrad_lds,
)
from kernels.common.conv_direct_dw_wgrad_mfma import (
    DwWgradMfmaSpec,
    build_dw_wgrad_mfma,
)
from kernels.common.conv_direct_grouped_wgrad_mfma import (
    GroupedWgradMfmaSpec,
    build_grouped_wgrad_mfma,
)
from kernels.common.conv_direct_wgrad_gather import (
    WgradGatherSpec,
    build_wgrad_gather,
)
from kernels.common.conv_direct_wgrad_tiled import (
    DirectWgradTiledSpec,
    build_direct_wgrad_tiled,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


# ---------------------------------------------------------------------------
# Non-grouped (groups == 1) DirectNongroupedConvSpec configs, indices 42..56.
#
# Each pins one branch of build_direct_conv_nongrouped: the four MFMA atoms,
# fp16/bf16, stride 1/2, 1x1 / 3x3 / 5x5 filters, partial W/H/K tiles, staging
# passes that do / do not divide the block evenly (the scratch-tail select),
# single vs double buffered LDS, chiplet swizzle on/off, iglp and waves_per_eu.
# ---------------------------------------------------------------------------
_NONGROUPED_CFG_BASE = 42
# Output-stationary depthwise (DirectDepthwiseTiledSpec) configs follow the
# 15 non-grouped ones.
_TILED_CFG_BASE = 57
# Dilated non-grouped configs follow the 6 tiled ones.
_NONGROUPED_DIL_CFG_BASE = 63
# Specs both engines must reject follow the 3 dilated non-grouped ones.
_REJECT_CFG_BASE = 66


# ---------------------------------------------------------------------------
# Parametric DirectConvSpec (the MFMA dgrad compute pass) and the dgrad weight
# transforms, indices 75..94. Each pins a branch of build_direct_conv: the
# plain-weight loops, the runtime K loop, the waves_k preload + LDS reduction,
# fold_k32, waves_q, single buffering, the persistent grid, partial K-atoms and
# M-tiles, a 5x5 filter; then the same with runtime channels; then the
# transpose / reorganize transforms baked and with runtime channels.
# ---------------------------------------------------------------------------
_GENERIC_CFG_BASE = 75
# Split-C non-grouped configs (split_c > 1) and the DirectSplitReduceSpec fold
# follow the 20 generic ones.
_NONGROUPED_SPLIT_CFG_BASE = 95
# Grouped (grouped=True, group_merge) DirectNongroupedConvSpec configs follow the
# 8 split-C ones.
_NONGROUPED_GROUPED_CFG_BASE = 103


# ---------------------------------------------------------------------------
# The direct wgrad kernels, indices 114..162: DirectWgradTiledSpec (wave tiles,
# unfold_n, row_unroll, q_groups with and without pingpong, 1x1 and 5x5), the
# DwWgradFoldSpec partial-slot fold, DwWgradColsSpec (small C), DwWgradLdsSpec
# (row rings, strided, cols_per_lane, a partial channel tile), WgradGatherSpec
# (gathered and transpose-read X, with and without a phase), then two specs
# both engines must reject; a small-C depthwise spec with more outputs than
# threads; then eight more rejections (LDS footprints over budget, targets
# without 16-byte LDS DMA, a negative row_unroll, no blocks, no dtype); then
# WgradGatherSpec at valid padding over three k-waves; from 148 DwWgradMfmaSpec
# (preshifted and plain windows, packed images, 3x3 tap groups, the
# single-prologue PAD = KH - 1 case, a partial channel tile) and
# GroupedWgradMfmaSpec (4 and 8 channels per group, wave splits, a partial
# channel tile), each with a reject; then both with no blocks and with a
# negative block budget. Follows the 11 grouped non-grouped configs.
# ---------------------------------------------------------------------------
_WGRAD_DIRECT_CFG_BASE = 114

# bf16 4c (mfma_f32_4x4x4_bf16) configs, indices 163..165, after the direct
# wgrad ones.
_FOURC_BF16_CFG_BASE = 163


def _fourc_bf16_spec(idx: int):
    """Return (kind, spec, arch) for config ``_FOURC_BF16_CFG_BASE + idx``, or
    None. 0: odd H/W (right-edge column mask, H-edge rows); 1: gfx942 (the
    ``_1k`` atom is CDNA2+, so the kernel stays arch-neutral); 2: two waves per
    block and four q-tiles per wave."""
    # (N, H, W, groups, block_q, block_groups, arch)
    table = {
        0: (2, 13, 13, 32, 4, 16, "gfx950"),
        1: (1, 8, 8, 16, 8, 16, "gfx942"),
        2: (2, 15, 15, 32, 16, 32, "gfx950"),
    }
    sel = table.get(idx)
    if sel is None:
        return None
    N, H, W, groups, bq, bg, arch = sel
    p = DirectConvProblem(
        N=N, H=H, W=W, groups=groups, cpg=4, kpg=4, KH=3, KW=3, PAD=1, dtype="bf16"
    )
    return ("4c", DirectConv4cSpec(problem=p, block_q=bq, block_groups=bg), arch)


def _wp(**over):
    kw = dict(
        N=2,
        H=16,
        W=32,
        groups=1,
        cpg=64,
        kpg=64,
        KH=3,
        KW=3,
        PAD=1,
        stride=1,
        dtype="bf16",
    )
    kw.update(over)
    return DirectConvProblem(**kw)


def _dwp(C, **over):
    return _wp(groups=C, cpg=1, kpg=1, **over)


def _wgrad_direct_spec(idx: int):
    """Return (kind, spec, arch) for config ``_WGRAD_DIRECT_CFG_BASE + idx``."""
    ts, ls, cs, gs, dm, gm = (
        DirectWgradTiledSpec,
        DwWgradLdsSpec,
        DwWgradColsSpec,
        WgradGatherSpec,
        DwWgradMfmaSpec,
        GroupedWgradMfmaSpec,
    )
    table = {
        0: ("wgrad_tiled", ts(problem=_wp(cpg=128, kpg=128, H=30))),
        1: (
            "wgrad_tiled",
            ts(
                problem=_wp(groups=2, cpg=64, kpg=32, dtype="fp16"),
                wave_k16=1,
                wave_c16=4,
                waves_k=2,
                waves_c=1,
                rows_per_seg=12,
                prefetch=1,
            ),
        ),
        2: (
            "wgrad_tiled",
            ts(
                problem=_wp(N=4, H=15, W=40),
                unfold_n=4,
                rows_per_seg=15,
                row_unroll=3,
            ),
        ),
        3: ("wgrad_tiled", ts(problem=_wp(), q_groups=2, waves_k=2, rows_per_seg=6)),
        4: (
            "wgrad_tiled",
            ts(
                problem=_wp(),
                q_groups=2,
                pingpong=True,
                waves_k=2,
                wave_k16=1,
                rows_per_seg=12,
                prefetch=3,
                target_blocks=64,
            ),
        ),
        5: (
            "wgrad_tiled",
            ts(
                problem=_wp(KH=1, KW=1, PAD=0, cpg=32, kpg=32, dtype="fp16"),
                waves_k=1,
                waves_c=1,
                rows_per_seg=6,
                unfold_n=2,
            ),
        ),
        6: (
            "wgrad_tiled",
            ts(
                problem=_wp(KH=5, KW=5, PAD=2, cpg=48, kpg=40),
                wave_k16=1,
                wave_c16=1,
                waves_k=2,
                waves_c=2,
                rows_per_seg=15,
                unfold_n=2,
            ),
        ),
        7: ("dw_wgrad_fold", DwWgradFoldSpec(taps=9)),
        8: ("dw_wgrad_fold", DwWgradFoldSpec(taps=27, name="fold27")),
        9: (
            "dw_wgrad_cols",
            cs(problem=_dwp(3, KH=7, KW=7, PAD=3, stride=2, H=20)),
        ),
        10: (
            "dw_wgrad_cols",
            cs(
                problem=_dwp(3, KH=11, KW=11, PAD=5, H=12, W=48, dtype="fp16"),
                strip=16,
                threads=256,
                target_blocks=4096,
            ),
        ),
        11: (
            "dw_wgrad_cols",
            cs(problem=_dwp(8, H=10, W=24), threads=128),
        ),
        12: ("dw_wgrad_lds", ls(problem=_dwp(64))),
        13: (
            "dw_wgrad_lds",
            ls(
                problem=_dwp(192, stride=2, H=30, W=40, dtype="fp16"),
                lanes_w=8,
                prefetch=1,
                target_blocks=2048,
            ),
        ),
        14: (
            "dw_wgrad_lds",
            ls(
                problem=_dwp(40, H=20, W=20),
                lanes_c=4,
                lanes_w=16,
                cols_per_lane=2,
                prefetch=3,
                rows_per_seg=8,
            ),
        ),
        15: (
            "dw_wgrad_lds",
            ls(
                problem=_dwp(16, KH=3, KW=1, PAD=0, H=9, W=12),
                lanes_c=2,
                rows_per_seg=5,
            ),
        ),
        # Gathered X (S * C = 6), the stem shape.
        16: ("wgrad_gather", gs(problem=_wp(cpg=3, kpg=24, stride=2, H=32, W=64))),
        # Transpose-read X, phase 0.
        17: (
            "wgrad_gather",
            gs(
                problem=_wp(cpg=24, kpg=96, stride=2, H=30, W=40),
                wave_k16=1,
                waves_k=6,
                rows_per_seg=8,
                target_blocks=256,
            ),
        ),
        # Transpose-read X at phase 4.
        18: (
            "wgrad_gather",
            gs(problem=_wp(cpg=4, kpg=32, dtype="fp16"), prefetch=1),
        ),
        # 7x7 stride 2: five leading X rows per segment, two k-waves.
        19: (
            "wgrad_gather",
            gs(
                problem=_wp(cpg=3, kpg=64, KH=7, KW=7, PAD=3, stride=2, H=40, W=64),
                waves_k=2,
                rows_per_seg=8,
            ),
        ),
        # Must reject: the tiled kernel is stride 1 only.
        20: ("wgrad_tiled", ts(problem=_wp(stride=2))),
        # Must reject: no ds_read_b64_tr_b16 on gfx942.
        21: ("wgrad_gather", gs(problem=_wp(cpg=3, kpg=24, stride=2, W=64)), "gfx942"),
        # KW * C = 72 outputs over 64 threads: a second epilogue pass.
        22: ("dw_wgrad_cols", cs(problem=_dwp(8, KH=9, KW=9, PAD=4))),
        # Must reject: a 512 x 88 fp32 reduction buffer is 176 KiB of LDS.
        23: (
            "dw_wgrad_cols",
            cs(problem=_dwp(8, KH=11, KW=11, PAD=5), threads=512),
        ),
        # Must reject: the padded rings are 200,704 B (logical rows 145,152).
        24: ("dw_wgrad_lds", ls(problem=_dwp(64, stride=2), prefetch=5)),
        # Must reject: gfx942 DMAs one dword per lane into LDS, gfx1250 is wave32.
        25: ("dw_wgrad_lds", ls(problem=_dwp(64)), "gfx942"),
        26: ("dw_wgrad_lds", ls(problem=_dwp(64)), "gfx1250"),
        # Must reject: sixteen waves pad each X slot to 16 KiB (196,864 B).
        27: (
            "wgrad_gather",
            gs(problem=_wp(cpg=3, kpg=64, stride=2), wave_k16=1, waves_k=16),
        ),
        # Must reject: six 32 KiB ring buffers.
        28: (
            "wgrad_tiled",
            ts(
                problem=_wp(cpg=256, kpg=256),
                waves_k=8,
                waves_c=2,
                prefetch=5,
                row_unroll=6,
            ),
        ),
        # Must reject: row_unroll -3 divides by 3 but runs no rows.
        29: ("wgrad_tiled", ts(problem=_wp(), row_unroll=-3)),
        30: ("wgrad_tiled", ts(problem=_wp(), target_blocks=0)),
        31: ("wgrad_tiled", ts(problem=_wp(dtype=None))),
        # Valid padding, three k-waves of one atom (two k tiles).
        32: (
            "wgrad_gather",
            gs(problem=_wp(cpg=8, kpg=96, PAD=0), wave_k16=1, waves_k=3),
        ),
        # Valid padding at stride 2, three k-waves of two atoms (a partial tile).
        33: (
            "wgrad_gather",
            gs(
                problem=_wp(cpg=3, kpg=104, PAD=0, stride=2, W=64),
                waves_k=3,
                prefetch=1,
            ),
        ),
        # Depthwise 4x4x4 MFMA: 7x7 (preshifted window, both prologue bodies).
        34: (
            "dw_wgrad_mfma",
            dm(problem=_dwp(64, KH=7, KW=7, PAD=3, W=20), q_tiles=4),
        ),
        # 3x3 fp16 (no preshift), a partial channel tile, prefetch 1.
        35: (
            "dw_wgrad_mfma",
            dm(
                problem=_dwp(72, H=9, W=30, dtype="fp16"),
                rows_per_chunk=4,
                prefetch=1,
            ),
        ),
        # Three 7x7 images packed in one 32-wide tile.
        36: (
            "dw_wgrad_mfma",
            dm(
                problem=_dwp(128, KH=7, KW=7, PAD=3, N=5, H=7, W=7),
                rows_per_chunk=14,
                unfold_n=3,
            ),
        ),
        # 5x5 "valid" padding, two packed images, three staging buffers.
        37: (
            "dw_wgrad_mfma",
            dm(
                problem=_dwp(64, KH=5, KW=5, PAD=0, H=10, W=6),
                q_tiles=5,
                unfold_n=2,
                prefetch=3,
            ),
        ),
        # 11x11: 3x3 tap groups.
        38: (
            "dw_wgrad_mfma",
            dm(
                problem=_dwp(64, KH=11, KW=11, PAD=5, H=12, W=12),
                q_tiles=2,
                rows_per_chunk=8,
                target_blocks=64,
            ),
        ),
        # PAD = KH - 1: one prologue body only.
        39: ("dw_wgrad_mfma", dm(problem=_dwp(64, PAD=2), q_tiles=4)),
        # Must reject: stride 2.
        40: ("dw_wgrad_mfma", dm(problem=_dwp(64, stride=2))),
        # Grouped 4x4x4 MFMA: 4 channels per group.
        41: ("grouped_wgrad_mfma", gm(problem=_wp(groups=16, cpg=4, kpg=4))),
        # 8 channels per group, fp16, a partial channel tile, 2x2 wave split.
        42: (
            "grouped_wgrad_mfma",
            gm(
                problem=_wp(groups=9, cpg=8, kpg=8, dtype="fp16", W=24),
                waves_q=2,
                wave_q4=2,
                rows_per_chunk=8,
                prefetch=3,
            ),
        ),
        # 5x5, one wave over four column groups, four staging buffers.
        43: (
            "grouped_wgrad_mfma",
            gm(
                problem=_wp(groups=16, cpg=4, kpg=4, KH=5, KW=5, PAD=2),
                waves_q=1,
                wave_q4=4,
                prefetch=4,
                target_blocks=64,
            ),
        ),
        # Must reject: 16 channels per group.
        44: ("grouped_wgrad_mfma", gm(problem=_wp(groups=4, cpg=16, kpg=16))),
        # Must reject: no blocks, and a negative block budget.
        45: ("dw_wgrad_mfma", dm(problem=_dwp(64), target_blocks=0)),
        46: ("dw_wgrad_mfma", dm(problem=_dwp(64), target_blocks=-1)),
        47: (
            "grouped_wgrad_mfma",
            gm(problem=_wp(groups=16, cpg=4, kpg=4), target_blocks=0),
        ),
        48: (
            "grouped_wgrad_mfma",
            gm(problem=_wp(groups=16, cpg=4, kpg=4), target_blocks=-1),
        ),
    }
    sel = table.get(idx)
    if sel is None:
        return None
    return (sel[0], sel[1], sel[2] if len(sel) > 2 else "gfx950")


# ---------------------------------------------------------------------------
# Depthwise forward on a wave32 target (gfx1250), indices 166..170: the
# preloading, spatial, tiled and column-streamed kernels at wave_size 32, then
# the preloading kernel at wave_size 64: a logical wave of 64 threads, which
# the wave32 target runs as two hardware waves.
# ---------------------------------------------------------------------------
_WAVE32_CFG_BASE = 166


def _dwp_fwd(N, H, W, groups, KH, PAD, stride, dtype):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=1,
        kpg=1,
        KH=KH,
        KW=KH,
        PAD=PAD,
        stride=stride,
        dtype=dtype,
    )


def _wave32_spec(idx: int):
    """Return (kind, spec, arch) for config ``_WAVE32_CFG_BASE + idx``, or None."""
    if idx == 0:
        p = _dwp_fwd(2, 14, 14, 64, 3, 1, 1, "fp16")
        return (
            "dw",
            DirectDepthwiseSpec(problem=p, block_w=8, block_waves=2, wave_size=32),
            "gfx1250",
        )
    if idx == 1:
        p = _dwp_fwd(2, 14, 14, 3, 3, 1, 2, "bf16")
        return (
            "spatial",
            DirectDepthwiseSpatialSpec(problem=p, block_waves=2, wave_size=32),
            "gfx1250",
        )
    if idx == 2:
        p = _dwp_fwd(2, 16, 16, 96, 5, 2, 1, "bf16")
        return (
            "dw_tiled",
            DirectDepthwiseTiledSpec(
                problem=p, block_w=8, block_h=2, block_waves=2, wave_size=32
            ),
            "gfx1250",
        )
    if idx == 3:
        p = _dwp_fwd(2, 12, 12, 96, 3, 1, 2, "fp16")
        return (
            "dwcol",
            DirectDepthwiseColSpec(
                problem=p,
                block_h=8,
                block_w=2,
                block_waves=1,
                dtype="fp16",
                wave_size=32,
            ),
            "gfx1250",
        )
    if idx == 4:
        # wave64 (logical) spec on the wave32 target.
        p = _dwp_fwd(2, 14, 14, 64, 3, 1, 1, "fp16")
        return (
            "dw",
            DirectDepthwiseSpec(problem=p, block_w=8, block_waves=1),
            "gfx1250",
        )
    return None


def _gp(groups, cpg, kpg, *, N=2, H=20, W=24, KH=3, PAD=1, dtype="fp16"):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=cpg,
        kpg=kpg,
        KH=KH,
        KW=KH,
        PAD=PAD,
        stride=1,
        dtype=dtype,
    )


def _generic_spec(idx: int):
    """Return (kind, spec, arch) for config ``_GENERIC_CFG_BASE + idx``, or None."""
    gs = DirectConvSpec
    table = {
        # Baked channels.
        0: ("generic", gs(problem=_gp(8, 16, 16))),
        1: (
            "generic",
            gs(
                problem=_gp(4, 24, 40, dtype="bf16"),
                block_q=32,
                block_groups=2,
                double_buffer=False,
                block_h=8,
            ),
        ),
        2: (
            "generic",
            gs(
                problem=_gp(1, 192, 48, dtype="bf16"),
                block_groups=1,
                block_h=8,
                waves_k=4,
            ),
        ),
        3: (
            "generic",
            gs(
                problem=_gp(1, 192, 48, dtype="bf16"),
                block_groups=1,
                block_h=8,
                waves_k=2,
                fold_k32=True,
            ),
        ),
        4: (
            "generic",
            gs(
                problem=_gp(2, 48, 20),
                block_q=32,
                block_groups=1,
                waves_q=2,
                runtime_k_loop=True,
            ),
        ),
        5: (
            "generic",
            gs(problem=_gp(16, 32, 32), block_groups=4, persistent_grid=True),
        ),
        6: ("generic", gs(problem=_gp(4, 16, 16, KH=5, PAD=2), block_groups=2)),
        # Runtime channels: the problem only has to fit k_atoms / m_tiles.
        7: (
            "generic",
            gs(
                problem=_gp(1, 24, 24, dtype="bf16"),
                block_q=32,
                block_groups=1,
                block_h=8,
                runtime_channels=True,
                k_atoms=2,
                m_tiles=2,
            ),
        ),
        8: (
            "generic",
            gs(
                problem=_gp(1, 192, 48, dtype="bf16"),
                block_groups=1,
                block_h=8,
                waves_k=2,
                fold_k32=True,
                runtime_channels=True,
                k_atoms=6,
                m_tiles=3,
            ),
        ),
        9: (
            "generic",
            gs(
                problem=_gp(1, 320, 4),
                block_groups=1,
                block_h=8,
                waves_k=4,
                runtime_channels=True,
                k_atoms=20,
                m_tiles=1,
            ),
        ),
        10: (
            "generic",
            gs(
                problem=_gp(4, 40, 20),
                block_q=32,
                block_groups=2,
                waves_q=2,
                runtime_k_loop=True,
                runtime_channels=True,
                k_atoms=3,
                m_tiles=2,
            ),
        ),
        11: (
            "generic",
            gs(
                problem=_gp(8, 64, 64),
                block_groups=2,
                persistent_grid=True,
                runtime_channels=True,
                k_atoms=4,
                m_tiles=4,
            ),
        ),
        12: (
            "generic",
            gs(
                problem=_gp(2, 16, 16, KH=5, PAD=2, dtype="bf16"),
                block_groups=1,
                double_buffer=False,
                runtime_channels=True,
                k_atoms=1,
                m_tiles=1,
            ),
        ),
        # Weight transforms (original dgrad problem).
        13: ("transpose", DirectTransposeWeightsDgradSpec(problem=_gp(2, 24, 40))),
        14: (
            "transpose",
            DirectTransposeWeightsDgradSpec(
                problem=_gp(1, 48, 192, dtype="bf16"), runtime_channels=True
            ),
        ),
        15: (
            "reorganize",
            DirectReorganizeWeightsSpec(problem=_gp(2, 24, 40, dtype="bf16")),
        ),
        16: (
            "reorganize",
            DirectReorganizeWeightsSpec(problem=_gp(1, 48, 192), fold_k32=True),
        ),
        17: (
            "reorganize",
            DirectReorganizeWeightsSpec(problem=_gp(1, 24, 40), runtime_channels=True),
        ),
        18: (
            "reorganize",
            DirectReorganizeWeightsSpec(
                problem=_gp(1, 4, 320, dtype="bf16"),
                fold_k32=True,
                runtime_channels=True,
            ),
        ),
        19: (
            "transpose",
            DirectTransposeWeightsDgradSpec(
                problem=_gp(1, 16, 16, KH=5, PAD=2), runtime_channels=True
            ),
        ),
    }
    sel = table.get(idx)
    if sel is None:
        return None
    return (sel[0], sel[1], "gfx950")


def _nongrouped_p(
    N, H, W, C, K, *, KH=3, KW=3, PAD=1, stride=1, dtype="bf16", dil_h=1, dil_w=1
):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=1,
        cpg=C,
        kpg=K,
        KH=KH,
        KW=KW,
        PAD=PAD,
        stride=stride,
        dtype=dtype,
        dil_h=dil_h,
        dil_w=dil_w,
    )


def _nongrouped_dil_spec(idx: int):
    """Return (spec, arch) for dilated non-grouped config ``idx`` (emitted as
    ``_NONGROUPED_DIL_CFG_BASE + idx``), or None past the last one."""
    if idx == 0:
        # Row dilation 3, PAD 1: a taller halo, tap rows 3 input rows apart.
        return (
            _nongrouped_s(_nongrouped_p(2, 16, 32, 64, 128, dil_h=3)),
            "gfx950",
        )
    if idx == 1:
        # Column dilation 4, PAD 4: a wider halo, tap columns 4 apart.
        return (
            _nongrouped_s(_nongrouped_p(1, 32, 32, 64, 64, PAD=4, dil_w=4)),
            "gfx950",
        )
    if idx == 2:
        # fp16, both axes dilated, stride 2, double-buffered LDS.
        return (
            _nongrouped_s(
                _nongrouped_p(
                    1, 32, 64, 64, 64, PAD=2, stride=2, dtype="fp16", dil_h=2, dil_w=2
                ),
                ck=16,
                double_buffer=True,
            ),
            "gfx950",
        )
    return None


# The geometry every non-grouped config shares unless it overrides it: the
# shape the sweep's best configs take (t8x32x64, ck32, 2x2 waves, 32x32x16,
# iglp0).
_NONGROUPED_BASE = dict(
    tile_h=8, tile_w=32, tile_k=64, ck=32, waves_m=2, waves_n=2, iglp=0
)


def _nongrouped_s(problem, **kw):
    return DirectNongroupedConvSpec(problem=problem, **{**_NONGROUPED_BASE, **kw})


def _nongrouped_split_spec(idx: int):
    """Return (kind, spec, arch) for split-C config ``idx`` (emitted as
    ``_NONGROUPED_SPLIT_CFG_BASE + idx``), or None past the last one. The
    non-grouped ones pin the split grid decode, the per-split chunk range, the
    split-offset prefetch and the fp32 workspace epilogue; 5 and 6 pin the
    reduction pass and 7 a split both engines must reject."""
    if idx == 0:
        # bf16, stride 1, swizzled, two splits; partial W and H tiles.
        return (
            "nongrouped",
            _nongrouped_s(_nongrouped_p(2, 14, 14, 256, 128), split_c=2),
            "gfx950",
        )
    if idx == 1:
        # fp16, stride 2, double-buffered LDS, four splits.
        return (
            "nongrouped",
            _nongrouped_s(
                _nongrouped_p(1, 24, 24, 128, 64, stride=2, dtype="fp16"),
                ck=16,
                double_buffer=True,
                split_c=4,
            ),
            "gfx950",
        )
    if idx == 2:
        # bf16, no swizzle, sixteen one-chunk splits, waves_per_eu.
        return (
            "nongrouped",
            _nongrouped_s(
                _nongrouped_p(1, 7, 7, 512, 64),
                chiplet_swizzle=False,
                waves_per_eu=3,
                split_c=16,
            ),
            "gfx950",
        )
    if idx == 3:
        # fp16, 12-row tile over four waves (three rows each), 16-wide atom
        # (one accumulator quad), splits that do not divide the chunk count.
        return (
            "nongrouped",
            _nongrouped_s(
                _nongrouped_p(1, 12, 12, 320, 64, dtype="fp16"),
                tile_h=12,
                tile_w=16,
                waves_n=4,
                atom="16x16x32",
                split_c=4,
            ),
            "gfx950",
        )
    if idx == 4:
        # gfx942 32x32x8, fp16, double buffer (one channel tile per wave to fit
        # the 64 KiB LDS), no swizzle, two splits.
        return (
            "nongrouped",
            _nongrouped_s(
                _nongrouped_p(1, 28, 28, 256, 64, dtype="fp16"),
                tile_k=32,
                waves_m=1,
                atom="32x32x8",
                ck=16,
                double_buffer=True,
                chiplet_swizzle=False,
                split_c=2,
            ),
            "gfx942",
        )
    if idx == 5:
        # The fold: fp16, two slabs, default block.
        return (
            "split_reduce",
            DirectSplitReduceSpec(dtype="fp16", split_c=2),
            "gfx950",
        )
    if idx == 6:
        # The fold: bf16, eight slabs, a 128-thread block.
        return (
            "split_reduce",
            DirectSplitReduceSpec(dtype="bf16", split_c=8, threads=128),
            "gfx950",
        )
    if idx == 7:
        # Must reject: 16 splits of a C that has only 8 chunks of ck=32.
        return (
            "nongrouped",
            _nongrouped_s(_nongrouped_p(1, 7, 7, 256, 64), split_c=16),
            "gfx950",
        )
    return None


def _grouped_p(N, H, W, groups, cpg, kpg, *, stride=1, dtype="bf16"):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=cpg,
        kpg=kpg,
        KH=3,
        KW=3,
        PAD=1,
        stride=stride,
        dtype=dtype,
    )


def _nongrouped_grouped_spec(idx: int):
    """Return (kind, spec, arch) for grouped config ``idx`` (emitted as
    ``_NONGROUPED_GROUPED_CFG_BASE + idx``), or None past the last one. They pin
    the runtime cpg / kpg, the merged-group cell axis, the A channel base, both
    weight paths (group_merge 1, and the per-4-channel pieces masked per chunk
    for group_merge > 1) and the epilogue channel offset; 7..10 are specs both
    engines must reject."""
    if idx == 0:
        # group_merge 1, fp16, cpg 32: plain weight path; tile_k above the
        # merged K, so the K_M masks bite.
        return (
            "nongrouped",
            _nongrouped_s(
                _grouped_p(2, 14, 20, 4, 32, 32, dtype="fp16"),
                grouped=True,
                group_merge=1,
            ),
            "gfx950",
        )
    if idx == 1:
        # group_merge 8 of cpg 4, bf16, 16x16x32 (two 4-channel pieces per
        # fragment), ck 32: one merged block per chunk.
        return (
            "nongrouped",
            _nongrouped_s(
                _grouped_p(1, 16, 16, 16, 4, 4),
                tile_k=32,
                atom="16x16x32",
                ck=32,
                grouped=True,
                group_merge=8,
            ),
            "gfx950",
        )
    if idx == 2:
        # group_merge 4 of cpg 8, fp16, stride 2, 32x32x16, ck 16: two chunks
        # per merged block.
        return (
            "nongrouped",
            _nongrouped_s(
                _grouped_p(1, 24, 24, 8, 8, 8, stride=2, dtype="fp16"),
                tile_k=32,
                waves_m=1,
                atom="32x32x16",
                ck=16,
                grouped=True,
                group_merge=4,
            ),
            "gfx950",
        )
    if idx == 3:
        # group_merge 4 of cpg 12, bf16, 16x16x16 (one piece per fragment):
        # 16-channel chunks straddle the 12-channel groups; two channel tiles.
        return (
            "nongrouped",
            _nongrouped_s(
                _grouped_p(1, 16, 16, 8, 12, 12),
                tile_k=32,
                atom="16x16x16",
                ck=16,
                grouped=True,
                group_merge=4,
            ),
            "gfx950",
        )
    if idx == 4:
        # Grouped + split_c 2 + double-buffered LDS: the split chunk range over
        # the merged C and the fp32 workspace at the group's channel offset.
        return (
            "nongrouped",
            _nongrouped_s(
                _grouped_p(1, 14, 14, 4, 64, 32, dtype="fp16"),
                double_buffer=True,
                split_c=2,
                grouped=True,
                group_merge=2,
            ),
            "gfx950",
        )
    if idx == 5:
        # Grouped without the chiplet swizzle (plain k_tile / cell decode),
        # waves_per_eu, bf16, group_merge 2 of cpg 16.
        return (
            "nongrouped",
            _nongrouped_s(
                _grouped_p(2, 12, 20, 6, 16, 16),
                tile_k=32,
                waves_m=1,
                chiplet_swizzle=False,
                waves_per_eu=3,
                grouped=True,
                group_merge=2,
            ),
            "gfx950",
        )
    if idx == 6:
        # gfx942 32x32x8, fp16, group_merge 4 of cpg 8 (one piece per
        # fragment), ck 16.
        return (
            "nongrouped",
            _nongrouped_s(
                _grouped_p(1, 20, 20, 8, 8, 8, dtype="fp16"),
                tile_k=32,
                waves_m=1,
                atom="32x32x8",
                ck=16,
                grouped=True,
                group_merge=4,
            ),
            "gfx942",
        )
    if idx == 7:
        # Must reject: group_merge 3 does not divide 8 groups.
        return (
            "nongrouped",
            _nongrouped_s(
                _grouped_p(1, 16, 16, 8, 8, 8),
                tile_k=32,
                grouped=True,
                group_merge=3,
            ),
            "gfx950",
        )
    if idx == 8:
        # Must reject: cpg 6 is not a whole number of 4-channel weight loads
        # (merged C 24 and K 32 pass their own checks).
        return (
            "nongrouped",
            _nongrouped_s(
                _grouped_p(1, 16, 16, 4, 6, 8, dtype="fp16"),
                tile_k=32,
                waves_m=1,
                atom="32x32x8",
                ck=8,
                grouped=True,
                group_merge=4,
            ),
            "gfx950",
        )
    if idx == 9:
        # Must reject: group_merge without grouped=True.
        return (
            "nongrouped",
            _nongrouped_s(_nongrouped_p(1, 16, 16, 64, 64), group_merge=2),
            "gfx950",
        )
    if idx == 10:
        # Must reject: grouped=True with no groups (0 % group_merge is 0).
        return (
            "nongrouped",
            _nongrouped_s(
                _grouped_p(1, 16, 16, 0, 8, 8),
                tile_k=32,
                grouped=True,
                group_merge=1,
            ),
            "gfx950",
        )
    return None


def _reject_spec(idx: int):
    """Return (kind, spec, arch) for must-reject config ``idx`` (emitted as
    ``_REJECT_CFG_BASE + idx``), or None past the last one. Each pins a
    validator branch whose failure mode the engines must share."""
    if idx == 0:
        # The column-streamed tap indexing has no dilation term.
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=128, cpg=1, kpg=1, KH=3, KW=3, PAD=1, dil_h=2
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=4, block_waves=2, dtype="fp16"),
            "gfx950",
        )
    if idx == 1:
        # Tiled: a filter larger than the padded input (Ho = Wo = -26).
        p = DirectConvProblem(
            N=1, H=4, W=4, groups=64, cpg=1, kpg=1, KH=31, KW=31, PAD=0
        )
        return (
            "dw_tiled",
            DirectDepthwiseTiledSpec(problem=p, block_w=8, block_h=2),
            "gfx950",
        )
    if idx == 2:
        # Tiled: H + 2*PAD - KH = -1 at stride 2. Ho floors to 0; a truncating
        # division would make it 1.
        p = DirectConvProblem(
            N=1, H=4, W=16, groups=64, cpg=1, kpg=1, KH=5, KW=5, PAD=0, stride=2
        )
        return (
            "dw_tiled",
            DirectDepthwiseTiledSpec(problem=p, block_w=4, block_h=2),
            "gfx950",
        )
    if idx == 3:
        # Non-grouped: a zero dilation is refused, not read as 1.
        return (
            "nongrouped",
            _nongrouped_s(_nongrouped_p(2, 16, 32, 64, 128, dil_h=0)),
            "gfx950",
        )
    if idx == 4:
        # 16c: any dilation other than 1, zero included, is refused.
        p = DirectConvProblem(N=2, H=16, W=16, groups=16, cpg=16, kpg=16, dil_w=0)
        return ("16c", DirectConv16cSpec(problem=p, block_groups=4), "gfx950")
    return None


def _tiled_spec(idx: int):
    """Return (spec, arch) for depthwise-tiled config ``idx`` (emitted as
    ``_TILED_CFG_BASE + idx``), or None past the last one."""

    def prob(N, H, W, groups, KH, PAD, stride=1, dtype="bf16"):
        return DirectConvProblem(
            N=N,
            H=H,
            W=W,
            groups=groups,
            cpg=1,
            kpg=1,
            KH=KH,
            KW=KH,
            PAD=PAD,
            stride=stride,
            dtype=dtype,
        )

    if idx == 0:
        # 31x31: the large-filter case the row-streaming kernel cannot compile.
        return (
            DirectDepthwiseTiledSpec(
                problem=prob(1, 56, 56, 192, 31, 15), block_w=8, block_h=2
            ),
            "gfx950",
        )
    if idx == 1:
        # fp16 7x7, filter-row loop unrolled.
        return (
            DirectDepthwiseTiledSpec(
                problem=prob(4, 14, 14, 128, 7, 3, dtype="fp16"),
                block_w=14,
                block_h=7,
                unroll_rows=True,
            ),
            "gfx950",
        )
    if idx == 2:
        # Stride 2: two phases with a window each; two waves; gfx942.
        return (
            DirectDepthwiseTiledSpec(
                problem=prob(2, 28, 28, 256, 5, 2, stride=2),
                block_w=8,
                block_h=4,
                block_waves=2,
            ),
            "gfx942",
        )
    if idx == 3:
        # block_h 1: no carried window.
        return (
            DirectDepthwiseTiledSpec(
                problem=prob(2, 16, 16, 64, 3, 1), block_w=16, block_h=1
            ),
            "gfx950",
        )
    if idx == 4:
        # Non-"same" padding: outputs are indexed directly.
        return (
            DirectDepthwiseTiledSpec(
                problem=prob(2, 16, 16, 64, 3, 0, dtype="fp16"), block_w=4, block_h=2
            ),
            "gfx950",
        )
    if idx == 5:
        # Over the register budget: both engines reject.
        return (
            DirectDepthwiseTiledSpec(
                problem=prob(1, 56, 56, 192, 31, 15), block_w=16, block_h=4
            ),
            "gfx950",
        )
    return None


def _nongrouped_spec(idx: int):
    """Return (spec, arch) for non-grouped config ``idx`` (emitted as
    ``_NONGROUPED_CFG_BASE + idx``), or None past the last one."""
    if idx == 0:
        # bf16 baseline: even staging passes, chiplet swizzle, iglp 0.
        return _nongrouped_s(_nongrouped_p(2, 16, 32, 64, 128)), "gfx950"
    if idx == 1:
        # fp16 operand / store path.
        return _nongrouped_s(_nongrouped_p(2, 16, 32, 64, 128, dtype="fp16")), "gfx950"
    if idx == 2:
        # Ping-pong LDS: parity-selected buffers, one barrier per chunk.
        return (
            _nongrouped_s(_nongrouped_p(2, 16, 32, 64, 128), double_buffer=True),
            "gfx950",
        )
    if idx == 3:
        # No chiplet swizzle: plain mod/div grid decode.
        return (
            _nongrouped_s(_nongrouped_p(2, 16, 32, 64, 128), chiplet_swizzle=False),
            "gfx950",
        )
    if idx == 4:
        # waves_per_eu attribute, no iglp_opt.
        return (
            _nongrouped_s(_nongrouped_p(2, 16, 32, 64, 128), iglp=None, waves_per_eu=3),
            "gfx950",
        )
    if idx == 5:
        # Stride 2: input-row sharing across taps, strided staging window.
        return _nongrouped_s(_nongrouped_p(1, 32, 64, 64, 64, stride=2)), "gfx950"
    if idx == 6:
        # 1x1 pointwise filter, PAD 0.
        return (
            _nongrouped_s(_nongrouped_p(1, 16, 32, 64, 64, KH=1, KW=1, PAD=0)),
            "gfx950",
        )
    if idx == 7:
        # 16x16x32 atom on a width that is not a multiple of 32 (partial W).
        return (
            _nongrouped_s(
                _nongrouped_p(2, 20, 40, 64, 64),
                tile_w=48,
                tile_k=32,
                waves_m=1,
                atom="16x16x32",
            ),
            "gfx950",
        )
    if idx == 8:
        # 16x16x16 atom (4-half fragments), ck 16, single-tile width.
        return (
            _nongrouped_s(
                _nongrouped_p(1, 16, 32, 64, 64, dtype="fp16"),
                tile_w=16,
                tile_k=32,
                ck=16,
                waves_m=1,
                atom="16x16x16",
            ),
            "gfx950",
        )
    if idx == 9:
        # 32x32x8 atom on gfx942 (no 32x32x16 there).
        return (
            _nongrouped_s(
                _nongrouped_p(1, 16, 32, 64, 64, dtype="fp16"), atom="32x32x8", ck=16
            ),
            "gfx942",
        )
    if idx == 10:
        # gfx942 bf16 16x16x16 with uneven staging passes.
        return (
            _nongrouped_s(
                _nongrouped_p(1, 16, 48, 32, 64),
                tile_w=48,
                tile_k=32,
                ck=16,
                waves_m=1,
                waves_n=4,
                atom="16x16x16",
            ),
            "gfx942",
        )
    if idx == 11:
        # 5x5 filter, PAD 2, partial K tile, swizzle_wgm 1.
        return (
            _nongrouped_s(
                _nongrouped_p(1, 16, 32, 64, 96, KH=5, KW=5, PAD=2), swizzle_wgm=1
            ),
            "gfx950",
        )
    if idx == 12:
        # Partial H tile + multi column-block tile (tile_w = 2 atoms).
        return (
            _nongrouped_s(
                _nongrouped_p(1, 20, 64, 64, 64), tile_h=16, tile_w=64, waves_n=4
            ),
            "gfx950",
        )
    if idx == 13:
        # A production-sized config: N4 C640 K640 64x64, t16x64x128 w2x4.
        return (
            _nongrouped_s(
                _nongrouped_p(4, 64, 64, 640, 640),
                tile_h=16,
                tile_w=64,
                tile_k=128,
                ck=16,
                waves_m=2,
                waves_n=4,
            ),
            "gfx950",
        )
    if idx == 14:
        # Double buffer + stride 2 + iglp 1 + fp16 + no swizzle together.
        return (
            _nongrouped_s(
                _nongrouped_p(1, 32, 64, 64, 64, stride=2, dtype="fp16"),
                ck=16,
                double_buffer=True,
                iglp=1,
                chiplet_swizzle=False,
            ),
            "gfx950",
        )
    return None


def _spec(idx: int):
    """Return (kind, spec, arch) for config index `idx`."""
    if idx == 0:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=16, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "16c",
            DirectConv16cSpec(problem=p, block_groups=4, fold_k32=True),
            "gfx950",
        )
    if idx == 1:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=16, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "16c",
            DirectConv16cSpec(problem=p, block_groups=8, fold_k32=True),
            "gfx950",
        )
    if idx == 2:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=64, cpg=4, kpg=4, KH=3, KW=3, PAD=1, stride=1
        )
        return ("4c", DirectConv4cSpec(problem=p, block_q=4, block_groups=16), "gfx950")
    if idx == 3:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=64, cpg=4, kpg=4, KH=3, KW=3, PAD=1, stride=1
        )
        return ("4c", DirectConv4cSpec(problem=p, block_q=8, block_groups=16), "gfx950")
    if idx == 4:
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "16c",
            DirectConv16cSpec(problem=p, block_groups=1, fold_k32=False),
            "gfx942",
        )
    if idx == 5:
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=16, cpg=4, kpg=4, KH=3, KW=3, PAD=1, stride=1
        )
        return ("4c", DirectConv4cSpec(problem=p, block_q=4, block_groups=16), "gfx950")
    if idx == 6:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=16, cpg=8, kpg=8, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "8c",
            DirectConv8cSpec(problem=p, block_q=16, block_groups=8, double_buffer=True),
            "gfx950",
        )
    if idx == 7:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=8, cpg=32, kpg=32, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "32c",
            DirectConv32cSpec(
                problem=p, block_q=32, block_groups=4, double_buffer=True
            ),
            "gfx950",
        )
    if idx == 8:
        # groups must be divisible by block_ch = block_waves * wave_size (2 * 64 = 128)
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=128, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "depthwise",
            DirectDepthwiseSpec(problem=p, block_w=16, block_waves=2),
            "gfx950",
        )
    if idx == 9:
        # depthwise with stride=2: exercises Ho/Wo output descriptors and
        # stride-aware flush (p_flush_val % stride == 0 guard)
        p = DirectConvProblem(
            N=2, H=14, W=14, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=2
        )
        return (
            "depthwise",
            DirectDepthwiseSpec(problem=p, block_w=8, block_waves=1),
            "gfx950",
        )
    if idx == 10:
        # spatial layout: groups=3 (non-power-of-two, exercises partial wave)
        p = DirectConvProblem(
            N=2, H=14, W=14, groups=3, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "spatial",
            DirectDepthwiseSpatialSpec(problem=p, block_waves=2),
            "gfx950",
        )
    if idx == 11:
        # spatial layout with stride=2: exercises Ho/Wo + spatial thread mapping
        p = DirectConvProblem(
            N=2, H=14, W=14, groups=3, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=2
        )
        return (
            "spatial",
            DirectDepthwiseSpatialSpec(problem=p, block_waves=1),
            "gfx950",
        )
    if idx == 12:
        # dgrad: baseline grouped dgrad stride=1
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dgrad",
            DirectConvDgradSpec(problem=p, block_q=16, block_groups=8),
            "gfx950",
        )
    if idx == 13:
        # dgrad: larger groups / different block_groups
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=8, cpg=32, kpg=32, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dgrad",
            DirectConvDgradSpec(problem=p, block_q=16, block_groups=4),
            "gfx950",
        )
    if idx == 14:
        # dgrad: gfx942 target
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dgrad",
            DirectConvDgradSpec(problem=p, block_q=16, block_groups=8),
            "gfx942",
        )
    if idx == 15:
        # depthwise_dgrad: stride=1
        p = DirectConvProblem(
            N=2, H=14, W=14, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dw_dgrad",
            DirectDepthwiseDgradSpec(problem=p, block_w=8, block_waves=1),
            "gfx950",
        )
    if idx == 16:
        # depthwise_dgrad: stride=2 exercises divisibility checks
        p = DirectConvProblem(
            N=2, H=14, W=14, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=2
        )
        return (
            "dw_dgrad",
            DirectDepthwiseDgradSpec(problem=p, block_w=8, block_waves=1),
            "gfx950",
        )
    if idx == 17:
        # 16c bf16: exercises bf16 I/O, bf16 load/store taps, mfma_f32_16x16x16_bf16
        p = DirectConvProblem(
            N=32,
            H=200,
            W=200,
            groups=16,
            cpg=16,
            kpg=16,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "16c",
            DirectConv16cSpec(problem=p, block_groups=8, fold_k32=False),
            "gfx950",
        )
    if idx == 18:
        # 8c bf16: exercises bf16 I/O, bf16 load/store taps, mfma_f32_16x16x16_bf16
        p = DirectConvProblem(
            N=32,
            H=200,
            W=200,
            groups=16,
            cpg=8,
            kpg=8,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "8c",
            DirectConv8cSpec(problem=p, block_q=16, block_groups=8, double_buffer=True),
            "gfx950",
        )
    if idx == 19:
        # dgrad bf16: exercises bf16 I/O on the scalar-FMA grouped dgrad path
        p = DirectConvProblem(
            N=2,
            H=8,
            W=8,
            groups=8,
            cpg=16,
            kpg=16,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "dgrad",
            DirectConvDgradSpec(problem=p, block_q=16, block_groups=8),
            "gfx950",
        )
    if idx == 20:
        # 16c bf16 with fold_k32=True: pins the non-default fold_k32 path under bf16
        p = DirectConvProblem(
            N=32,
            H=200,
            W=200,
            groups=16,
            cpg=16,
            kpg=16,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "16c",
            DirectConv16cSpec(problem=p, block_groups=8, fold_k32=True),
            "gfx950",
        )
    if idx == 21:
        # 32c bf16: exercises bf16 I/O on the 32c MFMA path
        p = DirectConvProblem(
            N=32,
            H=200,
            W=200,
            groups=32,
            cpg=32,
            kpg=32,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "32c",
            DirectConv32cSpec(problem=p, block_groups=8),
            "gfx950",
        )
    if idx == 22:
        # depthwise forward bf16: exercises bf16 I/O on the scalar-FMA depthwise path
        p = DirectConvProblem(
            N=2,
            H=14,
            W=14,
            groups=64,
            cpg=1,
            kpg=1,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "dw",
            DirectDepthwiseSpec(problem=p, block_w=8, block_waves=1),
            "gfx950",
        )
    if idx == 23:
        # depthwise spatial bf16: exercises bf16 I/O on the small-group spatial path
        p = DirectConvProblem(
            N=2,
            H=14,
            W=14,
            groups=16,
            cpg=1,
            kpg=1,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "spatial",
            DirectDepthwiseSpatialSpec(problem=p, block_waves=1),
            "gfx950",
        )
    if idx == 24:
        # depthwise dgrad bf16: exercises bf16 I/O on the scalar-FMA depthwise dgrad path
        p = DirectConvProblem(
            N=2,
            H=14,
            W=14,
            groups=64,
            cpg=1,
            kpg=1,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "dw_dgrad",
            DirectDepthwiseDgradSpec(problem=p, block_w=8, block_waves=1),
            "gfx950",
        )
    # ---- wgrad (backward weights) ----
    if idx == 25:
        # Defaults: mfma_k=32 (VEC_CH=8, two ds_read_tr per fragment),
        # waves_k=waves_c=waves_q=1, ho_per_block=4.
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return ("wgrad", DirectConvWgradSpec(problem=p), "gfx950")
    if idx == 26:
        # Narrow MFMA: mfma_k=16 (VEC_CH=4, one ds_read_tr per fragment).
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "wgrad",
            DirectConvWgradSpec(problem=p, mfma_k=16, ho_per_block=2),
            "gfx950",
        )
    if idx == 27:
        # Multi-wave: K/C/Q all split, so n_k_tiles = n_c_tiles = 2,
        # STRIP_GROUPS = 2 and the kernel name carries the _wq flag.
        p = DirectConvProblem(
            N=4, H=16, W=16, groups=4, cpg=64, kpg=64, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "wgrad",
            DirectConvWgradSpec(
                problem=p, waves_k=2, waves_c=2, waves_q=2, ho_per_block=3
            ),
            "gfx950",
        )
    if idx == 28:
        # gfx942 has no 16x16x32 f16 atom -> both engines reject (mfma_k=32).
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return ("wgrad", DirectConvWgradSpec(problem=p), "gfx942")
    if idx == 29:
        # gfx942 with mfma_k=16 clears the atom gate and is rejected one check
        # later, on the missing ds_read_tr16_b64 the LDS staging needs.
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return ("wgrad", DirectConvWgradSpec(problem=p, mfma_k=16), "gfx942")
    if idx == 30:
        # wgrad bf16: bf16 I/O and the bf16 MFMA atom, same LDS transpose
        # staging. Pins that only the atom and the element type move.
        p = DirectConvProblem(
            N=2,
            H=8,
            W=8,
            groups=8,
            cpg=16,
            kpg=16,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return ("wgrad", DirectConvWgradSpec(problem=p), "gfx950")
    if idx == 31:
        # wgrad bf16 at mfma_k=16: the narrow atom under bf16, one ds_read_tr
        # per fragment.
        p = DirectConvProblem(
            N=2,
            H=8,
            W=8,
            groups=8,
            cpg=16,
            kpg=16,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "wgrad",
            DirectConvWgradSpec(problem=p, mfma_k=16, ho_per_block=2),
            "gfx950",
        )
    if idx == 32:
        # column-streamed depthwise, stride=1 fp16, default 16-row tile. Exact
        # channel and W tiles, but the guards bound against kernargs (AOT), so
        # they are emitted all the same.
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=128, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=4, block_waves=2, dtype="fp16"),
            "gfx950",
        )
    if idx == 33:
        # col stride=2 bf16 with partial channel and W tiles (groups=70 % 64,
        # Wo=5 % 4) and two-row tiles, so Ho=5 needs a partial last row tile.
        p = DirectConvProblem(
            N=1, H=9, W=9, groups=70, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=2
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(
                problem=p, block_h=2, block_w=4, block_waves=1, dtype="bf16"
            ),
            "gfx950",
        )
    if idx == 34:
        # col stride=3: exercises the (y - r) % stride tap pruning at a stride
        # no other case reaches.
        p = DirectConvProblem(
            N=1, H=16, W=16, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=3
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=4, block_waves=1, dtype="bf16"),
            "gfx950",
        )
    if idx == 35:
        # col with a large filter (31x31): the regime the variant exists for --
        # KW rides the runtime loop so only KH weights are live. A 4-row tile
        # still unrolls (4-1)+31 input rows.
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=64, cpg=1, kpg=1, KH=31, KW=31, PAD=15, stride=1
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(
                problem=p, block_h=4, block_w=4, block_waves=1, dtype="fp16"
            ),
            "gfx950",
        )
    if idx == 36:
        # col 1x1 / PAD=0 degenerate with a non-power-of-two group count
        p = DirectConvProblem(
            N=2, H=6, W=6, groups=3, cpg=1, kpg=1, KH=1, KW=1, PAD=0, stride=1
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=2, block_waves=1, dtype="fp16"),
            "gfx950",
        )
    if idx == 37:
        # col with KH != KW (5x3): separates the unrolled axis from the runtime one
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=128, cpg=1, kpg=1, KH=5, KW=3, PAD=2, stride=1
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=4, block_waves=2, dtype="bf16"),
            "gfx950",
        )
    if idx == 38:
        # col stride=2 with valid padding (PAD=0) and an odd 3-row tile
        p = DirectConvProblem(
            N=1, H=13, W=13, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=0, stride=2
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(
                problem=p, block_h=3, block_w=6, block_waves=1, dtype="fp16"
            ),
            "gfx950",
        )
    if idx == 39:
        # col with block_w=1 and a channel tail (groups=100 % 128)
        p = DirectConvProblem(
            N=1, H=10, W=10, groups=100, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=1, block_waves=2, dtype="bf16"),
            "gfx950",
        )
    if idx == 40:
        # dwcol PAD-overhang: PAD=2 > (KH-1)/2=1 with stride=2 and two-row
        # tiles. n_iters = (block_h-1)*stride + KH is cross-verified by the
        # C/Python byte-identity gate.
        p = DirectConvProblem(
            N=1, H=10, W=10, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=2, stride=2
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(
                problem=p, block_h=2, block_w=4, block_waves=1, dtype="fp16"
            ),
            "gfx950",
        )
    if idx == 41:
        # depthwise 7x7 stride 2 bf16: PAD > 1 shifts the row stream's static
        # slots, and stride 2 gates its flushes on the row parity.
        p = DirectConvProblem(
            N=2,
            H=14,
            W=14,
            groups=128,
            cpg=1,
            kpg=1,
            KH=7,
            KW=7,
            PAD=3,
            stride=2,
            dtype="bf16",
        )
        return (
            "dw",
            DirectDepthwiseSpec(problem=p, block_w=8, block_waves=2),
            "gfx950",
        )
    if idx >= _WAVE32_CFG_BASE:
        sel = _wave32_spec(idx - _WAVE32_CFG_BASE)
        if sel is not None:
            return sel
        raise SystemExit(f"unknown config index {idx}")
    if idx >= _FOURC_BF16_CFG_BASE:
        sel = _fourc_bf16_spec(idx - _FOURC_BF16_CFG_BASE)
        if sel is not None:
            return sel
        raise SystemExit(f"unknown config index {idx}")
    if idx >= _WGRAD_DIRECT_CFG_BASE:
        sel = _wgrad_direct_spec(idx - _WGRAD_DIRECT_CFG_BASE)
        if sel is not None:
            return sel
        raise SystemExit(f"unknown config index {idx}")
    if idx >= _NONGROUPED_GROUPED_CFG_BASE:
        sel = _nongrouped_grouped_spec(idx - _NONGROUPED_GROUPED_CFG_BASE)
        if sel is not None:
            return sel
        raise SystemExit(f"unknown config index {idx}")
    if idx >= _NONGROUPED_SPLIT_CFG_BASE:
        sel = _nongrouped_split_spec(idx - _NONGROUPED_SPLIT_CFG_BASE)
        if sel is not None:
            return sel
        raise SystemExit(f"unknown config index {idx}")
    if idx >= _GENERIC_CFG_BASE:
        sel = _generic_spec(idx - _GENERIC_CFG_BASE)
        if sel is not None:
            return sel
    # ---- wgrad, runtime channels (indices 71..74) ----
    # cpg/kpg come from p_total_c / p_total_k / p_groups: runtime k/c tile
    # counts in the grid decode, a dynamic dW descriptor and runtime masks.
    if idx == 71:
        # Defaults otherwise: the baked config 25 with runtime channels.
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "wgrad",
            DirectConvWgradSpec(problem=p, runtime_channels=True),
            "gfx950",
        )
    if idx == 72:
        # bf16, K/C/Q split with a non-power-of-two block_c (waves_c=3).
        p = DirectConvProblem(
            N=2,
            H=8,
            W=8,
            groups=1,
            cpg=48,
            kpg=192,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "wgrad",
            DirectConvWgradSpec(
                problem=p,
                waves_k=2,
                waves_c=3,
                waves_q=2,
                ho_per_block=3,
                runtime_channels=True,
            ),
            "gfx950",
        )
    if idx == 73:
        # 1x1 filter, PAD 0, the narrow atom and four k-waves.
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=4, cpg=64, kpg=64, KH=1, KW=1, PAD=0, stride=1
        )
        return (
            "wgrad",
            DirectConvWgradSpec(
                problem=p,
                waves_k=4,
                mfma_k=16,
                ho_per_block=2,
                runtime_channels=True,
            ),
            "gfx950",
        )
    if idx == 74:
        # 5x5 filter with valid padding, bf16, a ragged channel count.
        p = DirectConvProblem(
            N=1,
            H=12,
            W=12,
            groups=1,
            cpg=24,
            kpg=24,
            KH=5,
            KW=5,
            PAD=0,
            stride=1,
            dtype="bf16",
        )
        return (
            "wgrad",
            DirectConvWgradSpec(
                problem=p, waves_c=2, ho_per_block=5, runtime_channels=True
            ),
            "gfx950",
        )
    if idx >= _REJECT_CFG_BASE:
        sel = _reject_spec(idx - _REJECT_CFG_BASE)
        if sel is not None:
            return sel
        raise SystemExit(f"unknown config index {idx}")
    if idx >= _NONGROUPED_DIL_CFG_BASE:
        sel = _nongrouped_dil_spec(idx - _NONGROUPED_DIL_CFG_BASE)
        if sel is not None:
            return ("nongrouped", *sel)
        raise SystemExit(f"unknown config index {idx}")
    if idx >= _TILED_CFG_BASE:
        sel = _tiled_spec(idx - _TILED_CFG_BASE)
        if sel is not None:
            return ("dw_tiled", *sel)
        raise SystemExit(f"unknown config index {idx}")
    if idx >= _NONGROUPED_CFG_BASE:
        sel = _nongrouped_spec(idx - _NONGROUPED_CFG_BASE)
        if sel is not None:
            return ("nongrouped", *sel)
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: conv_direct_grouped_emit.py <config_index>\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    kind, spec, arch = _spec(idx)
    if kind == "16c":
        kernel = build_direct_conv_16c(spec, arch=arch)
    elif kind == "4c":
        kernel = build_direct_conv_4c(spec, arch=arch)
    elif kind == "8c":
        kernel = build_direct_conv_8c(spec, arch=arch)
    elif kind == "32c":
        kernel = build_direct_conv_32c(spec, arch=arch)
    elif kind == "wgrad":
        kernel = build_direct_conv_wgrad(spec, arch=arch)
    elif kind == "spatial":
        kernel = build_direct_depthwise_spatial(spec, arch=arch)
    elif kind == "dwcol":
        kernel = build_direct_depthwise_col(spec, arch=arch)
    elif kind == "dgrad":
        kernel = build_direct_conv_dgrad(spec, arch=arch)
    elif kind == "dw_dgrad":
        kernel = build_direct_depthwise_dgrad(spec, arch=arch)
    elif kind == "nongrouped":
        kernel = build_direct_conv_nongrouped(spec, arch=arch)
    elif kind == "generic":
        kernel = build_direct_conv(spec, arch=arch)
    elif kind == "transpose":
        kernel = build_direct_transpose_weights_dgrad(spec, arch=arch)
    elif kind == "reorganize":
        kernel = build_direct_reorganize_weights(spec, arch=arch)
    elif kind == "dw_tiled":
        kernel = build_direct_depthwise_tiled(spec, arch=arch)
    elif kind == "wgrad_tiled":
        kernel = build_direct_wgrad_tiled(spec, arch=arch)
    elif kind == "dw_wgrad_fold":
        kernel = build_dw_wgrad_fold(spec, arch=arch)
    elif kind == "dw_wgrad_cols":
        kernel = build_dw_wgrad_cols(spec, arch=arch)
    elif kind == "dw_wgrad_lds":
        kernel = build_dw_wgrad_lds(spec, arch=arch)
    elif kind == "wgrad_gather":
        kernel = build_wgrad_gather(spec, arch=arch)
    elif kind == "split_reduce":
        kernel = build_direct_conv_split_reduce(spec, arch=arch)
    elif kind == "dw_wgrad_mfma":
        kernel = build_dw_wgrad_mfma(spec, arch=arch)
    elif kind == "grouped_wgrad_mfma":
        kernel = build_grouped_wgrad_mfma(spec, arch=arch)
    else:
        kernel = build_direct_depthwise(spec, arch=arch)
    if mode == "ll":
        text = _native_lower(kernel, arch=arch)
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    elif mode == "verify":
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    else:
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
