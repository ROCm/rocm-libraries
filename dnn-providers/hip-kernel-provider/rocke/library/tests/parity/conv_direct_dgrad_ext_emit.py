#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_dgrad_ext_emit.py -- Python reference emitter for the
# direct dgrad parity family that the AOT dgrad kernels added after
# conv_direct_grouped extend. Selects one config by argv[1], builds the kernel
# and prints _native_lower(arch=<cfg arch>) so it can be byte-compared with the
# C emitter conv_direct_dgrad_ext_emit.c.
#
# Indices 0..10: DirectDepthwiseDgradWindowedSpec, VALU and dot2 forms
# (10 is gfx942; 6 -- dot2 on gfx942 -- both engines reject).
# Indices 11..19: the Toeplitz MFMA form (15 -- gfx942 -- and 17 -- groups % 8
# != 0 -- both reject; 19, block_h = 0, both reject).
# Indices 20..29: the 4c dgrad kernel (DirectConv4cSpec on the transposed
# problem): 20..23 fused weights, gathered (20, 21 = gfx942) or LDS-staged
# with transpose reads (22, 23 = 1x1 with a partial staging pass); 24..27 the
# row-staged runtime row stream (stage_rows: odd W, 1x1 with a partial
# row-staging pass, two q waves x two channel waves, H = 1); 28 LDS weights on
# gfx942 and 29 a 5x5 filter (more than DCONV4C_MAX_TAPS taps) both reject.
from _emit_common import run_emit
from kernels.common.conv_direct_grouped import (
    DirectConv4cSpec,
    DirectConvProblem,
    DirectDepthwiseDgradWindowedSpec,
    build_direct_conv_4c,
    build_direct_depthwise_dgrad_windowed,
)

# (N, H, W, groups, KH, KW, PAD, dtype, arch, knobs)
_WIN_CFGS = (
    # 7x7, scalar f32 FMA path; the tile covers the image.
    (2, 14, 14, 64, 7, 7, 3, "bf16", "gfx950", dict(block_w=7, block_h=14)),
    # 7x7 with fdot2 tap pairing (odd KW pads a zero weight).
    (2, 14, 14, 64, 7, 7, 3, "bf16", "gfx950", dict(block_w=7, block_h=14, dot2=True)),
    # ch_per_lane=2 (dword loads/stores, packed f32 FMA).
    (
        2,
        12,
        12,
        128,
        3,
        3,
        1,
        "fp16",
        "gfx950",
        dict(block_w=6, block_waves=2, ch_per_lane=2, block_h=12),
    ),
    # Ragged row tiles (block_h=4, H=13) re-read the halo.
    (2, 13, 13, 64, 5, 5, 2, "fp16", "gfx950", dict(block_w=5, block_h=4)),
    # Even row tiles + dot2, W=13 not a multiple of block_w.
    (1, 8, 13, 40, 3, 3, 1, "bf16", "gfx950", dict(block_w=4, block_h=4, dot2=True)),
    # Even KW=4 with dot2 (no zero pad).
    (2, 9, 9, 72, 4, 4, 1, "fp16", "gfx950", dict(block_w=8, block_h=9, dot2=True)),
    # dot2 on gfx942: both engines reject.
    (2, 8, 8, 64, 3, 3, 1, "bf16", "gfx942", dict(block_w=4, block_h=8, dot2=True)),
    # ch_per_lane=4 (dwordx2) bf16, a tile taller than the image.
    (
        2,
        7,
        7,
        128,
        3,
        3,
        1,
        "bf16",
        "gfx950",
        dict(block_w=4, ch_per_lane=4, block_h=8),
    ),
    # KW=1 with dot2 (tail pair only, no full pairs).
    (2, 9, 11, 96, 3, 1, 1, "fp16", "gfx950", dict(block_w=5, block_h=9, dot2=True)),
    # Non-square 3x5 fp16 dot2 with ragged row tiles.
    (
        1,
        13,
        10,
        72,
        3,
        5,
        2,
        "fp16",
        "gfx950",
        dict(block_w=10, block_waves=2, block_h=5, dot2=True),
    ),
    # The VALU form on gfx942.
    (2, 9, 9, 64, 5, 5, 2, "fp16", "gfx942", dict(block_w=8, block_waves=4, block_h=7)),
    # Toeplitz MFMA form: odd N under w_fold 2 (empty second image of the last
    # tile).
    (
        3,
        14,
        14,
        64,
        7,
        7,
        3,
        "bf16",
        "gfx950",
        dict(mfma=True, block_waves=8, w_fold=2, prefetch_rows=2, block_h=14),
    ),
    # Ragged row tiles + 2 W tiles + a partial channel block, w_fold 1.
    (
        2,
        13,
        37,
        40,
        5,
        5,
        2,
        "fp16",
        "gfx950",
        dict(mfma=True, block_waves=4, w_fold=1, prefetch_rows=1, block_h=4),
    ),
    # 9x9, w_fold 4, three rows in flight.
    (
        2,
        10,
        11,
        16,
        9,
        9,
        4,
        "bf16",
        "gfx950",
        dict(mfma=True, block_waves=2, w_fold=4, prefetch_rows=3, block_h=5),
    ),
    # 11x11 ragged tiles (3 passes per filter row).
    (
        1,
        13,
        13,
        64,
        11,
        11,
        5,
        "fp16",
        "gfx950",
        dict(mfma=True, block_waves=8, w_fold=2, prefetch_rows=2, block_h=4),
    ),
    # gfx942: both engines reject.
    (
        2,
        14,
        14,
        64,
        7,
        7,
        3,
        "bf16",
        "gfx942",
        dict(mfma=True, block_waves=8, w_fold=2, prefetch_rows=2, block_h=14),
    ),
    # 3x3 pad 0 (single pass), one wave.
    (
        1,
        9,
        12,
        8,
        3,
        3,
        0,
        "fp16",
        "gfx950",
        dict(mfma=True, block_waves=1, w_fold=2, prefetch_rows=1, block_h=9),
    ),
    # groups % 8 != 0: both engines reject.
    (
        2,
        14,
        14,
        12,
        7,
        7,
        3,
        "bf16",
        "gfx950",
        dict(mfma=True, block_waves=1, w_fold=2, prefetch_rows=2, block_h=14),
    ),
    # Non-square 7x5.
    (
        2,
        11,
        15,
        32,
        7,
        5,
        2,
        "bf16",
        "gfx950",
        dict(mfma=True, block_waves=4, w_fold=2, prefetch_rows=2, block_h=12),
    ),
    # block_h = 0 (whole image): the image height is a kernarg, both reject.
    (2, 8, 8, 64, 3, 3, 1, "fp16", "gfx950", dict(block_w=8, block_h=0)),
)


# 4c dgrad configs from index _FOURC_CFG_BASE:
# (N, H, W, groups, KH, dtype, arch, block_q, block_groups, lds, stage_rows)
_FOURC_CFG_BASE = len(_WIN_CFGS)
_FOURC_CFGS = (
    (2, 13, 13, 32, 3, "bf16", "gfx950", 4, 16, False, False),
    (1, 8, 8, 16, 3, "fp16", "gfx942", 8, 16, False, False),
    (2, 15, 15, 32, 3, "bf16", "gfx950", 16, 32, True, False),
    (1, 7, 9, 64, 1, "fp16", "gfx950", 4, 64, True, False),
    (2, 13, 13, 32, 3, "bf16", "gfx950", 4, 16, True, True),
    (1, 7, 9, 64, 1, "fp16", "gfx950", 4, 64, True, True),
    (2, 9, 21, 64, 3, "bf16", "gfx950", 8, 32, True, True),
    (1, 1, 6, 16, 3, "fp16", "gfx950", 4, 16, True, True),
    (2, 13, 13, 32, 3, "bf16", "gfx942", 4, 16, True, False),
    (1, 8, 8, 16, 5, "fp16", "gfx950", 4, 16, False, False),
)


def _fourc_spec(idx: int):
    N, H, W, G, KH, dtype, arch, bq, bg, lds, sr = _FOURC_CFGS[idx]
    problem = DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=G,
        cpg=4,
        kpg=4,
        KH=KH,
        KW=KH,
        PAD=(KH - 1) // 2,
        stride=1,
        dtype=dtype,
    )
    spec = DirectConv4cSpec(
        problem=problem,
        name="direct_conv_4c_dgrad",
        block_q=bq,
        block_groups=bg,
        dgrad_fused_weights=True,
        dgrad_weights_lds=lds,
        stage_rows=sr,
        waves_q=bq // 4 if sr else 1,
    )
    return spec, arch


def _build(spec, arch):
    if isinstance(spec, DirectConv4cSpec):
        return build_direct_conv_4c(spec, arch=arch)
    return build_direct_depthwise_dgrad_windowed(spec, arch=arch)


def _spec(idx: int):
    if _FOURC_CFG_BASE <= idx < _FOURC_CFG_BASE + len(_FOURC_CFGS):
        return _fourc_spec(idx - _FOURC_CFG_BASE)
    if not 0 <= idx < len(_WIN_CFGS):
        raise SystemExit(f"unknown config index {idx}")
    N, H, W, G, KH, KW, PAD, dtype, arch, knobs = _WIN_CFGS[idx]
    problem = DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=G,
        cpg=1,
        kpg=1,
        KH=KH,
        KW=KW,
        PAD=PAD,
        stride=1,
        dtype=dtype,
    )
    return DirectDepthwiseDgradWindowedSpec(problem=problem, **knobs), arch


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: conv_direct_dgrad_ext_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
