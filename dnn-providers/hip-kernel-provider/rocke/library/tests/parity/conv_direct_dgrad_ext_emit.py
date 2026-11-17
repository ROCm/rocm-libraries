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
# Indices 30..44: the generic DirectConvSpec (build_direct_conv): the
# fused-weight dgrad knobs (30..32: gathered, LDS-staged with the waves-per-EU
# hint, plain preload; each with the row guard), the row-stream knobs (33..38:
# prefetch, LDS-only barrier, XCD tiles, LDS pad, staged output, split M,
# row exit, 5x5 and 1x1, gfx942), three rejects (39: LDS-staged weights and
# 40: xcd_tiles on gfx942, 44: waves_m = 2 with kpg = 16) and develop's forms
# (41: persistent grid with the runtime K loop, 42: waves_q = waves_k = 2,
# 43: runtime channels).
from _emit_common import run_emit
from kernels.common.conv_direct_grouped import (
    DirectConv4cSpec,
    DirectConvProblem,
    DirectConvSpec,
    DirectDepthwiseDgradWindowedSpec,
    build_direct_conv,
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


# Generic DirectConvSpec configs from index _GENERIC_CFG_BASE:
# (N, H, W, groups, cpg, kpg, K, dtype, arch, knobs); square K x K filter with
# "same" padding, stride 1.
_GENERIC_CFG_BASE = _FOURC_CFG_BASE + len(_FOURC_CFGS)
_FW = {"dgrad_fused_weights": True}
_FWL = {"dgrad_fused_weights": True, "dgrad_weights_lds": True}
_STREAM = {"prefetch_rows": 2, "lds_only_sync": True, "xcd_tiles": True}
_GENERIC_CFGS = (
    (2, 9, 17, 8, 16, 16, 3, "bf16", "gfx950", {"block_groups": 2, "row_guard": True, **_FW}),
    (2, 9, 17, 8, 16, 16, 3, "fp16", "gfx950", {"block_groups": 2, "waves_per_eu": 4, "row_guard": True, **_FWL}),
    (2, 9, 17, 4, 12, 20, 3, "fp16", "gfx950", {"block_groups": 2, "double_buffer": False, "preload_weights": True, "row_guard": True}),
    (9, 10, 13, 32, 16, 16, 3, "bf16", "gfx950", {"block_groups": 2, "waves_per_eu": 4, "lds_pad": 8, "stage_out": True, "row_guard": True, **_FWL, **_STREAM}),
    (8, 11, 19, 32, 16, 16, 3, "fp16", "gfx950", {"block_q": 32, "block_groups": 2, "block_h": 8, "lds_pad": 8, "row_guard": True, "row_exit": True, **_FWL, **_STREAM}),
    (8, 7, 14, 16, 32, 32, 3, "bf16", "gfx950", {"block_groups": 1, "fold_k32": True, "waves_m": 2, "lds_pad": 8, "stage_out": True, **_FWL, **_STREAM}),
    (2, 9, 14, 8, 8, 32, 5, "fp16", "gfx950", {"block_groups": 1, "block_h": 4, "prefetch_rows": 3, "lds_only_sync": True, "waves_m": 2, "stage_out": True, "row_guard": True, **_FW}),
    (2, 6, 21, 8, 20, 8, 1, "bf16", "gfx950", {"block_groups": 4, "block_h": 32, "row_guard": True, "row_exit": True, **_FW}),
    (2, 9, 17, 8, 16, 16, 3, "fp16", "gfx942", {"block_groups": 2, "block_h": 4, "prefetch_rows": 2, "lds_only_sync": True, "lds_pad": 8, "row_guard": True, **_FW}),
    (2, 9, 17, 8, 16, 16, 3, "bf16", "gfx942", {"block_groups": 2, **_FWL}),
    (9, 9, 17, 8, 16, 16, 3, "bf16", "gfx942", {"block_groups": 2, "xcd_tiles": True, **_FW}),
    (2, 9, 17, 4, 16, 16, 3, "fp16", "gfx950", {"block_groups": 1, "block_h": 4, "persistent_grid": True, "runtime_k_loop": True, "persistent_rounds": 8}),
    (2, 9, 17, 4, 32, 16, 3, "bf16", "gfx950", {"block_q": 32, "block_groups": 1, "waves_q": 2, "waves_k": 2}),
    (2, 9, 17, 4, 24, 24, 3, "fp16", "gfx950", {"block_groups": 1, "block_h": 8, "runtime_channels": True, "k_atoms": 2, "m_tiles": 2}),
    (2, 9, 17, 8, 16, 16, 3, "bf16", "gfx950", {"block_groups": 2, "waves_m": 2, **_FW}),
)  # fmt: skip


def _generic_spec(idx: int):
    N, H, W, G, cpg, kpg, K, dtype, arch, knobs = _GENERIC_CFGS[idx]
    problem = DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=G,
        cpg=cpg,
        kpg=kpg,
        KH=K,
        KW=K,
        PAD=(K - 1) // 2,
        stride=1,
        dtype=dtype,
    )
    return DirectConvSpec(problem=problem, **knobs), arch


def _build(spec, arch):
    if isinstance(spec, DirectConv4cSpec):
        return build_direct_conv_4c(spec, arch=arch)
    if isinstance(spec, DirectConvSpec):
        return build_direct_conv(spec, arch=arch)
    return build_direct_depthwise_dgrad_windowed(spec, arch=arch)


def _spec(idx: int):
    if _GENERIC_CFG_BASE <= idx < _GENERIC_CFG_BASE + len(_GENERIC_CFGS):
        return _generic_spec(idx - _GENERIC_CFG_BASE)
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
