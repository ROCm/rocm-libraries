#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_dgrad_wmma_emit.py -- Python reference emitter for
# the wave32 WMMA direct dgrad parity harness. Selects one of N sampled spec
# configs by argv[1], builds the DirectDgradWmmaSpec, builds the kernel via
# build_direct_dgrad_wmma(spec, arch=<cfg arch>) and prints the LLVM IR to
# stdout so it can be byte-compared with the C emitter
# conv_direct_dgrad_wmma_emit.c.
#
# Config index map (gfx1250 unless noted):
#   0  bf16, defaults (2x4 waves, 4x4 tiles, tile_w 16, ck 32, 2 stages)
#   1  fp16, 2 images of 8-wide tiles, 3 stages, prefetch + iterative-ilp
#   2  bf16, 3x3 PAD 0, 4 images of 4-wide tiles, one wave along channels
#   3  bf16, 5x5 PAD 2, 1x4 waves, 4x6 tiles, tile_w 32, prefetch + max-ilp
#   4  fp16, 1x2 waves, 1x1 tiles (smallest block), 3 stages
#   5  bf16, 2x4 waves, 4x8 tiles (upper VGPR banks), tile_w 64, prefetch
#   6  rejected by both engines: gfx950 (wave64, no TDM)
#   7  rejected by both engines: stride 2 at the default 4x4 tiles (four
#      phases of accumulators exceed the VGPR budget)
#   8  bf16, ck 64 (two WMMA k-steps per tap), prefetch + iterative-ilp
#   9  bf16, stride 2 3x3 PAD 1 (4 phases), 1x4 waves, 2x4 tiles
#  10  fp16, stride 2 3x3, 2 images of 8-wide tiles, ck 64, prefetch + iterative-ilp
#  11  bf16, 2x2 stride 2 PAD 0 (patch merge: 4 phases share one B fragment), ck 64, prefetch
#  12  fp16, 2x2 stride 2, 2 images of 8-wide tiles, 3 stages
#  13  bf16, dilation 1x2 PAD 2, prefetch + iterative-ilp
#  14  fp16, dilation 3x1 PAD 1, 2 images of 32-wide tiles
#  15  bf16, dilation 2x2 PAD 2 (no zero fill), 2x4 waves, 4x8 tiles
#  16  bf16, pack_groups 2 (cpg = kpg = 16, 4 groups), 2 images
#  17  fp16, pack_groups 4 (cpg = kpg = 8, 8 groups), 4 images
#  18  bf16, pack_groups 8 (cpg = kpg = 4, 16 groups), 32-wide tiles
#  19  bf16, pack_groups 16 (cpg = kpg = 2, 2-wide row copies), 3 stages
#  20  bf16, split_barrier, 2x2 waves, 4x8 tiles, prefetch + iterative-ilp
#  21  bf16, split_barrier + wave_epilogue, 1x4 waves, 4x6 tiles
#  22  fp16, split_barrier + wave_epilogue, 2 images of 8-wide tiles, 3 stages
#  23  bf16, split_barrier + wave_epilogue, tile_w 24 (not a power of two), 4 images
#  24  bf16, split_barrier, tile_w 56, 1x7 waves, 4x10 tiles, no prefetch
#  25  fp16, tile_w 40 (not a power of two), 1x5 waves, no split barrier
#  26  rejected by both engines: stride 3
#  27  rejected by both engines: stride 2 with dilation
#  28  rejected by both engines: split_barrier on a stride-2 tile
#  29  rejected by both engines: pack_groups 3 does not divide 4 groups
#  30  rejected by both engines: wave_epilogue without whole rows per wave
#  31  rejected by both engines: a 0x3 filter (no taps), prefetch
#  32  bf16, stride 2 pack_groups 2 (cpg = kpg = 16, 4 groups), 1x4 waves, 2x2 tiles
#  33  fp16, stride 2 pack_groups 4 (cpg = kpg = 8, 8 groups), 1x8 waves, 8-wide tiles
#  34  bf16, stride 2 pack_groups 8 (cpg = kpg = 4, 16 groups), 2x4 waves, 2 images
#  35  fp16, stride 2 pack_groups 2, 1x4 waves, 2x4 tiles, prefetch + iterative-ilp
#  36  bf16, stride 2 pack_groups 4, 1x2 waves, 2x4 tiles
#  37  fp16, stride 2 pack_groups 8, 1x4 waves, 2x3 tiles
from dataclasses import replace

from _emit_common import run_emit
from kernels.common.conv_direct_dgrad_wmma import (
    DirectDgradWmmaSpec,
    build_direct_dgrad_wmma,
)
from kernels.common.conv_direct_grouped import DirectConvProblem


def _p(
    *,
    KH=3,
    PAD=1,
    dtype="bf16",
    stride=1,
    groups=1,
    cpg=64,
    kpg=96,
    dil_h=1,
    dil_w=1,
):
    return DirectConvProblem(
        N=2,
        H=20,
        W=24,
        groups=groups,
        cpg=cpg,
        kpg=kpg,
        KH=KH,
        KW=KH,
        PAD=PAD,
        stride=stride,
        dtype=dtype,
        dil_h=dil_h,
        dil_w=dil_w,
    )


def _spec(idx: int):
    S = DirectDgradWmmaSpec
    cfgs = {
        0: (S(problem=_p()), "gfx1250"),
        1: (
            S(
                problem=_p(dtype="fp16"),
                waves_c=2,
                waves_p=4,
                tiles_c=2,
                tiles_p=2,
                tile_w=8,
                tile_n=2,
                stages=3,
                prefetch=True,
                sched="iterative-ilp",
            ),
            "gfx1250",
        ),
        2: (
            S(
                problem=_p(PAD=0),
                waves_c=1,
                waves_p=8,
                tiles_c=1,
                tiles_p=2,
                tile_w=4,
                tile_n=4,
            ),
            "gfx1250",
        ),
        3: (
            S(
                problem=_p(KH=5, PAD=2),
                waves_c=1,
                waves_p=4,
                tiles_c=4,
                tiles_p=6,
                tile_w=32,
                prefetch=True,
                sched="max-ilp",
            ),
            "gfx1250",
        ),
        4: (
            S(
                problem=_p(dtype="fp16"),
                waves_c=1,
                waves_p=2,
                tiles_c=1,
                tiles_p=1,
                stages=3,
            ),
            "gfx1250",
        ),
        5: (
            S(problem=_p(), tiles_c=4, tiles_p=8, tile_w=64, prefetch=True),
            "gfx1250",
        ),
        6: (S(problem=_p()), "gfx950"),
        7: (S(problem=_p(stride=2)), "gfx1250"),
        8: (
            S(
                problem=_p(),
                waves_c=1,
                waves_p=4,
                tiles_c=4,
                tiles_p=4,
                ck=64,
                prefetch=True,
                sched="iterative-ilp",
            ),
            "gfx1250",
        ),
        9: (
            S(
                problem=_p(stride=2),
                waves_c=1,
                waves_p=4,
                tiles_c=2,
                tiles_p=4,
            ),
            "gfx1250",
        ),
        10: (
            S(
                problem=_p(dtype="fp16", stride=2),
                waves_c=2,
                waves_p=4,
                tiles_c=2,
                tiles_p=4,
                tile_w=8,
                tile_n=2,
                ck=64,
                prefetch=True,
                sched="iterative-ilp",
            ),
            "gfx1250",
        ),
        11: (
            S(
                problem=_p(KH=2, PAD=0, stride=2),
                waves_c=2,
                waves_p=4,
                tiles_c=2,
                tiles_p=4,
                ck=64,
                prefetch=True,
                sched="iterative-ilp",
            ),
            "gfx1250",
        ),
        12: (
            S(
                problem=_p(KH=2, PAD=0, stride=2, dtype="fp16"),
                waves_c=1,
                waves_p=4,
                tiles_c=2,
                tiles_p=2,
                tile_w=8,
                tile_n=2,
                ck=64,
                stages=3,
            ),
            "gfx1250",
        ),
        13: (
            S(
                problem=_p(PAD=2, dil_w=2),
                waves_c=1,
                waves_p=4,
                tiles_c=4,
                tiles_p=4,
                prefetch=True,
                sched="iterative-ilp",
            ),
            "gfx1250",
        ),
        14: (
            S(
                problem=_p(dtype="fp16", dil_h=3),
                waves_c=1,
                waves_p=4,
                tiles_c=4,
                tiles_p=4,
                tile_w=32,
                tile_n=2,
            ),
            "gfx1250",
        ),
        15: (
            S(
                problem=_p(PAD=2, dil_h=2, dil_w=2),
                tiles_c=4,
                tiles_p=8,
                prefetch=True,
                sched="iterative-ilp",
            ),
            "gfx1250",
        ),
        16: (
            S(
                problem=_p(groups=4, cpg=16, kpg=16),
                waves_c=2,
                waves_p=4,
                tiles_c=1,
                tiles_p=8,
                tile_n=2,
                prefetch=True,
                sched="iterative-ilp",
                pack_groups=2,
            ),
            "gfx1250",
        ),
        17: (
            S(
                problem=_p(dtype="fp16", groups=8, cpg=8, kpg=8),
                waves_c=1,
                waves_p=8,
                tiles_c=2,
                tiles_p=4,
                tile_n=4,
                prefetch=True,
                sched="iterative-ilp",
                pack_groups=4,
            ),
            "gfx1250",
        ),
        18: (
            S(
                problem=_p(groups=16, cpg=4, kpg=4),
                waves_c=1,
                waves_p=4,
                tiles_c=2,
                tiles_p=8,
                tile_w=32,
                tile_n=2,
                prefetch=True,
                sched="iterative-ilp",
                pack_groups=8,
            ),
            "gfx1250",
        ),
        19: (
            S(
                problem=_p(groups=32, cpg=2, kpg=2),
                waves_c=1,
                waves_p=4,
                tiles_c=2,
                tiles_p=4,
                stages=3,
                pack_groups=16,
            ),
            "gfx1250",
        ),
        20: (
            S(
                problem=_p(),
                waves_c=2,
                waves_p=2,
                tiles_c=4,
                tiles_p=8,
                prefetch=True,
                sched="iterative-ilp",
                split_barrier=True,
            ),
            "gfx1250",
        ),
        21: (
            S(
                problem=_p(),
                waves_c=1,
                waves_p=4,
                tiles_c=4,
                tiles_p=6,
                prefetch=True,
                sched="iterative-ilp",
                split_barrier=True,
                wave_epilogue=True,
            ),
            "gfx1250",
        ),
        22: (
            S(
                problem=_p(dtype="fp16"),
                waves_c=2,
                waves_p=2,
                tiles_c=4,
                tiles_p=4,
                tile_w=8,
                tile_n=2,
                stages=3,
                prefetch=True,
                sched="iterative-ilp",
                split_barrier=True,
                wave_epilogue=True,
            ),
            "gfx1250",
        ),
        23: (
            S(
                problem=_p(),
                waves_c=1,
                waves_p=8,
                tiles_c=4,
                tiles_p=6,
                tile_w=24,
                tile_n=4,
                prefetch=True,
                sched="iterative-ilp",
                split_barrier=True,
                wave_epilogue=True,
            ),
            "gfx1250",
        ),
        24: (
            S(
                problem=_p(),
                waves_c=1,
                waves_p=7,
                tiles_c=4,
                tiles_p=10,
                tile_w=56,
                split_barrier=True,
            ),
            "gfx1250",
        ),
        25: (
            S(
                problem=_p(dtype="fp16"),
                waves_c=1,
                waves_p=5,
                tiles_c=4,
                tiles_p=5,
                tile_w=40,
                prefetch=True,
                sched="iterative-ilp",
            ),
            "gfx1250",
        ),
        26: (
            S(
                problem=_p(stride=3),
            ),
            "gfx1250",
        ),
        27: (
            S(
                problem=_p(stride=2, PAD=2, dil_h=2, dil_w=2),
                waves_c=1,
                waves_p=4,
                tiles_c=2,
                tiles_p=4,
            ),
            "gfx1250",
        ),
        28: (
            S(
                problem=_p(stride=2),
                waves_c=1,
                waves_p=4,
                tiles_c=2,
                tiles_p=4,
                split_barrier=True,
            ),
            "gfx1250",
        ),
        29: (
            S(
                problem=_p(groups=4, cpg=16, kpg=16),
                tiles_c=1,
                pack_groups=3,
            ),
            "gfx1250",
        ),
        30: (
            S(
                problem=_p(),
                waves_c=2,
                waves_p=4,
                tiles_c=4,
                tiles_p=1,
                tile_w=32,
                wave_epilogue=True,
            ),
            "gfx1250",
        ),
        31: (
            S(
                problem=replace(_p(), KH=0),
                prefetch=True,
                sched="iterative-ilp",
            ),
            "gfx1250",
        ),
        32: (
            S(
                problem=_p(stride=2, groups=4, cpg=16, kpg=16),
                waves_c=1,
                waves_p=4,
                tiles_c=2,
                tiles_p=2,
                pack_groups=2,
            ),
            "gfx1250",
        ),
        33: (
            S(
                problem=_p(dtype="fp16", stride=2, groups=8, cpg=8, kpg=8),
                waves_c=1,
                waves_p=8,
                tiles_c=2,
                tiles_p=4,
                tile_w=8,
                pack_groups=4,
            ),
            "gfx1250",
        ),
        34: (
            S(
                problem=_p(stride=2, groups=16, cpg=4, kpg=4),
                waves_c=2,
                waves_p=4,
                tiles_c=1,
                tiles_p=4,
                tile_n=2,
                pack_groups=8,
            ),
            "gfx1250",
        ),
        35: (
            S(
                problem=_p(dtype="fp16", stride=2, groups=4, cpg=16, kpg=16),
                waves_c=1,
                waves_p=4,
                tiles_c=2,
                tiles_p=4,
                prefetch=True,
                sched="iterative-ilp",
                pack_groups=2,
            ),
            "gfx1250",
        ),
        36: (
            S(
                problem=_p(stride=2, groups=8, cpg=8, kpg=8),
                waves_c=1,
                waves_p=2,
                tiles_c=2,
                tiles_p=4,
                pack_groups=4,
            ),
            "gfx1250",
        ),
        37: (
            S(
                problem=_p(dtype="fp16", stride=2, groups=16, cpg=4, kpg=4),
                waves_c=1,
                waves_p=4,
                tiles_c=2,
                tiles_p=3,
                pack_groups=8,
            ),
            "gfx1250",
        ),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    return cfgs[idx]


def main() -> int:
    return run_emit(
        _spec,
        build_direct_dgrad_wmma,
        usage="usage: conv_direct_dgrad_wmma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
