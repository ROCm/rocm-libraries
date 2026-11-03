#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_fwd_wmma_emit.py -- Python reference emitter for
# the wave32 WMMA direct forward parity harness. Selects one of N sampled spec
# configs by argv[1], builds the DirectFwdWmmaSpec, builds the kernel via
# build_direct_fwd_wmma(spec, arch=<cfg arch>) and prints the LLVM IR to
# stdout so it can be byte-compared with the C emitter
# conv_direct_fwd_wmma_emit.c.
#
# Config index map (gfx1250, 3x3 PAD 1 stride 1, dense cpg 64 / kpg 96 unless
# noted):
#   0  bf16, defaults (2x4 waves, 4x4 tiles, tile_w 16, ck 32, 2 stages)
#   1  fp16, 2 images of 8-wide tiles, 3 stages, prefetch + iterative-ilp
#   2  bf16, 2 images, split_barrier + wave_epilogue + prefetch
#   3  bf16, row_ring, 6x8 tiles (upper VGPR banks), tile_w 32, 3 stages
#   4  fp16, tap_ring, 3 stages
#   5  bf16, stride 2, no phase split, prefetch
#   6  bf16, stride 2, phase_split, ck 64, prefetch
#   7  bf16, 2x2 PAD 0 stride 2 (patch merge), phase_split
#   8  bf16, pack_groups 8 (groups 8, cpg = kpg = 4), prefetch
#   9  fp16, pack_groups 4 (groups 4, cpg = kpg = 8), stride 2, phase_split,
#      2 images
#  10  bf16, dilation 1x2 (PAD 2), ck 64, prefetch
#  11  bf16, dilation 3x1, stride 2, 2 images, prefetch
#  12  bf16, pingpong, 4x10 tiles
#  13  fp16, pingpong, pp_group 3, iterative-ilp
#  14  bf16, tap_reuse, prefetch
#  15  bf16, tap_reuse + tap_shift, tile_w 32
#  16  bf16, tap_reuse + tap_shift, dilation 1x2 (PAD 2), split_barrier
#  17  bf16, stride 2, phase_split + row_ring, 3 stages
#  18  bf16, stride 2, phase_split + pingpong, pp_group 2
#  19  rejected by both engines: gfx950 (wave64, no TDM)
#  20  rejected by both engines: stride 3
#  21  bf16, 5x5 PAD 2, prefetch + max-ilp
from _emit_common import run_emit
from kernels.common.conv_direct_fwd_wmma import (
    DirectFwdWmmaSpec,
    build_direct_fwd_wmma,
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


def _k(wk, wp, tk, tp, tw, tn, ck, st, **kw):
    return dict(
        waves_k=wk,
        waves_p=wp,
        tiles_k=tk,
        tiles_p=tp,
        tile_w=tw,
        tile_n=tn,
        ck=ck,
        stages=st,
        **kw,
    )


PF = dict(prefetch=True, sched="iterative-ilp")


def _spec(idx: int):
    S = DirectFwdWmmaSpec
    cfgs = {
        0: (S(problem=_p()), "gfx1250"),
        1: (
            S(problem=_p(dtype="fp16"), **_k(2, 4, 2, 2, 8, 2, 32, 3, **PF)),
            "gfx1250",
        ),
        2: (
            S(
                problem=_p(),
                **_k(
                    1,
                    4,
                    4,
                    6,
                    16,
                    2,
                    32,
                    2,
                    split_barrier=True,
                    wave_epilogue=True,
                    **PF,
                ),
            ),
            "gfx1250",
        ),
        3: (S(problem=_p(), **_k(1, 8, 6, 8, 32, 1, 32, 3, row_ring=True)), "gfx1250"),
        4: (
            S(problem=_p(dtype="fp16"), **_k(2, 4, 2, 4, 16, 1, 32, 3, tap_ring=True)),
            "gfx1250",
        ),
        5: (S(problem=_p(stride=2), **_k(1, 4, 6, 2, 16, 1, 32, 2, **PF)), "gfx1250"),
        6: (
            S(
                problem=_p(stride=2),
                **_k(2, 2, 1, 4, 16, 1, 64, 2, phase_split=True, **PF),
            ),
            "gfx1250",
        ),
        7: (
            S(
                problem=_p(KH=2, PAD=0, stride=2),
                **_k(1, 4, 2, 2, 16, 1, 64, 2, phase_split=True),
            ),
            "gfx1250",
        ),
        8: (
            S(
                problem=_p(groups=8, cpg=4, kpg=4),
                **_k(1, 4, 2, 4, 32, 1, 32, 2, pack_groups=8, **PF),
            ),
            "gfx1250",
        ),
        9: (
            S(
                problem=_p(dtype="fp16", stride=2, groups=4, cpg=8, kpg=8),
                **_k(2, 4, 1, 2, 32, 2, 32, 2, pack_groups=4, phase_split=True, **PF),
            ),
            "gfx1250",
        ),
        10: (
            S(problem=_p(PAD=2, dil_w=2), **_k(1, 4, 2, 6, 16, 1, 64, 2, **PF)),
            "gfx1250",
        ),
        11: (
            S(problem=_p(stride=2, dil_h=3), **_k(1, 4, 2, 2, 32, 2, 32, 2, **PF)),
            "gfx1250",
        ),
        12: (
            S(problem=_p(), **_k(2, 4, 4, 10, 16, 1, 32, 2, pingpong=True)),
            "gfx1250",
        ),
        13: (
            S(
                problem=_p(dtype="fp16"),
                **_k(
                    2,
                    4,
                    4,
                    6,
                    16,
                    1,
                    32,
                    2,
                    pingpong=True,
                    pp_group=3,
                    sched="iterative-ilp",
                ),
            ),
            "gfx1250",
        ),
        14: (
            S(problem=_p(), **_k(1, 4, 4, 4, 16, 1, 32, 2, tap_reuse=True, **PF)),
            "gfx1250",
        ),
        15: (
            S(
                problem=_p(),
                **_k(1, 4, 2, 4, 32, 1, 32, 2, tap_reuse=True, tap_shift=True),
            ),
            "gfx1250",
        ),
        16: (
            S(
                problem=_p(PAD=2, dil_w=2),
                **_k(
                    1,
                    4,
                    2,
                    4,
                    16,
                    1,
                    32,
                    2,
                    tap_reuse=True,
                    tap_shift=True,
                    split_barrier=True,
                ),
            ),
            "gfx1250",
        ),
        17: (
            S(
                problem=_p(stride=2),
                **_k(1, 4, 2, 2, 16, 1, 32, 3, phase_split=True, row_ring=True),
            ),
            "gfx1250",
        ),
        18: (
            S(
                problem=_p(stride=2),
                **_k(
                    2,
                    4,
                    2,
                    2,
                    16,
                    1,
                    32,
                    2,
                    phase_split=True,
                    pingpong=True,
                    pp_group=2,
                ),
            ),
            "gfx1250",
        ),
        19: (S(problem=_p()), "gfx950"),
        20: (S(problem=_p(stride=3)), "gfx1250"),
        21: (
            S(
                problem=_p(KH=5, PAD=2),
                **_k(1, 4, 2, 4, 16, 1, 32, 2, prefetch=True, sched="max-ilp"),
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
        build_direct_fwd_wmma,
        usage="usage: conv_direct_fwd_wmma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
