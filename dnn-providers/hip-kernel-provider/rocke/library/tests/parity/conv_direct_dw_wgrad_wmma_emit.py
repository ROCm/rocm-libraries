#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_dw_wgrad_wmma_emit.py -- Python reference emitter
# for the Toeplitz WMMA depthwise wgrad (direct_depthwise_wgrad_wmma). Selects
# one of N sampled spec configs by argv[1], builds the DwWgradWmmaSpec, builds
# the kernel via build_dw_wgrad_wmma(spec, arch=<cfg arch>) and prints the LLVM
# IR to stdout so it can be byte-compared with the C emitter
# conv_direct_dw_wgrad_wmma_emit.c.
#
# Config index map (gfx1250, N 2, groups 64, ws_replicas 64 unless noted;
# 7x7 on H 20 W 37, 1x31 on H 6 W 80 with column padding 15):
#  AOT grid axes (waves, tile_q, rows, target_blocks, prefetch):
#   0  bf16, 7x7 PAD 3, defaults (4 waves, tile_q 32, 8 rows, tb 1024, prefetch)
#   1  fp16, 7x7 PAD 3, 8 waves, tile_q 16, 4 rows, tb 512, prefetch
#   2  bf16, 7x7 PAD 0, 4 waves, tile_q 8, 8 rows, tb 512, no prefetch
#   3  fp16, 7x7 PAD 0, 8 waves, tile_q 32, 4 rows, tb 1024, no prefetch
#   4  bf16, 7x7 PAD 3, 8 waves, tile_q 8, 4 rows, tb 1024, no prefetch
#   5  fp16, 7x7 PAD 3, 4 waves, tile_q 16, 8 rows, tb 512, no prefetch
#   6  fp16, 7x7 PAD 0, 4 waves, tile_q 16, 8 rows, tb 1024, prefetch
#   7  bf16, 1x31 PAD 0 pad_w 15, 2 waves, tile_q 32, 2 rows, tb 1024, prefetch
#   8  fp16, 1x31 PAD 0 pad_w 15, 4 waves, tile_q 8, 2 rows, tb 512, no prefetch
#   9  bf16, 1x31 PAD 0 pad_w 15, 4 waves, tile_q 16, 2 rows, tb 512, prefetch
#  10  fp16, 1x31 PAD 0 pad_w 15, 2 waves, tile_q 16, 2 rows, tb 1024, no prefetch
#  other shapes the kernel accepts:
#  11  fp16, 3x3 PAD 1, defaults
#  12  bf16, 5x5 PAD 2, 1 wave, tile_q 4, 2 rows, no prefetch
#  13  fp16, 1x7 PAD 0 pad_w 3, 2 waves, tile_q 12, 6 rows
#  14  bf16, 3x5 PAD 1 pad_w 2, groups 24 (a partial channel block), 1 replica
#  15  fp16, 7x7 PAD 3, groups 40, H 13 W 19, 16 waves, tile_q 4, 2 rows
#  16  bf16, 2x9 PAD 0 (even KH, a second column tile)
#  rejected by both engines:
#  17  stride 2
#  18  cpg = kpg = 2
#  19  3 rows (odd)
#  20  tile_q 6 (not a multiple of 4)
#  21  gfx950 (wave64, no wave32 WMMA)
#  22  dilation 2
#  23  pad_w -2 (no column padding)
#  24  7x31, 16 rows, tile_q 64 (2048 unrolled WMMAs)
#  25  groups 12 (not a multiple of 8)
#  26  wave_size 64
#  27  7x7 PAD 3, 16 waves, 16 rows, tile_q 64 (past the LDS capacity)
from _emit_common import run_emit
from kernels.common.conv_direct_dw_wgrad_wmma import (
    DwWgradWmmaSpec,
    build_dw_wgrad_wmma,
)
from kernels.common.conv_direct_grouped import DirectConvProblem


def _p(
    KH=7,
    KW=7,
    PAD=3,
    dtype="bf16",
    pad_w=-1,
    G=64,
    H=20,
    W=37,
    stride=1,
    cpg=1,
    dil=1,
):
    return DirectConvProblem(
        N=2,
        H=H,
        W=W,
        groups=G,
        cpg=cpg,
        kpg=cpg,
        KH=KH,
        KW=KW,
        PAD=PAD,
        pad_w=pad_w,
        stride=stride,
        dtype=dtype,
        dil_h=dil,
        dil_w=dil,
    )


def _p31(dtype="bf16"):
    return _p(1, 31, 0, dtype, pad_w=15, H=6, W=80)


def _k(waves, tile_q, rows, tb, pf):
    return dict(waves=waves, tile_q=tile_q, rows=rows, target_blocks=tb, prefetch=pf)


def _spec(idx: int):
    S = DwWgradWmmaSpec
    cfgs = {
        0: (S(problem=_p()), "gfx1250"),
        1: (S(problem=_p(dtype="fp16"), **_k(8, 16, 4, 512, True)), "gfx1250"),
        2: (S(problem=_p(PAD=0), **_k(4, 8, 8, 512, False)), "gfx1250"),
        3: (S(problem=_p(PAD=0, dtype="fp16"), **_k(8, 32, 4, 1024, False)), "gfx1250"),
        4: (S(problem=_p(), **_k(8, 8, 4, 1024, False)), "gfx1250"),
        5: (S(problem=_p(dtype="fp16"), **_k(4, 16, 8, 512, False)), "gfx1250"),
        6: (S(problem=_p(PAD=0, dtype="fp16"), **_k(4, 16, 8, 1024, True)), "gfx1250"),
        7: (S(problem=_p31(), **_k(2, 32, 2, 1024, True)), "gfx1250"),
        8: (S(problem=_p31("fp16"), **_k(4, 8, 2, 512, False)), "gfx1250"),
        9: (S(problem=_p31(), **_k(4, 16, 2, 512, True)), "gfx1250"),
        10: (S(problem=_p31("fp16"), **_k(2, 16, 2, 1024, False)), "gfx1250"),
        11: (S(problem=_p(3, 3, 1, "fp16")), "gfx1250"),
        12: (S(problem=_p(5, 5, 2), **_k(1, 4, 2, 1024, False)), "gfx1250"),
        13: (
            S(problem=_p(1, 7, 0, "fp16", pad_w=3), waves=2, tile_q=12, rows=6),
            "gfx1250",
        ),
        14: (S(problem=_p(3, 5, 1, pad_w=2, G=24), ws_replicas=1), "gfx1250"),
        15: (
            S(problem=_p(dtype="fp16", G=40, H=13, W=19), waves=16, tile_q=4, rows=2),
            "gfx1250",
        ),
        16: (S(problem=_p(2, 9, 0)), "gfx1250"),
        17: (S(problem=_p(stride=2)), "gfx1250"),
        18: (S(problem=_p(cpg=2, G=32)), "gfx1250"),
        19: (S(problem=_p(), rows=3), "gfx1250"),
        20: (S(problem=_p(), tile_q=6), "gfx1250"),
        21: (S(problem=_p()), "gfx950"),
        22: (S(problem=_p(dil=2, PAD=6)), "gfx1250"),
        23: (S(problem=_p(pad_w=-2)), "gfx1250"),
        24: (
            S(problem=_p(7, 31, 3, pad_w=15, H=40, W=80), rows=16, tile_q=64),
            "gfx1250",
        ),
        25: (S(problem=_p(G=12)), "gfx1250"),
        26: (S(problem=_p(), wave_size=64), "gfx1250"),
        27: (S(problem=_p(), waves=16, rows=16, tile_q=64), "gfx1250"),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    return cfgs[idx]


def _build(spec, arch):
    return build_dw_wgrad_wmma(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: conv_direct_dw_wgrad_wmma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
