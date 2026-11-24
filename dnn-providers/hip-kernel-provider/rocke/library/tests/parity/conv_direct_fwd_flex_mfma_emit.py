#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_fwd_flex_mfma_emit.py -- Python reference emitter
# for the gfx950 flattened-pixel implicit-GEMM MFMA direct forward parity
# harness. Selects one of N sampled spec configs by argv[1], builds the
# FwdFlexMfmaSpec, builds the kernel via build_fwd_flex_mfma(spec, arch=<cfg
# arch>) and prints the LLVM IR to stdout so it can be byte-compared with the C
# emitter conv_direct_fwd_flex_mfma_emit.c.
#
# The problem is N=2, 20x24, C=512, K=96 with the filter / padding / stride /
# dilation of the row. Config index map (gfx950 unless noted):
#   0  bf16, 3x3 s2 pad 1, bk 256, split_c 1
#   1  fp16, 3x3 s2 pad 1, bk 128, split_c 1
#   2  bf16, 1x1 s2, bk 256, split_c 2
#   3  fp16, 2x2 s2, bk 128, split_c 4
#   4  bf16, 3x3 dil 2 pad 2, bk 128, split_c 8
#   5  fp16, 3x3 dil 12 pad 12, bk 256, split_c 1
#   6  bf16, 1x3 dil 1x3 pad 0x3, bk 256, split_c 2
#   7  fp16, 7x1 pad 3x0, bk 128, split_c 1
#   8  bf16, 1x11 dil 1x3 pad 0x15, bk 128, split_c 2
#   9  fp16, 1x7 dil 1x3 pad 0x9, bk 256, split_c 4
#  10  bf16, 3x3 dil 24 pad 24, bk 256, split_c 8
#  11  rejected by both engines: bk 192
#  12  rejected by both engines: gfx942 (no 16-byte LDS DMA)
#  13  rejected by both engines: C = 96 (not a multiple of 64)
#  14  rejected by both engines: K = 100 (not a multiple of 8)
#  15  rejected by both engines: split_c 16 > C / 64
#  16  rejected by both engines: 7x7 (more than 32 taps)
from _emit_common import run_emit
from kernels.common.conv_direct_fwd_flex_mfma import (
    FwdFlexMfmaSpec,
    build_fwd_flex_mfma,
)
from kernels.common.conv_direct_grouped import DirectConvProblem


def _p(*, KH=3, KW=3, PAD=1, pad_w=-1, stride=1, dil=(1, 1), dtype="bf16", C=512, K=96):
    return DirectConvProblem(
        N=2,
        H=20,
        W=24,
        groups=1,
        cpg=C,
        kpg=K,
        KH=KH,
        KW=KW,
        PAD=PAD,
        pad_w=pad_w,
        stride=stride,
        dil_h=dil[0],
        dil_w=dil[1],
        dtype=dtype,
    )


def _spec(idx: int):
    S = FwdFlexMfmaSpec
    h = "fp16"
    cfgs = {
        0: (S(problem=_p(stride=2)), "gfx950"),
        1: (S(problem=_p(stride=2, dtype=h), bk=128), "gfx950"),
        2: (S(problem=_p(KH=1, KW=1, PAD=0, stride=2), split_c=2), "gfx950"),
        3: (
            S(problem=_p(KH=2, KW=2, PAD=0, stride=2, dtype=h), bk=128, split_c=4),
            "gfx950",
        ),
        4: (S(problem=_p(PAD=2, dil=(2, 2)), bk=128, split_c=8), "gfx950"),
        5: (S(problem=_p(PAD=12, dil=(12, 12), dtype=h)), "gfx950"),
        6: (S(problem=_p(KH=1, PAD=0, pad_w=3, dil=(1, 3)), split_c=2), "gfx950"),
        7: (S(problem=_p(KH=7, KW=1, PAD=3, pad_w=0, dtype=h), bk=128), "gfx950"),
        8: (
            S(problem=_p(KH=1, KW=11, PAD=0, pad_w=15, dil=(1, 3)), bk=128, split_c=2),
            "gfx950",
        ),
        9: (
            S(problem=_p(KH=1, KW=7, PAD=0, pad_w=9, dil=(1, 3), dtype=h), split_c=4),
            "gfx950",
        ),
        10: (S(problem=_p(PAD=24, dil=(24, 24)), split_c=8), "gfx950"),
        11: (S(problem=_p(), bk=192), "gfx950"),
        12: (S(problem=_p()), "gfx942"),
        13: (S(problem=_p(C=96)), "gfx950"),
        14: (S(problem=_p(K=100)), "gfx950"),
        15: (S(problem=_p(), split_c=16), "gfx950"),
        16: (S(problem=_p(KH=7, KW=7, PAD=3)), "gfx950"),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    return cfgs[idx]


def main() -> int:
    return run_emit(
        _spec,
        build_fwd_flex_mfma,
        usage="usage: conv_direct_fwd_flex_mfma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
