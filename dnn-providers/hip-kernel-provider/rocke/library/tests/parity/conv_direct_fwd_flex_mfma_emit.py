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
# dilation of the row unless the row says otherwise. Config index map (gfx950
# unless noted):
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
#  11  bf16, 3x3 s1 pad 1, bk 192
#  12  rejected by both engines: gfx942 (no 16-byte LDS DMA)
#  13  rejected by both engines: C = 96 (not a multiple of 64)
#  14  rejected by both engines: K = 100 (not a multiple of 8)
#  15  rejected by both engines: split_c 16 > C / 64
#  16  rejected by both engines: 7x7 (more than 32 taps)
#  17  fp16, bk 64
#  18  bf16, s2, bk 192, split_c 2
#  19  fp16, bx 128, bk 128
#  20  bf16, bx 128, bk 256, split_c 2
#  21  fp16, cs 4 (bx 128, bk 128)
#  22  bf16, cs 4, wt, xcd, split_c 2
#  23  fp16, cs 4, xcd, kmaj, s2
#  24  bf16, cs 2 (bx 128, bk 256)
#  25  fp16, cs 2, wt, xcd, kmaj, split_c 2
#  26  bf16, wt, bk 256
#  27  fp16, wt, bk 192, split_c 3
#  28  bf16, wt, bx 128, bk 128
#  29  fp16, wt, xt8, c_tail (C=200), bk 256
#  30  bf16, wt, xt8, sw, xcd, bk 256
#  31  fp16, xcd, bk 256, split_c 2
#  32  bf16, xcd 2, sw, kmaj, bk 128
#  33  fp16, xcd 3, sw, bk 256
#  34  bf16, xcd, kmaj, bk 256
#  35  fp16, c_tail (C=200), bk 256
#  36  bf16, c_tail (C=200), sw, xcd, bk 128
#  37  fp16, flat (C=200), bk 128
#  38  bf16, flat (C=200), bk 192, split_c 4
#  39  fp16, k_tail (K=27), bk 64
#  40  bf16, k_tail (K=36), bk 64, split_c 2
#  41  fp16, sw, bk 128
#  42  bf16, sw, bk 256, split_c 2
#  43  fp16, grp (4 groups of 64 / 64), bk 64
#  44  bf16, grp (2 groups of 128 / 256), sw, wt, bx 128, bk 256
#  45  fp16, grp (2 groups of 56 / 64), c_tail, bk 64
#  46  bf16, ds dgrad 3x3 s2, wt, bx 128, bk 128
#  47  fp16, ds dgrad 1x16 s8 pad 0x4 (H=1), h1, wt, bk 128
#  48  bf16, ds dgrad 1x1 s2, zf, wt, bk 192
#  49  fp16, ds dgrad 1x1 s2, zf, zl 8, wt, bx 128, bk 256
#  50  bf16, ds dgrad 1x1 s2 (K=264), zf, zl 8, c_tail, wt, bx 128, bk 256
#  51  fp16, s2, bk 128, split_c 6
#  52  bf16, C=1024, bk 256, split_c 16
#  53  fp16, ds dgrad 3x3 s2, bk 256, split_c 2 (the transposed filter)
#  54  rejected by both engines: cs 2 on a 2x2 filter (4 taps)
#  55  rejected by both engines: zf without wt
#  56  rejected by both engines: xt8 with bx 128
#  57  rejected by both engines: grp on a dense problem
from _emit_common import run_emit
from kernels.common.conv_direct_fwd_flex_mfma import (
    FwdFlexMfmaSpec,
    build_fwd_flex_mfma,
)
from kernels.common.conv_direct_grouped import DirectConvProblem


def _p(
    *,
    KH=3,
    KW=3,
    PAD=1,
    pad_w=-1,
    stride=1,
    dil=(1, 1),
    dtype="bf16",
    C=512,
    K=96,
    N=2,
    H=20,
    W=24,
    groups=1,
):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
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
        17: (S(problem=_p(dtype=h), bk=64), "gfx950"),
        18: (S(problem=_p(stride=2), bk=192, split_c=2), "gfx950"),
        19: (S(problem=_p(dtype=h), bx=128, bk=128), "gfx950"),
        20: (S(problem=_p(), bx=128, split_c=2), "gfx950"),
        21: (S(problem=_p(dtype=h), bx=128, bk=128, cs=4), "gfx950"),
        22: (
            S(problem=_p(), bx=128, bk=128, cs=4, wt=True, xcd=True, split_c=2),
            "gfx950",
        ),
        23: (
            S(
                problem=_p(stride=2, dtype=h),
                bx=128,
                bk=128,
                cs=4,
                xcd=True,
                kmaj=True,
            ),
            "gfx950",
        ),
        24: (S(problem=_p(), bx=128, cs=2), "gfx950"),
        25: (
            S(
                problem=_p(dtype=h),
                bx=128,
                cs=2,
                wt=True,
                xcd=True,
                kmaj=True,
                split_c=2,
            ),
            "gfx950",
        ),
        26: (S(problem=_p(), wt=True), "gfx950"),
        27: (S(problem=_p(dtype=h), bk=192, wt=True, split_c=3), "gfx950"),
        28: (S(problem=_p(), bx=128, bk=128, wt=True), "gfx950"),
        29: (S(problem=_p(C=200, dtype=h), wt=True, xt8=True, c_tail=True), "gfx950"),
        30: (S(problem=_p(), wt=True, xt8=True, sw=True, xcd=True), "gfx950"),
        31: (S(problem=_p(dtype=h), xcd=True, split_c=2), "gfx950"),
        32: (S(problem=_p(), bk=128, xcd=2, sw=True, kmaj=True), "gfx950"),
        33: (S(problem=_p(dtype=h), xcd=3, sw=True), "gfx950"),
        34: (S(problem=_p(), xcd=True, kmaj=True), "gfx950"),
        35: (S(problem=_p(C=200, dtype=h), c_tail=True), "gfx950"),
        36: (S(problem=_p(C=200), bk=128, c_tail=True, sw=True, xcd=True), "gfx950"),
        37: (S(problem=_p(C=200, dtype=h), bk=128, flat=True), "gfx950"),
        38: (S(problem=_p(C=200), bk=192, flat=True, split_c=4), "gfx950"),
        39: (S(problem=_p(K=27, dtype=h), bk=64, k_tail=True), "gfx950"),
        40: (S(problem=_p(K=36), bk=64, k_tail=True, split_c=2), "gfx950"),
        41: (S(problem=_p(dtype=h), bk=128, sw=True), "gfx950"),
        42: (S(problem=_p(), sw=True, split_c=2), "gfx950"),
        43: (S(problem=_p(C=64, K=64, groups=4, dtype=h), bk=64, grp=True), "gfx950"),
        44: (
            S(
                problem=_p(C=128, K=256, groups=2),
                bx=128,
                grp=True,
                sw=True,
                wt=True,
            ),
            "gfx950",
        ),
        45: (
            S(
                problem=_p(C=56, K=64, groups=2, dtype=h),
                bk=64,
                grp=True,
                c_tail=True,
            ),
            "gfx950",
        ),
        46: (
            S(problem=_p(stride=2, C=256, K=128), bx=128, bk=128, wt=True, ds=True),
            "gfx950",
        ),
        47: (
            S(
                problem=_p(
                    KH=1,
                    KW=16,
                    PAD=0,
                    pad_w=4,
                    stride=8,
                    H=1,
                    W=64,
                    C=128,
                    K=128,
                    dtype=h,
                ),
                bk=128,
                wt=True,
                ds=True,
                h1=True,
            ),
            "gfx950",
        ),
        48: (
            S(
                problem=_p(KH=1, KW=1, PAD=0, stride=2, C=256, K=128),
                bk=192,
                wt=True,
                ds=True,
                zf=True,
            ),
            "gfx950",
        ),
        49: (
            S(
                problem=_p(KH=1, KW=1, PAD=0, stride=2, C=256, K=128, dtype=h),
                bx=128,
                wt=True,
                ds=True,
                zf=True,
                zl=8,
            ),
            "gfx950",
        ),
        50: (
            S(
                problem=_p(KH=1, KW=1, PAD=0, stride=2, C=256, K=264),
                bx=128,
                wt=True,
                ds=True,
                zf=True,
                zl=8,
                c_tail=True,
            ),
            "gfx950",
        ),
        51: (S(problem=_p(stride=2, dtype=h), bk=128, split_c=6), "gfx950"),
        52: (S(problem=_p(C=1024), split_c=16), "gfx950"),
        53: (
            S(problem=_p(stride=2, C=256, K=128, dtype=h), ds=True, split_c=2),
            "gfx950",
        ),
        54: (S(problem=_p(KH=2, KW=2, PAD=0), bx=128, cs=2), "gfx950"),
        55: (
            S(problem=_p(KH=1, KW=1, PAD=0, stride=2), ds=True, zf=True),
            "gfx950",
        ),
        56: (S(problem=_p(), bx=128, xt8=True), "gfx950"),
        57: (S(problem=_p(), grp=True), "gfx950"),
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
