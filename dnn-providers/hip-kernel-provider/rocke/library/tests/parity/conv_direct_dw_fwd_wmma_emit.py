#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_dw_fwd_wmma_emit.py -- Python reference emitter for
# the Toeplitz WMMA depthwise forward (direct_dw_fwd_wmma). Selects one of N
# sampled spec configs by argv[1], builds the DwFwdWmmaSpec, builds the kernel
# via build_dw_fwd_wmma(spec, arch=<cfg arch>) and prints the LLVM IR to stdout
# so it can be byte-compared with the C emitter conv_direct_dw_fwd_wmma_emit.c.
#
# Config index map (gfx1250, N 2, H 20, W 37, groups 64 unless noted):
#   0  bf16, 3x3 PAD 1, defaults (4 waves, 8 rows, w_fold 1, 2 prefetch rows)
#   1  fp16, 5x5 PAD 2, 2 waves, 16 rows, w_fold 2, 3 prefetch rows
#   2  bf16, 7x7 PAD 3, 8 waves, 4 rows, w_fold 4, 4 prefetch rows
#   3  fp16, 7x7 PAD 0 ("valid"), 4 waves, 7 rows
#   4  bf16, 3x3 PAD 1, groups 128, 16 waves, 1 row
#   5  fp16, 3x3 PAD 1, 56x56, a whole 56-row map, w_fold 2
#   6  bf16, 3x3 PAD 1, 13x19, groups 24 (a partial channel block), 8 waves,
#      2 rows, w_fold 4, 3 prefetch rows
#  rejected by both engines:
#   7  gfx950 (no wave32 WMMA form)
#   8  groups 12 (not a multiple of 8)
#   9  9x9 PAD 4 (fragments past the VGPR budget)
#  10  column padding 0 with PAD 1
#  11  5x5, 16 waves with 4 prefetch rows (past the 16-wave fragment budget)
#  built (rectangular filters, one forward PAD on both axes):
#  12  bf16, 3x5 PAD 1, 6 rows, w_fold 2
#  13  fp16, 1x7 PAD 3, 2 waves, 4 rows
from _emit_common import run_emit
from kernels.common.conv_direct_dw_fwd_wmma import DwFwdWmmaSpec, build_dw_fwd_wmma
from kernels.common.conv_direct_grouped import DirectConvProblem


def _p(K=3, PAD=1, dtype="bf16", G=64, H=20, W=37, pad_w=-1, KW=None):
    return DirectConvProblem(
        N=2,
        H=H,
        W=W,
        groups=G,
        cpg=1,
        kpg=1,
        KH=K,
        KW=K if KW is None else KW,
        PAD=PAD,
        pad_w=pad_w,
        stride=1,
        dtype=dtype,
    )


def _spec(idx: int):
    S = DwFwdWmmaSpec
    cfgs = {
        0: (S(problem=_p()), "gfx1250"),
        1: (
            S(
                problem=_p(5, 2, "fp16"),
                block_waves=2,
                block_h=16,
                w_fold=2,
                prefetch_rows=3,
            ),
            "gfx1250",
        ),
        2: (
            S(problem=_p(7, 3), block_waves=8, block_h=4, w_fold=4, prefetch_rows=4),
            "gfx1250",
        ),
        3: (S(problem=_p(7, 0, "fp16"), block_h=7), "gfx1250"),
        4: (S(problem=_p(G=128), block_waves=16, block_h=1), "gfx1250"),
        5: (S(problem=_p(dtype="fp16", H=56, W=56), block_h=56, w_fold=2), "gfx1250"),
        6: (
            S(
                problem=_p(G=24, H=13, W=19),
                block_waves=8,
                block_h=2,
                w_fold=4,
                prefetch_rows=3,
            ),
            "gfx1250",
        ),
        7: (S(problem=_p()), "gfx950"),
        8: (S(problem=_p(G=12)), "gfx1250"),
        9: (S(problem=_p(9, 4)), "gfx1250"),
        10: (S(problem=_p(pad_w=0)), "gfx1250"),
        11: (S(problem=_p(5, 2), block_waves=16, prefetch_rows=4), "gfx1250"),
        12: (S(problem=_p(3, 1, KW=5), block_h=6, w_fold=2), "gfx1250"),
        13: (S(problem=_p(1, 3, "fp16", KW=7), block_waves=2, block_h=4), "gfx1250"),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    return cfgs[idx]


def _build(spec, arch):
    return build_dw_fwd_wmma(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: conv_direct_dw_fwd_wmma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
