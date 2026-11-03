#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_fwd_small_wmma_emit.py -- Python reference emitter for the
# wave32 WMMA forward of tiny channel counts (fwd_smallc_wmma and
# fwd_smallk_wmma). Selects one of N sampled spec configs by argv[1], builds
# the FwdSmallcWmmaSpec / FwdSmallkWmmaSpec, builds the kernel via
# build_fwd_smallc_wmma / build_fwd_smallk_wmma(spec, arch=<cfg arch>) and
# prints the LLVM IR to stdout so it can be byte-compared with the C emitter
# conv_fwd_small_wmma_emit.c.
#
# Config index map (gfx1250 unless noted):
#  small C (fwd_smallc_wmma):
#   0  bf16, 3x3 s1 PAD 1, C=3, defaults (1x4 waves, 2x4 tiles, 8x32 tile)
#   1  fp16, 3x3 s1 PAD 1, C=4, 2x2 waves, 8x16 tile, tdm_store
#   2  bf16, 3x3 s2 PAD 1, C=3, 1x2 waves, 4x4 tiles, 4 tiles per block +
#      tdm_store (double-buffered window, LDS-staged output)
#   3  bf16, 7x7 s2 PAD 3, C=3, 1x1 waves, 4x4 tiles, 2x32 tile, 2 tiles per
#      block, iterative-ilp
#   4  fp16, 4x4 s4 PAD 0, C=3, 1x2 waves, 8x4 tiles, 4x32 tile, 4 tiles per
#      block + tdm_store
#   5  bf16, 14x14 s14 PAD 0, C=3, 2x2 waves, 2x4 tiles, 4x32 tile
#   6  fp16, 16x16 s16 PAD 0, C=3, 2x2 waves, 2x2 tiles, 4x16 tile, max-ilp
#   7  bf16, 3x3 s1 PAD 1, C=3, 3 tiles per block (odd tile count), custom name
#  small K (fwd_smallk_wmma):
#   8  bf16, 3x3 s1 PAD 1, K=4, 1x4 waves, 3x3 tiles, 6x16 tile, ck 64, prefetch
#   9  fp16, 3x3 s1 PAD 1, K=4, 3x2 waves, 1x5 tiles, 10x10 tile, ck 64,
#      3 stages, prefetch + max-ilp
#  10  bf16, 3x3 s1 PAD 1, K=8, 1x4 waves, 5x6 tiles, defaults otherwise
#  11  fp16, 3x3 s2 PAD 0, K=2, 1x2 waves, 2x3 tiles, 4x4 tile, 4 stages
#  12  bf16, 1x1 s1 PAD 0, K=1, 1x2 waves, 1x2 tiles, 8x8 tile, ck 96,
#      iterative-ilp
#  rejected by both engines:
#  13  fwd_smallc_wmma on gfx950 (wave64, no TDM)
#  14  fwd_smallk_wmma with C=36 (not a multiple of 8)
from _emit_common import run_emit
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_fwd_small_wmma import (
    FwdSmallcWmmaSpec,
    FwdSmallkWmmaSpec,
    build_fwd_smallc_wmma,
    build_fwd_smallk_wmma,
)


def _p(*, K=3, PAD=1, S=1, C=3, Kout=64, dtype="bf16", H=32, W=40):
    return DirectConvProblem(
        N=2,
        H=H,
        W=W,
        groups=1,
        cpg=C,
        kpg=Kout,
        KH=K,
        KW=K,
        PAD=PAD,
        stride=S,
        dtype=dtype,
    )


def _spec(idx: int):
    C = FwdSmallcWmmaSpec
    K = FwdSmallkWmmaSpec
    cfgs = {
        0: C(problem=_p()),
        1: C(
            problem=_p(C=4, dtype="fp16"),
            waves_k=2,
            waves_p=2,
            tiles_k=2,
            tiles_p=4,
            out_h=8,
            out_w=16,
            tdm_store=True,
        ),
        2: C(
            problem=_p(S=2),
            waves_k=1,
            waves_p=2,
            tiles_k=4,
            tiles_p=4,
            out_h=8,
            out_w=16,
            tiles_per_block=4,
            tdm_store=True,
        ),
        3: C(
            problem=_p(K=7, PAD=3, S=2),
            waves_k=1,
            waves_p=1,
            tiles_k=4,
            tiles_p=4,
            out_h=2,
            out_w=32,
            tiles_per_block=2,
            sched="iterative-ilp",
        ),
        4: C(
            problem=_p(K=4, PAD=0, S=4, dtype="fp16"),
            waves_k=1,
            waves_p=2,
            tiles_k=8,
            tiles_p=4,
            out_h=4,
            out_w=32,
            tiles_per_block=4,
            tdm_store=True,
        ),
        5: C(
            problem=_p(K=14, PAD=0, S=14, H=224, W=224),
            waves_k=2,
            waves_p=2,
            tiles_k=2,
            tiles_p=4,
            out_h=4,
            out_w=32,
        ),
        6: C(
            problem=_p(K=16, PAD=0, S=16, H=224, W=224, dtype="fp16"),
            waves_k=2,
            waves_p=2,
            tiles_k=2,
            tiles_p=2,
            out_h=4,
            out_w=16,
            sched="max-ilp",
        ),
        7: C(problem=_p(), name="rocke_fwd_smallc_wmma", tiles_per_block=3),
        8: K(
            problem=_p(C=128, Kout=4),
            waves_c=1,
            waves_p=4,
            tiles_c=3,
            tiles_p=3,
            out_h=6,
            out_w=16,
            ck=64,
            prefetch=True,
        ),
        9: K(
            problem=_p(C=128, Kout=4, dtype="fp16"),
            waves_c=3,
            waves_p=2,
            tiles_c=1,
            tiles_p=5,
            out_h=10,
            out_w=10,
            ck=64,
            stages=3,
            prefetch=True,
            sched="max-ilp",
        ),
        10: K(problem=_p(C=64, Kout=8), tiles_c=5),
        11: K(
            problem=_p(PAD=0, S=2, C=64, Kout=2, dtype="fp16"),
            waves_c=1,
            waves_p=2,
            tiles_c=2,
            tiles_p=3,
            out_h=4,
            out_w=4,
            stages=4,
        ),
        12: K(
            problem=_p(K=1, PAD=0, C=96, Kout=1),
            waves_c=1,
            waves_p=2,
            tiles_c=1,
            tiles_p=2,
            out_h=8,
            out_w=8,
            ck=96,
            sched="iterative-ilp",
        ),
        13: C(problem=_p()),
        14: K(problem=_p(C=36, Kout=4)),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    return cfgs[idx], ("gfx950" if idx == 13 else "gfx1250")


def _build(spec, arch):
    if isinstance(spec, FwdSmallcWmmaSpec):
        return build_fwd_smallc_wmma(spec, arch=arch)
    return build_fwd_smallk_wmma(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: conv_fwd_small_wmma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
