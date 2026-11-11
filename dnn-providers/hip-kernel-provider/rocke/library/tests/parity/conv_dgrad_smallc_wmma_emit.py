#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_dgrad_smallc_wmma_emit.py -- Python reference emitter for
# the wave32 WMMA small-C stem dgrad parity harness. Selects one of N sampled
# spec configs by argv[1], builds the DgradSmallcWmmaSpec, builds the kernel via
# build_dgrad_smallc_wmma(spec, arch=<cfg arch>) and prints the LLVM IR to
# stdout so it can be byte-compared with the C emitter
# conv_dgrad_smallc_wmma_emit.c.
#
# Config index map (gfx1250 unless noted; 64 dY channels):
#   0  bf16, 3x3 s1 p1 C4, AOT knobs 1x3 waves, 3x8 tiles, 8x32 tile, ck 64,
#      prefetch + iterative-ilp
#   1  fp16, 3x3 s2 p1 C3, AOT knobs 2x4 waves, 1x5 tiles, 32x32 tile (no halo
#      above / left: nothing to zero)
#   2  bf16, 7x7 s2 p3 C3, AOT knobs 2x4 waves, 5x6 tiles, ck 64, prefetch
#   3  fp16, 7x7 s2 p3 C3, AOT knobs 2x6 waves, 5x4 tiles, ck 64, prefetch
#   4  bf16, 3x3 s1 p1 C4, a 3-deep ring
#   5  bf16, 3x3 s1 p1 C1 (one-channel overlap-add)
#   6  rejected by both engines: gfx950 (wave64, no TDM)
#   7  rejected by both engines: C 8 > 4
#   8  rejected by both engines: a patch embedding (stride == filter, pad 0)
from _emit_common import run_emit
from kernels.common.conv_dgrad_smallc_wmma import (
    DgradSmallcWmmaSpec,
    build_dgrad_smallc_wmma,
)
from kernels.common.conv_direct_grouped import DirectConvProblem

AOT_NAME = "rocke_dgrad_smallc_wmma"


def _p(K, S, PAD, C, dtype="bf16"):
    return DirectConvProblem(
        N=2,
        H=64,
        W=64,
        groups=1,
        cpg=C,
        kpg=64,
        KH=K,
        KW=K,
        PAD=PAD,
        stride=S,
        dtype=dtype,
    )


def _k(wc, wp, tc, tp, oh, ow, ck, pf, **kw):
    knobs = dict(
        waves_c=wc, waves_p=wp, tiles_c=tc, tiles_p=tp, out_h=oh, out_w=ow, ck=ck
    )
    if pf:
        knobs.update(prefetch=True, sched="iterative-ilp")
    knobs.update(kw)
    return knobs


def _spec(idx: int):
    S = DgradSmallcWmmaSpec
    cfgs = {
        0: (
            S(problem=_p(3, 1, 1, 4), name=AOT_NAME, **_k(1, 3, 3, 8, 8, 32, 64, True)),
            "gfx1250",
        ),
        1: (
            S(
                problem=_p(3, 2, 1, 3, "fp16"),
                name=AOT_NAME,
                **_k(2, 4, 1, 5, 32, 32, 32, False),
            ),
            "gfx1250",
        ),
        2: (
            S(
                problem=_p(7, 2, 3, 3),
                name=AOT_NAME,
                **_k(2, 4, 5, 6, 32, 32, 64, True),
            ),
            "gfx1250",
        ),
        3: (
            S(
                problem=_p(7, 2, 3, 3, "fp16"),
                name=AOT_NAME,
                **_k(2, 6, 5, 4, 32, 32, 64, True),
            ),
            "gfx1250",
        ),
        4: (
            S(problem=_p(3, 1, 1, 4), **_k(1, 4, 3, 6, 8, 32, 32, False, stages=3)),
            "gfx1250",
        ),
        5: (S(problem=_p(3, 1, 1, 1), **_k(1, 4, 1, 5, 8, 16, 32, False)), "gfx1250"),
        6: (S(problem=_p(3, 1, 1, 4), **_k(1, 3, 3, 8, 8, 32, 64, True)), "gfx950"),
        7: (S(problem=_p(3, 1, 1, 8), **_k(1, 3, 3, 8, 8, 32, 64, True)), "gfx1250"),
        8: (S(problem=_p(4, 4, 0, 3), **_k(1, 3, 3, 8, 8, 32, 64, True)), "gfx1250"),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    return cfgs[idx]


def main() -> int:
    return run_emit(
        _spec,
        build_dgrad_smallc_wmma,
        usage="usage: conv_dgrad_smallc_wmma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
