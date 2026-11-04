#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_fwd_gstream_wmma_emit.py -- Python reference emitter
# for the small-group forward walking row bands (direct_fwd_gstream_wmma).
# Selects one of N sampled spec configs by argv[1], builds the
# FwdGstreamWmmaSpec, builds the kernel via build_fwd_gstream_wmma(spec,
# arch=<cfg arch>) and prints the LLVM IR to stdout so it can be byte-compared
# with the C emitter conv_direct_fwd_gstream_wmma_emit.c.
#
# Config index map (gfx1250, 3x3 PAD 1, N 2, 28x28 unless noted; knobs
# waves_k / waves_p / tiles_k / tiles_p / tile_w / walk / stages):
#   0  bf16, cpg 8 (two taps per WMMA K, two groups per M tile), 1/2/2/2/8/2/2
#   1  fp16, cpg 16, stride 2, phase_split, 2/2/2/2/16/4/2
#   2  bf16, cpg 32 (plain k-steps, one group per block), 2/2/4/1/16/12/2
#   3  fp16, cpg 64, stride 2, phase_split, pf_depth 2 (shared B), 2/2/2/1/16/8/2
#   4  bf16, cpg 56 (padded k-step and M tile), 8/1/2/4/32/4/2
#   5  fp16, cpg 56, stride 2, phase_split, 4/1/2/4/16/4/2
#   6  bf16, cpg 16, 19x23, 3 stages, no prefetch, 1/4/1/2/16/5/3
#   7  fp16, cpg 8, stride 2, iterative-ilp, 2/1/2/4/16/4/2
#   8  bf16, cpg 32, blocks of two groups, 4/2/2/2/16/12/2
#  rejected by both engines:
#   9  gfx950 (wave64)
#  10  cpg 24
#  11  phase_split at stride 1
#  12  cpg 16 != kpg 32
#  13  column padding 0 with PAD 1
from _emit_common import run_emit
from kernels.common.conv_direct_fwd_gstream_wmma import (
    FwdGstreamWmmaSpec,
    build_fwd_gstream_wmma,
)
from kernels.common.conv_direct_grouped import DirectConvProblem


def _p(c, g, s=1, dtype="bf16", H=28, W=28, k=None, pad_w=-1):
    return DirectConvProblem(
        N=2,
        H=H,
        W=W,
        groups=g,
        cpg=c,
        kpg=c if k is None else k,
        KH=3,
        KW=3,
        PAD=1,
        pad_w=pad_w,
        stride=s,
        dtype=dtype,
    )


def _k(wk, wp, tk, tp, tw, walk, st, ps=False, pd=1, **kw):
    return dict(
        waves_k=wk,
        waves_p=wp,
        tiles_k=tk,
        tiles_p=tp,
        tile_w=tw,
        walk=walk,
        stages=st,
        phase_split=ps,
        pf_depth=pd,
        **kw,
    )


def _spec(idx: int):
    S = FwdGstreamWmmaSpec
    cfgs = {
        0: (S(problem=_p(8, 8), **_k(1, 2, 2, 2, 8, 2, 2)), "gfx1250"),
        1: (
            S(problem=_p(16, 8, 2, "fp16"), **_k(2, 2, 2, 2, 16, 4, 2, True)),
            "gfx1250",
        ),
        2: (S(problem=_p(32, 8), **_k(2, 2, 4, 1, 16, 12, 2)), "gfx1250"),
        3: (
            S(problem=_p(64, 4, 2, "fp16"), **_k(2, 2, 2, 1, 16, 8, 2, True, 2)),
            "gfx1250",
        ),
        4: (S(problem=_p(56, 4), **_k(8, 1, 2, 4, 32, 4, 2)), "gfx1250"),
        5: (
            S(problem=_p(56, 4, 2, "fp16"), **_k(4, 1, 2, 4, 16, 4, 2, True)),
            "gfx1250",
        ),
        6: (
            S(
                problem=_p(16, 4, H=19, W=23),
                **_k(1, 4, 1, 2, 16, 5, 3, prefetch=False),
            ),
            "gfx1250",
        ),
        7: (
            S(
                problem=_p(8, 8, 2, "fp16"),
                **_k(2, 1, 2, 4, 16, 4, 2, sched="iterative-ilp"),
            ),
            "gfx1250",
        ),
        8: (S(problem=_p(32, 8), **_k(4, 2, 2, 2, 16, 12, 2)), "gfx1250"),
        9: (S(problem=_p(8, 8)), "gfx950"),
        10: (S(problem=_p(24, 8)), "gfx1250"),
        11: (S(problem=_p(16, 8), **_k(2, 2, 2, 2, 16, 4, 2, True)), "gfx1250"),
        12: (S(problem=_p(16, 8, k=32)), "gfx1250"),
        13: (S(problem=_p(16, 8, pad_w=0)), "gfx1250"),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    return cfgs[idx]


def _build(spec, arch):
    return build_fwd_gstream_wmma(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: conv_direct_fwd_gstream_wmma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
