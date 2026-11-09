#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_fwd_tapouter_wmma_emit.py -- Python reference
# emitter for the tap-outer direct forward of large dilations
# (direct_fwd_tapouter_wmma). Selects one of N sampled spec configs by
# argv[1], builds the DirectFwdTapOuterSpec, builds the kernel via
# build_direct_fwd_tapouter_wmma(spec, arch=<cfg arch>) and prints the LLVM IR
# to stdout so it can be byte-compared with the C emitter
# conv_direct_fwd_tapouter_wmma_emit.c.
#
# Config index map (gfx1250, 3x3 "same" with PAD = dilation, N 2, 33x33,
# ungrouped cpg 64 / kpg 96 unless noted; knobs waves_k / waves_p / tiles_k /
# tiles_p / tile_w / ck / stages):
#   0  bf16, dilation 2, 1/4/8/2/16/64/2
#   1  fp16, dilation 6, 2/2/4/4/8/128/2
#   2  bf16, dilation 12, 2/2/4/4/32/128/2
#   3  fp16, dilation 24, groups 2 (cpg 32 / kpg 48), 4/1/4/4/16/128/2
#   4  bf16, dilation 36, 40x44, 4/2/4/4/16/64/2
#   5  fp16, dilation 4, defaults (3 stages)
#   6  bf16, dilation 18, 2/2/4/2/16/64/2, prefetch, iterative-ilp, tdm_split 2
#   7  fp16, dilation 2x3 (PAD 3), defaults
#  rejected by both engines:
#   8  gfx950 (wave64)
#   9  stride 2
#  10  cpg 12
#  11  tdm_split 3
#  12  column padding 1 with PAD 2
from _emit_common import run_emit
from kernels.common.conv_direct_fwd_tapouter_wmma import (
    DirectFwdTapOuterSpec,
    build_direct_fwd_tapouter_wmma,
)
from kernels.common.conv_direct_grouped import DirectConvProblem


def _p(d, dtype="bf16", g=1, c=64, k=96, H=33, W=33, dh=None, pad=None, s=1, pad_w=-1):
    return DirectConvProblem(
        N=2,
        H=H,
        W=W,
        groups=g,
        cpg=c,
        kpg=k,
        KH=3,
        KW=3,
        PAD=d if pad is None else pad,
        pad_w=pad_w,
        stride=s,
        dtype=dtype,
        dil_h=d if dh is None else dh,
        dil_w=d,
    )


def _k(wk, wp, tk, tp, tw, ck, st, **kw):
    return dict(
        waves_k=wk,
        waves_p=wp,
        tiles_k=tk,
        tiles_p=tp,
        tile_w=tw,
        ck=ck,
        stages=st,
        **kw,
    )


def _spec(idx: int):
    S = DirectFwdTapOuterSpec
    cfgs = {
        0: (S(problem=_p(2), **_k(1, 4, 8, 2, 16, 64, 2)), "gfx1250"),
        1: (S(problem=_p(6, "fp16"), **_k(2, 2, 4, 4, 8, 128, 2)), "gfx1250"),
        2: (S(problem=_p(12), **_k(2, 2, 4, 4, 32, 128, 2)), "gfx1250"),
        3: (
            S(problem=_p(24, "fp16", g=2, c=32, k=48), **_k(4, 1, 4, 4, 16, 128, 2)),
            "gfx1250",
        ),
        4: (S(problem=_p(36, H=40, W=44), **_k(4, 2, 4, 4, 16, 64, 2)), "gfx1250"),
        5: (S(problem=_p(4, "fp16")), "gfx1250"),
        6: (
            S(
                problem=_p(18),
                **_k(
                    2,
                    2,
                    4,
                    2,
                    16,
                    64,
                    2,
                    prefetch=True,
                    sched="iterative-ilp",
                    tdm_split=2,
                ),
            ),
            "gfx1250",
        ),
        7: (S(problem=_p(3, "fp16", dh=2, pad=3)), "gfx1250"),
        8: (S(problem=_p(2)), "gfx950"),
        9: (S(problem=_p(2, s=2)), "gfx1250"),
        10: (S(problem=_p(2, c=12)), "gfx1250"),
        11: (S(problem=_p(2), tdm_split=3), "gfx1250"),
        12: (S(problem=_p(2, pad_w=1)), "gfx1250"),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    return cfgs[idx]


def _build(spec, arch):
    return build_direct_fwd_tapouter_wmma(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: conv_direct_fwd_tapouter_wmma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
