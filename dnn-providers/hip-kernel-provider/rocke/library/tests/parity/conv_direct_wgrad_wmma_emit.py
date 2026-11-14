#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_wgrad_wmma_emit.py -- Python reference emitter for
# the wave32 WMMA dense direct wgrad parity harness. Selects one of N sampled
# spec configs by argv[1], builds the DirectWgradWmmaSpec, builds the kernel
# via build_direct_wgrad_wmma(spec, arch=<cfg arch>) and prints the LLVM IR to
# stdout so it can be byte-compared with the C emitter
# conv_direct_wgrad_wmma_emit.c.
#
# Problem N=2 H=12 W=40 C=64 K=96 3x3 PAD 1 stride 1 unless noted. "AOT" is
# the shipped knob set: tiles 2x2, stages 4, rows 2, prefetch, sched, pad 16,
# xcd 8, TDM. Config index map (gfx1250 unless noted):
#   0  bf16, AOT, waves 4x2, target_blocks 256
#   1  fp16, AOT, waves 2x4, target_blocks 512
#   2  bf16, AOT, PAD 0, waves 2x4, target_blocks 256
#   3  fp16, AOT, PAD 0, waves 4x2, target_blocks 512
#   4  bf16, defaults (per-lane copies, rows 1, no prefetch / sched, xcd 0, pad 8)
#   5  fp16, per-lane copies, prefetch (rows 1), q_steps 2, tiles 1x1, 2 stages
#   6  bf16, per-lane copies, rows 2, prefetch, 3 stages, sched, xcd 8, pad 16
#   7  bf16, TDM, rows 1, no prefetch, waves 2x2, tiles 4x1
#   8  fp16, TDM, PAD 0, q_steps 2, tiles 1x4, 2 stages
#   9  bf16, TDM, 16 waves (4x4), rows 4, prefetch, 3 stages, target_blocks 64
#  10  fp16, per-lane copies, 2 waves (1x2), tiles 2x1, sched, target_blocks 1
#  11  bf16, TDM, 2 waves (2x1), rows 3 (3 transfers per wave), prefetch, 3 stages
#  12  fp16, 5x5 PAD 2, TDM, waves 2x2, tiles 1x1, prefetch
#  13  bf16, 1x3 PAD 0 (no dY ring), per-lane copies, xcd 3
#  14  fp16, TDM, rows 2, no prefetch, 2 stages, sched, pad 8, xcd 0
#  15  rejected by both engines: gfx950 (AOT knobs)
#  16  rejected by both engines: stride 2
#  17  rejected by both engines: groups 2
#  18  rejected by both engines: tiles 4x4 (too many VGPRs)
#  19  rejected by both engines: TDM rows 3 on 4 waves (uneven transfers)
#  20  rejected by both engines: rows 2 with prefetch on 2 stages
#  21  rejected by both engines: pad 4 (not a multiple of 8)
#  22  rejected by both engines: dilation 2x2
#  23  rejected by both engines: PAD 3 on a 3x3 filter
#  24  rejected by both engines: C 36 (not a multiple of 8)
#  25  rejected by both engines: LDS past 300 KiB (16 waves, rows 4, 8 stages)
#  26  rejected by both engines: PAD 1 with column padding 0 (one padding only)
#  27  rejected by both engines: 3x1 filter with PAD 1 (PAD > KW - 1)
from dataclasses import replace

from _emit_common import run_emit
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_direct_wgrad_wmma import (
    DirectWgradWmmaSpec,
    build_direct_wgrad_wmma,
)


def _p(*, KH=3, KW=3, PAD=1, dtype="bf16", stride=1, groups=1, C=64, K=96, dil=1):
    return DirectConvProblem(
        N=2,
        H=12,
        W=40,
        groups=groups,
        cpg=C // groups,
        kpg=K // groups,
        KH=KH,
        KW=KW,
        PAD=PAD,
        stride=stride,
        dtype=dtype,
        dil_h=dil,
        dil_w=dil,
    )


# The shipped (AOT) knob set; waves and target_blocks vary per job.
_AOT = dict(
    tiles_k=2,
    tiles_c=2,
    stages=4,
    rows=2,
    prefetch=True,
    sched=True,
    pad=16,
    xcd=8,
    tdm=True,
)


def _spec(idx: int):
    S = DirectWgradWmmaSpec
    cfgs = {
        0: (
            S(problem=_p(), waves_k=4, waves_c=2, target_blocks=256, **_AOT),
            "gfx1250",
        ),
        1: (
            S(
                problem=_p(dtype="fp16"),
                waves_k=2,
                waves_c=4,
                target_blocks=512,
                **_AOT,
            ),
            "gfx1250",
        ),
        2: (
            S(problem=_p(PAD=0), waves_k=2, waves_c=4, target_blocks=256, **_AOT),
            "gfx1250",
        ),
        3: (
            S(
                problem=_p(PAD=0, dtype="fp16"),
                waves_k=4,
                waves_c=2,
                target_blocks=512,
                **_AOT,
            ),
            "gfx1250",
        ),
        4: (S(problem=_p()), "gfx1250"),
        5: (
            S(
                problem=_p(dtype="fp16"),
                tiles_k=1,
                tiles_c=1,
                q_steps=2,
                stages=2,
                prefetch=True,
            ),
            "gfx1250",
        ),
        6: (
            S(
                problem=_p(),
                rows=2,
                stages=3,
                prefetch=True,
                sched=True,
                xcd=8,
                pad=16,
            ),
            "gfx1250",
        ),
        7: (
            S(problem=_p(), waves_k=2, waves_c=2, tiles_k=4, tiles_c=1, tdm=True),
            "gfx1250",
        ),
        8: (
            S(
                problem=_p(PAD=0, dtype="fp16"),
                tiles_k=1,
                tiles_c=4,
                q_steps=2,
                stages=2,
                tdm=True,
            ),
            "gfx1250",
        ),
        9: (
            S(
                problem=_p(),
                waves_k=4,
                waves_c=4,
                rows=4,
                stages=3,
                prefetch=True,
                target_blocks=64,
                tdm=True,
            ),
            "gfx1250",
        ),
        10: (
            S(
                problem=_p(dtype="fp16"),
                waves_k=1,
                waves_c=2,
                tiles_k=2,
                tiles_c=1,
                sched=True,
                target_blocks=1,
            ),
            "gfx1250",
        ),
        11: (
            S(
                problem=_p(),
                waves_k=2,
                waves_c=1,
                rows=3,
                stages=3,
                prefetch=True,
                tdm=True,
            ),
            "gfx1250",
        ),
        12: (
            S(
                problem=_p(KH=5, KW=5, PAD=2, dtype="fp16"),
                waves_k=2,
                waves_c=2,
                tiles_k=1,
                tiles_c=1,
                prefetch=True,
                tdm=True,
            ),
            "gfx1250",
        ),
        13: (S(problem=_p(KH=1, KW=3, PAD=0), xcd=3), "gfx1250"),
        14: (
            S(
                problem=_p(dtype="fp16"),
                rows=2,
                stages=2,
                sched=True,
                tdm=True,
            ),
            "gfx1250",
        ),
        15: (S(problem=_p(), **_AOT), "gfx950"),
        16: (S(problem=_p(stride=2)), "gfx1250"),
        17: (S(problem=_p(groups=2)), "gfx1250"),
        18: (S(problem=_p(), tiles_k=4, tiles_c=4), "gfx1250"),
        19: (
            S(problem=_p(), waves_k=2, waves_c=2, rows=3, tdm=True),
            "gfx1250",
        ),
        20: (S(problem=_p(), rows=2, stages=2, prefetch=True), "gfx1250"),
        21: (S(problem=_p(), pad=4), "gfx1250"),
        22: (S(problem=_p(PAD=2, dil=2)), "gfx1250"),
        23: (S(problem=_p(PAD=3)), "gfx1250"),
        24: (S(problem=_p(C=36)), "gfx1250"),
        25: (
            S(problem=_p(), waves_k=4, waves_c=4, rows=4, stages=8),
            "gfx1250",
        ),
        26: (S(problem=replace(_p(), pad_w=0)), "gfx1250"),
        27: (S(problem=_p(KH=3, KW=1)), "gfx1250"),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    return cfgs[idx]


def main() -> int:
    return run_emit(
        _spec,
        build_direct_wgrad_wmma,
        usage="usage: conv_direct_wgrad_wmma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
