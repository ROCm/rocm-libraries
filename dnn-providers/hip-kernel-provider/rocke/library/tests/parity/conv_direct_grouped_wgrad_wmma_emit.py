#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_grouped_wgrad_wmma_emit.py -- Python reference
# emitter for the gfx1250 wave32 WMMA grouped direct wgrad parity harness.
# Selects one of N sampled spec configs by argv[1], builds the
# GroupedWgradWmmaSpec, builds the kernel via
# build_grouped_wgrad_wmma(spec, arch=<cfg arch>) and prints the LLVM IR to
# stdout so it can be byte-compared with the C emitter
# conv_direct_grouped_wgrad_wmma_emit.c.
#
# Problem: N=2 H=19 W=45, C = K = 64 (groups = 64 / cpg) unless noted.
# Knobs as (waves_c, q_steps, rows_per_chunk, target_blocks, stages).
# Config index map (gfx1250 unless noted):
#   0  bf16, cpg 4, 3x3 PAD 1, defaults (4, 2, 32, 1024, 2)
#   1  fp16, cpg 8, 3x3 PAD 1, defaults
#   2  bf16, cpg 4, 3x3 PAD 0, (8, 1, 8, 2048, 3)
#   3  fp16, cpg 8, 3x3 PAD 0, (8, 2, 16, 1024, 4)
#   4  bf16, cpg 8, 3x3 PAD 1, (4, 1, 16, 2048, 2)
#   5  fp16, cpg 4, 3x3 PAD 1, (4, 2, 8, 1024, 3)
#   6  bf16, cpg 4, 3x3 PAD 0, (8, 2, 32, 2048, 4)
#   7  fp16, cpg 8, 3x3 PAD 1, (8, 1, 32, 1024, 2)
#   8  bf16, cpg 8, 3x3 PAD 0, (4, 2, 8, 2048, 4)
#   9  fp16, cpg 4, 3x3 PAD 0, (4, 1, 16, 2048, 3)
#  10  bf16, cpg 4, C = K = 80 (a partial channel tile), (4, 2, 16, 1024, 2)
#  11  bf16, cpg 8, (2, 3, 8, 512, 3): three WMMA steps, two waves
#  12  fp16, cpg 4, 1x1 PAD 0 (no dY ring), (1, 4, 4, 64, 2)
#  13  bf16, cpg 8, 5x3 PAD 1 (KH != KW), (2, 1, 8, 1024, 3)
#  14  rejected by both engines: (8, 4, 32, 1024, 4) needs more than 160 KiB LDS
#  15  rejected by both engines: gfx950 (no ds_load_tr16_b128 / wmma 16x16x32)
#  16  rejected by both engines: cpg = kpg = 16
#  17  rejected by both engines: stride 2
#  18  rejected by both engines: C = 24 (not whole 16-channel tiles)
#  19  rejected by both engines: stages 5
#  20  rejected by both engines: dilation 2x2
#  21  rejected by both engines: fp32
#  22  rejected by both engines: 3x3 PAD 3
#  23  rejected by both engines: q_steps 5
#  24  rejected by both engines: waves_c 3
#  25  rejected by both engines: PAD 1 with column padding 0 (one padding only)
#  26  rejected by both engines: 1x3 filter with PAD 1 (PAD > KH - 1)
#  27  rejected by both engines: cpg 4, kpg 8
from dataclasses import replace

from _emit_common import run_emit
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_direct_grouped_wgrad_wmma import (
    GroupedWgradWmmaSpec,
    build_grouped_wgrad_wmma,
)


def _p(
    *,
    cpg=4,
    C=64,
    KH=3,
    KW=3,
    PAD=1,
    dtype="bf16",
    stride=1,
    dil=1,
):
    return DirectConvProblem(
        N=2,
        H=19,
        W=45,
        groups=C // cpg,
        cpg=cpg,
        kpg=cpg,
        KH=KH,
        KW=KW,
        PAD=PAD,
        stride=stride,
        dtype=dtype,
        dil_h=dil,
        dil_w=dil,
    )


def _k(wc, qs, rpc, tb, st):
    return dict(waves_c=wc, q_steps=qs, rows_per_chunk=rpc, target_blocks=tb, stages=st)


def _spec(idx: int):
    S = GroupedWgradWmmaSpec
    cfgs = {
        0: (S(problem=_p()), "gfx1250"),
        1: (S(problem=_p(cpg=8, dtype="fp16")), "gfx1250"),
        2: (S(problem=_p(PAD=0), **_k(8, 1, 8, 2048, 3)), "gfx1250"),
        3: (
            S(problem=_p(cpg=8, PAD=0, dtype="fp16"), **_k(8, 2, 16, 1024, 4)),
            "gfx1250",
        ),
        4: (S(problem=_p(cpg=8), **_k(4, 1, 16, 2048, 2)), "gfx1250"),
        5: (S(problem=_p(dtype="fp16"), **_k(4, 2, 8, 1024, 3)), "gfx1250"),
        6: (S(problem=_p(PAD=0), **_k(8, 2, 32, 2048, 4)), "gfx1250"),
        7: (S(problem=_p(cpg=8, dtype="fp16"), **_k(8, 1, 32, 1024, 2)), "gfx1250"),
        8: (S(problem=_p(cpg=8, PAD=0), **_k(4, 2, 8, 2048, 4)), "gfx1250"),
        9: (
            S(problem=_p(PAD=0, dtype="fp16"), **_k(4, 1, 16, 2048, 3)),
            "gfx1250",
        ),
        10: (S(problem=_p(C=80), **_k(4, 2, 16, 1024, 2)), "gfx1250"),
        11: (S(problem=_p(cpg=8), **_k(2, 3, 8, 512, 3)), "gfx1250"),
        12: (
            S(problem=_p(KH=1, KW=1, PAD=0, dtype="fp16"), **_k(1, 4, 4, 64, 2)),
            "gfx1250",
        ),
        13: (S(problem=_p(cpg=8, KH=5), **_k(2, 1, 8, 1024, 3)), "gfx1250"),
        14: (S(problem=_p(), **_k(8, 4, 32, 1024, 4)), "gfx1250"),
        15: (S(problem=_p()), "gfx950"),
        16: (S(problem=_p(cpg=16)), "gfx1250"),
        17: (S(problem=_p(stride=2)), "gfx1250"),
        18: (S(problem=_p(C=24)), "gfx1250"),
        19: (S(problem=_p(), stages=5), "gfx1250"),
        20: (S(problem=_p(dil=2)), "gfx1250"),
        21: (S(problem=_p(dtype="fp32")), "gfx1250"),
        22: (S(problem=_p(PAD=3)), "gfx1250"),
        23: (S(problem=_p(), q_steps=5), "gfx1250"),
        24: (S(problem=_p(), waves_c=3), "gfx1250"),
        25: (S(problem=replace(_p(), pad_w=0)), "gfx1250"),
        26: (S(problem=_p(KH=1)), "gfx1250"),
        27: (S(problem=replace(_p(), kpg=8)), "gfx1250"),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    return cfgs[idx]


def main() -> int:
    return run_emit(
        _spec,
        build_grouped_wgrad_wmma,
        usage="usage: conv_direct_grouped_wgrad_wmma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
