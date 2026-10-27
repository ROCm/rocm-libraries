#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_dgrad_wmma_emit.py -- Python reference emitter for
# the wave32 WMMA direct dgrad parity harness. Selects one of N sampled spec
# configs by argv[1], builds the DirectDgradWmmaSpec, builds the kernel via
# build_direct_dgrad_wmma(spec, arch=<cfg arch>) and prints the LLVM IR to
# stdout so it can be byte-compared with the C emitter
# conv_direct_dgrad_wmma_emit.c.
#
# Config index map (gfx1250 unless noted):
#   0  bf16, defaults (2x4 waves, 4x4 tiles, tile_w 16, ck 32, 2 stages)
#   1  fp16, 2 images of 8-wide tiles, 3 stages, prefetch + iterative-ilp
#   2  bf16, 3x3 PAD 0, 4 images of 4-wide tiles, one wave along channels
#   3  bf16, 5x5 PAD 2, 1x4 waves, 4x6 tiles, tile_w 32, prefetch + max-ilp
#   4  fp16, 1x2 waves, 1x1 tiles (smallest block), 3 stages
#   5  bf16, 2x4 waves, 4x8 tiles (upper VGPR banks), tile_w 64, prefetch
#   6  rejected by both engines: gfx950 (wave64, no TDM)
#   7  rejected by both engines: stride 2
#   8  bf16, ck 64 (two WMMA k-steps per tap), prefetch + iterative-ilp
from _emit_common import run_emit
from kernels.common.conv_direct_dgrad_wmma import (
    DirectDgradWmmaSpec,
    build_direct_dgrad_wmma,
)
from kernels.common.conv_direct_grouped import DirectConvProblem


def _p(*, KH=3, PAD=1, dtype="bf16", stride=1):
    return DirectConvProblem(
        N=2,
        H=20,
        W=24,
        groups=1,
        cpg=64,
        kpg=96,
        KH=KH,
        KW=KH,
        PAD=PAD,
        stride=stride,
        dtype=dtype,
    )


def _spec(idx: int):
    S = DirectDgradWmmaSpec
    cfgs = {
        0: (S(problem=_p()), "gfx1250"),
        1: (
            S(
                problem=_p(dtype="fp16"),
                waves_c=2,
                waves_p=4,
                tiles_c=2,
                tiles_p=2,
                tile_w=8,
                tile_n=2,
                stages=3,
                prefetch=True,
                sched="iterative-ilp",
            ),
            "gfx1250",
        ),
        2: (
            S(
                problem=_p(PAD=0),
                waves_c=1,
                waves_p=8,
                tiles_c=1,
                tiles_p=2,
                tile_w=4,
                tile_n=4,
            ),
            "gfx1250",
        ),
        3: (
            S(
                problem=_p(KH=5, PAD=2),
                waves_c=1,
                waves_p=4,
                tiles_c=4,
                tiles_p=6,
                tile_w=32,
                prefetch=True,
                sched="max-ilp",
            ),
            "gfx1250",
        ),
        4: (
            S(
                problem=_p(dtype="fp16"),
                waves_c=1,
                waves_p=2,
                tiles_c=1,
                tiles_p=1,
                stages=3,
            ),
            "gfx1250",
        ),
        5: (
            S(problem=_p(), tiles_c=4, tiles_p=8, tile_w=64, prefetch=True),
            "gfx1250",
        ),
        6: (S(problem=_p()), "gfx950"),
        7: (S(problem=_p(stride=2)), "gfx1250"),
        8: (
            S(
                problem=_p(),
                waves_c=1,
                waves_p=4,
                tiles_c=4,
                tiles_p=4,
                ck=64,
                prefetch=True,
                sched="iterative-ilp",
            ),
            "gfx1250",
        ),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    return cfgs[idx]


def main() -> int:
    return run_emit(
        _spec,
        build_direct_dgrad_wmma,
        usage="usage: conv_direct_dgrad_wmma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
