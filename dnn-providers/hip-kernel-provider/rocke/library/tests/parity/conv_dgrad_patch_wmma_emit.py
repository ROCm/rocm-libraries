#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_dgrad_patch_wmma_emit.py -- Python reference emitter for
# the wave32 WMMA patch-embedding dgrad parity harness. Selects one of N sampled
# spec configs by argv[1], builds the DgradPatchWmmaSpec, builds the kernel via
# build_dgrad_patch_wmma(spec, arch=<cfg arch>) and prints the LLVM IR to
# stdout so it can be byte-compared with the C emitter
# conv_dgrad_patch_wmma_emit.c.
#
# Config index map (gfx1250 unless noted; 3 dX channels, 64 dY channels):
#   0  bf16, 4x4 patch, AOT knobs 3x1 waves, 1x8 tiles, tile_w 16, r_blk 4,
#      prefetch + iterative-ilp (scalar shuffle stores: P*C = 12)
#   1  fp16, 4x4 patch, AOT knobs 1x2 waves, 3x4 tiles, tile_w 32, ck 64, prefetch
#   2  bf16, 14x14 patch, AOT knobs 4x2 waves, 5x4 tiles, r_blk 7 (padded columns)
#   3  bf16, 16x16 patch, AOT knobs 4x2 waves, 6x4 tiles, r_blk 8 (vector stores)
#   4  fp16, 16x16 patch, AOT knobs 2x1 waves, 3x8 tiles, r_blk 2, ck 64, prefetch
#   5  bf16, 4x4 patch, a 3-deep ring
#   6  rejected by both engines: gfx950 (wave64, no TDM)
#   7  rejected by both engines: padding 1 (not a patch embedding)
#   8  rejected by both engines: block_c 64 over 48 GEMM columns
from _emit_common import run_emit
from kernels.common.conv_dgrad_patch_wmma import (
    DgradPatchWmmaSpec,
    build_dgrad_patch_wmma,
)
from kernels.common.conv_direct_grouped import DirectConvProblem

AOT_NAME = "rocke_dgrad_patch_wmma"


def _p(P, *, dtype="bf16", PAD=0, H=64, W=64):
    return DirectConvProblem(
        N=2,
        H=H,
        W=W,
        groups=1,
        cpg=3,
        kpg=64,
        KH=P,
        KW=P,
        PAD=PAD,
        stride=P,
        dtype=dtype,
    )


def _k(wc, wp, tc, tp, tw, rb, ck, pf, **kw):
    knobs = dict(
        waves_c=wc, waves_p=wp, tiles_c=tc, tiles_p=tp, tile_w=tw, r_blk=rb, ck=ck
    )
    if pf:
        knobs.update(prefetch=True, sched="iterative-ilp")
    knobs.update(kw)
    return knobs


def _spec(idx: int):
    S = DgradPatchWmmaSpec
    cfgs = {
        0: (
            S(problem=_p(4), name=AOT_NAME, **_k(3, 1, 1, 8, 16, 4, 32, True)),
            "gfx1250",
        ),
        1: (
            S(
                problem=_p(4, dtype="fp16"),
                name=AOT_NAME,
                **_k(1, 2, 3, 4, 32, 4, 64, True),
            ),
            "gfx1250",
        ),
        2: (
            S(
                problem=_p(14, H=56, W=56),
                name=AOT_NAME,
                **_k(4, 2, 5, 4, 32, 7, 32, False),
            ),
            "gfx1250",
        ),
        3: (
            S(problem=_p(16), name=AOT_NAME, **_k(4, 2, 6, 4, 16, 8, 32, False)),
            "gfx1250",
        ),
        4: (
            S(
                problem=_p(16, dtype="fp16"),
                name=AOT_NAME,
                **_k(2, 1, 3, 8, 16, 2, 64, True),
            ),
            "gfx1250",
        ),
        5: (S(problem=_p(4), **_k(3, 1, 1, 8, 16, 4, 32, False, stages=3)), "gfx1250"),
        6: (S(problem=_p(4), **_k(3, 1, 1, 8, 16, 4, 32, True)), "gfx950"),
        7: (S(problem=_p(4, PAD=1), **_k(3, 1, 1, 8, 16, 4, 32, False)), "gfx1250"),
        8: (S(problem=_p(4), **_k(4, 1, 1, 8, 16, 4, 32, False)), "gfx1250"),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    return cfgs[idx]


def main() -> int:
    return run_emit(
        _spec,
        build_dgrad_patch_wmma,
        usage="usage: conv_dgrad_patch_wmma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
