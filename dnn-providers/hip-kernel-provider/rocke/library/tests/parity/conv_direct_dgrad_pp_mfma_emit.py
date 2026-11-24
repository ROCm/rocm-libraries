#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_dgrad_pp_mfma_emit.py -- Python reference emitter for
# the gfx950 LDS-DMA ping-pong MFMA direct dgrad parity harness. Selects one of N
# sampled spec configs by argv[1], builds the DgradPpMfmaSpec over the
# transposed (forward-pass) problem of a dgrad, builds the kernel via
# build_dgrad_pp_mfma(spec, arch=<cfg arch>) and prints the LLVM IR to stdout so
# it can be byte-compared with the C emitter conv_direct_dgrad_pp_mfma_emit.c.
#
# The dgrad problem is N=2, 20x24, C=72, K=96 (the pass reduces over K=96 and
# writes C=72). Config index map (gfx950 unless noted):
#   0  bf16, 3x3, block_k 128
#   1  fp16, 3x3, block_k 256
#   2  bf16, 5x5, block_k 128
#   3  fp16, 5x5, block_k 128
#   4  bf16, 3x3, block_k 256
#   5  fp16, 3x3, block_k 128
#   6  rejected by both engines: block_k 64 (weight rows under 256 bytes)
#   7  rejected by both engines: 5x5 at block_k 256 (LDS)
#   8  rejected by both engines: 7x7 at block_k 128 (LDS)
#   9  rejected by both engines: gfx942 (no 16-byte LDS DMA)
#  10  rejected by both engines: stride 2
#  11  rejected by both engines: dgrad K = 12 (not a multiple of 8)
from dataclasses import replace

from _emit_common import run_emit
from kernels.common.conv_direct_dgrad_pp_mfma import (
    DgradPpMfmaSpec,
    build_dgrad_pp_mfma,
    dgrad_pp_mfma_problem,
)
from kernels.common.conv_direct_grouped import DirectConvProblem


def _p(*, KH=3, dtype="bf16", stride=1, kpg=96):
    return dgrad_pp_mfma_problem(
        DirectConvProblem(
            N=2,
            H=20,
            W=24,
            groups=1,
            cpg=72,
            kpg=kpg,
            KH=KH,
            KW=KH,
            PAD=(KH - 1) // 2,
            stride=stride,
            dtype=dtype,
        )
    )


def _spec(idx: int):
    S = DgradPpMfmaSpec
    cfgs = {
        0: (S(problem=_p()), "gfx950"),
        1: (S(problem=_p(dtype="fp16"), block_k=256), "gfx950"),
        2: (S(problem=_p(KH=5)), "gfx950"),
        3: (S(problem=_p(KH=5, dtype="fp16")), "gfx950"),
        4: (S(problem=_p(), block_k=256), "gfx950"),
        5: (S(problem=_p(dtype="fp16")), "gfx950"),
        6: (S(problem=_p(), block_k=64), "gfx950"),
        7: (S(problem=_p(KH=5), block_k=256), "gfx950"),
        8: (S(problem=_p(KH=7)), "gfx950"),
        9: (S(problem=_p()), "gfx942"),
        10: (S(problem=replace(_p(), stride=2)), "gfx950"),
        11: (S(problem=_p(kpg=12)), "gfx950"),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    return cfgs[idx]


def main() -> int:
    return run_emit(
        _spec,
        build_dgrad_pp_mfma,
        usage="usage: conv_direct_dgrad_pp_mfma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
