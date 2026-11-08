#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_fwd_small_mfma_emit.py -- Python reference emitter for the
# wave64 MFMA forward of tiny channel counts (fwd_smallc_mfma and
# fwd_smallk_mfma). Selects one of N sampled spec configs by argv[1], builds
# the FwdSmallcMfmaSpec / FwdSmallkMfmaSpec, builds the kernel via
# build_fwd_smallc_mfma / build_fwd_smallk_mfma(spec, arch=<cfg arch>) and
# prints the LLVM IR to stdout so it can be byte-compared with the C emitter
# conv_fwd_small_mfma_emit.c.
#
# Config index map (gfx950 unless noted):
#  small C (fwd_smallc_mfma):
#   0  bf16, 3x3 s1 PAD 1, C=3, defaults (1x4 waves, 4x2 tiles, 4x32 tile)
#   1  fp16, 3x3 s1 PAD 1, C=4, 2x2 waves, 2x2 tiles, 2x32 tile, 4 tiles per
#      block + prefetch
#   2  bf16, 3x3 s2 PAD 1, C=3, 1x4 waves, 1x4 tiles, 8x32 tile, 4 tiles per
#      block
#   3  fp16, 3x3 s2 PAD 0 ("valid" stem), C=3, 1x4 waves, 2x2 tiles, 4x32
#      tile, 8 tiles per block + prefetch
#   4  bf16, 7x7 s2 PAD 3, C=3 (5 MFMA steps), 1x8 waves, 4x2 tiles, 8x32
#      tile, 4 tiles per block + prefetch
#   5  fp16, 4x4 s4 PAD 0, C=3, 2x4 waves, 2x2 tiles, 2x64 tile
#   6  bf16, 3x3 s1 PAD 1, C=3, K=36 (partial channel block), 1x4 waves,
#      2x2 tiles, 1x128 tile, custom name
#   7  fp16, 7x7 s2 PAD 3, C=3, 1x4 waves, 4x1 tiles, 8x8 tile, 8 tiles per
#      block + prefetch
#  small K (fwd_smallk_mfma):
#   8  bf16, 3x3 s1 PAD 1, K=4, 4 waves, 8x8 tile
#   9  fp16, 3x3 s1 PAD 1, K=3 (per-channel stores), 8 waves, 16x32 tile
#  10  bf16, 1x7 s1 PAD 0 / column PAD 3, K=1, 1x300 row, 4 waves, 1x128 tile
#  11  fp16, 1x7 s1 PAD 0 / column PAD 3, K=1, 8 waves, 2x128 tile
#  12  bf16, 3x3 s2 PAD 1, K=8 (one 16-byte store per pixel), C=96 (a partial
#      32-channel step), 4 waves, 8x8 tile
#  13  fp16, 3x3 s1 PAD 1, K=5, 4 waves, 8x16 tile
#  14  bf16, 3x3 s1 PAD 1, K=2, 8 waves, 4x64 tile
#  rejected by both engines:
#  15  fwd_smallc_mfma on gfx1250 (wave32)
#  16  fwd_smallk_mfma with C=36 (not a multiple of 8)
from _emit_common import run_emit
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_fwd_small_mfma import (
    FwdSmallcMfmaSpec,
    FwdSmallkMfmaSpec,
    build_fwd_smallc_mfma,
    build_fwd_smallk_mfma,
)


def _p(*, K=3, PAD=1, S=1, C=3, Kout=64, dtype="bf16", H=32, W=40, KW=None, pad_w=-1):
    return DirectConvProblem(
        N=2,
        H=H,
        W=W,
        groups=1,
        cpg=C,
        kpg=Kout,
        KH=K,
        KW=K if KW is None else KW,
        PAD=PAD,
        pad_w=pad_w,
        stride=S,
        dtype=dtype,
    )


def _spec(idx: int):
    C = FwdSmallcMfmaSpec
    K = FwdSmallkMfmaSpec
    cfgs = {
        0: C(problem=_p(K=3, PAD=1, S=1, C=3, Kout=64, dtype="bf16")),
        1: C(
            problem=_p(K=3, PAD=1, S=1, C=4, Kout=32, dtype="fp16"),
            waves_k=2,
            waves_p=2,
            tiles_k=2,
            tiles_p=2,
            out_h=2,
            out_w=32,
            tiles_per_block=4,
            prefetch=True,
        ),
        2: C(
            problem=_p(K=3, PAD=1, S=2, C=3, Kout=64, dtype="bf16"),
            waves_k=1,
            waves_p=4,
            tiles_k=1,
            tiles_p=4,
            out_h=8,
            out_w=32,
            tiles_per_block=4,
        ),
        3: C(
            problem=_p(K=3, PAD=0, S=2, C=3, Kout=64, dtype="fp16"),
            waves_k=1,
            waves_p=4,
            tiles_k=2,
            tiles_p=2,
            out_h=4,
            out_w=32,
            tiles_per_block=8,
            prefetch=True,
        ),
        4: C(
            problem=_p(K=7, PAD=3, S=2, C=3, Kout=64, dtype="bf16"),
            waves_k=1,
            waves_p=8,
            tiles_k=4,
            tiles_p=2,
            out_h=8,
            out_w=32,
            tiles_per_block=4,
            prefetch=True,
        ),
        5: C(
            problem=_p(K=4, PAD=0, S=4, C=3, Kout=64, dtype="fp16"),
            waves_k=2,
            waves_p=4,
            tiles_k=2,
            tiles_p=2,
            out_h=2,
            out_w=64,
        ),
        6: C(
            problem=_p(K=3, PAD=1, S=1, C=3, Kout=36, dtype="bf16"),
            name="rocke_fwd_smallc_mfma",
            waves_k=1,
            waves_p=4,
            tiles_k=2,
            tiles_p=2,
            out_h=1,
            out_w=128,
        ),
        7: C(
            problem=_p(K=7, PAD=3, S=2, C=3, Kout=64, dtype="fp16"),
            waves_k=1,
            waves_p=4,
            tiles_k=4,
            tiles_p=1,
            out_h=8,
            out_w=8,
            tiles_per_block=8,
            prefetch=True,
        ),
        8: K(
            problem=_p(K=3, PAD=1, S=1, C=64, Kout=4, dtype="bf16"),
            waves=4,
            out_h=8,
            out_w=8,
        ),
        9: K(
            problem=_p(K=3, PAD=1, S=1, C=64, Kout=3, dtype="fp16"),
            waves=8,
            out_h=16,
            out_w=32,
        ),
        10: K(
            problem=_p(
                K=1, KW=7, PAD=0, pad_w=3, S=1, C=32, Kout=1, dtype="bf16", H=1, W=300
            ),
            waves=4,
            out_h=1,
            out_w=128,
        ),
        11: K(
            problem=_p(
                K=1, KW=7, PAD=0, pad_w=3, S=1, C=32, Kout=1, dtype="fp16", H=1, W=300
            ),
            waves=8,
            out_h=2,
            out_w=128,
        ),
        12: K(
            problem=_p(K=3, PAD=1, S=2, C=96, Kout=8, dtype="bf16"),
            waves=4,
            out_h=8,
            out_w=8,
        ),
        13: K(
            problem=_p(K=3, PAD=1, S=1, C=64, Kout=5, dtype="fp16"),
            waves=4,
            out_h=8,
            out_w=16,
        ),
        14: K(
            problem=_p(K=3, PAD=1, S=1, C=128, Kout=2, dtype="bf16"),
            waves=8,
            out_h=4,
            out_w=64,
        ),
        15: C(problem=_p(K=3, PAD=1, S=1, C=3, Kout=64, dtype="bf16")),
        16: K(problem=_p(K=3, PAD=1, S=1, C=36, Kout=4, dtype="bf16")),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    return cfgs[idx], ("gfx1250" if idx == 15 else "gfx950")


def _build(spec, arch):
    if isinstance(spec, FwdSmallcMfmaSpec):
        return build_fwd_smallc_mfma(spec, arch=arch)
    return build_fwd_smallk_mfma(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: conv_fwd_small_mfma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
