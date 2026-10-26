#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_winograd_gemm_wmma_emit.py -- Python reference emitter for
# the Winograd batched-GEMM (K3) parity harness. Selects one of N sampled spec
# configs by argv[1], builds the kernel via build_winograd_gemm_wmma (or the
# split-K reduce, build_winograd_gemm_reduce) for the config's arch and prints
# the LLVM IR to stdout so it can be byte-compared with the C emitter
# conv_winograd_gemm_wmma_emit.c.
#
# Config index map (gfx1250 unless noted):
#   0  fp16, defaults (2x4 waves, 8x4 fragments, tile_k 128, 2 stages, TDM epilogue)
#   1  bf16, defaults
#   2  fp16, tile_k 64, 3 stages
#   3  fp16, 128x128 tile (2x2 waves, 4x4 fragments)
#   4  fp16, tiny: 1x2 waves, 2x2 fragments, tile_k 64
#   5  fp16, tiny: 2x2 waves, 2x2 fragments, tile_k 32, 3 stages
#   6  fp16, direct epilogue
#   7  fp16, split-K (fp32 slab), TDM epilogue
#   8  bf16, split-K, direct epilogue
#   9  fp16, lds_pad clip, no fragment prefetch
#  10  fp16, no XCD remap, no scheduler strategy
#  11  fp16, 2 taps (segmented K3)
#  12  bf16, 3 taps, tile_k 64, 3 stages
#  13  split-K reduce, fp16
#  14  split-K reduce, bf16
#  15  rejected by both engines: gfx950 (wave64, no TDM)
#  16  rejected by both engines: tile_k 48 (not a multiple of 32)
#  17  fp16, one wave (both transfers from wave 0), 4x2 fragments, 3 stages
#  18  bf16, split-K, TDM epilogue, 2 taps, tile_k 64
from _emit_common import run_emit
from kernels.common.conv_winograd_gemm_wmma import (
    WinogradGemmReduceSpec,
    WinogradGemmWmmaSpec,
    build_winograd_gemm_reduce,
    build_winograd_gemm_wmma,
)


def _spec(idx: int):
    S = WinogradGemmWmmaSpec
    tiny = dict(waves_m=1, waves_n=2, frags_m=2, frags_n=2, tile_k=64)
    cfgs = {
        0: S(),
        1: S(dtype="bf16"),
        2: S(tile_k=64, stages=3),
        3: S(waves_m=2, waves_n=2, frags_m=4, frags_n=4),
        4: S(**tiny),
        5: S(waves_m=2, waves_n=2, frags_m=2, frags_n=2, tile_k=32, stages=3),
        6: S(epilogue="direct"),
        7: S(split_k=True),
        8: S(dtype="bf16", split_k=True, epilogue="direct"),
        9: S(lds_pad="clip", prefetch=False),
        10: S(xcd_remap=False, sched=None),
        11: S(taps=2),
        12: S(dtype="bf16", taps=3, tile_k=64, stages=3),
        13: WinogradGemmReduceSpec("fp16"),
        14: WinogradGemmReduceSpec("bf16"),
        15: (S(), "gfx950"),
        16: S(tile_k=48, lds_pad="clip"),
        17: S(waves_m=1, waves_n=1, frags_m=4, frags_n=2, tile_k=64, stages=3),
        18: S(dtype="bf16", split_k=True, taps=2, tile_k=64),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    cfg = cfgs[idx]
    return cfg if isinstance(cfg, tuple) else (cfg, "gfx1250")


def _build(spec, *, arch):
    if isinstance(spec, WinogradGemmReduceSpec):
        return build_winograd_gemm_reduce(spec, arch=arch)
    return build_winograd_gemm_wmma(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: conv_winograd_gemm_wmma_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
