#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_winograd_fwd_emit.py -- Python reference emitter for the
# Winograd F(m x m, 3 x 3) forward transform kernels (K1 filter, K2a per-pixel
# max, K2 input, K4 output). Selects one of N sampled configs by argv[1],
# builds the kernel with build_winograd_{filter,prescale,input,output}_transform
# and prints the LLVM IR, to be byte-compared with the C emitter
# conv_winograd_fwd_emit.c (same table, same order).
#
# A config row: (problem N, Hi, Wi, C, K, Y, X, pad, dilation, groups), the
# stage ("K1" | "K2a" | "K2" | "K4"), the arch, and the WinogradFwdSpec knobs.
# Index map:
#    0  F4 fp16 K1 [gfx1250]
#    1  F4 fp16 K2 [gfx1250]
#    2  F4 fp16 K4 [gfx1250]
#    3  F2 K1 [gfx1250]
#    4  F2 K2, 2 tiles per thread [gfx1250]
#    5  F2 K4 [gfx1250]
#    6  F4 bf16 fused prescale K2 [gfx1250]
#    7  F4 bf16 fused prescale K4 [gfx1250]
#    8  F4 bf16 split prescale K2a [gfx1250]
#    9  F4 bf16 split prescale K2 [gfx1250]
#   10  F2 bf16 operands K1 [gfx1250]
#   11  F2 bf16 operands K2 [gfx1250]
#   12  F2 bf16 operands K4 [gfx1250]
#   13  F2 LDS region 4x8, 4 tiles per thread, 16 lanes K2 [gfx1250]
#   14  F4 bf16 LDS region 2x4 + split prescale K2 [gfx1250]
#   15  F4 dilation 2 K1 (vec 4) [gfx1250]
#   16  F4 dilation 2 K2 [gfx1250]
#   17  F4 dilation 2 K4 [gfx1250]
#   18  F2 groups 4 K2, 2 tiles per thread, 2 tile rows [gfx1250]
#   19  F2 groups 4 K1 [gfx1250]
#   20  F2 groups 4 K4 [gfx1250]
#   21  F4 regular stores K1 [gfx1250]
#   22  F4 regular stores K2 [gfx1250]
#   23  F4 regular stores K4 [gfx1250]
#   24  F4 odd channels (vec 1) K1 [gfx1250]
#   25  F4 odd channels (vec 1) K2 [gfx1250]
#   26  F4 odd channels (vec 1) K4 [gfx1250]
#   27  F4 bf16 odd channels split prescale K2a [gfx1250]
#   28  F4 bf16 fused prescale over 128 lanes (LDS group max) K2 [gfx1250]
#   29  F4 K2 3 tiles per thread (partial runs) [gfx1250]
#   30  gfx950: F4 K2 [gfx950]
#   31  gfx950: F2 bf16 fused prescale K2 [gfx950]
#   32  rejected: m = 3 [gfx1250]
#   33  rejected: 5x5 filter [gfx1250]
#   34  rejected: tiles_per_thread does not divide the LDS block [gfx1250]
from _emit_common import run_emit
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_winograd_fwd import (
    WinogradFwdSpec,
    build_winograd_filter_transform,
    build_winograd_input_transform,
    build_winograd_output_transform,
    build_winograd_prescale,
)

# fmt: off
CONFIGS = [
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K1', 'gfx1250', {'m': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 0: F4 fp16 K1
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 1: F4 fp16 K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K4', 'gfx1250', {'m': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 2: F4 fp16 K4
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K1', 'gfx1250', {'m': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 3: F2 K1
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 2, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 4: F2 K2, 2 tiles per thread
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K4', 'gfx1250', {'m': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 5: F2 K4
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 6: F4 bf16 fused prescale K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K4', 'gfx1250', {'m': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 7: F4 bf16 fused prescale K4
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2a', 'gfx1250', {'m': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'split', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 8: F4 bf16 split prescale K2a
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'split', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 9: F4 bf16 split prescale K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K1', 'gfx1250', {'m': 2, 'io': 'bf16', 'op': 'bf16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 10: F2 bf16 operands K1
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 2, 'io': 'bf16', 'op': 'bf16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 11: F2 bf16 operands K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K4', 'gfx1250', {'m': 2, 'io': 'bf16', 'op': 'bf16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 12: F2 bf16 operands K4
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 4, 'pmode': 'fused', 'bh': 4, 'bw': 8, 'wgr': 1, 'nt': 1, 'lanes': 16}),  # 13: F2 LDS region 4x8, 4 tiles per thread, 16 lanes K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'fused', 'bh': 2, 'bw': 4, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 14: F4 bf16 LDS region 2x4 + split prescale K2
    ((1, 40, 40, 128, 64, 3, 3, 2, 2, 1), 'K1', 'gfx1250', {'m': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 4, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 15: F4 dilation 2 K1 (vec 4)
    ((1, 40, 40, 128, 64, 3, 3, 2, 2, 1), 'K2', 'gfx1250', {'m': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 4, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 16: F4 dilation 2 K2
    ((1, 40, 40, 128, 64, 3, 3, 2, 2, 1), 'K4', 'gfx1250', {'m': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 4, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 17: F4 dilation 2 K4
    ((2, 16, 16, 128, 128, 3, 3, 1, 1, 4), 'K2', 'gfx1250', {'m': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 2, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 2, 'nt': 1, 'lanes': 32}),  # 18: F2 groups 4 K2, 2 tiles per thread, 2 tile rows
    ((2, 16, 16, 128, 128, 3, 3, 1, 1, 4), 'K1', 'gfx1250', {'m': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 19: F2 groups 4 K1
    ((2, 16, 16, 128, 128, 3, 3, 1, 1, 4), 'K4', 'gfx1250', {'m': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 20: F2 groups 4 K4
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K1', 'gfx1250', {'m': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 0, 'lanes': 32}),  # 21: F4 regular stores K1
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 0, 'lanes': 32}),  # 22: F4 regular stores K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K4', 'gfx1250', {'m': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 0, 'lanes': 32}),  # 23: F4 regular stores K4
    ((1, 13, 11, 33, 35, 3, 3, 1, 1, 1), 'K1', 'gfx1250', {'m': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 24: F4 odd channels (vec 1) K1
    ((1, 13, 11, 33, 35, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 25: F4 odd channels (vec 1) K2
    ((1, 13, 11, 33, 35, 3, 3, 1, 1, 1), 'K4', 'gfx1250', {'m': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 26: F4 odd channels (vec 1) K4
    ((1, 13, 11, 33, 35, 3, 3, 1, 1, 1), 'K2a', 'gfx1250', {'m': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'split', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 27: F4 bf16 odd channels split prescale K2a
    ((1, 8, 8, 512, 64, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 4, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 28: F4 bf16 fused prescale over 128 lanes (LDS group max) K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 3, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 29: F4 K2 3 tiles per thread (partial runs)
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx950', {'m': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 30: gfx950: F4 K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx950', {'m': 2, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 31: gfx950: F2 bf16 fused prescale K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 3, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 32: rejected: m = 3
    ((1, 12, 12, 64, 64, 5, 5, 2, 1, 1), 'K1', 'gfx1250', {'m': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 33: rejected: 5x5 filter
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 4, 'pmode': 'fused', 'bh': 2, 'bw': 3, 'wgr': 1, 'nt': 1, 'lanes': 32}),  # 34: rejected: tiles_per_thread does not divide the LDS block
]
# fmt: on

_BUILD = {
    "K1": build_winograd_filter_transform,
    "K2a": build_winograd_prescale,
    "K2": build_winograd_input_transform,
    "K4": build_winograd_output_transform,
}


def _spec(idx: int):
    if not 0 <= idx < len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    prob, stage, arch, d = CONFIGS[idx]
    n, hi, wi, c, k, y, x, pad, dil, groups = prob
    problem = ConvProblem(
        N=n, Hi=hi, Wi=wi, C=c, K=k, Y=y, X=x,
        pH=pad, pW=pad, dH=dil, dW=dil, groups=groups,
    )  # fmt: skip
    spec = WinogradFwdSpec(
        problem=problem,
        m=d["m"],
        io_dtype=d["io"],
        operand_dtype=d["op"],
        vec=d["vec"],
        block_size=d["block"],
        prescale=bool(d["ps"]),
        tiles_per_thread=d["tpt"],
        prescale_mode=d["pmode"],
        input_block_tiles=(d["bh"], d["bw"]),
        wg_tile_rows=d["wgr"],
        nt_stores=bool(d["nt"]),
        lds_lanes=d["lanes"],
    )
    return (spec, stage), arch


def _build(sel, *, arch):
    spec, stage = sel
    return _BUILD[stage](spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: conv_winograd_fwd_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
