#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_winograd_fwd_emit.py -- Python reference emitter for the
# Winograd F(m x m_w, 3 x 3) forward transform kernels (K1 filter, K2a
# per-pixel max, K2 input, K4 output, and merged launches). Selects one of N
# sampled configs by argv[1], builds the kernel with
# build_winograd_{filter,prescale,input,output}_transform or
# build_winograd_merged and prints the LLVM IR, to be byte-compared with the C
# emitter conv_winograd_fwd_emit.c (same table, same order).
#
# A config row: (problem N, Hi, Wi, C, K, Y, X, pad, dilation, groups), the
# stage ("K1" | "K2a" | "K2" | "K4", or "A,B" for a merged launch), the arch,
# and the WinogradFwdSpec knobs. Index map:
#    0  F4 fp16 K1 [gfx1250]
#    1  F4 fp16 K2 [gfx1250]
#    2  F4 fp16 K4 [gfx1250]
#    3  F2 K1 [gfx1250]
#    4  F2 K2, 2 tiles per thread [gfx1250]
#    5  F2 K4 [gfx1250]
#    6  F(6x4) K1 [gfx1250]
#    7  F(6x4) K2 [gfx1250]
#    8  F(6x4) K4 [gfx1250]
#    9  F(4x2) K1, 2 filter rows [gfx1250]
#   10  F(4x2) K2 [gfx1250]
#   11  F(4x2) K4 [gfx1250]
#   12  F(2x6) K2 [gfx1250]
#   13  F(2x6) K4 [gfx1250]
#   14  F4 bf16 fused prescale K2 [gfx1250]
#   15  F4 bf16 fused prescale K4 [gfx1250]
#   16  F4 bf16 split prescale K2a [gfx1250]
#   17  F4 bf16 split prescale K2 [gfx1250]
#   18  F2 bf16 operands K1 [gfx1250]
#   19  F2 bf16 operands K2 [gfx1250]
#   20  F2 bf16 operands K4 [gfx1250]
#   21  F2 LDS region 4x8, 4 tiles per thread, 16 lanes K2 [gfx1250]
#   22  F4 bf16 LDS region 2x4 + split prescale K2 [gfx1250]
#   23  F4 dilation 2 K1 (vec 4, 2 filter rows) [gfx1250]
#   24  F4 dilation 2 K2 [gfx1250]
#   25  F4 dilation 2 K4 [gfx1250]
#   26  F2 groups 4 K2, 2 tiles per thread, 2 tile rows [gfx1250]
#   27  F2 groups 4 K1 [gfx1250]
#   28  F2 groups 4 K4 [gfx1250]
#   29  F4 regular stores K1 (filter_vec 2) [gfx1250]
#   30  F4 regular stores K2 [gfx1250]
#   31  F4 regular stores K4 [gfx1250]
#   32  F4 odd channels (vec 1) K1 [gfx1250]
#   33  F4 odd channels (vec 1) K2 [gfx1250]
#   34  F4 odd channels (vec 1) K4 [gfx1250]
#   35  F4 bf16 odd channels split prescale K2a [gfx1250]
#   36  merged K2+K1 F4 [gfx1250]
#   37  merged K1+K2 F2 [gfx1250]
#   38  merged K2a+K1 bf16 split prescale [gfx1250]
#   39  merged K2+K1 F2 LDS region [gfx1250]
#   40  F4 bf16 fused prescale over 128 lanes (LDS group max) K2 [gfx1250]
#   41  merged K2+K1 with a 64-thread K2 block (K1 split) [gfx1250]
#   42  F4 K2 3 tiles per thread (partial runs) [gfx1250]
#   43  F6 K2 [gfx1250]
#   44  F6 K4 [gfx1250]
#   45  gfx950: F4 K2 [gfx950]
#   46  gfx950: F2 bf16 fused prescale K2 [gfx950]
#   47  gfx950: merged K2+K1 [gfx950]
#   48  gfx950: F(6x4) bf16 prescale K4 [gfx950]
#   49  rejected: m = 3 [gfx1250]
#   50  rejected: 5x5 filter [gfx1250]
#   51  rejected: merged K2a without a split prescale [gfx1250]
#   52  rejected: tiles_per_thread does not divide the LDS block [gfx1250]
from _emit_common import run_emit
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_winograd_fwd import (
    WinogradFwdSpec,
    WinogradMergedSpec,
    build_winograd_filter_transform,
    build_winograd_input_transform,
    build_winograd_merged,
    build_winograd_output_transform,
    build_winograd_prescale,
)

# fmt: off
CONFIGS = [
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K1', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 0: F4 fp16 K1
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 1: F4 fp16 K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K4', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 2: F4 fp16 K4
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K1', 'gfx1250', {'m': 2, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 3: F2 K1
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 2, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 2, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 4: F2 K2, 2 tiles per thread
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K4', 'gfx1250', {'m': 2, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 5: F2 K4
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K1', 'gfx1250', {'m': 6, 'm_w': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 6: F(6x4) K1
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 6, 'm_w': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 7: F(6x4) K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K4', 'gfx1250', {'m': 6, 'm_w': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 8: F(6x4) K4
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K1', 'gfx1250', {'m': 4, 'm_w': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 2, 'fvec': 0}),  # 9: F(4x2) K1, 2 filter rows
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'm_w': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 10: F(4x2) K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K4', 'gfx1250', {'m': 4, 'm_w': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 11: F(4x2) K4
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 2, 'm_w': 6, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 12: F(2x6) K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K4', 'gfx1250', {'m': 2, 'm_w': 6, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 13: F(2x6) K4
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 14: F4 bf16 fused prescale K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K4', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 15: F4 bf16 fused prescale K4
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2a', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'split', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 16: F4 bf16 split prescale K2a
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'split', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 17: F4 bf16 split prescale K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K1', 'gfx1250', {'m': 2, 'm_w': 0, 'io': 'bf16', 'op': 'bf16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 18: F2 bf16 operands K1
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 2, 'm_w': 0, 'io': 'bf16', 'op': 'bf16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 19: F2 bf16 operands K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K4', 'gfx1250', {'m': 2, 'm_w': 0, 'io': 'bf16', 'op': 'bf16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 20: F2 bf16 operands K4
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 2, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 4, 'pmode': 'fused', 'bh': 4, 'bw': 8, 'wgr': 1, 'nt': 1, 'lanes': 16, 'frows': 1, 'fvec': 0}),  # 21: F2 LDS region 4x8, 4 tiles per thread, 16 lanes K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'fused', 'bh': 2, 'bw': 4, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 22: F4 bf16 LDS region 2x4 + split prescale K2
    ((1, 40, 40, 128, 64, 3, 3, 2, 2, 1), 'K1', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 4, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 2, 'fvec': 0}),  # 23: F4 dilation 2 K1 (vec 4, 2 filter rows)
    ((1, 40, 40, 128, 64, 3, 3, 2, 2, 1), 'K2', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 4, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 24: F4 dilation 2 K2
    ((1, 40, 40, 128, 64, 3, 3, 2, 2, 1), 'K4', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 4, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 25: F4 dilation 2 K4
    ((2, 16, 16, 128, 128, 3, 3, 1, 1, 4), 'K2', 'gfx1250', {'m': 2, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 2, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 2, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 26: F2 groups 4 K2, 2 tiles per thread, 2 tile rows
    ((2, 16, 16, 128, 128, 3, 3, 1, 1, 4), 'K1', 'gfx1250', {'m': 2, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 27: F2 groups 4 K1
    ((2, 16, 16, 128, 128, 3, 3, 1, 1, 4), 'K4', 'gfx1250', {'m': 2, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 28: F2 groups 4 K4
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K1', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 0, 'lanes': 32, 'frows': 1, 'fvec': 2}),  # 29: F4 regular stores K1 (filter_vec 2)
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 0, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 30: F4 regular stores K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K4', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 0, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 31: F4 regular stores K4
    ((1, 13, 11, 33, 35, 3, 3, 1, 1, 1), 'K1', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 32: F4 odd channels (vec 1) K1
    ((1, 13, 11, 33, 35, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 33: F4 odd channels (vec 1) K2
    ((1, 13, 11, 33, 35, 3, 3, 1, 1, 1), 'K4', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 34: F4 odd channels (vec 1) K4
    ((1, 13, 11, 33, 35, 3, 3, 1, 1, 1), 'K2a', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'split', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 35: F4 bf16 odd channels split prescale K2a
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2,K1', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 36: merged K2+K1 F4
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K1,K2', 'gfx1250', {'m': 2, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 37: merged K1+K2 F2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2a,K1', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'split', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 38: merged K2a+K1 bf16 split prescale
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2,K1', 'gfx1250', {'m': 2, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 4, 'pmode': 'fused', 'bh': 4, 'bw': 8, 'wgr': 1, 'nt': 1, 'lanes': 16, 'frows': 1, 'fvec': 0}),  # 39: merged K2+K1 F2 LDS region
    ((1, 8, 8, 512, 64, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'bf16', 'op': 'fp16', 'vec': 4, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 40: F4 bf16 fused prescale over 128 lanes (LDS group max) K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2,K1', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 64, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 41: merged K2+K1 with a 64-thread K2 block (K1 split)
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 3, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 42: F4 K2 3 tiles per thread (partial runs)
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 6, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 43: F6 K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K4', 'gfx1250', {'m': 6, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 44: F6 K4
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx950', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 45: gfx950: F4 K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx950', {'m': 2, 'm_w': 0, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 46: gfx950: F2 bf16 fused prescale K2
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2,K1', 'gfx950', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 47: gfx950: merged K2+K1
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K4', 'gfx950', {'m': 6, 'm_w': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 1, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 48: gfx950: F(6x4) bf16 prescale K4
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 3, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 49: rejected: m = 3
    ((1, 12, 12, 64, 64, 5, 5, 2, 1, 1), 'K1', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 50: rejected: 5x5 filter
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2a,K1', 'gfx1250', {'m': 4, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 1, 'pmode': 'fused', 'bh': 0, 'bw': 0, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 51: rejected: merged K2a without a split prescale
    ((2, 20, 24, 64, 96, 3, 3, 1, 1, 1), 'K2', 'gfx1250', {'m': 2, 'm_w': 0, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'block': 256, 'ps': 0, 'tpt': 4, 'pmode': 'fused', 'bh': 2, 'bw': 3, 'wgr': 1, 'nt': 1, 'lanes': 32, 'frows': 1, 'fvec': 0}),  # 52: rejected: tiles_per_thread does not divide the LDS block
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
        m_w=d["m_w"],
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
        filter_rows=d["frows"],
        filter_vec=d["fvec"],
    )
    return (spec, stage), arch


def _build(sel, *, arch):
    spec, stage = sel
    if "," in stage:
        names = tuple(stage.split(","))
        return build_winograd_merged(WinogradMergedSpec(spec, names), arch=arch)
    return _BUILD[stage](spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: conv_winograd_fwd_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
