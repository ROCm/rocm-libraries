#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_winograd_fwd_rt_emit.py -- Python reference emitter for the
# shape-generic Winograd F(mh x mw, 3 x 3) forward transform kernels (K1
# filter, K2a per-pixel max, K2 input, K4 output) of conv_winograd_fwd_rt.
# Selects one of N sampled configs by argv[1], builds the stage's spec and
# kernel (build_stage) and prints the LLVM IR, to be byte-compared with the C
# emitter conv_winograd_fwd_rt_emit.c (same table, same order).
#
# A config row: the stage ("K1" | "K2a" | "K2" | "K4"), the arch, and the
# spec knobs (a stage reads the ones its spec class has). Index map:
#    0  F4 fp16 K1 (2 filter rows) [gfx1250]
#    1  F2 bf16 operands K1 (all rows) [gfx1250]
#    2  F(6x4) K1 vec 4 [gfx1250]
#    3  F(2x6) K1 vec 1, 1 filter row [gfx1250]
#    4  F(4x2) bf16 I/O K1 vec 2 [gfx1250]
#    5  K2a fp16 vec 8 [gfx1250]
#    6  K2a bf16 vec 1 [gfx1250]
#    7  K2a bf16 vec 4 [gfx1250]
#    8  F4 K2 register, 1 tile [gfx1250]
#    9  F2 K2 register, 2 tiles per thread [gfx1250]
#   10  F(6x4) K2 register vec 4 [gfx1250]
#   11  F(2x4) K2 register 3 tiles per thread (partial runs) [gfx1250]
#   12  F4 K2 regular stores, block 64 [gfx1250]
#   13  F4 bf16 K2 split prescale [gfx1250]
#   14  F(4x6) bf16 K2 split prescale 2 tiles [gfx1250]
#   15  F4 bf16 K2 fused prescale block 256 (LDS group max) [gfx1250]
#   16  F(6x4) bf16 K2 fused prescale block 1024 [gfx1250]
#   17  F4 bf16 K2 fused prescale block 32 (shuffles only) [gfx1250]
#   18  F2 bf16 operands K2 register [gfx1250]
#   19  F2 K2 LDS 4x8, 4 tiles per thread, 16 lanes [gfx1250]
#   20  F2 bf16 K2 LDS 4x8 vec 1 [gfx1250]
#   21  F4 bf16 K2 LDS 2x4 + split prescale, 8 lanes [gfx1250]
#   22  F(4x2) K2 LDS 4x1 (one tile column), 1 lane [gfx1250]
#   23  F4 K4 [gfx1250]
#   24  F2 K4 vec 1 block 64 [gfx1250]
#   25  F(6x4) bf16 K4 prescale [gfx1250]
#   26  F(2x6) K4 regular stores [gfx1250]
#   27  F6 K4 [gfx1250]
#   28  gfx950: F4 K2 register [gfx950]
#   29  gfx950: F4 bf16 K2 fused prescale [gfx950]
#   30  gfx950: F2 K2 LDS [gfx950]
#   31  gfx950: F4 K1 [gfx950]
#   32  gfx950: F4 bf16 K4 prescale [gfx950]
#   33  gfx950: K2a [gfx950]
#   34  rejected: tile 3 [gfx1250]
#   35  rejected: bf16 operands with F4 [gfx1250]
#   36  rejected: vec 3 [gfx1250]
#   37  rejected: fused without prescale [gfx1250]
#   38  rejected: fused on the LDS path [gfx1250]
#   39  rejected: tiles_per_thread does not divide the LDS block [gfx1250]
#   40  rejected: LDS region over 1024 threads [gfx1250]
#   41  rejected: filter_rows does not divide n [gfx1250]
#   42  rejected: block not a power of two [gfx1250]
#   43  rejected: K2a vec 16 [gfx1250]
#   44  rejected: filter_rows -1 [gfx1250]
#   45  rejected: K2 block over 1024 [gfx1250]
#   46  rejected: K4 block over 1024 [gfx1250]
#   47  rejected: LDS lanes not a power of two [gfx1250]
#   48  rejected: LDS lanes 0 [gfx1250]
#   49  rejected: LDS region of zero width [gfx1250]
from _emit_common import run_emit
from kernels.common.conv_winograd_fwd_rt import (
    RtFilterSpec,
    RtInputSpec,
    RtOutputSpec,
    RtPrescaleSpec,
    build_stage,
)

# fmt: off
CONFIGS = [
    ('K1', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 2, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 0: F4 fp16 K1 (2 filter rows)
    ('K1', 'gfx1250', {'mh': 2, 'mw': 2, 'io': 'bf16', 'op': 'bf16', 'vec': 8, 'fr': 4, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 1: F2 bf16 operands K1 (all rows)
    ('K1', 'gfx1250', {'mh': 6, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 4, 'fr': 2, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 2: F(6x4) K1 vec 4
    ('K1', 'gfx1250', {'mh': 2, 'mw': 6, 'io': 'fp16', 'op': 'fp16', 'vec': 1, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 3: F(2x6) K1 vec 1, 1 filter row
    ('K1', 'gfx1250', {'mh': 4, 'mw': 2, 'io': 'bf16', 'op': 'fp16', 'vec': 2, 'fr': 3, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 4: F(4x2) bf16 I/O K1 vec 2
    ('K2a', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 5: K2a fp16 vec 8
    ('K2a', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 1, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 6: K2a bf16 vec 1
    ('K2a', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 4, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 7: K2a bf16 vec 4
    ('K2', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 4, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 8: F4 K2 register, 1 tile
    ('K2', 'gfx1250', {'mh': 2, 'mw': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 2, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 9: F2 K2 register, 2 tiles per thread
    ('K2', 'gfx1250', {'mh': 6, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 4, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 10: F(6x4) K2 register vec 4
    ('K2', 'gfx1250', {'mh': 2, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 2, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 3, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 11: F(2x4) K2 register 3 tiles per thread (partial runs)
    ('K2', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 64, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 0, 'fused': 0}),  # 12: F4 K2 regular stores, block 64
    ('K2', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 4, 'fr': 1, 'block': 256, 'ps': 1, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 13: F4 bf16 K2 split prescale
    ('K2', 'gfx1250', {'mh': 4, 'mw': 6, 'io': 'bf16', 'op': 'fp16', 'vec': 2, 'fr': 1, 'block': 256, 'ps': 1, 'tpt': 2, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 14: F(4x6) bf16 K2 split prescale 2 tiles
    ('K2', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 4, 'fr': 1, 'block': 256, 'ps': 1, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 1}),  # 15: F4 bf16 K2 fused prescale block 256 (LDS group max)
    ('K2', 'gfx1250', {'mh': 6, 'mw': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 2, 'fr': 1, 'block': 1024, 'ps': 1, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 1}),  # 16: F(6x4) bf16 K2 fused prescale block 1024
    ('K2', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 1, 'fr': 1, 'block': 32, 'ps': 1, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 1}),  # 17: F4 bf16 K2 fused prescale block 32 (shuffles only)
    ('K2', 'gfx1250', {'mh': 2, 'mw': 2, 'io': 'bf16', 'op': 'bf16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 18: F2 bf16 operands K2 register
    ('K2', 'gfx1250', {'mh': 2, 'mw': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 4, 'bh': 4, 'bw': 8, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 19: F2 K2 LDS 4x8, 4 tiles per thread, 16 lanes
    ('K2', 'gfx1250', {'mh': 2, 'mw': 2, 'io': 'bf16', 'op': 'bf16', 'vec': 1, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 4, 'bh': 4, 'bw': 8, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 20: F2 bf16 K2 LDS 4x8 vec 1
    ('K2', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 4, 'fr': 1, 'block': 256, 'ps': 1, 'tpt': 1, 'bh': 2, 'bw': 4, 'lanes': 8, 'nt': 1, 'fused': 0}),  # 21: F4 bf16 K2 LDS 2x4 + split prescale, 8 lanes
    ('K2', 'gfx1250', {'mh': 4, 'mw': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 2, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 4, 'bw': 1, 'lanes': 1, 'nt': 1, 'fused': 0}),  # 22: F(4x2) K2 LDS 4x1 (one tile column), 1 lane
    ('K4', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 23: F4 K4
    ('K4', 'gfx1250', {'mh': 2, 'mw': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 1, 'fr': 1, 'block': 64, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 24: F2 K4 vec 1 block 64
    ('K4', 'gfx1250', {'mh': 6, 'mw': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 4, 'fr': 1, 'block': 256, 'ps': 1, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 25: F(6x4) bf16 K4 prescale
    ('K4', 'gfx1250', {'mh': 2, 'mw': 6, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 0, 'fused': 0}),  # 26: F(2x6) K4 regular stores
    ('K4', 'gfx1250', {'mh': 6, 'mw': 6, 'io': 'fp16', 'op': 'fp16', 'vec': 2, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 27: F6 K4
    ('K2', 'gfx950', {'mh': 4, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 28: gfx950: F4 K2 register
    ('K2', 'gfx950', {'mh': 4, 'mw': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 1, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 1}),  # 29: gfx950: F4 bf16 K2 fused prescale
    ('K2', 'gfx950', {'mh': 2, 'mw': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 4, 'bh': 4, 'bw': 8, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 30: gfx950: F2 K2 LDS
    ('K1', 'gfx950', {'mh': 4, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 2, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 31: gfx950: F4 K1
    ('K4', 'gfx950', {'mh': 4, 'mw': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 1, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 32: gfx950: F4 bf16 K4 prescale
    ('K2a', 'gfx950', {'mh': 4, 'mw': 4, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 33: gfx950: K2a
    ('K2', 'gfx1250', {'mh': 3, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 34: rejected: tile 3
    ('K1', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'bf16', 'op': 'bf16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 35: rejected: bf16 operands with F4
    ('K4', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 3, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 36: rejected: vec 3
    ('K2', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 1}),  # 37: rejected: fused without prescale
    ('K2', 'gfx1250', {'mh': 2, 'mw': 2, 'io': 'bf16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 1, 'tpt': 4, 'bh': 4, 'bw': 8, 'lanes': 16, 'nt': 1, 'fused': 1}),  # 38: rejected: fused on the LDS path
    ('K2', 'gfx1250', {'mh': 2, 'mw': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 4, 'bh': 2, 'bw': 3, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 39: rejected: tiles_per_thread does not divide the LDS block
    ('K2', 'gfx1250', {'mh': 2, 'mw': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 8, 'bw': 8, 'lanes': 32, 'nt': 1, 'fused': 0}),  # 40: rejected: LDS region over 1024 threads
    ('K1', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 4, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 41: rejected: filter_rows does not divide n
    ('K4', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 96, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 42: rejected: block not a power of two
    ('K2a', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 16, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 43: rejected: K2a vec 16
    ('K1', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': -1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 44: rejected: filter_rows -1
    ('K2', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 2048, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 45: rejected: K2 block over 1024
    ('K4', 'gfx1250', {'mh': 4, 'mw': 4, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 2048, 'ps': 0, 'tpt': 1, 'bh': 0, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 46: rejected: K4 block over 1024
    ('K2', 'gfx1250', {'mh': 2, 'mw': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 4, 'bh': 4, 'bw': 8, 'lanes': 3, 'nt': 1, 'fused': 0}),  # 47: rejected: LDS lanes not a power of two
    ('K2', 'gfx1250', {'mh': 2, 'mw': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 4, 'bh': 4, 'bw': 8, 'lanes': 0, 'nt': 1, 'fused': 0}),  # 48: rejected: LDS lanes 0
    ('K2', 'gfx1250', {'mh': 2, 'mw': 2, 'io': 'fp16', 'op': 'fp16', 'vec': 8, 'fr': 1, 'block': 256, 'ps': 0, 'tpt': 1, 'bh': 4, 'bw': 0, 'lanes': 16, 'nt': 1, 'fused': 0}),  # 49: rejected: LDS region of zero width
]
# fmt: on


def _spec(idx: int):
    if not 0 <= idx < len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    stage, arch, d = CONFIGS[idx]
    if stage == "K1":
        spec = RtFilterSpec(
            mh=d["mh"], mw=d["mw"], io_dtype=d["io"], operand_dtype=d["op"],
            vec=d["vec"], filter_rows=d["fr"],
        )  # fmt: skip
    elif stage == "K2a":
        spec = RtPrescaleSpec(io_dtype=d["io"], vec=d["vec"])
    elif stage == "K2":
        spec = RtInputSpec(
            mh=d["mh"], mw=d["mw"], io_dtype=d["io"], operand_dtype=d["op"],
            vec=d["vec"], block_size=d["block"], prescale=bool(d["ps"]),
            tiles_per_thread=d["tpt"], lds=(d["bh"], d["bw"]),
            lds_lanes=d["lanes"], nt_stores=bool(d["nt"]), fused=bool(d["fused"]),
        )  # fmt: skip
    else:
        spec = RtOutputSpec(
            mh=d["mh"], mw=d["mw"], io_dtype=d["io"], operand_dtype=d["op"],
            vec=d["vec"], block_size=d["block"], prescale=bool(d["ps"]),
            nt_stores=bool(d["nt"]),
        )  # fmt: skip
    return spec, arch


def main() -> int:
    return run_emit(
        _spec,
        build_stage,
        usage="usage: conv_winograd_fwd_rt_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
