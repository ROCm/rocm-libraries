#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_grouped_emit.py -- Python reference emitter for the
# direct grouped convolution parity harness. Selects one of N sampled spec
# configs by argv[1], builds the DirectConv16cSpec / DirectConv4cSpec /
# DirectConv8cSpec / DirectConv32cSpec / DirectDepthwiseSpec /
# DirectConvDgradSpec / DirectDepthwiseDgradSpec /
# DirectDepthwiseDgradWindowedSpec / DirectConvWgradSpec,
# builds the kernel via the matching build_direct_conv_* function
# (arch=<cfg arch>) and prints _native_lower(arch=<cfg arch>) to stdout so it
# can be byte-compared with the C emitter conv_direct_grouped_emit.c.
import sys

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    DirectConv16cSpec,
    DirectConv4cSpec,
    DirectConv8cSpec,
    DirectConv32cSpec,
    DirectConvWgradSpec,
    DirectDepthwiseSpec,
    DirectDepthwiseSpatialSpec,
    DirectConvDgradSpec,
    DirectDepthwiseDgradSpec,
    DirectDepthwiseDgradWindowedSpec,
    build_direct_conv_16c,
    build_direct_conv_4c,
    build_direct_conv_8c,
    build_direct_conv_32c,
    build_direct_conv_wgrad,
    build_direct_depthwise,
    build_direct_depthwise_spatial,
    build_direct_conv_dgrad,
    build_direct_depthwise_dgrad,
    build_direct_depthwise_dgrad_windowed,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


def _spec(idx: int):
    """Return (kind, spec, arch) for config index `idx`."""
    if idx == 0:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=16, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "16c",
            DirectConv16cSpec(problem=p, block_groups=4, fold_k32=True),
            "gfx950",
        )
    if idx == 1:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=16, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "16c",
            DirectConv16cSpec(problem=p, block_groups=8, fold_k32=True),
            "gfx950",
        )
    if idx == 2:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=64, cpg=4, kpg=4, KH=3, KW=3, PAD=1, stride=1
        )
        return ("4c", DirectConv4cSpec(problem=p, block_q=4, block_groups=16), "gfx950")
    if idx == 3:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=64, cpg=4, kpg=4, KH=3, KW=3, PAD=1, stride=1
        )
        return ("4c", DirectConv4cSpec(problem=p, block_q=8, block_groups=16), "gfx950")
    if idx == 4:
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "16c",
            DirectConv16cSpec(problem=p, block_groups=1, fold_k32=False),
            "gfx942",
        )
    if idx == 5:
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=16, cpg=4, kpg=4, KH=3, KW=3, PAD=1, stride=1
        )
        return ("4c", DirectConv4cSpec(problem=p, block_q=4, block_groups=16), "gfx950")
    if idx == 6:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=16, cpg=8, kpg=8, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "8c",
            DirectConv8cSpec(problem=p, block_q=16, block_groups=8, double_buffer=True),
            "gfx950",
        )
    if idx == 7:
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=8, cpg=32, kpg=32, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "32c",
            DirectConv32cSpec(
                problem=p, block_q=32, block_groups=4, double_buffer=True
            ),
            "gfx950",
        )
    if idx == 8:
        # groups must be divisible by block_ch = block_waves * wave_size (2 * 64 = 128)
        p = DirectConvProblem(
            N=32, H=200, W=200, groups=128, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "depthwise",
            DirectDepthwiseSpec(problem=p, block_w=16, block_waves=2),
            "gfx950",
        )
    if idx == 9:
        # depthwise with stride=2: exercises Ho/Wo output descriptors and
        # stride-aware flush (p_flush_val % stride == 0 guard)
        p = DirectConvProblem(
            N=2, H=14, W=14, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=2
        )
        return (
            "depthwise",
            DirectDepthwiseSpec(problem=p, block_w=8, block_waves=1),
            "gfx950",
        )
    if idx == 10:
        # spatial layout: groups=3 (non-power-of-two, exercises partial wave)
        p = DirectConvProblem(
            N=2, H=14, W=14, groups=3, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "spatial",
            DirectDepthwiseSpatialSpec(problem=p, block_waves=2),
            "gfx950",
        )
    if idx == 11:
        # spatial layout with stride=2: exercises Ho/Wo + spatial thread mapping
        p = DirectConvProblem(
            N=2, H=14, W=14, groups=3, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=2
        )
        return (
            "spatial",
            DirectDepthwiseSpatialSpec(problem=p, block_waves=1),
            "gfx950",
        )
    if idx == 12:
        # dgrad: baseline grouped dgrad stride=1
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dgrad",
            DirectConvDgradSpec(problem=p, block_q=16, block_groups=8),
            "gfx950",
        )
    if idx == 13:
        # dgrad: larger groups / different block_groups
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=8, cpg=32, kpg=32, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dgrad",
            DirectConvDgradSpec(problem=p, block_q=16, block_groups=4),
            "gfx950",
        )
    if idx == 14:
        # dgrad: gfx942 target
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dgrad",
            DirectConvDgradSpec(problem=p, block_q=16, block_groups=8),
            "gfx942",
        )
    if idx == 15:
        # depthwise_dgrad: stride=1
        p = DirectConvProblem(
            N=2, H=14, W=14, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dw_dgrad",
            DirectDepthwiseDgradSpec(problem=p, block_w=8, block_waves=1),
            "gfx950",
        )
    if idx == 16:
        # depthwise_dgrad: stride=2 exercises divisibility checks
        p = DirectConvProblem(
            N=2, H=14, W=14, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=2
        )
        return (
            "dw_dgrad",
            DirectDepthwiseDgradSpec(problem=p, block_w=8, block_waves=1),
            "gfx950",
        )
    if idx == 17:
        # 16c bf16: exercises bf16 I/O, bf16 load/store taps, mfma_f32_16x16x16_bf16
        p = DirectConvProblem(
            N=32,
            H=200,
            W=200,
            groups=16,
            cpg=16,
            kpg=16,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "16c",
            DirectConv16cSpec(problem=p, block_groups=8, fold_k32=False),
            "gfx950",
        )
    if idx == 18:
        # 8c bf16: exercises bf16 I/O, bf16 load/store taps, mfma_f32_16x16x16_bf16
        p = DirectConvProblem(
            N=32,
            H=200,
            W=200,
            groups=16,
            cpg=8,
            kpg=8,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "8c",
            DirectConv8cSpec(problem=p, block_q=16, block_groups=8, double_buffer=True),
            "gfx950",
        )
    if idx == 19:
        # dgrad bf16: exercises bf16 I/O on the scalar-FMA grouped dgrad path
        p = DirectConvProblem(
            N=2,
            H=8,
            W=8,
            groups=8,
            cpg=16,
            kpg=16,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "dgrad",
            DirectConvDgradSpec(problem=p, block_q=16, block_groups=8),
            "gfx950",
        )
    if idx == 20:
        # 16c bf16 with fold_k32=True: pins the non-default fold_k32 path under bf16
        p = DirectConvProblem(
            N=32,
            H=200,
            W=200,
            groups=16,
            cpg=16,
            kpg=16,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "16c",
            DirectConv16cSpec(problem=p, block_groups=8, fold_k32=True),
            "gfx950",
        )
    if idx == 21:
        # 32c bf16: exercises bf16 I/O on the 32c MFMA path
        p = DirectConvProblem(
            N=32,
            H=200,
            W=200,
            groups=32,
            cpg=32,
            kpg=32,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "32c",
            DirectConv32cSpec(problem=p, block_groups=8),
            "gfx950",
        )
    if idx == 22:
        # depthwise forward bf16: exercises bf16 I/O on the scalar-FMA depthwise path
        p = DirectConvProblem(
            N=2,
            H=14,
            W=14,
            groups=64,
            cpg=1,
            kpg=1,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "dw",
            DirectDepthwiseSpec(problem=p, block_w=8, block_waves=1),
            "gfx950",
        )
    if idx == 23:
        # depthwise spatial bf16: exercises bf16 I/O on the small-group spatial path
        p = DirectConvProblem(
            N=2,
            H=14,
            W=14,
            groups=16,
            cpg=1,
            kpg=1,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "spatial",
            DirectDepthwiseSpatialSpec(problem=p, block_waves=1),
            "gfx950",
        )
    if idx == 24:
        # depthwise dgrad bf16: exercises bf16 I/O on the scalar-FMA depthwise dgrad path
        p = DirectConvProblem(
            N=2,
            H=14,
            W=14,
            groups=64,
            cpg=1,
            kpg=1,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "dw_dgrad",
            DirectDepthwiseDgradSpec(problem=p, block_w=8, block_waves=1),
            "gfx950",
        )
    # ---- wgrad (backward weights) ----
    if idx == 25:
        # Defaults: mfma_k=32 (VEC_CH=8, two ds_read_tr per fragment),
        # waves_k=waves_c=waves_q=1, ho_per_block=4.
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return ("wgrad", DirectConvWgradSpec(problem=p), "gfx950")
    if idx == 26:
        # Narrow MFMA: mfma_k=16 (VEC_CH=4, one ds_read_tr per fragment).
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "wgrad",
            DirectConvWgradSpec(problem=p, mfma_k=16, ho_per_block=2),
            "gfx950",
        )
    if idx == 27:
        # Multi-wave: K/C/Q all split, so n_k_tiles = n_c_tiles = 2,
        # STRIP_GROUPS = 2 and the kernel name carries the _wq flag.
        p = DirectConvProblem(
            N=4, H=16, W=16, groups=4, cpg=64, kpg=64, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "wgrad",
            DirectConvWgradSpec(
                problem=p, waves_k=2, waves_c=2, waves_q=2, ho_per_block=3
            ),
            "gfx950",
        )
    if idx == 28:
        # gfx942 has no 16x16x32 f16 atom -> both engines reject (mfma_k=32).
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return ("wgrad", DirectConvWgradSpec(problem=p), "gfx942")
    if idx == 29:
        # gfx942 with mfma_k=16 clears the atom gate and is rejected one check
        # later, on the missing ds_read_tr16_b64 the LDS staging needs.
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        return ("wgrad", DirectConvWgradSpec(problem=p, mfma_k=16), "gfx942")
    if idx == 30:
        # wgrad bf16: bf16 I/O and the bf16 MFMA atom, same LDS transpose
        # staging. Pins that only the atom and the element type move.
        p = DirectConvProblem(
            N=2,
            H=8,
            W=8,
            groups=8,
            cpg=16,
            kpg=16,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return ("wgrad", DirectConvWgradSpec(problem=p), "gfx950")
    if idx == 31:
        # wgrad bf16 at mfma_k=16: the narrow atom under bf16, one ds_read_tr
        # per fragment.
        p = DirectConvProblem(
            N=2,
            H=8,
            W=8,
            groups=8,
            cpg=16,
            kpg=16,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "wgrad",
            DirectConvWgradSpec(problem=p, mfma_k=16, ho_per_block=2),
            "gfx950",
        )
    if idx == 32:
        # windowed depthwise dgrad: 7x7, scalar f32 FMA path
        p = DirectConvProblem(
            N=2,
            H=14,
            W=14,
            groups=64,
            cpg=1,
            kpg=1,
            KH=7,
            KW=7,
            PAD=3,
            stride=1,
            dtype="bf16",
        )
        return (
            "dw_dgrad_win",
            DirectDepthwiseDgradWindowedSpec(
                problem=p,
                block_w=7,
                block_waves=1,
                ch_per_lane=1,
                block_h=0,
                dot2=False,
            ),
            "gfx950",
        )
    if idx == 33:
        # windowed depthwise dgrad: 7x7 with fdot2 tap pairing (odd KW pads a zero weight)
        p = DirectConvProblem(
            N=2,
            H=14,
            W=14,
            groups=64,
            cpg=1,
            kpg=1,
            KH=7,
            KW=7,
            PAD=3,
            stride=1,
            dtype="bf16",
        )
        return (
            "dw_dgrad_win",
            DirectDepthwiseDgradWindowedSpec(
                problem=p,
                block_w=7,
                block_waves=1,
                ch_per_lane=1,
                block_h=0,
                dot2=True,
            ),
            "gfx950",
        )
    if idx == 34:
        # windowed depthwise dgrad: ch_per_lane=2 (dword loads/stores, packed f32 FMA)
        p = DirectConvProblem(
            N=2,
            H=12,
            W=12,
            groups=128,
            cpg=1,
            kpg=1,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="fp16",
        )
        return (
            "dw_dgrad_win",
            DirectDepthwiseDgradWindowedSpec(
                problem=p,
                block_w=6,
                block_waves=2,
                ch_per_lane=2,
                block_h=0,
                dot2=False,
            ),
            "gfx950",
        )
    if idx == 35:
        # windowed depthwise dgrad: ragged H split (block_h=4, H=13) re-reads the halo
        p = DirectConvProblem(
            N=2,
            H=13,
            W=13,
            groups=64,
            cpg=1,
            kpg=1,
            KH=5,
            KW=5,
            PAD=2,
            stride=1,
            dtype="fp16",
        )
        return (
            "dw_dgrad_win",
            DirectDepthwiseDgradWindowedSpec(
                problem=p,
                block_w=5,
                block_waves=1,
                ch_per_lane=1,
                block_h=4,
                dot2=False,
            ),
            "gfx950",
        )
    if idx == 36:
        # windowed depthwise dgrad: even H split + dot2, W=13 not a multiple of block_w
        p = DirectConvProblem(
            N=1,
            H=8,
            W=13,
            groups=40,
            cpg=1,
            kpg=1,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "dw_dgrad_win",
            DirectDepthwiseDgradWindowedSpec(
                problem=p,
                block_w=4,
                block_waves=1,
                ch_per_lane=1,
                block_h=4,
                dot2=True,
            ),
            "gfx950",
        )
    if idx == 37:
        # windowed depthwise dgrad: even KW=4 with dot2 (no zero pad)
        p = DirectConvProblem(
            N=2,
            H=9,
            W=9,
            groups=72,
            cpg=1,
            kpg=1,
            KH=4,
            KW=4,
            PAD=1,
            stride=1,
            dtype="fp16",
        )
        return (
            "dw_dgrad_win",
            DirectDepthwiseDgradWindowedSpec(
                problem=p,
                block_w=8,
                block_waves=1,
                ch_per_lane=1,
                block_h=0,
                dot2=True,
            ),
            "gfx950",
        )
    if idx == 38:
        # windowed depthwise dgrad: dot2 on gfx942 -> both engines reject
        p = DirectConvProblem(
            N=2,
            H=8,
            W=8,
            groups=64,
            cpg=1,
            kpg=1,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "dw_dgrad_win",
            DirectDepthwiseDgradWindowedSpec(
                problem=p,
                block_w=4,
                block_waves=1,
                ch_per_lane=1,
                block_h=0,
                dot2=True,
            ),
            "gfx942",
        )
    if idx == 39:
        # windowed depthwise dgrad: ch_per_lane=4 (dwordx2) bf16
        p = DirectConvProblem(
            N=2,
            H=7,
            W=7,
            groups=128,
            cpg=1,
            kpg=1,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "dw_dgrad_win",
            DirectDepthwiseDgradWindowedSpec(
                problem=p,
                block_w=4,
                block_waves=1,
                ch_per_lane=4,
                block_h=0,
                dot2=False,
            ),
            "gfx950",
        )
    if idx == 40:
        # windowed depthwise dgrad: KW=1 with dot2 (tail pair only, no full pairs)
        p = DirectConvProblem(
            N=2,
            H=9,
            W=11,
            groups=96,
            cpg=1,
            kpg=1,
            KH=3,
            KW=1,
            PAD=1,
            stride=1,
            dtype="fp16",
        )
        return (
            "dw_dgrad_win",
            DirectDepthwiseDgradWindowedSpec(
                problem=p,
                block_w=5,
                block_waves=1,
                ch_per_lane=1,
                block_h=0,
                dot2=True,
            ),
            "gfx950",
        )
    if idx == 41:
        # windowed depthwise dgrad: non-square 3x5 fp16 dot2 with a ragged H split
        p = DirectConvProblem(
            N=1,
            H=13,
            W=10,
            groups=72,
            cpg=1,
            kpg=1,
            KH=3,
            KW=5,
            PAD=2,
            stride=1,
            dtype="fp16",
        )
        return (
            "dw_dgrad_win",
            DirectDepthwiseDgradWindowedSpec(
                problem=p,
                block_w=10,
                block_waves=2,
                ch_per_lane=1,
                block_h=5,
                dot2=True,
            ),
            "gfx950",
        )
    if idx == 42:
        # 4c bf16: bf16 I/O and the mfma_f32_4x4x4_bf16 (`_1k`) atom; odd H/W
        # so the right-edge column mask and the H-edge rows are exercised.
        p = DirectConvProblem(
            N=2,
            H=13,
            W=13,
            groups=32,
            cpg=4,
            kpg=4,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "4c",
            DirectConv4cSpec(problem=p, block_q=4, block_groups=16),
            "gfx950",
        )
    if idx == 43:
        # 4c bf16 on gfx942: the 4x4x4 bf16 `_1k` atom is CDNA2+, so the
        # kernel stays arch-neutral.
        p = DirectConvProblem(
            N=1,
            H=8,
            W=8,
            groups=16,
            cpg=4,
            kpg=4,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "4c",
            DirectConv4cSpec(problem=p, block_q=8, block_groups=16),
            "gfx942",
        )
    if idx == 44:
        # 4c bf16 with two waves per block (block_groups=32) and four q-tiles
        # per wave: the shape the 4c dgrad entry builds for cpg=kpg=4.
        p = DirectConvProblem(
            N=2,
            H=14,
            W=14,
            groups=32,
            cpg=4,
            kpg=4,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        return (
            "4c",
            DirectConv4cSpec(
                problem=p, name="direct_conv_4c_dgrad", block_q=16, block_groups=32
            ),
            "gfx950",
        )
    if idx in (45, 46, 47, 48):
        # 4c fused dgrad weights (the 4c dgrad entry with dgrad_fused_weights):
        # B is the original weight, read flipped / k<->c transposed in the
        # prologue -- 45/46 per-element gathers (46 = the gfx942 path),
        # 47/48 LDS staging + ds_read_b64_tr_b16 (48 = 1x1, partial pass).
        n, h, w, g, kh, pad, dt, arch, bq, bg, lds = {
            45: (2, 13, 13, 32, 3, 1, "bf16", "gfx950", 4, 16, False),
            46: (1, 8, 8, 16, 3, 1, "fp16", "gfx942", 8, 16, False),
            47: (2, 14, 14, 32, 3, 1, "bf16", "gfx950", 16, 32, True),
            48: (1, 7, 9, 64, 1, 0, "fp16", "gfx950", 4, 64, True),
        }[idx]
        p = DirectConvProblem(
            N=n,
            H=h,
            W=w,
            groups=g,
            cpg=4,
            kpg=4,
            KH=kh,
            KW=kh,
            PAD=pad,
            stride=1,
            dtype=dt,
        )
        return (
            "4c",
            DirectConv4cSpec(
                problem=p,
                name="direct_conv_4c_dgrad",
                block_q=bq,
                block_groups=bg,
                dgrad_fused_weights=True,
                dgrad_weights_lds=lds,
            ),
            arch,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: conv_direct_grouped_emit.py <config_index>\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    kind, spec, arch = _spec(idx)
    if kind == "16c":
        kernel = build_direct_conv_16c(spec, arch=arch)
    elif kind == "4c":
        kernel = build_direct_conv_4c(spec, arch=arch)
    elif kind == "8c":
        kernel = build_direct_conv_8c(spec, arch=arch)
    elif kind == "32c":
        kernel = build_direct_conv_32c(spec, arch=arch)
    elif kind == "wgrad":
        kernel = build_direct_conv_wgrad(spec, arch=arch)
    elif kind == "spatial":
        kernel = build_direct_depthwise_spatial(spec, arch=arch)
    elif kind == "dgrad":
        kernel = build_direct_conv_dgrad(spec, arch=arch)
    elif kind == "dw_dgrad":
        kernel = build_direct_depthwise_dgrad(spec, arch=arch)
    elif kind == "dw_dgrad_win":
        kernel = build_direct_depthwise_dgrad_windowed(spec, arch=arch)
    else:
        kernel = build_direct_depthwise(spec, arch=arch)
    if mode == "ll":
        text = _native_lower(kernel, arch=arch)
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    elif mode == "verify":
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    else:
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
