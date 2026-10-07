#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_grouped_emit.py -- Python reference emitter for the
# direct grouped convolution parity harness. Selects one of N sampled spec
# configs by argv[1], builds the DirectConv16cSpec / DirectConv4cSpec /
# DirectConv8cSpec / DirectConv32cSpec / DirectDepthwiseSpec /
# DirectDepthwiseColSpec /
# DirectConvDgradSpec / DirectDepthwiseDgradSpec / DirectConvWgradSpec (and,
# from index 42, the non-grouped DirectNongroupedConvSpec, from index 57 the
# output-stationary DirectDepthwiseTiledSpec, from index 63 the dilated
# non-grouped DirectNongroupedConvSpec and from index 66 specs both engines
# must reject), builds the kernel
# via the matching build_direct_conv_* function (arch=<cfg arch>) and prints
# _native_lower(arch=<cfg arch>) to stdout so it can be byte-compared with
# the C emitter conv_direct_grouped_emit.c.
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
    DirectDepthwiseColSpec,
    DirectDepthwiseTiledSpec,
    DirectConvDgradSpec,
    DirectDepthwiseDgradSpec,
    build_direct_conv_16c,
    build_direct_conv_4c,
    build_direct_conv_8c,
    build_direct_conv_32c,
    build_direct_conv_wgrad,
    build_direct_depthwise,
    build_direct_depthwise_spatial,
    build_direct_depthwise_col,
    build_direct_depthwise_tiled,
    build_direct_conv_dgrad,
    build_direct_depthwise_dgrad,
)
from kernels.common.conv_direct_nongrouped import (
    DirectNongroupedConvSpec,
    build_direct_conv_nongrouped,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


# ---------------------------------------------------------------------------
# Non-grouped (groups == 1) DirectNongroupedConvSpec configs, indices 42..56.
#
# Each pins one branch of build_direct_conv_nongrouped: the four MFMA atoms,
# fp16/bf16, stride 1/2, 1x1 / 3x3 / 5x5 filters, partial W/H/K tiles, staging
# passes that do / do not divide the block evenly (the scratch-tail select),
# single vs double buffered LDS, chiplet swizzle on/off, iglp and waves_per_eu.
# ---------------------------------------------------------------------------
_NONGROUPED_CFG_BASE = 42
# Output-stationary depthwise (DirectDepthwiseTiledSpec) configs follow the
# 15 non-grouped ones.
_TILED_CFG_BASE = 57
# Dilated non-grouped configs follow the 6 tiled ones.
_NONGROUPED_DIL_CFG_BASE = 63
# Specs both engines must reject follow the 3 dilated non-grouped ones.
_REJECT_CFG_BASE = 66


def _nongrouped_p(
    N, H, W, C, K, *, KH=3, KW=3, PAD=1, stride=1, dtype="bf16", dil_h=1, dil_w=1
):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=1,
        cpg=C,
        kpg=K,
        KH=KH,
        KW=KW,
        PAD=PAD,
        stride=stride,
        dtype=dtype,
        dil_h=dil_h,
        dil_w=dil_w,
    )


def _nongrouped_dil_spec(idx: int):
    """Return (spec, arch) for dilated non-grouped config ``idx`` (emitted as
    ``_NONGROUPED_DIL_CFG_BASE + idx``), or None past the last one."""
    if idx == 0:
        # Row dilation 3, PAD 1: a taller halo, tap rows 3 input rows apart.
        return (
            _nongrouped_s(_nongrouped_p(2, 16, 32, 64, 128, dil_h=3)),
            "gfx950",
        )
    if idx == 1:
        # Column dilation 4, PAD 4: a wider halo, tap columns 4 apart.
        return (
            _nongrouped_s(_nongrouped_p(1, 32, 32, 64, 64, PAD=4, dil_w=4)),
            "gfx950",
        )
    if idx == 2:
        # fp16, both axes dilated, stride 2, double-buffered LDS.
        return (
            _nongrouped_s(
                _nongrouped_p(
                    1, 32, 64, 64, 64, PAD=2, stride=2, dtype="fp16", dil_h=2, dil_w=2
                ),
                ck=16,
                double_buffer=True,
            ),
            "gfx950",
        )
    return None


# The geometry every non-grouped config shares unless it overrides it: the
# shape the sweep's best configs take (t8x32x64, ck32, 2x2 waves, 32x32x16,
# iglp0).
_NONGROUPED_BASE = dict(
    tile_h=8, tile_w=32, tile_k=64, ck=32, waves_m=2, waves_n=2, iglp=0
)


def _nongrouped_s(problem, **kw):
    return DirectNongroupedConvSpec(problem=problem, **{**_NONGROUPED_BASE, **kw})


def _reject_spec(idx: int):
    """Return (kind, spec, arch) for must-reject config ``idx`` (emitted as
    ``_REJECT_CFG_BASE + idx``), or None past the last one. Each pins a
    validator branch whose failure mode the engines must share."""
    if idx == 0:
        # The column-streamed tap indexing has no dilation term.
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=128, cpg=1, kpg=1, KH=3, KW=3, PAD=1, dil_h=2
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=4, block_waves=2, dtype="fp16"),
            "gfx950",
        )
    if idx == 1:
        # Tiled: a filter larger than the padded input (Ho = Wo = -26).
        p = DirectConvProblem(
            N=1, H=4, W=4, groups=64, cpg=1, kpg=1, KH=31, KW=31, PAD=0
        )
        return (
            "dw_tiled",
            DirectDepthwiseTiledSpec(problem=p, block_w=8, block_h=2),
            "gfx950",
        )
    if idx == 2:
        # Tiled: H + 2*PAD - KH = -1 at stride 2. Ho floors to 0; a truncating
        # division would make it 1.
        p = DirectConvProblem(
            N=1, H=4, W=16, groups=64, cpg=1, kpg=1, KH=5, KW=5, PAD=0, stride=2
        )
        return (
            "dw_tiled",
            DirectDepthwiseTiledSpec(problem=p, block_w=4, block_h=2),
            "gfx950",
        )
    if idx == 3:
        # Non-grouped: a zero dilation is refused, not read as 1.
        return (
            "nongrouped",
            _nongrouped_s(_nongrouped_p(2, 16, 32, 64, 128, dil_h=0)),
            "gfx950",
        )
    if idx == 4:
        # 16c: any dilation other than 1, zero included, is refused.
        p = DirectConvProblem(N=2, H=16, W=16, groups=16, cpg=16, kpg=16, dil_w=0)
        return ("16c", DirectConv16cSpec(problem=p, block_groups=4), "gfx950")
    return None


def _tiled_spec(idx: int):
    """Return (spec, arch) for depthwise-tiled config ``idx`` (emitted as
    ``_TILED_CFG_BASE + idx``), or None past the last one."""

    def prob(N, H, W, groups, KH, PAD, stride=1, dtype="bf16"):
        return DirectConvProblem(
            N=N,
            H=H,
            W=W,
            groups=groups,
            cpg=1,
            kpg=1,
            KH=KH,
            KW=KH,
            PAD=PAD,
            stride=stride,
            dtype=dtype,
        )

    if idx == 0:
        # 31x31: the large-filter case the row-streaming kernel cannot compile.
        return (
            DirectDepthwiseTiledSpec(
                problem=prob(1, 56, 56, 192, 31, 15), block_w=8, block_h=2
            ),
            "gfx950",
        )
    if idx == 1:
        # fp16 7x7, filter-row loop unrolled.
        return (
            DirectDepthwiseTiledSpec(
                problem=prob(4, 14, 14, 128, 7, 3, dtype="fp16"),
                block_w=14,
                block_h=7,
                unroll_rows=True,
            ),
            "gfx950",
        )
    if idx == 2:
        # Stride 2: two phases with a window each; two waves; gfx942.
        return (
            DirectDepthwiseTiledSpec(
                problem=prob(2, 28, 28, 256, 5, 2, stride=2),
                block_w=8,
                block_h=4,
                block_waves=2,
            ),
            "gfx942",
        )
    if idx == 3:
        # block_h 1: no carried window.
        return (
            DirectDepthwiseTiledSpec(
                problem=prob(2, 16, 16, 64, 3, 1), block_w=16, block_h=1
            ),
            "gfx950",
        )
    if idx == 4:
        # Non-"same" padding: outputs are indexed directly.
        return (
            DirectDepthwiseTiledSpec(
                problem=prob(2, 16, 16, 64, 3, 0, dtype="fp16"), block_w=4, block_h=2
            ),
            "gfx950",
        )
    if idx == 5:
        # Over the register budget: both engines reject.
        return (
            DirectDepthwiseTiledSpec(
                problem=prob(1, 56, 56, 192, 31, 15), block_w=16, block_h=4
            ),
            "gfx950",
        )
    return None


def _nongrouped_spec(idx: int):
    """Return (spec, arch) for non-grouped config ``idx`` (emitted as
    ``_NONGROUPED_CFG_BASE + idx``), or None past the last one."""
    if idx == 0:
        # bf16 baseline: even staging passes, chiplet swizzle, iglp 0.
        return _nongrouped_s(_nongrouped_p(2, 16, 32, 64, 128)), "gfx950"
    if idx == 1:
        # fp16 operand / store path.
        return _nongrouped_s(_nongrouped_p(2, 16, 32, 64, 128, dtype="fp16")), "gfx950"
    if idx == 2:
        # Ping-pong LDS: parity-selected buffers, one barrier per chunk.
        return (
            _nongrouped_s(_nongrouped_p(2, 16, 32, 64, 128), double_buffer=True),
            "gfx950",
        )
    if idx == 3:
        # No chiplet swizzle: plain mod/div grid decode.
        return (
            _nongrouped_s(_nongrouped_p(2, 16, 32, 64, 128), chiplet_swizzle=False),
            "gfx950",
        )
    if idx == 4:
        # waves_per_eu attribute, no iglp_opt.
        return (
            _nongrouped_s(_nongrouped_p(2, 16, 32, 64, 128), iglp=None, waves_per_eu=3),
            "gfx950",
        )
    if idx == 5:
        # Stride 2: input-row sharing across taps, strided staging window.
        return _nongrouped_s(_nongrouped_p(1, 32, 64, 64, 64, stride=2)), "gfx950"
    if idx == 6:
        # 1x1 pointwise filter, PAD 0.
        return (
            _nongrouped_s(_nongrouped_p(1, 16, 32, 64, 64, KH=1, KW=1, PAD=0)),
            "gfx950",
        )
    if idx == 7:
        # 16x16x32 atom on a width that is not a multiple of 32 (partial W).
        return (
            _nongrouped_s(
                _nongrouped_p(2, 20, 40, 64, 64),
                tile_w=48,
                tile_k=32,
                waves_m=1,
                atom="16x16x32",
            ),
            "gfx950",
        )
    if idx == 8:
        # 16x16x16 atom (4-half fragments), ck 16, single-tile width.
        return (
            _nongrouped_s(
                _nongrouped_p(1, 16, 32, 64, 64, dtype="fp16"),
                tile_w=16,
                tile_k=32,
                ck=16,
                waves_m=1,
                atom="16x16x16",
            ),
            "gfx950",
        )
    if idx == 9:
        # 32x32x8 atom on gfx942 (no 32x32x16 there).
        return (
            _nongrouped_s(
                _nongrouped_p(1, 16, 32, 64, 64, dtype="fp16"), atom="32x32x8", ck=16
            ),
            "gfx942",
        )
    if idx == 10:
        # gfx942 bf16 16x16x16 with uneven staging passes.
        return (
            _nongrouped_s(
                _nongrouped_p(1, 16, 48, 32, 64),
                tile_w=48,
                tile_k=32,
                ck=16,
                waves_m=1,
                waves_n=4,
                atom="16x16x16",
            ),
            "gfx942",
        )
    if idx == 11:
        # 5x5 filter, PAD 2, partial K tile, swizzle_wgm 1.
        return (
            _nongrouped_s(
                _nongrouped_p(1, 16, 32, 64, 96, KH=5, KW=5, PAD=2), swizzle_wgm=1
            ),
            "gfx950",
        )
    if idx == 12:
        # Partial H tile + multi column-block tile (tile_w = 2 atoms).
        return (
            _nongrouped_s(
                _nongrouped_p(1, 20, 64, 64, 64), tile_h=16, tile_w=64, waves_n=4
            ),
            "gfx950",
        )
    if idx == 13:
        # A production-sized config: N4 C640 K640 64x64, t16x64x128 w2x4.
        return (
            _nongrouped_s(
                _nongrouped_p(4, 64, 64, 640, 640),
                tile_h=16,
                tile_w=64,
                tile_k=128,
                ck=16,
                waves_m=2,
                waves_n=4,
            ),
            "gfx950",
        )
    if idx == 14:
        # Double buffer + stride 2 + iglp 1 + fp16 + no swizzle together.
        return (
            _nongrouped_s(
                _nongrouped_p(1, 32, 64, 64, 64, stride=2, dtype="fp16"),
                ck=16,
                double_buffer=True,
                iglp=1,
                chiplet_swizzle=False,
            ),
            "gfx950",
        )
    return None


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
        # column-streamed depthwise, stride=1 fp16, default 16-row tile. Exact
        # channel and W tiles, but the guards bound against kernargs (AOT), so
        # they are emitted all the same.
        p = DirectConvProblem(
            N=2, H=8, W=8, groups=128, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=4, block_waves=2, dtype="fp16"),
            "gfx950",
        )
    if idx == 33:
        # col stride=2 bf16 with partial channel and W tiles (groups=70 % 64,
        # Wo=5 % 4) and two-row tiles, so Ho=5 needs a partial last row tile.
        p = DirectConvProblem(
            N=1, H=9, W=9, groups=70, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=2
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(
                problem=p, block_h=2, block_w=4, block_waves=1, dtype="bf16"
            ),
            "gfx950",
        )
    if idx == 34:
        # col stride=3: exercises the (y - r) % stride tap pruning at a stride
        # no other case reaches.
        p = DirectConvProblem(
            N=1, H=16, W=16, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=3
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=4, block_waves=1, dtype="bf16"),
            "gfx950",
        )
    if idx == 35:
        # col with a large filter (31x31): the regime the variant exists for --
        # KW rides the runtime loop so only KH weights are live. A 4-row tile
        # still unrolls (4-1)+31 input rows.
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=64, cpg=1, kpg=1, KH=31, KW=31, PAD=15, stride=1
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(
                problem=p, block_h=4, block_w=4, block_waves=1, dtype="fp16"
            ),
            "gfx950",
        )
    if idx == 36:
        # col 1x1 / PAD=0 degenerate with a non-power-of-two group count
        p = DirectConvProblem(
            N=2, H=6, W=6, groups=3, cpg=1, kpg=1, KH=1, KW=1, PAD=0, stride=1
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=2, block_waves=1, dtype="fp16"),
            "gfx950",
        )
    if idx == 37:
        # col with KH != KW (5x3): separates the unrolled axis from the runtime one
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=128, cpg=1, kpg=1, KH=5, KW=3, PAD=2, stride=1
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=4, block_waves=2, dtype="bf16"),
            "gfx950",
        )
    if idx == 38:
        # col stride=2 with valid padding (PAD=0) and an odd 3-row tile
        p = DirectConvProblem(
            N=1, H=13, W=13, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=0, stride=2
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(
                problem=p, block_h=3, block_w=6, block_waves=1, dtype="fp16"
            ),
            "gfx950",
        )
    if idx == 39:
        # col with block_w=1 and a channel tail (groups=100 % 128)
        p = DirectConvProblem(
            N=1, H=10, W=10, groups=100, cpg=1, kpg=1, KH=3, KW=3, PAD=1, stride=1
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(problem=p, block_w=1, block_waves=2, dtype="bf16"),
            "gfx950",
        )
    if idx == 40:
        # dwcol PAD-overhang: PAD=2 > (KH-1)/2=1 with stride=2 and two-row
        # tiles. n_iters = (block_h-1)*stride + KH is cross-verified by the
        # C/Python byte-identity gate.
        p = DirectConvProblem(
            N=1, H=10, W=10, groups=64, cpg=1, kpg=1, KH=3, KW=3, PAD=2, stride=2
        )
        return (
            "dwcol",
            DirectDepthwiseColSpec(
                problem=p, block_h=2, block_w=4, block_waves=1, dtype="fp16"
            ),
            "gfx950",
        )
    if idx == 41:
        # depthwise 7x7 stride 2 bf16: PAD > 1 shifts the row stream's static
        # slots, and stride 2 gates its flushes on the row parity.
        p = DirectConvProblem(
            N=2,
            H=14,
            W=14,
            groups=128,
            cpg=1,
            kpg=1,
            KH=7,
            KW=7,
            PAD=3,
            stride=2,
            dtype="bf16",
        )
        return (
            "dw",
            DirectDepthwiseSpec(problem=p, block_w=8, block_waves=2),
            "gfx950",
        )
    if idx >= _REJECT_CFG_BASE:
        sel = _reject_spec(idx - _REJECT_CFG_BASE)
        if sel is not None:
            return sel
        raise SystemExit(f"unknown config index {idx}")
    if idx >= _NONGROUPED_DIL_CFG_BASE:
        sel = _nongrouped_dil_spec(idx - _NONGROUPED_DIL_CFG_BASE)
        if sel is not None:
            return ("nongrouped", *sel)
        raise SystemExit(f"unknown config index {idx}")
    if idx >= _TILED_CFG_BASE:
        sel = _tiled_spec(idx - _TILED_CFG_BASE)
        if sel is not None:
            return ("dw_tiled", *sel)
        raise SystemExit(f"unknown config index {idx}")
    if idx >= _NONGROUPED_CFG_BASE:
        sel = _nongrouped_spec(idx - _NONGROUPED_CFG_BASE)
        if sel is not None:
            return ("nongrouped", *sel)
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
    elif kind == "dwcol":
        kernel = build_direct_depthwise_col(spec, arch=arch)
    elif kind == "dgrad":
        kernel = build_direct_conv_dgrad(spec, arch=arch)
    elif kind == "dw_dgrad":
        kernel = build_direct_depthwise_dgrad(spec, arch=arch)
    elif kind == "nongrouped":
        kernel = build_direct_conv_nongrouped(spec, arch=arch)
    elif kind == "dw_tiled":
        kernel = build_direct_depthwise_tiled(spec, arch=arch)
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
