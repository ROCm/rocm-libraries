#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_direct_grouped_emit.py -- Python reference emitter for the
# direct grouped convolution parity harness. Selects one of N sampled spec
# configs by argv[1], builds the DirectConv16cSpec / DirectConv4cSpec /
# DirectConv8cSpec / DirectConv32cSpec / DirectDepthwiseSpec /
# DirectConvDgradSpec / DirectDepthwiseDgradSpec (and, from index 25, the
# non-grouped DirectNhwcConvSpec), builds the kernel via the
# matching build_direct_conv_* function (arch=<cfg arch>) and prints
# _native_lower(arch=<cfg arch>) to stdout so it can be byte-compared with
# the C emitter conv_direct_grouped_emit.c.
import sys

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    DirectConv16cSpec,
    DirectConv4cSpec,
    DirectConv8cSpec,
    DirectConv32cSpec,
    DirectDepthwiseSpec,
    DirectDepthwiseSpatialSpec,
    DirectConvDgradSpec,
    DirectDepthwiseDgradSpec,
    build_direct_conv_16c,
    build_direct_conv_4c,
    build_direct_conv_8c,
    build_direct_conv_32c,
    build_direct_depthwise,
    build_direct_depthwise_spatial,
    build_direct_conv_dgrad,
    build_direct_depthwise_dgrad,
)
from kernels.common.conv_direct_nhwc import DirectNhwcConvSpec, build_direct_conv_nhwc

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


# ---------------------------------------------------------------------------
# Non-grouped (groups == 1) DirectNhwcConvSpec configs, indices 25..39.
#
# Each pins one branch of build_direct_conv_nhwc: the four MFMA atoms,
# fp16/bf16, stride 1/2, 1x1 / 3x3 / 5x5 filters, partial W/H/K tiles, staging
# passes that do / do not divide the block evenly (the scratch-tail select),
# single vs double buffered LDS, chiplet swizzle on/off, iglp and waves_per_eu.
# ---------------------------------------------------------------------------
_NHWC_CFG_BASE = 25


def _nhwc_p(N, H, W, C, K, *, KH=3, KW=3, PAD=1, stride=1, dtype="bf16"):
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
    )


# The geometry every non-grouped config shares unless it overrides it: the
# shape the sweep's best configs take (t8x32x64, ck32, 2x2 waves, 32x32x16,
# iglp0).
_NHWC_BASE = dict(tile_h=8, tile_w=32, tile_k=64, ck=32, waves_m=2, waves_n=2, iglp=0)


def _nhwc_s(problem, **kw):
    return DirectNhwcConvSpec(problem=problem, **{**_NHWC_BASE, **kw})


def _nhwc_spec(idx: int):
    """Return (spec, arch) for non-grouped config ``idx`` (emitted as
    ``_NHWC_CFG_BASE + idx``), or None past the last one."""
    if idx == 0:
        # bf16 baseline: even staging passes, chiplet swizzle, iglp 0.
        return _nhwc_s(_nhwc_p(2, 16, 32, 64, 128)), "gfx950"
    if idx == 1:
        # fp16 operand / store path.
        return _nhwc_s(_nhwc_p(2, 16, 32, 64, 128, dtype="fp16")), "gfx950"
    if idx == 2:
        # Ping-pong LDS: parity-selected buffers, one barrier per chunk.
        return _nhwc_s(_nhwc_p(2, 16, 32, 64, 128), double_buffer=True), "gfx950"
    if idx == 3:
        # No chiplet swizzle: plain mod/div grid decode.
        return _nhwc_s(_nhwc_p(2, 16, 32, 64, 128), chiplet_swizzle=False), "gfx950"
    if idx == 4:
        # waves_per_eu attribute, no iglp_opt.
        return (
            _nhwc_s(_nhwc_p(2, 16, 32, 64, 128), iglp=None, waves_per_eu=3),
            "gfx950",
        )
    if idx == 5:
        # Stride 2: input-row sharing across taps, strided staging window.
        return _nhwc_s(_nhwc_p(1, 32, 64, 64, 64, stride=2)), "gfx950"
    if idx == 6:
        # 1x1 pointwise filter, PAD 0.
        return _nhwc_s(_nhwc_p(1, 16, 32, 64, 64, KH=1, KW=1, PAD=0)), "gfx950"
    if idx == 7:
        # 16x16x32 atom on a width that is not a multiple of 32 (partial W).
        return (
            _nhwc_s(
                _nhwc_p(2, 20, 40, 64, 64),
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
            _nhwc_s(
                _nhwc_p(1, 16, 32, 64, 64, dtype="fp16"),
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
            _nhwc_s(_nhwc_p(1, 16, 32, 64, 64, dtype="fp16"), atom="32x32x8", ck=16),
            "gfx942",
        )
    if idx == 10:
        # gfx942 bf16 16x16x16 with uneven staging passes.
        return (
            _nhwc_s(
                _nhwc_p(1, 16, 48, 32, 64),
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
            _nhwc_s(_nhwc_p(1, 16, 32, 64, 96, KH=5, KW=5, PAD=2), swizzle_wgm=1),
            "gfx950",
        )
    if idx == 12:
        # Partial H tile + multi column-block tile (tile_w = 2 atoms).
        return (
            _nhwc_s(_nhwc_p(1, 20, 64, 64, 64), tile_h=16, tile_w=64, waves_n=4),
            "gfx950",
        )
    if idx == 13:
        # A production-sized config: N4 C640 K640 64x64, t16x64x128 w2x4.
        return (
            _nhwc_s(
                _nhwc_p(4, 64, 64, 640, 640),
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
            _nhwc_s(
                _nhwc_p(1, 32, 64, 64, 64, stride=2, dtype="fp16"),
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
    if idx >= _NHWC_CFG_BASE:
        sel = _nhwc_spec(idx - _NHWC_CFG_BASE)
        if sel is not None:
            return ("nhwc", *sel)
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
    elif kind == "spatial":
        kernel = build_direct_depthwise_spatial(spec, arch=arch)
    elif kind == "dgrad":
        kernel = build_direct_conv_dgrad(spec, arch=arch)
    elif kind == "dw_dgrad":
        kernel = build_direct_depthwise_dgrad(spec, arch=arch)
    elif kind == "nhwc":
        kernel = build_direct_conv_nhwc(spec, arch=arch)
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
