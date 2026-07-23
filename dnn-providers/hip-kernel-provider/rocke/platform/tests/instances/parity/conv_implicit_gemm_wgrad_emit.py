#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_implicit_gemm_wgrad_emit.py -- Python reference emitter
# for the implicit-GEMM backward-weight convolution parity harness.
# Selects one of N sampled spec configs by argv[1], builds the WgradConvSpec,
# builds the kernel via build_implicit_gemm_conv_wgrad(spec, arch=<cfg arch>)
# and prints lower_kernel_to_llvm(arch=<cfg arch>) to stdout so it can be
# byte-compared with the C emitter conv_implicit_gemm_wgrad_emit.c.
from rocke.instances.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    build_implicit_gemm_conv_wgrad,
)
from rocke.instances.common._conv_implicit_gemm_common import ConvProblem
from _emit_common import run_emit


def _spec(idx: int):
    """Return (spec, arch) for config index `idx`."""
    if idx == 0:
        # Baseline: 3x3 conv, mem pipeline, default epilogue, gfx950.
        p = ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3)
        return (
            WgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="mem",
                epilogue="default",
            ),
            "gfx950",
        )
    if idx == 1:
        # cshuffle epilogue, gfx950.
        p = ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3)
        return (
            WgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="mem",
                epilogue="cshuffle",
            ),
            "gfx950",
        )
    if idx == 2:
        # Split-K=4 with fp16 output, gfx950.
        p = ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3)
        return (
            WgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="mem",
                epilogue="default",
                split_k=4,
            ),
            "gfx950",
        )
    if idx == 3:
        # 1x1 conv, mem pipeline, gfx950.
        p = ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=1, X=1)
        return (
            WgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="mem",
                epilogue="default",
            ),
            "gfx950",
        )
    if idx == 4:
        # Larger tile, compv4 pipeline, gfx950.
        p = ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3)
        return (
            WgradConvSpec(
                problem=p,
                tile_m=128,
                tile_n=128,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="compv4",
                epilogue="default",
            ),
            "gfx950",
        )
    if idx in (5, 6):
        # WMMA wave32 RDNA targets: 16x16x16 / mem / default.
        arch = {5: "gfx1151", 6: "gfx1201"}[idx]
        p = ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3)
        return (
            WgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
                wave_size=32,
                pipeline="mem",
                epilogue="default",
            ),
            arch,
        )
    if idx == 7:
        # Split-K=4 with fp32 output, gfx950.
        p = ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3)
        from rocke.instances.common._conv_implicit_gemm_common import ConvDataSpec

        return (
            WgradConvSpec(
                problem=p,
                data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp32"),
                tile_m=64,
                tile_n=64,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="mem",
                epilogue="default",
                split_k=4,
            ),
            "gfx950",
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_implicit_gemm_conv_wgrad,
        usage="usage: conv_implicit_gemm_wgrad_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
