#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_implicit_gemm_dgrad_emit.py -- Python reference emitter for
# the implicit-GEMM backward-data convolution parity harness.  Selects one of N
# sampled spec configs by argv[1], builds the DgradConvSpec, calls
# build_implicit_gemm_conv_dgrad(spec, arch=<cfg arch>) and prints
# lower_kernel_to_llvm(arch=<cfg arch>) to stdout so it can be byte-compared
# with the C emitter conv_implicit_gemm_dgrad_emit.c.
from rocke.instances.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    build_implicit_gemm_conv_dgrad,
)
from rocke.instances.common._conv_implicit_gemm_common import ConvProblem
from _emit_common import run_emit


def _cp(N, Hi, Wi, C, K, Y, X, sH=1, sW=1, pH=0, pW=0, dH=1, dW=1):
    return ConvProblem(
        N=N, Hi=Hi, Wi=Wi, C=C, K=K, Y=Y, X=X, sH=sH, sW=sW, pH=pH, pW=pW, dH=dH, dW=dW
    )


def _spec(idx: int):
    """Return (spec, arch) for config index `idx`."""

    # Config 0: baseline stride=1, default tile geometry, gfx950
    if idx == 0:
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
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

    # Config 1: larger tile, compv4 pipeline, gfx950
    if idx == 1:
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
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

    # Config 2: cshuffle epilogue, gfx950
    if idx == 2:
        p = _cp(N=16, Hi=112, Wi=112, C=128, K=128, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
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

    # Config 3: async_dma pipeline, gfx950
    if idx == 3:
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
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
                async_dma=True,
            ),
            "gfx950",
        )

    # Config 4: 1×1 filter (no spatial reduction), gfx950
    if idx == 4:
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=1, X=1)
        return (
            DgradConvSpec(
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

    # Config 5: stride=2, pad=1, 3×3 — tilde path (2×2 = 4 sub-GEMMs), gfx950
    if idx == 5:
        p = _cp(N=4, Hi=28, Wi=28, C=64, K=64, Y=3, X=3, sH=2, sW=2, pH=1, pW=1)
        return (
            DgradConvSpec(
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

    # Config 6: stride=2, pad=1, 4×4 — JIRA ticket scope, gfx950
    if idx == 6:
        p = _cp(N=4, Hi=28, Wi=28, C=64, K=64, Y=4, X=4, sH=2, sW=2, pH=1, pW=1)
        return (
            DgradConvSpec(
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

    # Config 7: split_k=2, stride=1, gfx950
    if idx == 7:
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
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
                split_k=2,
            ),
            "gfx950",
        )

    # Config 8: WMMA wave32, gfx1151
    if idx == 8:
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
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
            "gfx1151",
        )

    # Config 9: WMMA wave32, gfx1201
    if idx == 9:
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
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
            "gfx1201",
        )

    # Config 10: chiplet_swizzle, gfx950
    if idx == 10:
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=128,
                tile_n=128,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="mem",
                epilogue="default",
                chiplet_swizzle=True,
                chiplet_wgm=8,
                chiplet_num_xcds=8,
                chiplet_chunk_size=64,
            ),
            "gfx950",
        )

    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_implicit_gemm_conv_dgrad,
        usage="usage: conv_implicit_gemm_dgrad_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
