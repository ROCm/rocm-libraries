#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_implicit_gemm_emit.py -- Python reference emitter for the
# implicit-GEMM convolution parity harness. Selects one of N sampled spec
# configs by argv[1], builds the ImplicitGemmConvSpec, builds the kernel via
# build_implicit_gemm_conv(spec, arch=<cfg arch>) and prints
# lower_kernel_to_llvm(arch=<cfg arch>) to stdout so it can be byte-compared
# with the C emitter conv_implicit_gemm_emit.c.
import sys

from ck_dsl.instances.common.conv_implicit_gemm import (
    ConvProblem,
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
)
from ck_dsl import lower_kernel_to_llvm


def _spec(idx: int):
    """Return (spec, arch) for config index `idx`."""
    if idx == 0:
        p = ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, R=3, S=3)
        return (
            ImplicitGemmConvSpec(
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
        p = ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, R=3, S=3)
        return (
            ImplicitGemmConvSpec(
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
    if idx == 2:
        p = ConvProblem(N=16, Hi=112, Wi=112, C=128, K=128, R=3, S=3)
        return (
            ImplicitGemmConvSpec(
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
    if idx == 3:
        p = ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, R=3, S=3)
        return (
            ImplicitGemmConvSpec(
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
    if idx == 4:
        p = ConvProblem(N=1, Hi=224, Wi=224, C=3, K=64, R=7, S=7)
        return (
            ImplicitGemmConvSpec(
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
            ),
            "gfx950",
        )
    if idx == 5:
        p = ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, R=1, S=1)
        return (
            ImplicitGemmConvSpec(
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
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: conv_implicit_gemm_emit.py <config_index>\n")
        return 2
    idx = int(sys.argv[1])
    spec, arch = _spec(idx)
    kernel = build_implicit_gemm_conv(spec, arch=arch)
    text = lower_kernel_to_llvm(kernel, arch=arch)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
