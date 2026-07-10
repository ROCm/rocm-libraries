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
from rocke.instances.common.conv_implicit_gemm import (
    ConvProblem,
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
)
from _emit_common import run_emit

import inspect as _inspect

# The filter-window dimensions are spelled R/S in some trees and Y/X in others
# (same dimensions: filter height/width). Build against whichever this tree's
# ConvProblem exposes so the emitter is not tied to one spelling.
_CONV_PARAMS = set(_inspect.signature(ConvProblem.__init__).parameters)


def _cp(*, fy: int, fx: int, **kw):
    if "Y" in _CONV_PARAMS:
        kw["Y"], kw["X"] = fy, fx
    else:
        kw["R"], kw["S"] = fy, fx
    return ConvProblem(**kw)


def _spec(idx: int):
    """Return (spec, arch) for config index `idx`."""
    if idx == 0:
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, fy=3, fx=3)
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
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, fy=3, fx=3)
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
        p = _cp(N=16, Hi=112, Wi=112, C=128, K=128, fy=3, fx=3)
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
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, fy=3, fx=3)
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
        p = _cp(N=1, Hi=224, Wi=224, C=3, K=64, fy=7, fx=7)
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
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, fy=1, fx=1)
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
    if idx in (6, 7, 8):
        # WMMA wave32 RDNA targets: 16x16x16 / mem / default, w32.
        arch = {6: "gfx1151", 7: "gfx1201", 8: "gfx11-generic"}[idx]
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, fy=3, fx=3)
        return (
            ImplicitGemmConvSpec(
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
    if idx == 9:
        # chiplet_swizzle gfx950.
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, fy=3, fx=3)
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
        build_implicit_gemm_conv,
        usage="usage: conv_implicit_gemm_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
