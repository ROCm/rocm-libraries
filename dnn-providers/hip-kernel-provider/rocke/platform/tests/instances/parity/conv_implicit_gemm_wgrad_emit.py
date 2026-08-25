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
#
# Config index map:
#   0  -- 3x3 2-D, mem/default, gfx950
#   1  -- 3x3 2-D, mem/cshuffle, gfx950
#   2  -- split-K=4 fp16 output, gfx950
#   3  -- 1x1 2-D, mem/default, gfx950
#   4  -- 128x128 tile, compv4/default, gfx950
#   5  -- WMMA gfx1151
#   6  -- WMMA gfx1201
#   7  -- split-K=4 fp32 output, gfx950
#   8  -- 3-D conv (Z/Di), mem/default, gfx950
#   9  -- split-K=4 bf16 output (packed bf16 atomic + accumulation-error path), gfx950
#   10 -- chiplet swizzle enabled, gfx950
#   (async_dma omitted: C++ async load path does not yet honour the wgrad A-descriptor
#    override, so it would produce different IR and break the byte-identity gate)
#
# Negative cases (configs 100+) verify that invalid specs are rejected:
#   100 -- odd C with fp16 split-K (must raise ValueError)
#   102 -- split_k > 1 on RDNA gfx1151 (must raise ValueError)
# (These illustrate the validator contract. The C emitter defines only cases
# 0-10, so run_diff.py stops at the shared END before reaching 100+; these
# configs are not exercised by the differential gate.)
from rocke.instances.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    build_implicit_gemm_conv_wgrad,
)
from rocke.instances.common._conv_implicit_gemm_common import ConvProblem
from _emit_common import run_emit


def _spec(idx: int):
    """Return (spec, arch) for config index `idx`.

    For negative cases (idx >= 100) the spec is expected to raise ValueError
    when built (run this file with that index to confirm it raises). These are
    illustrative of the validator contract; no differential harness consumes
    them (see the module header).
    """
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
    if idx == 8:
        # 3-D convolution (Z/Di path), mem/default, gfx950.
        p = ConvProblem(
            N=4,
            Hi=14,
            Wi=14,
            Di=14,
            C=32,
            K=32,
            Y=3,
            X=3,
            Z=3,
            sH=1,
            sW=1,
            sD=1,
            pH=1,
            pW=1,
            pD=1,
            dH=1,
            dW=1,
            dD=1,
        )
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
    if idx == 9:
        # Split-K=4 with bf16 output -- exercises packed bf16 atomic +
        # accumulation-error path (the newest codegen, previously untested).
        p = ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3)
        from rocke.instances.common._conv_implicit_gemm_common import ConvDataSpec

        return (
            WgradConvSpec(
                problem=p,
                data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="bf16"),
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
    if idx == 10:
        # Chiplet swizzle enabled, gfx950.
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
                chiplet_swizzle=True,
            ),
            "gfx950",
        )

    # ----------------------------------------------------------------
    # Negative cases: these specs must be REJECTED by the validator.
    # The harness (run_emit) expects a ValueError / SystemExit when
    # the config index is >= 100 and "expect_fail=True" is set.
    # ----------------------------------------------------------------
    if idx == 100:
        # Odd C with fp16 split-K -- must raise (packed atomic OOB).
        p = ConvProblem(N=8, Hi=56, Wi=56, C=3, K=64, Y=2, X=2)
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
    if idx == 102:
        # split_k > 1 on RDNA gfx1151 -- must raise (CDNA-only).
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
                split_k=4,
            ),
            "gfx1151",
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
