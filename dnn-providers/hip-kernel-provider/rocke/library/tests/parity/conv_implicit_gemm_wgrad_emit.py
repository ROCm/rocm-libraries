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
#   11 -- K-outer LDS + ds_read_b64_tr_b16 transpose reads, gfx950
#   12 -- K-outer + async_dma (direct global->LDS load), gfx950
#   13 -- K-outer with the 16x16x16 atom (4 operand elements per lane), gfx950
#   14 -- pipeline="basic" unrolled global-read/compute overlap, gfx950
#   15 -- split-K=4, two_stage=True (workspace-store epilogue), fp16, gfx950
#   16 -- split-K=4, two_stage=True (workspace-store epilogue), fp16, gfx942
#   17 -- gfx1250 wave32 WMMA 16x16x32 K-outer (ds_load_tr16_b128 transpose reads)
#   18 -- unroll_k double-buffered loop under split-K=4 (odd-tail prefetch guard), gfx950
#   19 -- split-K (degree at launch) + two_stage, odd wg_N (C=3), fp16, gfx950
#   20 -- K-outer + async_dma + split-K=2 with the fp16 cshuffle atomic epilogue,
#         3 K tiles per slice (odd: the phase-B prefetch is redirected to the
#         zero-fill offset), gfx950
#   21 -- stream-K dp_sk, linear fixup, fp16, gfx950
#   22 -- stream-K dp_sk, tree fixup, fp16, gfx950
#   23 -- stream-K dp_sk, atomic reduction, fp32, gfx950
#   24 -- stream-K dp_sk, workspace reduction, fp16, gfx950
#   25 -- stream-K persistent, linear fixup, 16x16x16 atom, gfx942
#   26 -- stream-K dp_sk, linear, groups=3 with a partial last M tile, gfx950
#   27 -- stream-K persistent, tree, bf16 dW, groups=2, gfx950
#   28 -- stream-K persistent, atomic, fp32, groups=2, gfx950
#   29 -- stream-K persistent, workspace, fp16, gfx942
#   30 -- stream-K dp_sk, linear, K-outer LDS tile, groups=2, gfx950
#   31-48 -- prefetch_addr=True on a base config (see _PFA_BASES): the
#         carried-offset K loop across mem/compv3/compv4/basic, M-outer and
#         K-outer, split-K atomic, two-stage, scalar loads, pointwise, 3-D,
#         gfx942 and chiplet swizzle; 45-46 pair it with async_dma and
#         unroll_k, which both engines must reject; 47 is WMMA gfx1151 (the
#         carried-offset loop is family-agnostic); 48 is a scalar-load
#         128x128 tile (more carried offsets than accumulators)
#   49 -- gfx1250 WMMA split-K=4, bf16 dW (zero-fill packed bf16 atomic), M-outer
#   50 -- gfx1250 WMMA split-K=4, fp32 dW (scalar atomic), K-outer
#   51 -- gfx1250 WMMA split-K=2, fp16 dW, C=24 (partial N tiles)
#   52-54 -- WMMA split-K both engines must reject: gfx1201 bf16 dW (arch not
#         enabled), gfx1250 bf16 dW with the cshuffle epilogue (outside the
#         WMMA wgrad subset), gfx1151 fp32 dW (arch not enabled, any dtype)
#   55 -- gfx1250 pipeline="wmma_v1" (WMMA ds_read/wmma interleave hints), K-outer,
#         64x64x64 2x2 warps, split-K=4 fp32 dW
#   56 -- gfx1250 unroll_k ping-pong, K-outer, split-K=4 fp32 dW (odd slices:
#         the phase-B prefetch takes the zero-fill redirect)
#   57 -- gfx1250 prefetch_addr carried-offset loop, K-outer, split-K=2 fp16 dW
#   58 -- gfx1250 async_dma (global_load_async_to_lds ping-pong), K-outer,
#         split-K=2 fp32 dW, C=24 (partial N tiles: zero-filled chunks) and odd
#         slices (zero-fill redirect)
#   59 -- gfx1250 async_dma + "wmma_v1", K-outer, 64x64x64 2x2 warps
#   60-61 -- both engines must reject: "wmma_v1" on gfx950 (MFMA), and gfx1250
#         async_dma with C=3 (a one-element chunk is narrower than the copy)
#   (Grouped wgrad reaches the C++ engine only through stream-K, and its tile
#    loader has no elem_dtype, so no case loads bf16 operands. The stream-K
#    partition and CTA pool are kernargs, so no case varies them. async_dma is
#    rejected with stream-K, and so is prefetch_addr.)
#
# Negative cases (configs 100+) verify that invalid specs are rejected:
#   100 -- odd C with fp16 split-K (must raise ValueError)
#   102 -- 16-bit split_k > 1 on RDNA gfx1151 (must raise ValueError)
#   103 -- two_stage=True with split_k=1 (must raise ValueError)
#   104 -- streamk with split_k=4 (must raise ValueError)
#   105 -- streamk on gfx1250 (must raise ValueError)
#   106 -- streamk with the cshuffle epilogue (must raise ValueError)
#   107 -- streamk with unroll_k (must raise ValueError)
#   108 -- streamk atomic reduction into fp16 dW (must raise ValueError)
# (These illustrate the validator contract. The C emitter defines only cases
# 0-54, so run_diff.py stops at the shared END before reaching 100+; these
# configs are not exercised by the differential gate.)
from kernels.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    build_implicit_gemm_conv_wgrad,
)
from kernels.common._conv_implicit_gemm_common import ConvProblem
from _emit_common import run_emit


# prefetch_addr configs: index -> (base config index, extra overrides).
_PFA_BASES = {
    31: (1, {}),  # mem/cshuffle, M-outer
    32: (11, {}),  # mem/cshuffle, K-outer
    33: (4, {}),  # compv4, 128x128 tile
    34: (13, {}),  # K-outer, 16x16x16 atom, tile_k 16
    35: (7, {}),  # split-K atomic, fp32 dW
    36: (11, {"split_k": 2}),  # split-K packed fp16 atomic, K-outer cshuffle
    37: (15, {}),  # two-stage
    38: (19, {}),  # two-stage, C=3 (scalar loads)
    39: (14, {}),  # pipeline="basic", split-K fp32
    40: (3, {}),  # pointwise 1x1
    41: (8, {}),  # 3-D
    42: (16, {}),  # gfx942 two-stage
    43: (0, {"pipeline": "compv3"}),  # compv3
    44: (10, {}),  # chiplet swizzle
    45: (12, {}),  # + async_dma: rejected
    46: (18, {}),  # + unroll_k: rejected
    47: (5, {}),  # WMMA gfx1151: rejected
    # 128x128x64 with C = K = 3, so both operands load scalar: 4 accumulators
    # + 64 carried offsets, past the C++ engine's accumulator cap.
    48: (4, {"problem": ConvProblem(N=8, Hi=56, Wi=56, C=3, K=3, Y=3, X=3)}),
}


def _spec(idx: int):
    """Return (spec, arch) for config index `idx`.

    For negative cases (idx >= 100) the spec is expected to raise ValueError
    when built (run this file with that index to confirm it raises). These are
    illustrative of the validator contract; no differential harness consumes
    them (see the module header).
    """
    if idx in _PFA_BASES:
        from dataclasses import replace

        base_idx, extra = _PFA_BASES[idx]
        spec, arch = _spec(base_idx)
        return replace(spec, prefetch_addr=True, **extra), arch

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
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

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
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

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

    if idx == 11:
        # K-outer LDS tile + gfx950 ds_read_b64_tr_b16 transpose reads.
        # Exercises the swapped loader axes, the K-outer smem shapes and the
        # transpose-read fragment feed in one config.
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
                # cshuffle, not default: the validator rejects dtype_d in
                # (fp16, bf16) with epilogue="default", so a "default" config
                # would land as BOTH_REJECTED and the gate would never actually
                # compare the two engines.
                epilogue="cshuffle",
                lds_k_outer=True,
            ),
            "gfx950",
        )

    if idx == 12:
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        # K-outer + direct load. Exercises the async loader on the swapped tile
        # axes, the contig_cols guard and the packed (pad-0) LDS shape.
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
                # fp32 output + the direct-store epilogue: the async / K-outer
                # path under split-K on its own. Config 20 covers the same path
                # with the 16-bit cshuffle atomic epilogue.
                epilogue="default",
                data=ConvDataSpec(dtype_d="fp32"),
                lds_k_outer=True,
                async_dma=True,
                split_k=4,
            ),
            "gfx950",
        )

    if idx == 13:
        # K-outer with the 16x16x16 atom: the only 16-bit atom whose MFMA
        # operand is 4 elements per lane rather than 8, so the transpose-read
        # k-stride between lane groups is 4. Configs 11 and 12 both use
        # 32x32x16 (8 per lane) and cannot catch a stride hardcoded to 8.
        p = ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3)
        return (
            WgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=16,
                warp_m=2,
                warp_n=2,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
                pipeline="mem",
                epilogue="cshuffle",
                lds_k_outer=True,
            ),
            "gfx950",
        )

    if idx == 14:
        # pipeline="basic": the CK pipeline_basic loop, unrolled at build time
        # with the global read for tile it+1 issued before the MFMA for tile it
        # and the LDS write deferred past the second barrier. Needs a
        # compile-time trip count, hence the fixed split_k.
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

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
                pipeline="basic",
                epilogue="default",
                data=ConvDataSpec(dtype_d="fp32"),
                split_k=4,
            ),
            "gfx950",
        )

    if idx == 15:
        # Two-stage deterministic: workspace-store epilogue instead of atomic-add.
        # split_k=4, two_stage=True, fp16 output, gfx950.
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
                two_stage=True,
            ),
            "gfx950",
        )

    if idx == 16:
        # Two-stage deterministic, gfx942 (16x16x16 MFMA only).
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
                pipeline="mem",
                epilogue="default",
                split_k=4,
                two_stage=True,
            ),
            "gfx942",
        )

    if idx == 17:
        # gfx1250 wave32 WMMA 16x16x32 K-outer: the transpose read lowers to
        # ds_load_tr16_b128 (8 per lane), so a 16-element fragment is two reads.
        # dtype_d=fp32 because WMMA wgrad supports only the 'default' epilogue,
        # which rejects 16-bit dW.
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        p = ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        return (
            WgradConvSpec(
                problem=p,
                data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp32"),
                tile_m=32,
                tile_n=32,
                tile_k=32,
                warp_m=1,
                warp_n=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
                wave_size=32,
                pipeline="mem",
                epilogue="default",
                lds_k_outer=True,
            ),
            "gfx1250",
        )

    if idx == 18:
        # unroll_k: the double-buffered loop computes two tiles per step, so
        # under split-K the prefetch past an odd slice end has to be redirected
        # to wg_K or it pulls in the next slice's first tile.
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

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
                data=ConvDataSpec(dtype_d="fp32"),
                unroll_k=True,
                split_k=4,
            ),
            "gfx950",
        )

    if idx == 20:
        # wg_K = 10*6*6 = 360; at tile_k 64 and split_k 2 each slice is 3
        # tiles, so the ping-pong's last phase B lies past the slice end and
        # its prefetch takes the k_zero_fill redirect.
        p = ConvProblem(N=10, Hi=8, Wi=8, C=64, K=64, Y=3, X=3)
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
                lds_k_outer=True,
                async_dma=True,
                split_k=2,
            ),
            "gfx950",
        )
    if idx == 19:
        # Runtime split-K degree with the two-stage scratch: one binary for
        # every degree > 1. wg_N = 3*3*3 = 27 is odd, which the packed 16-bit
        # atomic cannot address -- this is the shape that needs it.
        p = ConvProblem(N=8, Hi=56, Wi=56, C=3, K=64, Y=3, X=3)
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
                split_k=2,
                two_stage=True,
            ),
            "gfx950",
        )

    if 21 <= idx <= 29:
        return _streamk_spec(idx)
    if idx == 30:
        from dataclasses import replace

        spec, arch = _streamk_spec(26)
        return (
            replace(
                spec,
                problem=ConvProblem(
                    N=4, Hi=14, Wi=14, C=64, K=96, Y=3, X=3, pH=1, pW=1, groups=2
                ),
                streamk_ctas=7,
                lds_k_outer=True,
            ),
            arch,
        )

    if 49 <= idx <= 51:
        # gfx1250 wave32 WMMA split-K through the WMMA atomic epilogue: a packed
        # 16-bit atomic per element (zero partner) or a scalar fp32 atomic.
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        problem, dtype_d, split_k, lds_k_outer = {
            49: (
                ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1),
                "bf16",
                4,
                False,
            ),
            50: (
                ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1),
                "fp32",
                4,
                True,
            ),
            51: (
                ConvProblem(N=4, Hi=12, Wi=12, C=24, K=32, Y=3, X=3, pH=1, pW=1),
                "fp16",
                2,
                False,
            ),
        }[idx]
        return (
            WgradConvSpec(
                problem=problem,
                data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d=dtype_d),
                tile_m=32,
                tile_n=32,
                tile_k=32,
                warp_m=1,
                warp_n=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
                wave_size=32,
                pipeline="mem",
                epilogue="default",
                split_k=split_k,
                lds_k_outer=lds_k_outer,
            ),
            "gfx1250",
        )

    if 52 <= idx <= 54:
        # WMMA split-K specs both engines must reject.
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        arch, dtype_d, epilogue, warp_tile_k = {
            52: ("gfx1201", "bf16", "default", 16),
            53: ("gfx1250", "bf16", "cshuffle", 32),
            54: ("gfx1151", "fp32", "default", 16),
        }[idx]
        return (
            WgradConvSpec(
                problem=ConvProblem(
                    N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1
                ),
                data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d=dtype_d),
                tile_m=32,
                tile_n=32,
                tile_k=32,
                warp_m=1,
                warp_n=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=warp_tile_k,
                wave_size=32,
                pipeline="mem",
                epilogue=epilogue,
                split_k=4,
            ),
            arch,
        )

    if 55 <= idx <= 61:
        # gfx1250 WMMA K-loop variants: the hint schedule, the unroll_k and
        # prefetch_addr loops, and the global_load_async_to_lds ping-pong.
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        p = ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        p_small = ConvProblem(N=4, Hi=12, Wi=12, C=24, K=32, Y=3, X=3, pH=1, pW=1)
        p_c3 = ConvProblem(N=4, Hi=12, Wi=12, C=3, K=32, Y=3, X=3, pH=1, pW=1)
        problem, dtype_d, split_k, big, extra, arch = {
            55: (p, "fp32", 4, True, {"pipeline": "wmma_v1"}, "gfx1250"),
            56: (p_small, "fp32", 4, False, {"unroll_k": True}, "gfx1250"),
            57: (p, "fp16", 2, False, {"prefetch_addr": True}, "gfx1250"),
            58: (p_small, "fp32", 2, False, {"async_dma": True}, "gfx1250"),
            59: (
                p,
                "fp32",
                4,
                True,
                {"async_dma": True, "pipeline": "wmma_v1"},
                "gfx1250",
            ),
            60: (p, "fp32", 1, False, {"pipeline": "wmma_v1"}, "gfx950"),
            61: (p_c3, "fp32", 2, False, {"async_dma": True}, "gfx1250"),
        }[idx]
        wave32 = arch == "gfx1250"
        kw = dict(
            problem=problem,
            data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d=dtype_d),
            tile_m=64 if big else 32,
            tile_n=64 if big else 32,
            tile_k=64 if big else 32,
            warp_m=2 if big else 1,
            warp_n=2 if big else 1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32 if wave32 else 16,
            wave_size=32 if wave32 else 64,
            pipeline="mem",
            epilogue="default",
            split_k=split_k,
            lds_k_outer=wave32,
        )
        kw.update(extra)
        return WgradConvSpec(**kw), arch

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
        # 16-bit split_k > 1 on RDNA gfx1151 -- must raise (WMMA split-K is
        # not enabled on gfx1151).
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
    if idx == 103:
        # two_stage=True with split_k=1 -- must raise ValueError.
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
                split_k=1,
                two_stage=True,
            ),
            "gfx950",
        )
    if 104 <= idx <= 108:
        from dataclasses import replace

        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        base, arch = _streamk_spec(21)
        bad = {
            104: (dict(split_k=4), arch),
            105: (dict(), "gfx1250"),
            106: (dict(epilogue="cshuffle"), arch),
            107: (dict(unroll_k=True), arch),
            108: (dict(streamk_reduction="atomic", data=ConvDataSpec()), arch),
        }[idx]
        return replace(base, **bad[0]), bad[1]
    raise SystemExit(f"unknown config index {idx}")


# (problem kwargs, data dtypes (a/b, d), atom edge, streamk, reduction, ctas, arch)
_STREAMK_CFGS = {
    21: (
        dict(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3),
        ("fp16", "fp16"),
        32,
        "dp_sk",
        "linear",
        16,
        "gfx950",
    ),
    22: (
        dict(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3),
        ("fp16", "fp16"),
        32,
        "dp_sk",
        "tree",
        16,
        "gfx950",
    ),
    23: (
        dict(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3),
        ("fp16", "fp32"),
        32,
        "dp_sk",
        "atomic",
        4,
        "gfx950",
    ),
    24: (
        dict(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3),
        ("fp16", "fp16"),
        32,
        "dp_sk",
        "workspace",
        4,
        "gfx950",
    ),
    25: (
        dict(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3),
        ("fp16", "fp16"),
        16,
        "persistent",
        "linear",
        4,
        "gfx942",
    ),
    26: (
        dict(N=4, Hi=28, Wi=28, C=96, K=120, Y=3, X=3, pH=1, pW=1, groups=3),
        ("fp16", "fp16"),
        32,
        "dp_sk",
        "linear",
        6,
        "gfx950",
    ),
    27: (
        dict(N=4, Hi=28, Wi=28, C=64, K=128, Y=3, X=3, pH=1, pW=1, groups=2),
        ("fp16", "bf16"),
        32,
        "persistent",
        "tree",
        5,
        "gfx950",
    ),
    28: (
        dict(N=4, Hi=28, Wi=28, C=64, K=128, Y=3, X=3, pH=1, pW=1, groups=2),
        ("fp16", "fp32"),
        32,
        "persistent",
        "atomic",
        3,
        "gfx950",
    ),
    29: (
        dict(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3),
        ("fp16", "fp16"),
        16,
        "persistent",
        "workspace",
        6,
        "gfx942",
    ),
}


def _streamk_spec(idx: int):
    """Stream-K configs; mirrored field-for-field by the C emitter."""
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec

    prob, (dt_ab, dt_d), edge, mode, reduction, ctas, arch = _STREAMK_CFGS[idx]
    return (
        WgradConvSpec(
            problem=ConvProblem(**prob),
            data=ConvDataSpec(dtype_a=dt_ab, dtype_b=dt_ab, dtype_d=dt_d),
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=edge,
            warp_tile_n=edge,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="default",
            streamk=mode,
            streamk_reduction=reduction,
            streamk_ctas=ctas,
        ),
        arch,
    )


def main() -> int:
    return run_emit(
        _spec,
        build_implicit_gemm_conv_wgrad,
        usage="usage: conv_implicit_gemm_wgrad_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
