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
#   (Grouped wgrad reaches the C++ engine only through stream-K, and its tile
#    loader has no elem_dtype, so no case loads bf16 operands. The stream-K
#    partition and CTA pool are kernargs, so no case varies them. async_dma is
#    rejected with stream-K.)
#
# Negative cases (configs 100+) verify that invalid specs are rejected:
#   100 -- odd C with fp16 split-K (must raise ValueError)
#   102 -- split_k > 1 on RDNA gfx1151 (must raise ValueError)
#   103 -- two_stage=True with split_k=1 (must raise ValueError)
#   104 -- streamk with split_k=4 (must raise ValueError)
#   105 -- streamk on gfx1250 (must raise ValueError)
#   106 -- streamk with the cshuffle epilogue (must raise ValueError)
#   107 -- streamk with unroll_k (must raise ValueError)
#   108 -- streamk atomic reduction into fp16 dW (must raise ValueError)
# (These illustrate the validator contract. The C emitter defines only cases
# 0-30, so run_diff.py stops at the shared END before reaching 100+; these
# configs are not exercised by the differential gate.)
from kernels.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    build_implicit_gemm_conv_wgrad,
)
from kernels.common._conv_implicit_gemm_common import ConvProblem
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
