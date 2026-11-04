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
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    build_implicit_gemm_conv_dgrad,
)
from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from _emit_common import run_emit


def _cp(N, Hi, Wi, C, K, Y, X, sH=1, sW=1, pH=0, pW=0, dH=1, dW=1, groups=1):
    return ConvProblem(
        N=N,
        Hi=Hi,
        Wi=Wi,
        C=C,
        K=K,
        Y=Y,
        X=X,
        sH=sH,
        sW=sW,
        pH=pH,
        pW=pW,
        dH=dH,
        dW=dW,
        groups=groups,
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

    if idx == 11:
        # K-outer B tile + ds_read_tr16_b64 transpose read, 32x32x16 atom.
        # cshuffle deliberately: the validator rejects 16-bit dtype_d with the
        # default epilogue, and a rejected config lands as BOTH_REJECTED, which
        # the gate counts as a pass -- it would compare nothing.
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
                epilogue="cshuffle",
                lds_k_outer=True,
            ),
            "gfx950",
        )

    if idx == 12:
        # Same path on the 16x16x16 atom, where b_frag_len is 4 rather than 8:
        # one ds_read_tr16_b64 per fragment instead of two. Pins the per-atom
        # fragment length -- hardcoding 8 reads past the end of the tile.
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=32,
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

    if idx == 13:
        # gfx1250 wave32 WMMA 16x16x32 K-outer. The shared transpose-read helper
        # takes its wave32 branch here and lowers to ds_load_tr16_b128 (8 per
        # lane), so a 16-element fragment is two reads. Configs 11 and 12 are
        # both wave64, so without this the dgrad wave32 path shipped with no
        # cross-engine coverage at all.
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
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

    if idx == 14:
        # Rejected by both validators: an explicit vector_size_a of 8 on a
        # 16x32 dY tile gives 64 chunks for a 128-thread block, which the
        # loader cannot split evenly (coalesced_load_reason).
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=16,
                tile_n=32,
                tile_k=32,
                warp_m=1,
                warp_n=2,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
                pipeline="mem",
                epilogue="default",
                vector_size_a=8,
            ),
            "gfx950",
        )

    if idx == 15:
        # uniform_k_decode, stride 1: the per-tile decode loop plus the plain
        # fallback loop, sharing one MFMA phase.
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
                uniform_k_decode=True,
            ),
            "gfx950",
        )

    if idx == 16:
        # uniform_k_decode on the strided tilde path with the K-outer B tile
        # (the swapped W descriptor) and the cshuffle epilogue.
        p = _cp(N=4, Hi=28, Wi=28, C=64, K=64, Y=3, X=3, sH=2, sW=2, pH=1, pW=1)
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
                pipeline="compv3",
                epilogue="cshuffle",
                lds_k_outer=True,
                uniform_k_decode=True,
            ),
            "gfx950",
        )

    if idx == 17:
        # uniform_k_decode with split-K: the decode starts from a non-zero
        # k_lo, and the B (y, x) test zeroes the tiles past gemm_k.
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="mem",
                epilogue="default",
                split_k=2,
                uniform_k_decode=True,
            ),
            "gfx950",
        )

    if idx == 18:
        # uniform_k_decode on a pointwise problem: the fallback loop keeps the
        # pointwise descriptors, the decode loop does not use them.
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
                uniform_k_decode=True,
            ),
            "gfx950",
        )

    if idx == 19:
        # uniform_k_decode on WMMA wave32: the shared MFMA phase takes its
        # WMMA branch in both loops.
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
                uniform_k_decode=True,
            ),
            "gfx1201",
        )

    if idx == 20:
        # Rejected by both validators: the wavelet pipeline has its own K
        # loop, so uniform_k_decode would only rename the kernel.
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
                wave_size=32,
                pipeline="wavelet",
                epilogue="default",
                uniform_k_decode=True,
            ),
            "gfx1250",
        )

    # Configs 21-33: grouped convolution (groups > 1). The conv group rides
    # blockIdx.y; the reduction decodes by kpg, the dY/W channel is offset by
    # g*kpg, W's channel stride is cpg and dX's channel is offset by g*cpg.
    # Every epilogue leaf, the uniform_k_decode loop and the per-group vector
    # widths are covered, plus the two grouped rejections. All use fp16
    # operands: the C++ tile loader stages through f16-typed LDS stores, so a
    # bf16 operand differs from Python in that IR type alone, grouped or not.

    if idx == 21:
        # Grouped stride 1, direct epilogue.
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=4)
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

    if idx == 22:
        # Grouped stride 1 with uniform_k_decode: the decode divides by kpg,
        # and both per-tile offsets carry the group's output-channel base.
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=4)
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
                uniform_k_decode=True,
            ),
            "gfx950",
        )

    if idx == 23:
        # Grouped cshuffle with cpg = kpg = 6: the load and store widths
        # follow the per-group channel run (2), not the dense C (8).
        p = _cp(N=4, Hi=28, Wi=28, C=24, K=24, Y=3, X=3, pH=1, pW=1, groups=4)
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

    if idx == 24:
        # Grouped uniform_k_decode with the K-outer B tile and cshuffle.
        p = _cp(N=8, Hi=28, Wi=28, C=128, K=128, Y=3, X=3, pH=1, pW=1, groups=4)
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
                epilogue="cshuffle",
                lds_k_outer=True,
                uniform_k_decode=True,
            ),
            "gfx950",
        )

    if idx == 25:
        # Grouped stride 2: the tilde path with the direct tilde epilogue.
        p = _cp(
            N=4, Hi=28, Wi=28, C=64, K=64, Y=3, X=3, sH=2, sW=2, pH=1, pW=1, groups=4
        )
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

    if idx == 26:
        # Grouped stride 2 with uniform_k_decode and the tilde cshuffle
        # epilogue. cpg = 12, so the tilde store is 4 wide: it follows the
        # per-group channel run like the stride-1 cshuffle, not a fixed 8.
        p = _cp(
            N=4, Hi=28, Wi=28, C=24, K=24, Y=3, X=3, sH=2, sW=2, pH=1, pW=1, groups=2
        )
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
                pipeline="compv3",
                epilogue="cshuffle",
                uniform_k_decode=True,
            ),
            "gfx950",
        )

    if idx == 27:
        # Grouped split-K with uniform_k_decode: the packed fp16 atomic
        # epilogue on the absolute dX channel.
        p = _cp(N=8, Hi=28, Wi=28, C=128, K=128, Y=3, X=3, pH=1, pW=1, groups=2)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="mem",
                epilogue="default",
                split_k=2,
                uniform_k_decode=True,
            ),
            "gfx950",
        )

    if idx == 28:
        # Grouped split-K on the strided tilde path (packed fp16 atomics).
        p = _cp(
            N=4, Hi=28, Wi=28, C=64, K=64, Y=3, X=3, sH=2, sW=2, pH=1, pW=1, groups=4
        )
        return (
            DgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=32,
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

    if idx == 29:
        # Grouped 1x1 with uniform_k_decode: a grouped problem never takes
        # the ungrouped pointwise fast path.
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=1, X=1, groups=4)
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
                uniform_k_decode=True,
            ),
            "gfx950",
        )

    if idx == 30:
        # Grouped WMMA wave32 stride 1 with uniform_k_decode (the WMMA direct
        # epilogue).
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=4)
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
                uniform_k_decode=True,
            ),
            "gfx1201",
        )

    if idx == 31:
        # Grouped WMMA wave32 stride 2 (the WMMA tilde direct epilogue).
        p = _cp(
            N=4, Hi=28, Wi=28, C=64, K=64, Y=3, X=3, sH=2, sW=2, pH=1, pW=1, groups=4
        )
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

    if idx == 32:
        # Grouped depthwise (channels-per-group == 1): a one-column GEMM per
        # group, every operand scalar. cshuffle so the config is accepted (the
        # default epilogue rejects a 16-bit dX).
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=64)
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

    if idx == 33:
        # Rejected by both validators: packed fp16 split-K atomics need an even
        # channels-per-group (C = 12 is even, cpg = 3 is not).
        p = _cp(N=8, Hi=56, Wi=56, C=12, K=12, Y=3, X=3, pH=1, pW=1, groups=4)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=32,
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

    if idx == 34:
        # patch: a patch embedding (stride == filter, no padding) as one GEMM
        # over the filter's (y, x, c) -- the tilde cshuffle store splits each
        # column into a filter row and its run of X*C.
        p = _cp(N=4, Hi=32, Wi=32, C=3, K=64, Y=4, X=4, sH=4, sW=4)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=128,
                tile_n=64,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
                pipeline="mem",
                epilogue="cshuffle",
                vector_size_a=8,
                vector_size_b=4,
                vector_size_c=4,
                lds_k_outer=True,
                uniform_k_decode=True,
                patch=True,
            ),
            "gfx950",
        )

    if idx == 35:
        # patch with the scalar (default) tilde store and stride > filter.
        p = _cp(N=2, Hi=30, Wi=30, C=4, K=32, Y=2, X=2, sH=3, sW=3)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="mem",
                epilogue="default",
                patch=True,
            ),
            "gfx950",
        )

    # Configs 36-39: validator hardening. 36 and 37 are rejected by both
    # validators; 38 and 39 are accepted and built.
    if idx == 36:
        # Rejected: the dgrad builder has no K-unrolled loop (unroll_k used to
        # be accepted and silently ignored).
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
                unroll_k=True,
            ),
            "gfx950",
        )

    if idx == 37:
        # Rejected: 1x2 warps of the 16x16x32 atom on a 256x256 tile hold 512
        # fp32 accumulators per lane (> 256, the whole register file).
        p = _cp(N=4, Hi=12, Wi=12, C=384, K=384, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=256,
                tile_n=256,
                tile_k=64,
                warp_m=1,
                warp_n=2,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
                pipeline="mem",
                epilogue="default",
            ),
            "gfx950",
        )

    if idx == 38:
        # Accepted: compv4 is charged one A and one B tile (the builder never
        # double-buffers), so a 128x128x64 tile fits gfx942's LDS; a doubled
        # charge rejected it.
        p = _cp(N=4, Hi=28, Wi=28, C=128, K=128, Y=3, X=3, pH=1, pW=1)
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
                warp_tile_k=8,
                pipeline="compv4",
                epilogue="default",
            ),
            "gfx942",
        )

    if idx == 39:
        # Accepted: a non-default A row pad (lds_k_pad=0) reaches the kernel
        # name ("_kp0").
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
                lds_k_pad=0,
            ),
            "gfx950",
        )

    # Configs 40-43: stride 1 builds its single tilde record from the
    # kernargs (DgradConvSpec.folds_sub_gemm_record), at every dilation, pad,
    # filter and grouping; configs 0, 4 and 21-24 fold too.
    if idx == 40:
        # Stride 1, dilation 2: folded record, tilde epilogue (a_embed = -dH).
        p = _cp(N=4, Hi=20, Wi=20, C=64, K=64, Y=3, X=3, pH=2, pW=2, dH=2, dW=2)
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
                uniform_k_decode=True,
            ),
            "gfx950",
        )

    if idx == 41:
        # Grouped 5x5, pad 0 (Ho < Hi): the AOT shape of an entry --
        # K-outer, uniform_k_decode, cshuffle, explicit widths.
        p = _cp(N=2, Hi=24, Wi=20, C=128, K=256, Y=5, X=5, groups=2)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=128,
                tile_n=64,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="compv3",
                epilogue="cshuffle",
                lds_k_outer=True,
                uniform_k_decode=True,
                vector_size_a=8,
                vector_size_b=8,
                vector_size_c=8,
            ),
            "gfx950",
        )

    if idx == 42:
        # Stride 1 split-K: the folded gemm_k_padded rounds dg_K up to
        # tile_k * split_k in the kernel.
        p = _cp(N=4, Hi=14, Wi=14, C=64, K=96, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="mem",
                epilogue="cshuffle",
                split_k=2,
            ),
            "gfx950",
        )

    if idx == 43:
        # gfx942, grouped 1x3 filter with pad (0, 1).
        p = _cp(N=2, Hi=12, Wi=16, C=64, K=128, Y=1, X=3, pH=0, pW=1, groups=4)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=8,
                pipeline="mem",
                epilogue="default",
            ),
            "gfx942",
        )

    # Configs 44-48: the tap-outer K loop (DgradConvSpec.tap_outer_k).
    # 44-46 build; 47 and 48 are rejected by both validators (the loop has no
    # fallback for a problem it cannot compute).
    if idx == 44:
        # Grouped 3x3: the AOT shape of an entry (K-outer, cshuffle, widths).
        p = _cp(N=2, Hi=14, Wi=14, C=128, K=128, Y=3, X=3, pH=1, pW=1, groups=2)
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
                lds_k_outer=True,
                vector_size_a=8,
                vector_size_b=8,
                vector_size_c=8,
                tap_outer_k=True,
            ),
            "gfx950",
        )

    if idx == 45:
        # Ungrouped 5x5, pad 2, M-outer, several chunks per tap.
        p = _cp(N=2, Hi=13, Wi=9, C=64, K=128, Y=5, X=5, pH=2, pW=2)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="compv4",
                epilogue="default",
                tap_outer_k=True,
            ),
            "gfx950",
        )

    if idx == 46:
        # gfx942, grouped 1x3 filter with pad (0, 1).
        p = _cp(N=2, Hi=12, Wi=16, C=64, K=128, Y=1, X=3, pH=0, pW=1, groups=4)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=64,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=8,
                pipeline="mem",
                epilogue="default",
                tap_outer_k=True,
            ),
            "gfx942",
        )

    if idx == 47:
        # Rejected: kpg = 96 is not a multiple of tile_k = 64.
        p = _cp(N=2, Hi=14, Wi=14, C=64, K=96, Y=3, X=3, pH=1, pW=1)
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
                tap_outer_k=True,
            ),
            "gfx950",
        )

    if idx == 48:
        # Rejected: stride 2 (the tilde slice is not the plain filter tap).
        p = _cp(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, sH=2, sW=2, pH=1, pW=1)
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
                tap_outer_k=True,
            ),
            "gfx950",
        )

    # Configs 49-50: the XCD-contiguous tile order of grouped stride-1
    # kernels on gfx950 (DgradConvSpec.xcd_tile_order); 49 emits the remap
    # (the default), 50 is the launch-order A/B build.
    if idx in (49, 50):
        p = _cp(N=1, Hi=8, Wi=8, C=256, K=256, Y=3, X=3, pH=1, pW=1, groups=4)
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
                xcd_tile_order=(idx == 49),
            ),
            "gfx950",
        )

    # Configs 51-56: the dY halo mode of the tap-outer loop (dy_halo).
    # 51-54 build; 55 (no tap-outer loop) and 56 (Wo > halo_max_wo) are
    # rejected by both validators.
    halo = {
        51: (
            _cp(N=2, Hi=10, Wi=14, C=128, K=128, Y=3, X=3, pH=1, pW=1, groups=2),
            {
                "epilogue": "cshuffle",
                "lds_k_outer": True,
                "vector_size_a": 8,
                "vector_size_b": 8,
                "vector_size_c": 8,
                "dy_halo": 2,
                "halo_max_wo": 16,
            },
        ),
        52: (
            _cp(N=1, Hi=9, Wi=20, C=64, K=128, Y=5, X=5, pH=2, pW=2),
            {
                "tile_m": 128,
                "warp_m": 4,
                "warp_n": 1,
                "lds_k_outer": True,
                "dy_halo": 1,
                "dy_halo_setprio": 1,
                "dy_halo_kouter_pad": 32,
                "halo_max_wo": 32,
            },
        ),
        53: (
            _cp(N=2, Hi=8, Wi=8, C=64, K=64, Y=7, X=7, pH=3, pW=3),
            {
                "tile_m": 256,
                "warp_m": 4,
                "warp_n": 1,
                "epilogue": "cshuffle",
                "lds_k_outer": True,
                "dy_halo": 2,
                "dy_halo_setprio": 2,
                "halo_max_wo": 8,
            },
        ),
        54: (
            _cp(N=1, Hi=6, Wi=19, C=256, K=256, Y=3, X=5, pH=1, pW=2, groups=4),
            {"dy_halo": 2, "halo_max_wo": 24},
        ),
        55: (
            _cp(N=2, Hi=10, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1),
            {
                "lds_k_outer": True,
                "dy_halo": 2,
                "halo_max_wo": 16,
                "tap_outer_k": False,
            },
        ),
        56: (
            _cp(N=2, Hi=10, Wi=40, C=64, K=64, Y=3, X=3, pH=1, pW=1),
            {"lds_k_outer": True, "dy_halo": 2, "halo_max_wo": 32},
        ),
    }
    if idx in halo:
        p, kw = halo[idx]
        base = {
            "problem": p,
            "tile_m": 64,
            "tile_n": 64,
            "tile_k": 64,
            "warp_m": 2,
            "warp_n": 2,
            "warp_tile_m": 32,
            "warp_tile_n": 32,
            "warp_tile_k": 16,
            "pipeline": "mem",
            "epilogue": "default",
            "tap_outer_k": True,
        }
        base.update(kw)
        return DgradConvSpec(**base), "gfx950"

    if 57 <= idx <= 66:
        return _tf32_spec(idx)
    if 67 <= idx <= 70:
        return _split_k_cshuffle_spec(idx)
    if idx >= 57 + _GFX942_3D_OFFSET:
        return _gfx942_3d_any(idx - _GFX942_3D_OFFSET)

    raise SystemExit(f"unknown config index {idx}")


# TF32 compute (ConvDataSpec(mma_dtype="tf32")): fp32 dY / W / dX, each LDS
# fragment bitcast to vec<tf32xN> for the arch's TF32 atom -- native XF32 on
# gfx942 (16x16x8, 32x32x4), BF16x3 emulation on gfx950 (16x16x32, 32x32x16).
# N=2 16x16 C=K=64 3x3 pad 1, 64x64 tile, 2x2 warps. Keep in lockstep with
# make_cfg_tf32() in conv_implicit_gemm_dgrad_emit.c.
#   57: gfx950 16x16x32, stride 1, mem/default
#   58: gfx950 32x32x16, stride 2 (tilde sub-GEMMs), compv4/cshuffle
#   59: gfx942 16x16x8, stride 1, compv3/cshuffle
#   60: gfx942 32x32x4, stride 2, mem/default
#   61: gfx950 16x16x32, stride 2, split_k=2 (atomic fp32 dX), cshuffle
#   62: gfx942 16x16x8, stride 1, groups=2, split_k=2
#   63: gfx950 16x16x32, stride 1, tap_outer_k + dy_halo=2
#   64: gfx942 32x32x4, stride 2, uniform_k_decode, compv4
#   65: gfx950 lds_k_outer: rejected by both engines
#   66: gfx950 with gfx942's 16x16x8 atom: rejected by both engines
_TF32 = ConvDataSpec("fp32", "fp32", "fp32", mma_dtype="tf32")


def _tf32_spec(idx: int):
    arch, wt, tile_k, stride, kw = {
        57: ("gfx950", (16, 16, 32), 64, 1, {}),
        58: (
            "gfx950",
            (32, 32, 16),
            32,
            2,
            dict(pipeline="compv4", epilogue="cshuffle"),
        ),
        59: (
            "gfx942",
            (16, 16, 8),
            32,
            1,
            dict(pipeline="compv3", epilogue="cshuffle"),
        ),
        60: ("gfx942", (32, 32, 4), 32, 2, {}),
        61: ("gfx950", (16, 16, 32), 64, 2, dict(split_k=2, epilogue="cshuffle")),
        62: ("gfx942", (16, 16, 8), 32, 1, dict(split_k=2)),
        63: (
            "gfx950",
            (16, 16, 32),
            64,
            1,
            dict(tap_outer_k=True, dy_halo=2, halo_max_wo=16),
        ),
        64: (
            "gfx942",
            (32, 32, 4),
            32,
            2,
            dict(uniform_k_decode=True, pipeline="compv4"),
        ),
        65: ("gfx950", (16, 16, 32), 64, 1, dict(lds_k_outer=True)),
        66: ("gfx950", (16, 16, 8), 32, 1, {}),
    }[idx]
    groups = 2 if idx == 62 else 1
    base = {
        "problem": _cp(
            N=2,
            Hi=16,
            Wi=16,
            C=64 * groups,
            K=64 * groups,
            Y=3,
            X=3,
            sH=stride,
            sW=stride,
            pH=1,
            pW=1,
            groups=groups,
        ),
        "data": _TF32,
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": tile_k,
        "warp_m": 2,
        "warp_n": 2,
        "warp_tile_m": wt[0],
        "warp_tile_n": wt[1],
        "warp_tile_k": wt[2],
        "pipeline": "mem",
        "epilogue": "default",
    }
    base.update(kw)
    return DgradConvSpec(**base), arch


# Split-K with the cshuffle epilogue on gfx950 (the cached split-K form): the
# staged tile is atomic-added into dX through the tilde address function, in
# packed <2 x dtype> pairs for a 16-bit dX. 16x16x32 atom, 8-wide loads, K-outer
# LDS. Keep in lockstep with make_cfg_split_k_cshuffle() in
# conv_implicit_gemm_dgrad_emit.c.
#   67: fp16, stride 2 (tilde sub-GEMMs), 32x128x64 tile, 2x4 warps, split_k=4,
#       dX width 2 (one pair per thread per row run)
#   68: bf16, stride 1, groups=2 (the group base in the dX address), compv3,
#       split_k=2, dX width 2
#   69: fp16, stride 1, 64x128x64 tile, 2x4 warps, split_k=8, dX width 8
#       (four pairs per thread)
#   70: fp16, split_k=2, dX width 1 (no packed pair): rejected by both engines
def _split_k_cshuffle_spec(idx: int):
    stride, groups, kw = {
        67: (2, 1, dict(tile_m=32, tile_n=128, split_k=4)),
        68: (
            1,
            2,
            dict(data=ConvDataSpec("bf16", "bf16", "bf16"), pipeline="compv3"),
        ),
        69: (1, 1, dict(tile_n=128, split_k=8, vector_size_c=8)),
        70: (1, 1, dict(vector_size_c=1)),
    }[idx]
    base = dict(
        problem=_cp(
            N=2,
            Hi=16,
            Wi=16,
            C=64 * groups,
            K=128 * groups,
            Y=3,
            X=3,
            sH=stride,
            sW=stride,
            pH=1,
            pW=1,
            groups=groups,
        ),
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=4,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
        pipeline="mem",
        epilogue="cshuffle",
        lds_k_outer=True,
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=2,
        split_k=2,
    )
    base.update(kw)
    return DgradConvSpec(**base), "gfx950"


# gfx942 3-D (NDHWC) tap-outer configs on the grouped 3x3x3 / pad-1 problem
# the AOT cache builds its dgrad binaries against: (tile, overrides). Keep in
# lockstep with make_cfg_3d() in conv_implicit_gemm_dgrad_emit.c. The numbers
# below are _gfx942_3d_any's own; it serves parity config N as its config
# N - _GFX942_3D_OFFSET (they follow the split-K cshuffle configs, which end
# at 70).
#   57: t128x256x64 w2x4 cshuffle, the plain tap-outer loop
#   58: 57 + tapk_prefetch 1, b_kpack 4, xcd_remap 6
#   59: 57 + tapk_prefetch 3, b_kpack 4
#   60: t256x64x32 w4x1 + tapk_prefetch 3, b_kpack 2, xcd_remap 6, wpe2
#   61: 60 with tapk_prefetch 2 (two LDS buffers)
#   62: 60 with tapk_prefetch 1, no b_kpack, fp16
#   63: 58 on the 2-D problem
#   64: 58 ungrouped (groups 1; xcd_remap also serves it)
#   65: 57 with pad (0, 1, 1)
#   66-70: 58 + tapk_pingpong (see the table)
_GFX942_3D = {
    57: ("s6", {}),
    58: ("s6", dict(tapk_prefetch=1, b_kpack=4, xcd_remap=6)),
    59: ("s6", dict(tapk_prefetch=3, b_kpack=4)),
    60: ("s1", dict(tapk_prefetch=3, b_kpack=2, xcd_remap=6, waves_per_eu=2)),
    61: ("s1", dict(tapk_prefetch=2, b_kpack=2, xcd_remap=6, waves_per_eu=2)),
    62: ("s1", dict(tapk_prefetch=1, xcd_remap=6, waves_per_eu=2)),
    63: ("s6", dict(tapk_prefetch=1, b_kpack=4, xcd_remap=6)),
    64: ("s6", dict(tapk_prefetch=1, b_kpack=4, xcd_remap=6)),
    65: ("s6", {}),
    # 66-70: the tapk_pingpong block ping-pong (8 waves) on 58
    66: (
        "s6",
        dict(
            tapk_prefetch=1,
            b_kpack=4,
            xcd_remap=6,
            tapk_pingpong=4,
            tapk_pp_sched=2,
            lds_xor=True,
        ),
    ),
    67: (
        "s6",
        dict(
            tapk_prefetch=1,
            b_kpack=4,
            xcd_remap=6,
            tapk_pingpong=4,
            tapk_pp_sched=5,
            lds_xor=True,
            tapk_pp_bslab=3,
        ),
    ),
    68: (
        "s6",
        dict(
            tapk_prefetch=1,
            b_kpack=4,
            xcd_remap=6,
            tapk_pingpong=4,
            tapk_pp_sched=12,
            lds_xor=True,
            tapk_pp_bslab=2,
        ),
    ),
    69: (
        "s6",
        dict(
            tapk_prefetch=1,
            b_kpack=4,
            xcd_remap=6,
            tapk_pingpong=5,
            tapk_pp_sched=1,
            lds_xor=True,
        ),
    ),
    70: (
        "s6",
        dict(
            tapk_prefetch=1,
            b_kpack=4,
            xcd_remap=6,
            tapk_pingpong=3,
            tapk_pp_sched=2,
            lds_xor=True,
        ),
    ),
}


# 71-76: the ping-pong / box record kernels (tapk_pingpong 4 over pp_boxes 7 /
# 77 records, split-K with pp_box_det 1, tapk_pp_epi 2, cshuffle_m_chunks) on
# the 3-D grouped problem; keep in lockstep with make_cfg_3d_pp() in
# conv_implicit_gemm_dgrad_emit.c.
_PP_COMMON = dict(
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=16,
    vector_size_a=8,
    vector_size_b=8,
    vector_size_c=8,
    lds_k_pad=0,
    pipeline="mem",
    epilogue="cshuffle",
    tap_outer_k=True,
    tapk_prefetch=1,
    b_kpack=4,
    tapk_pingpong=4,
    lds_xor=True,
    xcd_remap=6,
)
_T256 = dict(tile_m=256, tile_n=256, tile_k=32, warp_m=4, warp_n=2)
_T128 = dict(tile_m=128, tile_n=256, tile_k=64, warp_m=2, warp_n=4)
_GFX942_3D_PP = {
    71: dict(
        _T256,
        tapk_pp_sched=5,
        tapk_pp_bslab=3,
        cshuffle_m_chunks=2,
        split_k=8,
        pp_boxes=7,
        pp_box_order=2,
        pp_box_det=1,
    ),
    72: dict(
        _T256,
        tapk_pp_sched=5,
        cshuffle_m_chunks=4,
        split_k=8,
        pp_boxes=7,
        pp_box_order=2,
        pp_box_det=1,
    ),
    73: dict(
        _T128,
        tapk_pp_sched=2,
        tapk_pp_epi=2,
        tapk_pp_bslab=3,
        pp_boxes=77,
        pp_box_order=2,
        epi_lds_barriers=True,
    ),
    74: dict(
        _T256,
        tapk_pp_sched=12,
        tapk_pp_bslab=3,
        cshuffle_m_chunks=2,
        split_k=3,
        pp_boxes=77,
        pp_box_det=1,
    ),
    75: dict(
        _T128, tapk_pp_sched=5, split_k=4, pp_boxes=77, pp_box_order=3, pp_box_det=1
    ),
    76: dict(
        _T128, tapk_pp_sched=2, split_k=8, pp_boxes=7, pp_box_order=1, pp_box_det=1
    ),
    # 84: no boxes -- tapk_ztrim, the partial dY tile (224 rows) and the
    # stride-1 pp_epi 2 epilogue.
    84: dict(
        tile_m=224,
        tile_n=256,
        tile_k=32,
        warp_m=2,
        warp_n=4,
        tapk_pp_sched=5,
        tapk_pp_epi=2,
        tapk_ztrim=1,
        tapk_pp_bslab=3,
        cshuffle_m_chunks=7,
    ),
    # 85-87: split_atomic_pairs on pixel-list boxes (the packed-atomic box
    # store: pair lanes, batched row map, LDS-only barriers).
    85: dict(
        _T128,
        tapk_pp_sched=2,
        split_k=8,
        pp_boxes=7,
        pp_box_order=1,
        split_atomic_pairs=True,
    ),
    86: dict(
        _T256,
        tapk_pp_sched=12,
        tapk_pp_bslab=3,
        cshuffle_m_chunks=2,
        split_k=8,
        pp_boxes=77,
        split_atomic_pairs=True,
        epi_lds_barriers=True,
    ),
    87: dict(
        _T256,
        tapk_pp_sched=12,
        tapk_pp_bslab=3,
        cshuffle_m_chunks=4,
        split_k=8,
        pp_boxes=7,
        pp_box_cus=456,
        pp_box_order=2,
        split_atomic_pairs=True,
        epi_lds_barriers=True,
    ),
    # 88-92: the other two-ahead schedules (pps 4 / 3 / 6 / 11) and m_band on
    # the non-box ping-pong.
    88: dict(
        tile_m=256,
        tile_n=128,
        tile_k=64,
        warp_m=4,
        warp_n=2,
        tapk_pp_sched=4,
        tapk_pp_epi=2,
        tapk_ztrim=1,
    ),
    89: dict(_T128, tapk_pp_sched=3),
    90: dict(_T128, tapk_pp_sched=6),
    91: dict(_T128, tapk_pp_sched=11),
    92: dict(
        tile_m=512,
        tile_n=128,
        tile_k=32,
        warp_m=4,
        warp_n=2,
        tapk_pp_sched=1,
        tapk_pp_epi=2,
        tapk_pp_bslab=1,
        m_band=11,
        cshuffle_m_chunks=2,
    ),
    # 93-94: the non-box ping-pong through the plain cshuffle store in m_chunks
    # (93) and the pp_box_det 2 (bf16 slot) split box store (94).
    93: dict(_T128, tapk_pp_sched=5, cshuffle_m_chunks=2),
    94: dict(
        _T128, tapk_pp_sched=2, split_k=8, pp_boxes=7, pp_box_order=1, pp_box_det=2
    ),
    # 97-98: tapk_pp_sched 7 (three tiles ahead), split boxes and plain.
    97: dict(
        _T128, tapk_pp_sched=7, split_k=8, pp_boxes=7, pp_box_order=2, pp_box_det=1
    ),
    98: dict(_T128, tapk_pp_sched=7),
    # 99-100: split-K ping-pong without boxes (the plain packed-atomic store;
    # 100: pair lanes).
    99: dict(_T128, tapk_pp_sched=5, split_k=4),
    100: dict(_T128, tapk_pp_sched=2, split_k=3, split_atomic_pairs=True),
    # 101-105: pp_zskip (plain, split boxes) and the split box stores through
    # the record's (n, h, w) decode (plain atomics, sapl) and the pixel list.
    101: dict(_T128, tapk_pp_sched=5, pp_zskip=True),
    102: dict(
        _T128,
        tapk_pp_sched=2,
        split_k=3,
        pp_boxes=5,
        pp_box_order=2,
        pp_box_det=2,
        pp_zskip=True,
    ),
    103: dict(
        _T128,
        tapk_pp_sched=2,
        split_k=4,
        pp_boxes=3,
        pp_box_order=2,
        epi_lds_barriers=True,
    ),
    104: dict(
        _T128,
        tapk_pp_sched=7,
        split_k=8,
        pp_boxes=113,
        pp_box_order=1,
        split_atomic_pairs=True,
        pp_zskip=True,
    ),
    105: dict(
        _T256,
        tapk_pp_sched=12,
        tapk_pp_bslab=3,
        cshuffle_m_chunks=2,
        split_k=4,
        pp_boxes=7,
        pp_box_order=2,
    ),
    # 106: xh_epi_lin on the dense pp_epi 2 store (+ epi_lds_barriers).
    106: dict(
        _T128, tapk_pp_sched=5, tapk_pp_epi=2, xh_epi_lin=True, epi_lds_barriers=True
    ),
    # 107-109: tapk_pingpong 1 / 2 / 3 with the plain loop (tapk_pp_sched 0;
    # 109 split-K).
    107: dict(_T128, tapk_pingpong=1),
    108: dict(_T128, tapk_pingpong=2, lds_xor=False),
    109: dict(_T128, tapk_pingpong=3, split_k=4),
}
# 116-117: pp_boxes 8 (the pixel-list binary of pp_boxes 7; the records come
# from the host-side makespan plan) on 105 and on 76 (pp_box_det 1).
_GFX942_3D_PP[116] = dict(_GFX942_3D_PP[105], pp_boxes=8)
_GFX942_3D_PP[117] = dict(_GFX942_3D_PP[76], pp_boxes=8)


# 77-80: the x-halo (tapk_xhalo) record kernel family on the W-extended
# pW = 0 view of the grouped 3-D problem; keep in lockstep with
# make_cfg_3d_xh() in conv_implicit_gemm_dgrad_emit.c.
_XH_COMMON = dict(
    tile_m=256,
    tile_n=64,
    tile_k=32,
    warp_m=4,
    warp_n=1,
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=16,
    vector_size_a=8,
    vector_size_b=8,
    vector_size_c=8,
    lds_k_pad=0,
    pipeline="mem",
    epilogue="cshuffle",
    tap_outer_k=True,
    tapk_xhalo=True,
    b_kpack=2,
    xhalo_exact=True,
    xh_a_k4=True,
    xh_b_wide=True,
    xh_b_kpad=4,
    lds_xor=True,
    waves_per_eu=2,
)
_XH_FULL = dict(
    tapk_prefetch=3,
    tapk_ilv=4,
    tapk_lds_ahead=1,
    tapk_lds_barrier=True,
    xh_order=6,
    xh_defer=4,
    xh_fast_a=1,
    xh_epi=1,
    tapk_setprio=1,
    xcd_remap=6,
    m_band=11,
    xh_stride_m=True,
    xh_epi_lin=True,
    epi_lds_barriers=True,
    xh_tap_rsrc=True,
    tapk_ztrim=2,
    cshuffle_m_chunks=2,
)
_GFX942_3D_XH = {
    77: dict(_XH_FULL),
    78: dict(
        tapk_prefetch=3, xh_order=1, tapk_setprio=1, tapk_lds_barrier=True, tapk_ilv=4
    ),
    79: dict(
        tapk_prefetch=1,
        tapk_ztrim=1,
        xh_fast_a=1,
        xh_tap_rsrc=True,
        xh_epi=1,
        tapk_lds_ahead=1,
        xh_defer=2,
    ),
    80: dict(_XH_FULL, tapk_ztrim=0),
    # 81-83: the y-ring (xh_yring) forms
    81: dict(_XH_FULL, xh_order=7, xh_yring=130, tapk_ztrim=1, no_agpr=True),
    82: dict(_XH_FULL, xh_yring=66, tapk_ztrim=1, no_agpr=True, xh_defer=0),
    83: dict(_XH_FULL, xh_yring=66, tapk_ztrim=1, xh_order=1, xh_defer=2),
}
# 95-96: xh_fast_a without xh_tap_rsrc (the tap offset per load) on 77 and 79.
_GFX942_3D_XH[95] = dict(_GFX942_3D_XH[77], xh_tap_rsrc=False)
_GFX942_3D_XH[96] = dict(_GFX942_3D_XH[79], xh_tap_rsrc=False)
# 110: tapk_setprio 2 (phase-clustered loads / LDS reads / MFMAs) on 77's loop
# without tapk_ilv / tapk_lds_ahead / xh_defer / xh_stride_m.
_XH_PLAIN = dict(tapk_ilv=0, tapk_lds_ahead=0, xh_defer=0, xh_stride_m=False)
_GFX942_3D_XH[110] = dict(_GFX942_3D_XH[77], tapk_setprio=2, **_XH_PLAIN)
# 111: the y-ring (82) through the per-tap MFMA phase (no lds_ahead / defer).
_GFX942_3D_XH[111] = dict(_GFX942_3D_XH[82], **_XH_PLAIN)
# 112: 77 with the B tiles swizzled under lds_xor instead of padded (xh_b_kpad 0).
_GFX942_3D_XH[112] = dict(_GFX942_3D_XH[77], xh_b_kpad=0)
# 113-115: the W-column store mask (xh_dx_wmask) on 77 (linear store with row
# and plane gaps), on 77 without xh_epi_lin (masked descriptor store) and on
# the y-ring 81.
_GFX942_3D_XH[113] = dict(_GFX942_3D_XH[77], xh_dx_wmask=True)
_GFX942_3D_XH[114] = dict(_GFX942_3D_XH[77], xh_dx_wmask=True, xh_epi_lin=False)
_GFX942_3D_XH[115] = dict(_GFX942_3D_XH[81], xh_dx_wmask=True)
# 118-121: the direct x-halo epilogues on 77's loop: xh_epi 2 (8-byte
# stores; no xh_stride_m), xh_epi 3 (16-byte stores through the xor-16 lane
# exchange), and both with xh_dx_wmask.
_GFX942_3D_XH[118] = dict(
    _GFX942_3D_XH[77], xh_epi=2, xh_epi_lin=False, xh_stride_m=False
)
_GFX942_3D_XH[119] = dict(_GFX942_3D_XH[77], xh_epi=3, xh_epi_lin=False)
_GFX942_3D_XH[120] = dict(_GFX942_3D_XH[118], xh_dx_wmask=True)
_GFX942_3D_XH[121] = dict(_GFX942_3D_XH[119], xh_dx_wmask=True)


def _gfx942_3d_xh_spec(idx: int):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec

    if idx == 80:
        p = ConvProblem(N=1, Hi=16, Wi=16, C=128, K=128, Y=3, X=3, pH=1, pW=0, groups=2)
    else:
        p = ConvProblem(
            N=1,
            Di=16,
            Hi=16,
            Wi=16,
            C=128,
            K=128,
            Z=3,
            Y=3,
            X=3,
            sD=1,
            sH=1,
            sW=1,
            pD=1,
            pH=1,
            pW=0,
            dD=1,
            dH=1,
            dW=1,
            groups=2,
        )
    data = ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16")
    return (
        DgradConvSpec(problem=p, data=data, **{**_XH_COMMON, **_GFX942_3D_XH[idx]}),
        "gfx942",
    )


def _gfx942_3d_pp_spec(idx: int):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec

    p = ConvProblem(
        N=1,
        Di=16,
        Hi=16,
        Wi=16,
        C=128,
        K=128,
        Z=3,
        Y=3,
        X=3,
        sD=1,
        sH=1,
        sW=1,
        pD=1,
        pH=1,
        pW=1,
        dD=1,
        dH=1,
        dW=1,
        groups=2,
    )
    data = ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16")
    return (
        DgradConvSpec(problem=p, data=data, **{**_PP_COMMON, **_GFX942_3D_PP[idx]}),
        "gfx942",
    )


def _gfx942_3d_spec(idx: int):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec

    tile, kw = _GFX942_3D[idx]
    groups = 1 if idx == 64 else 2
    pad_d = 0 if idx == 65 else 1
    if idx == 63:
        p = ConvProblem(
            N=1, Hi=16, Wi=16, C=128, K=128, Y=3, X=3, pH=1, pW=1, groups=groups
        )
    else:
        p = ConvProblem(
            N=1,
            Di=16,
            Hi=16,
            Wi=16,
            C=128,
            K=128,
            Z=3,
            Y=3,
            X=3,
            sD=1,
            sH=1,
            sW=1,
            pD=pad_d,
            pH=1,
            pW=1,
            dD=1,
            dH=1,
            dW=1,
            groups=groups,
        )
    dt = "fp16" if idx == 62 else "bf16"
    base = dict(
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=8,
        lds_k_pad=0,
        pipeline="mem",
        epilogue="cshuffle",
        tap_outer_k=True,
    )
    if tile == "s6":
        base.update(tile_m=128, tile_n=256, tile_k=64, warp_m=2, warp_n=4)
    else:
        base.update(tile_m=256, tile_n=64, tile_k=32, warp_m=4, warp_n=1)
    base.update(kw)
    data = ConvDataSpec(dtype_a=dt, dtype_b=dt, dtype_d=dt)
    return DgradConvSpec(problem=p, data=data, **base), "gfx942"


_GFX942_3D_OFFSET = 14


def _gfx942_3d_any(idx: int):
    """The gfx942 3-D configs below by their own number (from 57)."""
    if idx in _GFX942_3D:
        return _gfx942_3d_spec(idx)
    if idx in _GFX942_3D_PP:
        return _gfx942_3d_pp_spec(idx)
    if idx in _GFX942_3D_XH:
        return _gfx942_3d_xh_spec(idx)
    raise SystemExit(f"unknown config index {idx + _GFX942_3D_OFFSET}")


def main() -> int:
    return run_emit(
        _spec,
        build_implicit_gemm_conv_dgrad,
        usage="usage: conv_implicit_gemm_dgrad_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
