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
from kernels.common._conv_implicit_gemm_common import ConvProblem
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

    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_implicit_gemm_conv_dgrad,
        usage="usage: conv_implicit_gemm_dgrad_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
