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
from kernels.common.conv_implicit_gemm import (
    ConvAccumulatorEpilogue,
    ConvDataSpec,
    ConvProblem,
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
)
from _emit_common import run_emit

import inspect as _inspect
from dataclasses import replace

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
    """Return (spec, arch) for config index `idx`.

    The default epilogue stores one element per lane, so the validator rejects
    it with the vec_c the spec would auto-derive from kpg (8 here). Configs that
    leave vector_size_c unset therefore get vector_size_c=1 -- otherwise they
    do not build and the gate only compares two rejections. Mirrored in
    make_cfg() of conv_implicit_gemm_emit.c.
    """
    spec, arch = _spec_raw(idx)
    if spec.epilogue == "default" and spec.vector_size_c is None:
        spec = replace(spec, vector_size_c=1)
    return spec, arch


def _spec_raw(idx: int):
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
    if idx == 10:
        # cshuffle epilogue with cshuffle_no_alias=True (idx 2 shape): the C tile
        # gets its own exclusive LDS bytes and the step-0 reuse barrier is elided.
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
                cshuffle_no_alias=True,
            ),
            "gfx950",
        )
    if idx == 11:
        # basic pipeline, default epilogue (single-buffer global-read/compute overlap).
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
                pipeline="basic",
                epilogue="default",
            ),
            "gfx950",
        )
    if idx == 12:
        # basic pipeline, cshuffle epilogue.
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
                pipeline="basic",
                epilogue="cshuffle",
            ),
            "gfx950",
        )
    if idx == 13:
        # wavelet pipeline, default epilogue (gfx1250 WMMA).
        # warp_tile_k=32: gfx1250 WMMA requires the 16x16x32 atom.
        # vector_size_c=1: auto-derives vec_c=8 from K=64 which blocks default epilogue.
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, fy=3, fx=3)
        return (
            ImplicitGemmConvSpec(
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
                pipeline="wavelet",
                epilogue="default",
                vector_size_c=1,
                num_load_waves=2,
            ),
            "gfx1250",
        )
    if idx == 14:
        # wavelet pipeline, cshuffle epilogue (gfx1250 WMMA).
        # warp_tile_k=32: gfx1250 WMMA requires the 16x16x32 atom.
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, fy=3, fx=3)
        return (
            ImplicitGemmConvSpec(
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
                pipeline="wavelet",
                epilogue="cshuffle",
                num_load_waves=2,
            ),
            "gfx1250",
        )
    if idx == 15:
        # Grouped conv: groups=4, C=64 -> cpg=16, K=64 -> kpg=16.
        p = _cp(N=2, Hi=14, Wi=14, C=64, K=64, fy=3, fx=3, pH=1, pW=1, groups=4)
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
    if idx == 16:
        # Cardinality-grouped conv: groups=32, C=256 -> cpg=8, K=256 -> kpg=8.
        p = _cp(N=2, Hi=8, Wi=8, C=256, K=256, fy=3, fx=3, pH=1, pW=1, groups=32)
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
    if idx == 17:
        # Pointwise + cshuffle: the pointwise d_addr closure on the cshuffle
        # store (idx 5 covers it on the direct store).
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
                epilogue="cshuffle",
            ),
            "gfx950",
        )
    if idx == 19:
        # unroll_k: the double-buffered 2x K loop over a runtime extent, and its
        # "unroll" name tag. K_gemm = 3*3*64 = 576 is 9 tiles of 64 -- odd, so
        # the last step's second tile reads past K_gemm and must read zero.
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
                epilogue="cshuffle",
                unroll_k=True,
            ),
            "gfx950",
        )
    if idx == 18:
        # Rejected by both validators: an explicit vector_size_a/b of 8 on a
        # 16x32 A tile gives 64 chunks for a 128-thread block, which the loader
        # cannot split evenly (coalesced_load_reason).
        p = _cp(N=8, Hi=56, Wi=56, C=64, K=64, fy=3, fx=3)
        return (
            ImplicitGemmConvSpec(
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
                vector_size_b=8,
            ),
            "gfx950",
        )
    if idx == 20:
        # gfx942 (CDNA3) async_dma: same shape as config 3, but on an arch whose
        # buffer_load_lds moves only a dword (not CDNA4's b128) per lane --
        # covers the arch-width cap in the C++ async-conv port, which previously
        # had parity coverage only on gfx950. warp_tile is 16x16x16 (not config
        # 3's 32x32x16): that atom is CDNA4-only, gfx942's MFMA catalog tops
        # out at 16x16x16 / 32x32x8.
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
                pipeline="mem",
                epilogue="default",
                async_dma=True,
            ),
            "gfx942",
        )
    # --- depthwise + merged groups -------------------------------------------
    # Every config below is depthwise (C == K == groups, so cpg == kpg == 1),
    # which is the only shape group_merge admits. 21 is the unmerged control:
    # without it the merged configs would have nothing to differ *from*, and the
    # depthwise path itself carried no structural coverage at all.
    #
    # 22-26 span the axes that change emitted IR under merge:
    #   - the shift/mask width (log2 Gm) of the merged k decode,
    #   - Gm == groups (one merged group, the grouped decode still engaged),
    #   - which epilogue consumes the merged dims,
    #   - and the unroll_k K loop over the in-kernel merged extent.
    if idx in (21, 22, 23, 24, 25, 26):
        # groups=64, C=64 -> cpg=1, K=64 -> kpg=1. M = 2*14*14 = 392.
        p = _cp(N=2, Hi=14, Wi=14, C=64, K=64, fy=3, fx=3, pH=1, pW=1, groups=64)
        # 21 is the unmerged depthwise control. 24 keeps the *direct* epilogue by
        # pinning vector_size_c=1 -- merged kpg would otherwise auto-derive
        # vec_c > 1, which the validator turns into a cshuffle requirement, so
        # without the pin no merged config would exercise the direct epilogue.
        gm = {21: 1, 22: 8, 23: 32, 24: 4, 25: 64, 26: 8}[idx]
        epi = "default" if idx in (21, 24) else "cshuffle"
        kw = {"vector_size_c": 1} if idx == 24 else {}
        if idx == 26:
            kw["unroll_k"] = True
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
                epilogue=epi,
                group_merge=gm,
                **kw,
            ),
            "gfx950",
        )
    if idx in (27, 28):
        # The 1-wide cshuffle store (store_vec == 1): 27 is merged Gm=8 pinned
        # to vector_size_c=1, 28 the unmerged depthwise control whose open
        # store width derives from kpg == 1. Same depthwise problem as 21-26.
        p = _cp(N=2, Hi=14, Wi=14, C=64, K=64, fy=3, fx=3, pH=1, pW=1, groups=64)
        kw = {"group_merge": 8, "vector_size_c": 1} if idx == 27 else {}
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
                **kw,
            ),
            "gfx950",
        )
    if idx in (29, 30, 31, 32):
        # Group merging on WMMA (wave32). Same depthwise problem as 21-28.
        # 29: gfx1250 mem/cshuffle, 30: gfx1250 wavelet/cshuffle, 31: gfx1250
        # direct epilogue (vector_size_c pinned to 1), 32: gfx1151 (gfx11
        # WMMA, 16x16x16 atom) mem/cshuffle.
        p = _cp(N=2, Hi=14, Wi=14, C=64, K=64, fy=3, fx=3, pH=1, pW=1, groups=64)
        gm = {29: 8, 30: 16, 31: 4, 32: 8}[idx]
        kw = {"vector_size_c": 1} if idx == 31 else {}
        if idx == 30:
            kw["num_load_waves"] = 2
        return (
            ImplicitGemmConvSpec(
                problem=p,
                tile_m=32,
                tile_n=32,
                tile_k=32,
                warp_m=1,
                warp_n=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16 if idx == 32 else 32,
                wave_size=32,
                pipeline="wavelet" if idx == 30 else "mem",
                epilogue="default" if idx == 31 else "cshuffle",
                group_merge=gm,
                **kw,
            ),
            "gfx1151" if idx == 32 else "gfx1250",
        )
    if idx in (33, 34, 35):
        # Grid-per-group WMMA (unmerged) on gfx1250 grouped problems.
        # 33: groups=4 (cpg=kpg=16) mem/cshuffle, 34:
        # depthwise (cpg=kpg=1) mem/default, 35: groups=4 wavelet/cshuffle.
        if idx == 34:
            p = _cp(N=2, Hi=14, Wi=14, C=64, K=64, fy=3, fx=3, pH=1, pW=1, groups=64)
        else:
            p = _cp(N=2, Hi=14, Wi=14, C=64, K=64, fy=3, fx=3, pH=1, pW=1, groups=4)
        kw = {"num_load_waves": 2} if idx == 35 else {}
        return (
            ImplicitGemmConvSpec(
                problem=p,
                tile_m=32,
                tile_n=32,
                tile_k=32,
                warp_m=2,
                warp_n=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
                wave_size=32,
                pipeline="wavelet" if idx == 35 else "mem",
                epilogue="default" if idx == 34 else "cshuffle",
                **kw,
            ),
            "gfx1250",
        )
    if idx in (36, 37, 38):
        # Grouped pointwise (1x1, stride 1, no pad): the flat fast path with the
        # group's channel slab on A, the within-group filter bound on B and the
        # group's columns in D's K-wide rows. groups=4, cpg=kpg=16. 36: direct
        # store, 37: cshuffle store, 38: async_dma loaders.
        p = _cp(N=2, Hi=14, Wi=14, C=64, K=64, fy=1, fx=1, groups=4)
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
                epilogue="cshuffle" if idx == 37 else "default",
                async_dma=(idx == 38),
            ),
            "gfx950",
        )
    if idx == 39:
        # Grouped pointwise on the WMMA direct store, whose C++ port spells the
        # pointwise D offset out inline rather than through d_addr.
        p = _cp(N=2, Hi=14, Wi=14, C=64, K=64, fy=1, fx=1, groups=4)
        return (
            ImplicitGemmConvSpec(
                problem=p,
                tile_m=32,
                tile_n=32,
                tile_k=32,
                warp_m=2,
                warp_n=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
                wave_size=32,
                pipeline="mem",
                epilogue="default",
            ),
            "gfx1250",
        )
    if 40 <= idx <= 53:
        return _split_k_spec(idx)
    if 54 <= idx <= 65:
        return _tf32_spec(idx)
    if 66 <= idx <= 73:
        return _mfma_split_k_spec(idx)
    if 74 <= idx <= 81:
        return _mfma_gate_spec(idx)
    if 82 <= idx <= 85:
        return _tf32_split_k_spec(idx)
    raise SystemExit(f"unknown config index {idx}")


# Split-K forward (split_k > 1) and its single-launch cluster form
# (split_k_cluster) on gfx1250 WMMA, plus one gfx950 MFMA split. Keep in
# lockstep with make_cfg_split_k() in conv_implicit_gemm_emit.c.
#   40: wavelet/cshuffle, 16x16x64 tile (small M, deep K)
#   41: mem/cshuffle, 64x64 tile, 2x2 warps
#   42: mem/default (vector_size_c=1), 32x32x32 tile
#   43: grouped (groups=2) wavelet/cshuffle
#   44: cluster S=4, mem/cshuffle, 16x16x64 tile
#   45: cluster S=8, mem/cshuffle, 64x64 tile
#   46: cluster S=15, mem/cshuffle, 16x16x64 tile
#   47: cluster S=4, grouped (groups=2), 32x32 tile, 1x2 warps
#   48: cluster S=12, mem/default, 32x32x32 tile
#   49: split_k=2 on gfx950 (MFMA mem/cshuffle, 64x64 tile)
#   50: split_k=2 with a bias acc_epilogue: rejected by both engines
#   51: split_k=0: rejected by both engines
#   52: cluster S=4 with K=kpg=10 (not a multiple of 4): rejected by both
#   53: cluster S=4 on gfx1201 (no cluster launch): rejected by both engines
_SMALL_M = dict(N=1, Hi=7, Wi=7, C=256, K=64, fy=3, fx=3, pH=1, pW=1)
_CSH_64 = dict(N=2, Hi=12, Wi=12, C=128, K=96, fy=3, fx=3, pH=1, pW=1)
_DEFAULT = dict(N=1, Hi=9, Wi=9, C=64, K=40, fy=3, fx=3, pH=1, pW=1)
_GROUPED = dict(N=1, Hi=10, Wi=10, C=128, K=64, fy=3, fx=3, pH=1, pW=1, groups=2)
_ODD_K = dict(N=1, Hi=7, Wi=7, C=256, K=10, fy=3, fx=3, pH=1, pW=1)


def _split_k_spec(idx: int):
    base = dict(
        tile_m=16,
        tile_n=16,
        tile_k=64,
        warp_m=1,
        warp_n=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
        wave_size=32,
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=8,
        pipeline="wavelet",
        epilogue="cshuffle",
        num_load_waves=1,
        split_k=2,
    )
    tile_64 = dict(tile_m=64, tile_n=64, warp_m=2, warp_n=2)
    tile_def = dict(
        tile_m=32,
        tile_n=32,
        tile_k=32,
        warp_m=2,
        warp_n=1,
        epilogue="default",
        vector_size_c=1,
    )
    prob, kw = {
        40: (_SMALL_M, {}),
        41: (_CSH_64, dict(tile_64, pipeline="mem")),
        42: (_DEFAULT, dict(tile_def, pipeline="mem")),
        43: (_GROUPED, {}),
        44: (_SMALL_M, dict(pipeline="mem", split_k_cluster=4)),
        45: (_CSH_64, dict(tile_64, pipeline="mem", split_k_cluster=8)),
        46: (_SMALL_M, dict(pipeline="mem", split_k_cluster=15)),
        47: (
            _GROUPED,
            dict(
                tile_m=32,
                tile_n=32,
                warp_m=1,
                warp_n=2,
                pipeline="mem",
                split_k_cluster=4,
            ),
        ),
        48: (_DEFAULT, dict(tile_def, pipeline="mem", split_k_cluster=12)),
        49: (
            _CSH_64,
            dict(
                tile_m=64,
                tile_n=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                wave_size=64,
                pipeline="mem",
            ),
        ),
        50: (
            _SMALL_M,
            dict(pipeline="mem", acc_epilogue=ConvAccumulatorEpilogue(bias=1.0)),
        ),
        51: (_SMALL_M, dict(split_k=0)),
        52: (_ODD_K, dict(pipeline="mem", split_k_cluster=4, vector_size_c=2)),
        53: (_SMALL_M, dict(pipeline="mem", split_k_cluster=4, warp_tile_k=16)),
    }[idx]
    cfg = dict(base, **kw)
    return (
        ImplicitGemmConvSpec(problem=_cp(**prob), **cfg),
        {49: "gfx950", 53: "gfx1201"}.get(idx, "gfx1250"),
    )


# TF32 compute (ConvDataSpec(mma_dtype="tf32")): fp32 storage, each LDS
# fragment bitcast to vec<tf32xN> for the arch's TF32 atom -- native XF32 on
# gfx942 (16x16x8, 32x32x4), BF16x3 emulation on gfx950 (16x16x32, 32x32x16).
# 64x64x32 tile, 2x2 warps. Keep in lockstep with make_cfg_tf32() in
# conv_implicit_gemm_emit.c.
#   54: gfx950 16x16x32, 2-D, mem/cshuffle
#   55: gfx950 32x32x16, 2-D, compv4/default (the scheduling-hint counts)
#   56: gfx942 16x16x8, 2-D, compv3/cshuffle
#   57: gfx942 32x32x4, 2-D, compv4/default
#   58: gfx950 16x16x32, 3-D, compv3/cshuffle
#   59: gfx942 16x16x8, 3-D, mem/default
#   60: gfx950 32x32x16, 2-D, async_dma mem/cshuffle
#   61: gfx942 32x32x4, 2-D, unroll_k compv4/cshuffle
#   62: gfx950 16x16x32, 2-D, basic/default
#   63: gfx942 16x16x8, depthwise group_merge=4, mem/cshuffle
#   64: gfx942 16x16x32 (gfx950's atom): rejected by both engines
#   65: gfx950 16x16x32 with lds_k_pad=2 (misaligned row stride): rejected
_TF32 = ConvDataSpec("fp32", "fp32", "fp32", mma_dtype="tf32")
_TF32_2D = dict(N=2, Hi=14, Wi=14, C=32, K=64, fy=3, fx=3, pH=1, pW=1)
_TF32_3D = dict(
    N=1, Hi=6, Wi=6, C=32, K=32, fy=3, fx=3, pH=1, pW=1, Di=4, Z=3, sD=1, pD=1, dD=1
)
_TF32_DW = dict(N=2, Hi=14, Wi=14, C=64, K=64, fy=3, fx=3, pH=1, pW=1, groups=64)


def _tf32_spec(idx: int):
    prob, arch, m, k, pipeline, epilogue, kw = {
        54: (_TF32_2D, "gfx950", 16, 32, "mem", "cshuffle", {}),
        55: (_TF32_2D, "gfx950", 32, 16, "compv4", "default", {}),
        56: (_TF32_2D, "gfx942", 16, 8, "compv3", "cshuffle", {}),
        57: (_TF32_2D, "gfx942", 32, 4, "compv4", "default", {}),
        58: (_TF32_3D, "gfx950", 16, 32, "compv3", "cshuffle", {}),
        59: (_TF32_3D, "gfx942", 16, 8, "mem", "default", {}),
        60: (_TF32_2D, "gfx950", 32, 16, "mem", "cshuffle", dict(async_dma=True)),
        61: (_TF32_2D, "gfx942", 32, 4, "compv4", "cshuffle", dict(unroll_k=True)),
        62: (_TF32_2D, "gfx950", 16, 32, "basic", "default", {}),
        63: (_TF32_DW, "gfx942", 16, 8, "mem", "cshuffle", dict(group_merge=4)),
        64: (_TF32_2D, "gfx942", 16, 32, "mem", "cshuffle", {}),
        65: (_TF32_2D, "gfx950", 16, 32, "mem", "cshuffle", dict(lds_k_pad=2)),
    }[idx]
    return (
        ImplicitGemmConvSpec(
            problem=_cp(**prob),
            data=_TF32,
            tile_m=64,
            tile_n=64,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=m,
            warp_tile_n=m,
            warp_tile_k=k,
            pipeline=pipeline,
            epilogue=epilogue,
            **kw,
        ),
        arch,
    )


# Split-K forward on gfx950 MFMA (wave64): the plain mem / compv3 / compv4 K
# loops over one slice, the tile / warp-tile axes the AOT grid varies, a
# 4-wide-vector variant, the direct epilogue, and the MFMA loops the split is
# not wired for.
# fp16 only: the C++ forward builder stages A/B as f16 whatever the dtype (see
# test_conv_fwd_split_k.TestEngineAgreement), so bf16 is covered by the
# representative golden and the GPU numerics instead. Keep in lockstep with
# make_cfg_split_k_mfma() in conv_implicit_gemm_emit.c.
#   66: mem/cshuffle, 32x32x64 tile, 2x2 warps, 16x16x32 warp tile
#   67: compv3/cshuffle, 64x128x64 tile, 2x4 warps
#   68: mem/cshuffle, grouped (groups=2), 64x64x64 tile, 4-wide vectors
#   69: compv4/cshuffle, 128x64x64 tile, 4x2 warps
#   70: compv3/default (vector_size_c=1), 32x32x32 tile, 2x1 warps
#   71: basic pipeline: rejected by both engines (mem/compv3/compv4 only)
#   72: mem with async_dma: rejected by both engines
#   73: compv3/cshuffle, 128x128x64 tile, 4x2 warps


def _mfma_split_k_spec(idx: int):
    base = dict(
        tile_m=32,
        tile_n=32,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
        wave_size=64,
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=8,
        pipeline="mem",
        epilogue="cshuffle",
        split_k=2,
    )
    wt32 = dict(warp_tile_m=32, warp_tile_n=32, warp_tile_k=16)
    prob, kw = {
        66: (_SMALL_M, {}),
        67: (
            _CSH_64,
            dict(wt32, tile_m=64, tile_n=128, warp_n=4, pipeline="compv3"),
        ),
        68: (
            _GROUPED,
            dict(
                wt32,
                tile_m=64,
                tile_n=64,
                vector_size_a=4,
                vector_size_b=4,
                vector_size_c=4,
            ),
        ),
        69: (
            _SMALL_M,
            dict(wt32, tile_m=128, tile_n=64, warp_m=4, warp_n=2, pipeline="compv4"),
        ),
        70: (
            _DEFAULT,
            dict(
                tile_k=32,
                warp_n=1,
                pipeline="compv3",
                epilogue="default",
                vector_size_c=1,
            ),
        ),
        71: (_SMALL_M, dict(pipeline="basic")),
        72: (_SMALL_M, dict(async_dma=True)),
        73: (
            _SMALL_M,
            dict(wt32, tile_m=128, tile_n=128, warp_m=4, pipeline="compv3"),
        ),
    }[idx]
    cfg = dict(base, **kw)
    return ImplicitGemmConvSpec(problem=_cp(**prob), **cfg), "gfx950"


# The gfx950 spec gate around the MFMA split: a missing (None) pipeline or
# epilogue, split or not, and the LDS budget that charges compv4 one A/B tile.
# Built on config 66 (mem/cshuffle, 32x32x64 tile, split_k=2). Keep in
# lockstep with make_cfg_mfma_gate() in conv_implicit_gemm_emit.c.
#   74: unsplit, pipeline None: rejected by both engines
#   75: unsplit, epilogue None: rejected by both engines
#   76: split, pipeline None: rejected by both engines
#   77: split, epilogue None: rejected by both engines
#   78: split compv4/cshuffle, 128x64x256 tile, 4x2 warps, lds_k_pad=0 (fits
#       only because compv4 is charged one A/B tile)
#   79: config 78 unsplit
#   80: config 78 unsplit on compv3 with unroll_k (two A/B tiles): rejected by
#       both engines on the LDS budget
#   81: config 78 at 256x128x256: rejected by both engines on the LDS budget


def _mfma_gate_spec(idx: int):
    spec, arch = _mfma_split_k_spec(66)
    if idx >= 78:
        spec = replace(
            spec,
            tile_m=128,
            tile_n=64,
            tile_k=256,
            warp_m=4,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            lds_k_pad=0,
            pipeline="compv4",
        )
    kw = {
        74: dict(split_k=1, pipeline=None),
        75: dict(split_k=1, epilogue=None),
        76: dict(pipeline=None),
        77: dict(epilogue=None),
        78: {},
        79: dict(split_k=1),
        80: dict(split_k=1, pipeline="compv3", unroll_k=True),
        81: dict(tile_m=256, tile_n=128),
    }[idx]
    return replace(spec, **kw), arch


# fp32 output under the two-stage split: Stage 2 casts to fp16 / bf16 only,
# on every family, and the cluster form folds the slabs with the same cast,
# so both engines reject either. Keep in lockstep with
# make_cfg_tf32_split_k() in conv_implicit_gemm_emit.c.
#   82: config 54 (gfx950 TF32 16x16x32, mem/cshuffle) with split_k=2
#   83: config 56 (gfx942 TF32 16x16x8, compv3/cshuffle) with split_k=2
#   84: config 41 (gfx1250 WMMA, mem/cshuffle, split_k=2) in fp32 storage on
#       the 16x16x4 fp32 atom, 4-wide vectors
#   85: config 84 in the cluster form (split_k_cluster=4): rejected by both


def _tf32_split_k_spec(idx: int):
    if idx in (84, 85):
        spec, arch = _split_k_spec(41)
        return (
            replace(
                spec,
                data=ConvDataSpec("fp32", "fp32", "fp32"),
                warp_tile_k=4,
                vector_size_a=4,
                vector_size_b=4,
                vector_size_c=4,
                split_k_cluster=4 if idx == 85 else 0,
            ),
            arch,
        )
    spec, arch = _tf32_spec({82: 54, 83: 56}[idx])
    return replace(spec, split_k=2), arch


def main() -> int:
    return run_emit(
        _spec,
        build_implicit_gemm_conv,
        usage="usage: conv_implicit_gemm_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
