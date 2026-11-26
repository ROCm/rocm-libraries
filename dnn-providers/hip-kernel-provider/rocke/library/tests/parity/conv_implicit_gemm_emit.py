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
    if idx >= _BIAS_FIRST:
        return _bias_spec(idx)
    if idx >= 54 + _GFX942_3D_OFFSET:
        return _gfx942_3d_spec(idx - _GFX942_3D_OFFSET)
    raise SystemExit(f"unknown config index {idx}")


# Per-output-channel bias (ImplicitGemmConvSpec.bias): an existing config, by
# index, built with bias=True. Keep in lockstep with make_cfg_bias() in
# conv_implicit_gemm_emit.c.
#   192-195: gfx950 MFMA: mem/default, compv4/default, depthwise merged
#            (group_merge 8) cshuffle, grouped pointwise cshuffle
#   196-200: WMMA: gfx1250 grouped mem/cshuffle, grouped wavelet/cshuffle,
#            depthwise mem/default, grouped pointwise direct; gfx1151 merged
#   201:     gfx950 TF32 (fp32 storage: an fp32 bias)
#   202-207: gfx942 3-D: fp16 cshuffle, bf16 grouped cshuffle, bf16 32x32
#            default, epi_quad_xpose 1, the x-halo loop (epi_quad_xpose 4),
#            epi_store_fast
#   208-210: split-K: gfx950 workspace split (the Stage 2 fold adds the
#            bias), gfx942 3-D atomic split (rejected by both engines),
#            gfx1250 wavelet workspace split (Stage 2 adds the bias)
#   211:     192 with an acc_epilogue (scalar bias + relu) after the bias
#   212-219: the two-stage and cluster split-K forward: gfx1250 WMMA
#            mem/cshuffle, mem/default, grouped wavelet (Stage 2 adds the
#            bias); cluster S=4, grouped cluster, cluster mem/default (the
#            cluster fold adds it); gfx950 MFMA grouped 4-wide, compv3/default
#            (Stage 2 adds it)
#   220-224: gfx942 3-D split_det: mode 1, mode 2 + split_tail (the owner's
#            fold and the whole-K tiles add the bias), mode 2 + split_tail +
#            m_band, mode 3 (the workspace fold adds it), mode 3 +
#            split_xcd_remap
#   225-229: rejected by both engines: the gfx942 3-D atomic split, stream-K,
#            split_hybrid + split_det 1, split_hybrid + split_det 3, the 2-D
#            split_atomic
_BIAS_FIRST = 192
_BIAS_BASE = {
    192: 0,
    193: 1,
    194: 22,
    195: 37,
    196: 33,
    197: 35,
    198: 34,
    199: 39,
    200: 32,
    201: 54,
    202: 86,
    203: 87,
    204: 88,
    205: 186,
    206: 146,
    207: 168,
    208: 49,
    209: 189,
    210: 40,
    211: 0,
    212: 41,
    213: 42,
    214: 43,
    215: 44,
    216: 47,
    217: 48,
    218: 68,
    219: 70,
    220: 126,  # gfx942 3-D config 94
    221: 124,  # gfx942 3-D config 92
    222: 128,  # gfx942 3-D config 96
    223: 137,  # gfx942 3-D config 105
    224: 136,  # gfx942 3-D config 104
    225: 95,  # gfx942 3-D config 63
    226: 142,  # gfx942 3-D config 110
    227: 143,  # gfx942 3-D config 111
    228: 140,  # gfx942 3-D config 108
    229: 183,  # gfx942 3-D config 151
}


def _bias_spec(idx: int):
    if idx not in _BIAS_BASE:
        raise SystemExit(f"unknown config index {idx}")
    spec, arch = _spec(_BIAS_BASE[idx])
    kw = dict(bias=True)
    if idx == 211:
        kw["acc_epilogue"] = ConvAccumulatorEpilogue(bias=1.0, relu=True)
    return replace(spec, **kw), arch


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


_GFX942_3D_OFFSET = 32


# gfx942 (wave64 MFMA) 3-D NDHWC configs on the 3x3x3 / pad-1 problem the
# AOT cache builds its 3-D binaries against: the 3-D split A/B addressing
# (every non-pointwise 3-D spec), bf16 operands through the MFMA path, and the
# opt-in K-loop knobs layered on top. Keep in lockstep with make_cfg_3d() in
# conv_implicit_gemm_emit.c. The numbers below are _gfx942_3d_spec's own; it
# serves parity config N as its config N - _GFX942_3D_OFFSET (they follow the
# TF32 and MFMA split-K configs 54-85).
#   54: fp16, groups=1, t256x64x32 w4x1 a16x16x16, mem/cshuffle
#   55: bf16, groups=2, t256x64x32 w4x1 a16x16x16, mem/cshuffle
#   56: bf16, groups=1, stride 2, t128x128x32 w2x2 a32x32x8, mem/default
#   57: bf16, groups=2, 1x1x1 pointwise (no split addressing), t128x64x32 w2x2
#   58: 55 + xcd_remap=6
#   59: 55 + xcd_remap=6, m_band=32
#   60: 54 + xcd_remap=8, waves_per_eu=2
#   61: 55 + xcd_remap=6 + chiplet_swizzle: rejected by both engines
#   62: 55 + m_band=32 without xcd_remap: rejected by both engines
#   63: 55 + split_k=2 (wave64 MFMA atomic split-K: ks/ks_count, atomics into D)
#   64: 54 + split_k=4
#   65: 55 + split_k=2, xcd_remap=6, m_band=32
#   66: 56 + split_k=2 (default epilogue): rejected by both engines
#   67: 55 + reg_prefetch=1
#   68: 55 + reg_prefetch=2
#   69: 55 + reg_prefetch=1, lds_xor
#   70: 55 + reg_prefetch=1, lds_xor, lds_kpack_read, lds_xor_rowkey, xcd_remap=6, m_band=32
#   71: 55 + split_k=2, reg_prefetch=1, lds_xor, lds_kpack_read
#   72: 54 + compv3, reg_prefetch=2, lds_kpack_read (schedule hints, k-packed counts)
#   73: 55 + lds_xor without a reg_prefetch loop: rejected by both engines
#   74: 55 + reg_prefetch=1, lds_xor, lds_xor_rowkey without lds_kpack_read: rejected
#   75: 70 + uniform_k_decode (the uniform-tap loop + plain fallback)
#   76: 55 + uniform_k_decode (no register prefetch)
#   77: 55 + uniform_k_decode, reg_prefetch=2, uk_tap_fast, uk_nopad, fb_skip_empty
#   78: 71 + uniform_k_decode, uk_tap_fast, uk_nopad, fb_skip_empty, xcd_remap=6 (split slices)
#   79: 55 + uniform_k_decode, reg_prefetch=1, lds_xor, uk_tap_fast, uk_chunk_group=2, uk_nopad
#   80: 56 + uniform_k_decode, reg_prefetch=1, lds_kpack_read (stride 2, 32x32x8, default epilogue)
#   81: 57 + uniform_k_decode (pointwise): rejected by both engines
#   82: 71 + uniform_k_decode, uk_tap_fast, uk_chunk_group=2: rejected by both engines
#   83: 55 at t256x128x64 w4x2 + split_k=2, uniform_k_decode, reg_prefetch=1, lds_xor,
#       lds_kpack_read, uk_tap_fast, uk_nopad, fb_skip_empty, xcd_remap=6 (8 waves)
#   84: 83 + rpf_pingpong=5, vmem_interleave=5, pp_b_double, pp_split_fetch=3
#   85: 83 + rpf_pingpong=5, vmem_interleave=6, pp_b_double, pp_split_fetch=4
#   86: 83 + rpf_pingpong=4, rpf_setprio=1, pp_pre_mfma=2
#   87: 83 + rpf_pingpong=1
#   88: 83 + rpf_pingpong=3, pp_b_double: rejected by both engines (modes 2 / 3
#       do not wait for the [mem A] LDS reads)
#   89: 83 + rpf_pingpong=5, vmem_interleave=5, pp_split_fetch=10
#   90: 54 + compv3, reg_prefetch=1, vmem_interleave=2, rpf_setprio=1
#   91: 83 + rpf_pingpong=5 without vmem_interleave: rejected by both engines
#   92: 84 + split_tail, split_atomic_pairs, split_det=2
#   93: 84 + split_tail, split_atomic_pairs (atomic pair stores, plain stores of full tiles)
#   94: 85 + split_det=1
#   95: 83 + split_atomic_pairs
#   96: 83 + split_tail, split_det=2, m_band=32
#   97: 84 + uk_chunk_group=2 (split slices in the grouped tap order)
#   98: 54 + split_k=2, split_det=2 (fp16 D): rejected by both engines
#   99: 92 + cshuffle_m_chunks=2, epi_lds_barriers
#  100: 93 + cshuffle_m_chunks=4 (chunked atomic pairs)
#  101: 83 + cshuffle_m_chunks=2 (chunked packed atomics)
#  102: 55 + reg_prefetch=1, cshuffle_m_chunks=2, epi_lds_barriers (chunked store)
#  103: 54 + cshuffle_m_chunks=2 (fp16 D): rejected by both engines
#  104: 85 without xcd_remap + split_xcd_remap=6, split_atomic_pairs, split_det=3
#       (per-slice f32 slabs)
#  105: 83 + split_det=3, cshuffle_m_chunks=2
#  106: 84 without xcd_remap + split_xcd_remap=6
#  107: 83 + split_det=3, split_tail: rejected by both engines
#  108: t256x256x32 w4x2, split_k=2, uniform_k_decode, reg_prefetch=1, lds_xor,
#       uk_tap_fast, cshuffle_m_chunks=4, uk_nopad, vmem_interleave=5,
#       lds_kpack_read, uk_chunk_group=2, rpf_pingpong=5, lds_xor_rowkey,
#       split_atomic_pairs, pp_split_fetch=10, fb_skip_empty, split_hybrid=16,
#       split_det=3
#  109: 108 with split_hybrid=8, split_det=2
#  110: 84 without xcd_remap + split_streamk (one segment per tile touched)
#  111: 83 without xcd_remap + split_hybrid=4, split_det=1
#  112: 110 + split_det=2: rejected by both engines
#  113: 108 with split_streamk in place of split_hybrid: rejected by both engines
#  114: 55 + uniform_k_decode, reg_prefetch=1, xcd_remap=6, lds_xor, uk_xhalo=8,
#       rpf_setprio=2, lds_xor_rowkey, xh_wait_smem, epi_quad_xpose=4, xh_stride_m,
#       fxh_dma=1036, lds_kpack_read (the wave-private x-halo DMA loop and the
#       quad-transpose epilogue)
#  115: 114 with uk_xhalo=4, rpf_setprio=1, no_agpr, m_band=32, fxh_dma=2016
#  116: 114 + uk_tap_fast
#  117: 114 with epi_quad_xpose=3
#  118: 55 + xh_stride_m without uk_xhalo: rejected by both engines
#  119: 116 + xh_yring=14 (the input-row ring, early reload)
#  120: 116 + xh_yring=2
#  121-123: 55 + reg_prefetch=1 with rpf_lds_barrier / rpf_fetch_late / sched_strategy
#  124-127: 56 / 55 + lds_ahead / defer_mfma / defer_mfma_per_write
#  128-130: 55 + the plain uk_xhalo loop (defer_mfma, lds_ahead, chunk groups)
#  131-132: 55 + the plain uk_xhalo loop with xh_stride_m
#  133-134: 55 + split_k 2 with split_atomic_agent (+ split_atomic_pairs)
#  135-141: 55 + split_atomic_fast, epi_store_fast, split_flag_store / split_zero_first
#  142-143: 55 + split_tail_compact (+ split_zero_first)
#  144-145: 84 / 87 + pp_end_guard
#  146-147: 55 / 54 + split_addr_k_decode
#  148-150: 115 / 119 / 114 + xh_pad (the x-halo DMA loop on padded inputs)
#  151: 55 on the 2-D 16x16 problem + split_k=2, split_atomic (the wave64 MFMA
#       atomic split-K on a 2-D problem)
#  152: 55 + split_k=2, split_atomic: rejected by both engines (3-D)
#  153: 151 without split_atomic (the two-stage workspace split) +
#       uniform_k_decode: rejected by both engines (an atomic-split K loop)
#  154: 57 + epi_quad_xpose=1 (the quad-transpose store through the pointwise
#       D address)
#  155: 114 with warp_m=1 (one wave: the x-halo B DMA list of the
#       wave-private loop without the 4-wave form; a 65-segment wave range)
#  156: 114 with tile_n=128 (the same B DMA list with four waves)
#  157: 57 + split_k=2: rejected by both engines (pointwise atomic split)
#  158: 111 with split_det=0 + split_zero_first: rejected by both engines
#       (split_hybrid runs the stream-K loop)
#  159: 94 without xcd_remap + split_xcd_remap=6: rejected by both engines
#       (split_det 1 owner dispatched before its slices)
def _gfx942_3d_spec(idx: int):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec

    def prob(groups=1, f=3, s=1):
        pad = (f - 1) // 2
        return ConvProblem(
            N=1,
            Di=16,
            Hi=16,
            Wi=16,
            C=128,
            K=128,
            Z=f,
            Y=f,
            X=f,
            sD=s,
            sH=s,
            sW=s,
            pD=pad,
            pH=pad,
            pW=pad,
            dD=1,
            dH=1,
            dW=1,
            groups=groups,
        )

    def data(dt):
        return ConvDataSpec(dtype_a=dt, dtype_b=dt, dtype_d=dt)

    t256 = dict(
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
        pipeline="mem",
        epilogue="cshuffle",
    )
    cfgs = {
        54: (prob(), "fp16", t256),
        55: (prob(groups=2), "bf16", t256),
        56: (
            prob(s=2),
            "bf16",
            dict(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=8,
                pipeline="mem",
                epilogue="default",
            ),
        ),
        57: (
            prob(groups=2, f=1),
            "bf16",
            dict(
                tile_m=128,
                tile_n=64,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
                vector_size_a=8,
                vector_size_b=8,
                vector_size_c=8,
                pipeline="mem",
                epilogue="cshuffle",
            ),
        ),
    }
    knobs = {
        58: (55, dict(xcd_remap=6)),
        59: (55, dict(xcd_remap=6, m_band=32)),
        60: (54, dict(xcd_remap=8, waves_per_eu=2)),
        61: (55, dict(xcd_remap=6, chiplet_swizzle=True)),
        62: (55, dict(m_band=32)),
        63: (55, dict(split_k=2)),
        64: (54, dict(split_k=4)),
        65: (55, dict(split_k=2, xcd_remap=6, m_band=32)),
        66: (56, dict(split_k=2)),
        67: (55, dict(reg_prefetch=1)),
        68: (55, dict(reg_prefetch=2)),
        69: (55, dict(reg_prefetch=1, lds_xor=True)),
        70: (
            55,
            dict(
                reg_prefetch=1,
                lds_xor=True,
                lds_kpack_read=True,
                lds_xor_rowkey=True,
                xcd_remap=6,
                m_band=32,
            ),
        ),
        71: (55, dict(split_k=2, reg_prefetch=1, lds_xor=True, lds_kpack_read=True)),
        72: (54, dict(pipeline="compv3", reg_prefetch=2, lds_kpack_read=True)),
        73: (55, dict(lds_xor=True)),
        74: (55, dict(reg_prefetch=1, lds_xor=True, lds_xor_rowkey=True)),
        75: (
            55,
            dict(
                uniform_k_decode=True,
                reg_prefetch=1,
                lds_xor=True,
                lds_kpack_read=True,
                lds_xor_rowkey=True,
                xcd_remap=6,
                m_band=32,
            ),
        ),
        76: (55, dict(uniform_k_decode=True)),
        77: (
            55,
            dict(
                uniform_k_decode=True,
                reg_prefetch=2,
                uk_tap_fast=True,
                uk_nopad=True,
                fb_skip_empty=True,
            ),
        ),
        78: (
            55,
            dict(
                split_k=2,
                reg_prefetch=1,
                lds_xor=True,
                lds_kpack_read=True,
                uniform_k_decode=True,
                uk_tap_fast=True,
                uk_nopad=True,
                fb_skip_empty=True,
                xcd_remap=6,
            ),
        ),
        79: (
            55,
            dict(
                uniform_k_decode=True,
                reg_prefetch=1,
                lds_xor=True,
                uk_tap_fast=True,
                uk_chunk_group=2,
                uk_nopad=True,
            ),
        ),
        80: (56, dict(uniform_k_decode=True, reg_prefetch=1, lds_kpack_read=True)),
        81: (57, dict(uniform_k_decode=True)),
        82: (
            55,
            dict(
                split_k=2,
                reg_prefetch=1,
                lds_xor=True,
                lds_kpack_read=True,
                uniform_k_decode=True,
                uk_tap_fast=True,
                uk_chunk_group=2,
            ),
        ),
        90: (
            54,
            dict(pipeline="compv3", reg_prefetch=1, vmem_interleave=2, rpf_setprio=1),
        ),
    }
    _s3 = dict(
        tile_n=128,
        tile_k=64,
        warp_n=2,
        split_k=2,
        uniform_k_decode=True,
        reg_prefetch=1,
        lds_xor=True,
        lds_kpack_read=True,
        uk_tap_fast=True,
        uk_nopad=True,
        fb_skip_empty=True,
        xcd_remap=6,
    )
    _pp = {
        83: {},
        84: dict(rpf_pingpong=5, vmem_interleave=5, pp_b_double=True, pp_split_fetch=3),
        85: dict(rpf_pingpong=5, vmem_interleave=6, pp_b_double=True, pp_split_fetch=4),
        86: dict(rpf_pingpong=4, rpf_setprio=1, pp_pre_mfma=2),
        87: dict(rpf_pingpong=1),
        88: dict(rpf_pingpong=3, pp_b_double=True),
        89: dict(rpf_pingpong=5, vmem_interleave=5, pp_split_fetch=10),
        91: dict(rpf_pingpong=5),
    }
    _p84 = dict(rpf_pingpong=5, vmem_interleave=5, pp_b_double=True, pp_split_fetch=3)
    _p85 = dict(rpf_pingpong=5, vmem_interleave=6, pp_b_double=True, pp_split_fetch=4)
    _pp.update(
        {
            92: dict(_p84, split_tail=True, split_atomic_pairs=True, split_det=2),
            93: dict(_p84, split_tail=True, split_atomic_pairs=True),
            94: dict(_p85, split_det=1),
            95: dict(split_atomic_pairs=True),
            96: dict(split_tail=True, split_det=2, m_band=32),
            97: dict(_p84, uk_chunk_group=2),
        }
    )
    knobs[98] = (54, dict(split_k=2, split_det=2))
    _pp.update(
        {
            99: dict(
                _p84,
                split_tail=True,
                split_atomic_pairs=True,
                split_det=2,
                cshuffle_m_chunks=2,
                epi_lds_barriers=True,
            ),
            100: dict(
                _p84, split_tail=True, split_atomic_pairs=True, cshuffle_m_chunks=4
            ),
            101: dict(cshuffle_m_chunks=2),
        }
    )
    knobs[102] = (55, dict(reg_prefetch=1, cshuffle_m_chunks=2, epi_lds_barriers=True))
    _pp.update(
        {
            104: dict(
                _p85,
                xcd_remap=0,
                split_xcd_remap=6,
                split_atomic_pairs=True,
                split_det=3,
            ),
            105: dict(split_det=3, cshuffle_m_chunks=2),
            106: dict(_p84, xcd_remap=0, split_xcd_remap=6),
            107: dict(split_det=3, split_tail=True),
        }
    )
    knobs[103] = (54, dict(cshuffle_m_chunks=2))
    _s8 = dict(
        tile_n=256,
        tile_k=32,
        warp_n=2,
        split_k=2,
        uniform_k_decode=True,
        reg_prefetch=1,
        lds_xor=True,
        uk_tap_fast=True,
        cshuffle_m_chunks=4,
        uk_nopad=True,
        vmem_interleave=5,
        lds_kpack_read=True,
        uk_chunk_group=2,
        rpf_pingpong=5,
        lds_xor_rowkey=True,
        split_atomic_pairs=True,
        pp_split_fetch=10,
        fb_skip_empty=True,
        split_hybrid=16,
        split_det=3,
    )
    knobs[108] = (55, _s8)
    knobs[109] = (55, dict(_s8, split_hybrid=8, split_det=2))
    knobs[113] = (55, dict(_s8, split_hybrid=0, split_streamk=True, split_det=0))
    _s4 = dict(
        uniform_k_decode=True,
        reg_prefetch=1,
        xcd_remap=6,
        lds_xor=True,
        uk_xhalo=8,
        rpf_setprio=2,
        lds_xor_rowkey=True,
        xh_wait_smem=True,
        epi_quad_xpose=4,
        xh_stride_m=True,
        fxh_dma=1036,
        lds_kpack_read=True,
    )
    knobs[114] = (55, _s4)
    knobs[115] = (
        55,
        dict(_s4, uk_xhalo=4, rpf_setprio=1, no_agpr=True, m_band=32, fxh_dma=2016),
    )
    knobs[116] = (55, dict(_s4, uk_tap_fast=True))
    knobs[117] = (55, dict(_s4, epi_quad_xpose=3))
    knobs[118] = (55, dict(xh_stride_m=True))
    knobs[119] = (55, dict(_s4, uk_tap_fast=True, xh_yring=14))
    knobs[120] = (55, dict(_s4, uk_tap_fast=True, xh_yring=2))
    # 121-123: reg_prefetch=1 loop knobs (rpf_lds_barrier, rpf_fetch_late) and
    # the scheduler strategy attribute.
    knobs[121] = (55, dict(reg_prefetch=1, rpf_lds_barrier=True))
    knobs[122] = (55, dict(uniform_k_decode=True, reg_prefetch=1, rpf_fetch_late=True))
    knobs[123] = (
        55,
        dict(
            reg_prefetch=1, rpf_fetch_late=True, rpf_lds_barrier=True, sched_strategy=2
        ),
    )
    # 124-127: lds_ahead / defer_mfma (+ defer_mfma_per_write) K-loop schedules.
    _lah = dict(
        uniform_k_decode=True,
        reg_prefetch=1,
        xcd_remap=6,
        lds_xor=True,
        uk_tap_fast=True,
        lds_kpack_read=True,
        lds_ahead=True,
    )
    knobs[124] = (56, dict(_lah, pipeline="compv4", defer_mfma=3, vmem_interleave=4))
    knobs[125] = (
        56,
        dict(
            pipeline="compv3",
            reg_prefetch=1,
            lds_ahead=True,
            defer_mfma=2,
            defer_mfma_per_write=3,
        ),
    )
    knobs[126] = (56, dict(pipeline="compv3", reg_prefetch=2, lds_ahead=True))
    knobs[127] = (55, dict(reg_prefetch=1, defer_mfma=1, rpf_setprio=2))
    # 128-130: the plain (register-staged) uk_xhalo loop, with defer_mfma.
    _xp = dict(
        uniform_k_decode=True,
        reg_prefetch=1,
        xcd_remap=6,
        lds_xor=True,
        uk_xhalo=8,
        rpf_setprio=2,
        lds_xor_rowkey=True,
        xh_wait_smem=True,
        epi_quad_xpose=4,
        lds_kpack_read=True,
    )
    knobs[128] = (
        55,
        dict(_xp, defer_mfma=2, vmem_interleave=2, no_agpr=True, m_band=32),
    )
    knobs[129] = (
        55,
        dict(
            _xp,
            uk_xhalo=4,
            rpf_setprio=0,
            uk_tap_fast=True,
            uk_chunk_group=2,
            lds_ahead=True,
            defer_mfma=2,
            vmem_interleave=2,
        ),
    )
    knobs[130] = (55, dict(_xp, epi_quad_xpose=0))
    # 133-134: split_atomic_agent (device-scope packed atomics), plain and
    # with lane-contiguous pairs.
    knobs[133] = (55, dict(split_k=2, split_atomic_agent=True))
    knobs[134] = (55, dict(split_k=2, split_atomic_agent=True, split_atomic_pairs=True))
    # 135: split_atomic_fast (the in-bounds tile's unchecked pair walk).
    knobs[135] = (55, dict(split_k=2, split_atomic_pairs=True, split_atomic_fast=True))
    # 136-137: epi_store_fast (the packed in-bounds tile's constant-step
    # store), plain and on the split-K binary (predicate only).
    knobs[136] = (55, dict(epi_store_fast=True))
    knobs[137] = (
        55,
        dict(reg_prefetch=1, epi_store_fast=True, split_k=2, split_atomic_pairs=True),
    )
    # 138-141: split_flag_store / split_zero_first (no zeroed D), with
    # split_tail and with pair lanes.
    knobs[138] = (55, dict(split_k=2, split_flag_store=True))
    knobs[139] = (55, dict(split_k=2, split_zero_first=True))
    knobs[140] = (55, dict(split_k=2, split_zero_first=True, split_tail=True))
    knobs[141] = (
        55,
        dict(reg_prefetch=1, split_k=2, split_flag_store=True, split_atomic_pairs=True),
    )
    # 142-143: split_tail_compact (the flat grid decoded into block ids),
    # plain and with split_zero_first.
    knobs[142] = (55, dict(split_k=2, split_tail=True, split_tail_compact=True))
    knobs[143] = (
        55,
        dict(
            split_k=2, split_tail=True, split_tail_compact=True, split_zero_first=True
        ),
    )
    # 131-132: the plain uk_xhalo loop with xh_stride_m (the A store's _xa
    # map, each tap's fragments shifted from the previous tap's).
    knobs[131] = (55, dict(_xp, xh_stride_m=True, defer_mfma=2, vmem_interleave=2))
    knobs[132] = (
        55,
        dict(
            _xp,
            xh_stride_m=True,
            epi_quad_xpose=3,
            uk_xhalo=4,
            uk_tap_fast=True,
            uk_chunk_group=2,
            lds_ahead=True,
        ),
    )
    _pp.update(
        {
            110: dict(_p84, xcd_remap=0, split_streamk=True),
            111: dict(xcd_remap=0, split_hybrid=4, split_det=1),
            112: dict(_p84, xcd_remap=0, split_streamk=True, split_det=2),
            # 144-145: pp_end_guard (the uniform-tap prefetch masked past the
            # loop's end) on 84 and on the rpf_pingpong 1 loop (87).
            144: dict(_p84, pp_end_guard=True),
            145: dict(rpf_pingpong=1, pp_end_guard=True),
        }
    )
    # 146-147: split_addr_k_decode (the tile's tap decoded once from its k
    # origin) on the plain 3-D split-addressing loop, grouped and not.
    knobs[146] = (55, dict(split_addr_k_decode=True))
    knobs[147] = (54, dict(split_addr_k_decode=True))
    knobs[148] = (55, dict(knobs[115][1], xh_pad=True))
    knobs[149] = (55, dict(knobs[119][1], xh_pad=True))
    knobs[150] = (55, dict(_s4, xh_pad=True))
    # 151-152: split_atomic, which selects the atomic split-K form on a 2-D
    # wave64 problem (a 3-D split is atomic already, so 152 is rejected).
    cfgs[151] = (
        _cp(N=1, Hi=16, Wi=16, C=128, K=128, fy=3, fx=3, pH=1, pW=1, groups=2),
        "bf16",
        dict(t256, split_k=2, split_atomic=True),
    )
    knobs[152] = (55, dict(split_k=2, split_atomic=True))
    cfgs[153] = (
        cfgs[151][0],
        "bf16",
        dict(t256, split_k=2, uniform_k_decode=True),
    )
    if idx in _pp:
        knobs[idx] = (55, dict(_s3, **_pp[idx]))
    knobs[154] = (57, dict(epi_quad_xpose=1))
    knobs[155] = (55, dict(_s4, warp_m=1))
    knobs[156] = (55, dict(_s4, tile_n=128))
    knobs[157] = (57, dict(split_k=2))
    knobs[158] = (55, dict(_s3, xcd_remap=0, split_hybrid=4, split_zero_first=True))
    knobs[159] = (
        55,
        dict(_s3, **_p85, split_det=1, xcd_remap=0, split_xcd_remap=6),
    )
    extra = {}
    if idx in knobs:
        idx, extra = knobs[idx]
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    p, dt, kw = cfgs[idx]
    kw = dict(kw, **extra)
    return ImplicitGemmConvSpec(problem=p, data=data(dt), **kw), "gfx942"


def main() -> int:
    return run_emit(
        _spec,
        build_implicit_gemm_conv,
        usage="usage: conv_implicit_gemm_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
