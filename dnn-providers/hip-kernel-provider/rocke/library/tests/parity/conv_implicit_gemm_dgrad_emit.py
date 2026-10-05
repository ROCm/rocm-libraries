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

    # Config 3: runtime sub-GEMM record on a stride-1 problem (static_sub_gemm
    # off): binary search + record loads, flat K loop. (Previously async_dma,
    # which dgrad now rejects instead of silently ignoring.)
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
                static_sub_gemm=False,
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

    # Config 10: folded record with the flat K loop (tap_outer_k off); not a
    # dispatch warp tile, so no waves_per_eu accumulator hint.
    # (Previously chiplet_swizzle, which dgrad now rejects instead of ignoring.)
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
                tap_outer_k=False,
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
        # K not a multiple of tile_k: a K tile straddles two taps, so the
        # folded record keeps the flat K loop.
        p = _cp(N=2, Hi=14, Wi=14, C=64, K=48, Y=3, X=3, pH=1, pW=1)
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
            ),
            "gfx950",
        )

    if idx == 15:
        # Tap-outer K loop on an odd, non-square, partial-tile problem: N=1,
        # 13x17, pad 2 with a 5x5 filter, C not a multiple of tile_n.
        p = _cp(N=1, Hi=13, Wi=17, C=48, K=64, Y=5, X=5, pH=2, pW=2)
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

    if idx == 16:
        # Tap-outer K loop with the M-outer B tile, compv3 schedule hints.
        p = _cp(N=4, Hi=28, Wi=28, C=128, K=128, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=128,
                tile_n=64,
                tile_k=64,
                warp_m=4,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="compv3",
                epilogue="cshuffle",
            ),
            "gfx950",
        )

    if idx == 17:
        # Ungrouped pointwise (1x1, stride 1, pad 0): static_sub_gemm stays on
        # but the record is not folded (runtime record, opaque trip count);
        # C not a multiple of tile_n.
        p = _cp(N=2, Hi=14, Wi=14, C=96, K=128, Y=1, X=1, pH=0, pW=0)
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
            ),
            "gfx950",
        )

    if idx == 18:
        # Folded record, flat K loop, 4-element dY and W loads (K and C not
        # multiples of 8): the waves_per_eu accumulator hint is withheld.
        p = _cp(N=2, Hi=13, Wi=11, C=100, K=68, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
                pipeline="mem",
                epilogue="cshuffle",
                lds_k_outer=True,
            ),
            "gfx950",
        )

    if idx == 19:
        # Folded record, flat K loop, 8-element loads, explicit waves_per_eu:
        # the explicit value wins over the accumulator hint.
        p = _cp(N=2, Hi=14, Wi=14, C=64, K=48, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
                pipeline="mem",
                epilogue="cshuffle",
                lds_k_outer=True,
                waves_per_eu=1,
            ),
            "gfx950",
        )

    if idx == 20:
        # The large-problem dispatch tile: 128x128x64, 2x2 waves, 16x16x32
        # atom, K-outer B, tap-outer K loop. fp16: the C++ CoalescedTileLoader
        # mirror has no elem_dtype yet, so a bf16 config cannot be compared.
        p = _cp(N=2, Hi=16, Wi=16, C=256, K=256, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=128,
                tile_n=128,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
                pipeline="mem",
                epilogue="cshuffle",
                lds_k_outer=True,
            ),
            "gfx950",
        )

    if idx == 21:
        # Flat K loop (kpg not a multiple of tile_k), 8-element loads, 256 fp32
        # accumulators per lane (256x128 tile, 1x2 waves, 32x32 atom): above
        # 128 the record is not folded and the waves_per_eu accumulator hint
        # is withheld (both would add spills).
        p = _cp(N=2, Hi=14, Wi=14, C=128, K=48, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=256,
                tile_n=128,
                tile_k=64,
                warp_m=1,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="mem",
                epilogue="default",
                lds_k_outer=True,
            ),
            "gfx950",
        )

    if idx == 22:
        # Flat K loop, 8-element loads, 128 fp32 accumulators per lane with the
        # 16x16x32 atom on a single warp (64x128 tile, 1x1 waves): the record
        # is folded, but the waves_per_eu accumulator hint is withheld (not a
        # dispatch warp tile; under its 256-register cap this tile spills).
        p = _cp(N=2, Hi=14, Wi=14, C=128, K=48, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=64,
                tile_n=128,
                tile_k=64,
                warp_m=1,
                warp_n=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
                pipeline="mem",
                epilogue="default",
                lds_k_outer=True,
            ),
            "gfx950",
        )

    if idx == 23:
        # Flat K loop, 8-element loads, 64 fp32 accumulators per lane on a
        # warp tile that is not a dispatch tile (256x64x64, 2x2 waves, 32x32
        # atom): the record is folded but the waves_per_eu accumulator hint
        # is withheld (it is applied only to the two dispatch tiles).
        p = _cp(N=2, Hi=14, Wi=14, C=128, K=48, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=256,
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
            ),
            "gfx950",
        )

    if idx == 24:
        # Tap-aligned K, 256 fp32 accumulators per lane (256x128 tile, 1x2
        # waves, 32x32 atom) and an N tile wider than the input channels
        # (C=64 < tile_n): the folded record with the tap-outer K loop (the
        # 128-accumulator fold limit applies to the flat loop only).
        p = _cp(N=2, Hi=14, Wi=14, C=64, K=128, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=256,
                tile_n=128,
                tile_k=64,
                warp_m=1,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="mem",
                epilogue="default",
                lds_k_outer=True,
            ),
            "gfx950",
        )

    if idx == 25:
        # The large-problem dispatch tile (128x128x64, 2x2 waves, 16x16x32
        # atom) on a flat K loop (kpg 96 not a multiple of tile_k): folded
        # record with the waves_per_eu accumulator hint. fp16, see config 20.
        p = _cp(N=2, Hi=16, Wi=16, C=256, K=96, Y=3, X=3, pH=1, pW=1)
        return (
            DgradConvSpec(
                problem=p,
                tile_m=128,
                tile_n=128,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
                pipeline="mem",
                epilogue="cshuffle",
                lds_k_outer=True,
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
