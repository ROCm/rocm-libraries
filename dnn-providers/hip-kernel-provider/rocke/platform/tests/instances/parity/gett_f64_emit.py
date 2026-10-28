#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gett_f64_emit.py -- Python reference emitter for the f64 /
# complex-f64 GETT family. gett_f64_emit.c builds the same specs through the C
# API so run_diff.py can byte-compare the two engines.
#
# _CONFIGS cover both load directions, K digits that split / do not split at
# the K tile, batch and C strides, complex with conjugation, the padding (no
# swizzle) path, every swizzle mode, one and two LDS stages, every schedule
# knob, blocked K accumulation, runtime sizes and split_k (the partial-sum
# kernel and the reduce kernel), on gfx950 / gfx942 / gfx90a.
# _TUNED adds every TUNED_CONFIGS entry on a problem of its own shape class (the
# configuration tuned_spec selects for it), so each shipped configuration is
# compared as dispatch builds it. _F32_CONFIGS follow: the kernel with f32 /
# complex-f32 words.
#
# The C cases are generated from these specs: `python gett_f64_emit.py
# --c-cases` prints the body of make_spec() in gett_f64_emit.c.
import sys

from rocke.instances.common.gett_f64 import (
    TUNED_CONFIGS,
    GettF64Problem,
    GettF64Spec,
    build_gett_f64,
    build_gett_f64_reduce,
    gett_f64_shape_class,
    tuned_spec,
)

from _emit_common import run_emit


def _gemm(M, N, K, a, b, d, **kw):
    return dict(
        M=M,
        N=N,
        K=K,
        a_m=((M, a[0]),),
        a_k=((K, a[1]),),
        b_n=((N, b[0]),),
        b_k=((K, b[1]),),
        d_m=((M, d[0]),),
        d_n=((N, d[1]),),
        **kw,
    )


# (problem kwargs, spec kwargs, arch)
_CONFIGS = (
    # 0: A and B K-contiguous (along-K loads), K tail.
    (_gemm(256, 192, 100, (100, 1), (100, 1), (192, 1)), {}, "gfx950"),
    # 1: both M/N-contiguous, 64x64 tiles, row-major raster, XCD remap.
    (
        _gemm(256, 256, 64, (1, 256), (1, 256), (256, 1)),
        dict(tile_m=64, tile_n=64, group_m=1, xcds=8),
        "gfx950",
    ),
    # 2: complex, A conjugated.
    (
        _gemm(96, 64, 36, (36, 1), (1, 64), (64, 1), complex_=True, conj_a=True),
        dict(tile_m=64, tile_n=64),
        "gfx950",
    ),
    # 3: complex, both conjugated, scale only (no C), gfx942 K tile 8.
    (
        _gemm(
            64,
            64,
            32,
            (1, 64),
            (32, 1),
            (64, 1),
            complex_=True,
            conj_a=True,
            conj_b=True,
        ),
        dict(tile_m=64, tile_n=64, tile_k=8, bilinear=False),
        "gfx942",
    ),
    # 4: multi-digit (hipTensor P1 style); K digits (6, 8) do not split at 8.
    (
        dict(
            M=84,
            N=120,
            K=48,
            a_m=((14, 1), (6, 84)),
            a_k=((6, 14), (8, 504)),
            b_n=((12, 1), (10, 72)),
            b_k=((6, 12), (8, 720)),
            d_m=((14, 12), (6, 1680)),
            d_n=((12, 1), (10, 168)),
        ),
        dict(tile_m=64, tile_n=64, tile_k=8),
        "gfx90a",
    ),
    # 5: batched, C with its own strides.
    (
        _gemm(
            64,
            48,
            32,
            (32, 1),
            (1, 48),
            (48, 1),
            L=3,
            a_l=((3, 2048),),
            b_l=((3, 1536),),
            d_l=((3, 3072),),
            c_m=((64, 1),),
            c_n=((48, 64),),
            c_l=((3, 3072),),
        ),
        dict(tile_m=64, tile_n=64),
        "gfx950",
    ),
    # 6: padding instead of swizzle, no prefetch, no fences, occupancy hint.
    (
        _gemm(256, 256, 64, (64, 1), (1, 256), (256, 1)),
        dict(lds_swizzle=False, prefetch=False, sched_fence=False, waves_per_eu=2),
        "gfx950",
    ),
    # 7: 48-row tiles (pad path), K tile 8 on gfx942.
    (
        _gemm(200, 136, 68, (68, 1), (68, 1), (136, 1)),
        dict(tile_m=48, tile_n=64, warps_m=1, warps_n=2, tile_k=8, lds_pad=8),
        "gfx942",
    ),
    # 8: one LDS stage, K tile 16, both K-contiguous (along-K loads), 8 waves.
    (
        _gemm(256, 128, 96, (96, 1), (96, 1), (1, 256)),
        dict(lds_stages=1, tile_k=16, warps_m=4, warps_n=2, group_m=4),
        "gfx90a",
    ),
    # 9: swizzle mode 1 (K-pair groups + odd-row flip), one stage, A along K.
    (
        _gemm(128, 192, 64, (64, 1), (1, 128), (192, 1)),
        dict(lds_swizzle_mode=1, lds_stages=1, tile_k=16, tile_n=64),
        "gfx942",
    ),
    # 10: swizzle mode 2 (eight K-pair groups), two stages, B along K.
    (
        _gemm(128, 128, 48, (1, 128), (48, 1), (1, 128)),
        dict(lds_swizzle_mode=2, tile_k=8, waves_per_eu=4),
        "gfx950",
    ),
    # 11: complex, one LDS stage (the swizzle mode is inert for complex).
    (
        _gemm(64, 96, 40, (1, 64), (40, 1), (1, 64), complex_=True, conj_b=True),
        dict(
            tile_m=64,
            tile_n=32,
            warps_m=1,
            warps_n=2,
            tile_k=8,
            lds_stages=1,
            lds_swizzle_mode=2,
        ),
        "gfx950",
    ),
    # 12: prefetch two K tiles ahead (registers carried through the loop), one stage.
    (
        _gemm(256, 128, 96, (96, 1), (1, 256), (1, 256)),
        dict(prefetch_depth=2, lds_stages=1, tile_k=16),
        "gfx90a",
    ),
    # 13: complex with four accumulators per atom, both conjugated, prefetch depth 2.
    (
        _gemm(
            64,
            64,
            40,
            (40, 1),
            (1, 64),
            (1, 64),
            complex_=True,
            conj_a=True,
            conj_b=True,
        ),
        dict(tile_m=64, tile_n=64, tile_k=8, complex_accs=4, prefetch_depth=2),
        "gfx942",
    ),
    # 14: complex, four accumulators, B conjugated only (the mixed-sign epilogue).
    (
        _gemm(96, 64, 24, (1, 96), (24, 1), (64, 1), complex_=True, conj_b=True),
        dict(tile_m=32, tile_n=64, warps_m=1, warps_n=4, tile_k=8, complex_accs=4),
        "gfx950",
    ),
    # 15: complex, four accumulators, A conjugated only.
    (
        _gemm(64, 96, 24, (1, 64), (1, 96), (1, 64), complex_=True, conj_a=True),
        dict(tile_m=64, tile_n=32, tile_k=8, complex_accs=4, lds_stages=1),
        "gfx90a",
    ),
    # 16: K loop unrolled by two tiles, odd tile count (tail after the loop).
    (
        _gemm(128, 128, 80, (1, 128), (80, 1), (1, 128)),
        dict(unroll2=True, tile_k=16, lds_swizzle_mode=1),
        "gfx950",
    ),
    # 17: unroll2, complex, even tile count, no fences.
    (
        _gemm(64, 64, 32, (32, 1), (1, 64), (1, 64), complex_=True, conj_a=True),
        dict(unroll2=True, tile_m=64, tile_n=64, tile_k=8, sched_fence=False),
        "gfx90a",
    ),
    # 18: K stagger with one LDS stage and prefetch depth 2 (wrapped K-tile order).
    (
        _gemm(256, 256, 160, (160, 1), (160, 1), (1, 256)),
        dict(k_stagger=2, lds_stages=1, tile_k=16, prefetch_depth=2),
        "gfx90a",
    ),
    # 19: K stagger with the unrolled loop, odd tile count.
    (
        _gemm(128, 128, 48, (1, 128), (1, 128), (1, 128)),
        dict(k_stagger=1, unroll2=True, tile_k=16),
        "gfx950",
    ),
    # 20: real, scale only (no C): the alpha == 0 select without C.
    (
        _gemm(128, 96, 40, (40, 1), (1, 96), (96, 1)),
        dict(tile_m=64, tile_n=32, warps_m=2, warps_n=1, bilinear=False),
        "gfx942",
    ),
    # 21: blocked K accumulation, odd block count (last block shorter).
    (
        _gemm(256, 192, 200, (200, 1), (200, 1), (192, 1)),
        dict(k_block=3),
        "gfx950",
    ),
    # 22: blocked K accumulation with the unrolled loop and an odd tile count.
    (
        _gemm(128, 128, 88, (1, 128), (88, 1), (1, 128)),
        dict(k_block=4, unroll2=True, tile_k=8),
        "gfx942",
    ),
    # 23: blocked K with prefetch depth 2 (registers carried through both
    # loops), one stage and K stagger.
    (
        _gemm(256, 128, 160, (160, 1), (1, 128), (1, 256)),
        dict(k_block=2, prefetch_depth=2, lds_stages=1, tile_k=16, k_stagger=3),
        "gfx90a",
    ),
    # 24: runtime sizes, both operands K-contiguous, two-element loads.
    (
        _gemm(256, 192, 100, (100, 1), (100, 1), (192, 1)),
        dict(runtime_sizes=True),
        "gfx950",
    ),
    # 25: runtime sizes, odd M and K runs (one-element loads), M-contiguous A,
    # XCD remap, raster groups, K stagger, unrolled loop.
    (
        _gemm(255, 193, 99, (1, 255), (99, 1), (1, 255)),
        dict(runtime_sizes=True, xcds=8, group_m=16, k_stagger=1, unroll2=True),
        "gfx942",
    ),
    # 26: runtime sizes, batched with C strides, row-major raster, blocked K,
    # prefetch depth 2.
    (
        _gemm(
            64,
            48,
            32,
            (32, 1),
            (1, 48),
            (48, 1),
            L=3,
            a_l=((3, 2048),),
            b_l=((3, 1536),),
            d_l=((3, 3072),),
            c_m=((64, 1),),
            c_n=((48, 64),),
            c_l=((3, 3072),),
        ),
        dict(
            runtime_sizes=True,
            tile_m=64,
            tile_n=64,
            group_m=1,
            k_block=2,
            prefetch_depth=2,
            lds_stages=1,
        ),
        "gfx90a",
    ),
    # 27: runtime sizes, complex, both conjugated, four accumulators.
    (
        _gemm(
            96,
            64,
            40,
            (40, 1),
            (1, 64),
            (1, 96),
            complex_=True,
            conj_a=True,
            conj_b=True,
        ),
        dict(runtime_sizes=True, tile_m=64, tile_n=64, tile_k=8, complex_accs=4),
        "gfx950",
    ),
    # 28: runtime sizes, complex scale only (no C), unrolled loop + blocked K.
    (
        _gemm(64, 96, 48, (1, 64), (48, 1), (96, 1), complex_=True, conj_b=True),
        dict(
            runtime_sizes=True,
            tile_m=64,
            tile_n=64,
            tile_k=8,
            bilinear=False,
            unroll2=True,
            k_block=2,
        ),
        "gfx942",
    ),
    # 29 / 30: split_k, partial sums and the reduce kernel; K-contiguous real.
    (
        _gemm(256, 192, 520, (520, 1), (520, 1), (192, 1)),
        dict(split_k=4),
        "gfx950",
    ),
    (
        _gemm(256, 192, 520, (520, 1), (520, 1), (192, 1)),
        dict(split_k=4),
        "gfx950",
        "reduce",
    ),
    # 31 / 32: split_k with runtime sizes, prefetch depth 2 and one LDS stage.
    (
        _gemm(200, 136, 1040, (1040, 1), (1, 136), (136, 1)),
        dict(split_k=5, runtime_sizes=True, prefetch_depth=2, lds_stages=1),
        "gfx90a",
    ),
    (
        _gemm(200, 136, 1040, (1040, 1), (1, 136), (136, 1)),
        dict(split_k=5, runtime_sizes=True, prefetch_depth=2, lds_stages=1),
        "gfx90a",
        "reduce",
    ),
    # 33 / 34: split_k, complex with four accumulators, A conjugated.
    (
        _gemm(96, 64, 264, (264, 1), (1, 64), (64, 1), complex_=True, conj_a=True),
        dict(split_k=3, tile_m=64, tile_n=64, tile_k=8, complex_accs=4),
        "gfx942",
    ),
    (
        _gemm(96, 64, 264, (264, 1), (1, 64), (64, 1), complex_=True, conj_a=True),
        dict(split_k=3, tile_m=64, tile_n=64, tile_k=8, complex_accs=4),
        "gfx942",
        "reduce",
    ),
    # 35 / 36: split_k, complex runtime sizes, scale only (no C).
    (
        _gemm(64, 96, 200, (1, 64), (200, 1), (96, 1), complex_=True, conj_b=True),
        dict(
            split_k=2,
            runtime_sizes=True,
            tile_m=64,
            tile_n=64,
            tile_k=8,
            bilinear=False,
        ),
        "gfx950",
    ),
    (
        _gemm(64, 96, 200, (1, 64), (200, 1), (96, 1), complex_=True, conj_b=True),
        dict(
            split_k=2,
            runtime_sizes=True,
            tile_m=64,
            tile_n=64,
            tile_k=8,
            bilinear=False,
        ),
        "gfx950",
        "reduce",
    ),
    # 37 / 38: split_k on a multi-digit contraction (the reduce kernel writes D through mixed-radix maps).
    (
        dict(
            M=84,
            N=120,
            K=48,
            a_m=((14, 1), (6, 84)),
            a_k=((6, 14), (8, 504)),
            b_n=((12, 1), (10, 72)),
            b_k=((6, 12), (8, 720)),
            d_m=((14, 12), (6, 1680)),
            d_n=((12, 1), (10, 168)),
        ),
        dict(split_k=3, tile_m=64, tile_n=64, tile_k=8),
        "gfx90a",
    ),
    (
        dict(
            M=84,
            N=120,
            K=48,
            a_m=((14, 1), (6, 84)),
            a_k=((6, 14), (8, 504)),
            b_n=((12, 1), (10, 72)),
            b_k=((6, 12), (8, 720)),
            d_m=((14, 12), (6, 1680)),
            d_n=((12, 1), (10, 168)),
        ),
        dict(split_k=3, tile_m=64, tile_n=64, tile_k=8),
        "gfx90a",
        "reduce",
    ),
    # 39 / 40: split_k, batched with C strides.
    (
        _gemm(
            64,
            48,
            96,
            (96, 1),
            (1, 48),
            (48, 1),
            L=3,
            a_l=((3, 6144),),
            b_l=((3, 4608),),
            d_l=((3, 3072),),
            c_m=((64, 1),),
            c_n=((48, 64),),
            c_l=((3, 3072),),
        ),
        dict(split_k=2, tile_m=64, tile_n=64, tile_k=16),
        "gfx950",
    ),
    (
        _gemm(
            64,
            48,
            96,
            (96, 1),
            (1, 48),
            (48, 1),
            L=3,
            a_l=((3, 6144),),
            b_l=((3, 4608),),
            d_l=((3, 3072),),
            c_m=((64, 1),),
            c_n=((48, 64),),
            c_l=((3, 3072),),
        ),
        dict(split_k=2, tile_m=64, tile_n=64, tile_k=16),
        "gfx950",
        "reduce",
    ),
)

# CU counts the shape classes of _TUNED are computed with.
_CUS = {"gfx90a": 104, "gfx942": 304, "gfx950": 256}
_MN_CONTIG = lambda M, N, K: ((1, M), (1, N), (1, M))  # noqa: E731
_K_CONTIG = lambda M, N, K: ((K, 1), (K, 1), (N, 1))  # noqa: E731
_K_MIXED = lambda M, N, K: ((K, 1), (1, N), (1, M))  # noqa: E731

# (arch, kind, class) -> (M, N, K, layout): a problem of that shape class.
_TUNED_PROBLEMS = {
    ("gfx950", "real", "small"): (1024, 1024, 512, _K_CONTIG),
    ("gfx950", "real", "medium"): (4096, 4096, 512, _K_CONTIG),
    ("gfx950", "real", "large"): (8192, 8192, 512, _MN_CONTIG),
    ("gfx950", "complex", "small"): (1024, 1024, 512, _K_CONTIG),
    ("gfx950", "complex", "medium"): (4096, 4096, 1024, _MN_CONTIG),
    ("gfx950", "complex", "large"): (4096, 4096, 4096, _K_MIXED),
    ("gfx90a", "real", "small"): (1536, 1536, 512, _K_CONTIG),
    ("gfx90a", "real", "medium"): (4096, 4096, 512, _MN_CONTIG),
    ("gfx90a", "real", "kmixed"): (4096, 4096, 512, _K_MIXED),
    ("gfx90a", "real", "kk"): (4096, 4096, 512, _K_CONTIG),
    ("gfx90a", "real", "wide"): (12288, 12288, 768, _MN_CONTIG),
    ("gfx90a", "real", "pow2"): (8192, 8192, 8192, _MN_CONTIG),
    ("gfx90a", "complex", "medium"): (1024, 1024, 512, _K_CONTIG),
    ("gfx90a", "complex", "wide"): (6144, 6144, 768, _MN_CONTIG),
    ("gfx90a", "complex", "large"): (4096, 4096, 2048, _K_MIXED),
    ("gfx942", "real", "small"): (1024, 1024, 512, _K_CONTIG),
    ("gfx942", "real", "medium"): (4096, 4096, 512, _MN_CONTIG),
    ("gfx942", "complex", "small"): (1024, 1024, 512, _K_CONTIG),
    ("gfx942", "complex", "medium"): (4096, 4096, 512, _K_MIXED),
}

# Every shipped configuration, in a stable order.
_TUNED = tuple(sorted(TUNED_CONFIGS))

# f32 / complex-f32 words (problem dtype "f32"): the same kernel family with
# the f32 atom (gfx950 / gfx942). Four-, two- and one-element loads (K runs of
# 4k, 4k + 2 and odd), both load directions, complex with conjugation, the
# padding and swizzle paths, one and two LDS stages, every schedule knob both
# engines build, blocked K, runtime sizes and a tile only f32 accumulators fit.
_F32_CONFIGS = (
    # f0: A and B K-contiguous, K run of 4k (four-element loads along K).
    (_gemm(256, 192, 100, (100, 1), (100, 1), (192, 1), dtype="f32"), {}, "gfx950"),
    # f1: both M/N-contiguous (four-element loads along M / N), XCD remap.
    (
        _gemm(256, 256, 64, (1, 256), (1, 256), (256, 1), dtype="f32"),
        dict(tile_m=64, tile_n=64, group_m=1, xcds=8),
        "gfx950",
    ),
    # f2: complex, A conjugated.
    (
        _gemm(
            96,
            64,
            36,
            (36, 1),
            (1, 64),
            (64, 1),
            complex_=True,
            conj_a=True,
            dtype="f32",
        ),
        dict(tile_m=64, tile_n=64),
        "gfx950",
    ),
    # f3: complex, both conjugated, scale only, gfx942 K tile 8.
    (
        _gemm(
            64,
            64,
            32,
            (1, 64),
            (32, 1),
            (64, 1),
            complex_=True,
            conj_a=True,
            conj_b=True,
            dtype="f32",
        ),
        dict(tile_m=64, tile_n=64, tile_k=8, bilinear=False),
        "gfx942",
    ),
    # f4: multi-digit; K digits (6, 8) do not split at 8.
    (
        dict(
            M=84,
            N=120,
            K=48,
            a_m=((14, 1), (6, 84)),
            a_k=((6, 14), (8, 504)),
            b_n=((12, 1), (10, 72)),
            b_k=((6, 12), (8, 720)),
            d_m=((14, 12), (6, 1680)),
            d_n=((12, 1), (10, 168)),
            dtype="f32",
        ),
        dict(tile_m=64, tile_n=64, tile_k=8),
        "gfx942",
    ),
    # f5: batched, C with its own strides.
    (
        _gemm(
            64,
            48,
            32,
            (32, 1),
            (1, 48),
            (48, 1),
            L=3,
            a_l=((3, 2048),),
            b_l=((3, 1536),),
            d_l=((3, 3072),),
            c_m=((64, 1),),
            c_n=((48, 64),),
            c_l=((3, 3072),),
            dtype="f32",
        ),
        dict(tile_m=64, tile_n=64),
        "gfx950",
    ),
    # f6: padding instead of swizzle, no prefetch, no fences, occupancy hint.
    (
        _gemm(256, 256, 64, (64, 1), (1, 256), (256, 1), dtype="f32"),
        dict(lds_swizzle=False, prefetch=False, sched_fence=False, waves_per_eu=2),
        "gfx950",
    ),
    # f7: 48-row tiles (pad path), explicit pad, K tile 8 on gfx942.
    (
        _gemm(200, 136, 68, (68, 1), (68, 1), (136, 1), dtype="f32"),
        dict(tile_m=48, tile_n=64, warps_m=1, warps_n=2, tile_k=8, lds_pad=8),
        "gfx942",
    ),
    # f8: K run of 4k + 2 (two-element loads), one LDS stage, 8 waves.
    (
        _gemm(256, 128, 98, (98, 1), (98, 1), (1, 256), dtype="f32"),
        dict(lds_stages=1, tile_k=16, warps_m=4, warps_n=2, group_m=4),
        "gfx942",
    ),
    # f9: odd K run (one-element loads along K), swizzle mode 1, one stage.
    (
        _gemm(128, 192, 97, (97, 1), (1, 128), (192, 1), dtype="f32"),
        dict(lds_swizzle_mode=1, lds_stages=1, tile_k=16, tile_n=64),
        "gfx950",
    ),
    # f10: swizzle mode 2, two stages, B along K.
    (
        _gemm(128, 128, 48, (1, 128), (48, 1), (1, 128), dtype="f32"),
        dict(lds_swizzle_mode=2, tile_k=8, waves_per_eu=4),
        "gfx942",
    ),
    # f11: prefetch two K tiles ahead, one stage.
    (
        _gemm(256, 128, 96, (96, 1), (1, 256), (1, 256), dtype="f32"),
        dict(prefetch_depth=2, lds_stages=1, tile_k=16),
        "gfx950",
    ),
    # f12: complex with four accumulators, both conjugated, prefetch depth 2.
    (
        _gemm(
            64,
            64,
            40,
            (40, 1),
            (1, 64),
            (1, 64),
            complex_=True,
            conj_a=True,
            conj_b=True,
            dtype="f32",
        ),
        dict(tile_m=64, tile_n=64, tile_k=8, complex_accs=4, prefetch_depth=2),
        "gfx942",
    ),
    # f13: K loop unrolled by two tiles, odd tile count.
    (
        _gemm(128, 128, 80, (1, 128), (80, 1), (1, 128), dtype="f32"),
        dict(unroll2=True, tile_k=16, lds_swizzle_mode=1),
        "gfx950",
    ),
    # f14: K stagger with the unrolled loop.
    (
        _gemm(128, 128, 48, (1, 128), (1, 128), (1, 128), dtype="f32"),
        dict(k_stagger=1, unroll2=True, tile_k=16),
        "gfx942",
    ),
    # f15: blocked K accumulation, odd block count.
    (
        _gemm(256, 192, 200, (200, 1), (200, 1), (192, 1), dtype="f32"),
        dict(k_block=3),
        "gfx950",
    ),
    # f16: runtime sizes, both K-contiguous, four-element loads.
    (
        _gemm(256, 192, 100, (100, 1), (100, 1), (192, 1), dtype="f32"),
        dict(runtime_sizes=True),
        "gfx950",
    ),
    # f17: runtime sizes, odd M and K runs, XCD remap, groups, stagger, unroll2.
    (
        _gemm(255, 193, 99, (1, 255), (99, 1), (1, 255), dtype="f32"),
        dict(runtime_sizes=True, xcds=8, group_m=16, k_stagger=1, unroll2=True),
        "gfx942",
    ),
    # f18: runtime sizes, batched with C strides, blocked K, prefetch depth 2.
    (
        _gemm(
            64,
            48,
            32,
            (32, 1),
            (1, 48),
            (48, 1),
            L=3,
            a_l=((3, 2048),),
            b_l=((3, 1536),),
            d_l=((3, 3072),),
            c_m=((64, 1),),
            c_n=((48, 64),),
            c_l=((3, 3072),),
            dtype="f32",
        ),
        dict(
            runtime_sizes=True,
            tile_m=64,
            tile_n=64,
            group_m=1,
            k_block=2,
            prefetch_depth=2,
            lds_stages=1,
        ),
        "gfx950",
    ),
    # f19: runtime sizes, complex, both conjugated, four accumulators.
    (
        _gemm(
            96,
            64,
            40,
            (40, 1),
            (1, 64),
            (1, 96),
            complex_=True,
            conj_a=True,
            conj_b=True,
            dtype="f32",
        ),
        dict(runtime_sizes=True, tile_m=64, tile_n=64, tile_k=8, complex_accs=4),
        "gfx942",
    ),
    # f20: a 256 x 256 tile (only the f32 accumulators fit).
    (
        _gemm(512, 512, 64, (64, 1), (64, 1), (512, 1), dtype="f32"),
        dict(tile_m=256, tile_n=256),
        "gfx950",
    ),
)


def _tuned_problem(key) -> GettF64Problem:
    M, N, K, layout = _TUNED_PROBLEMS[key]
    a, b, d = layout(M, N, K)
    return GettF64Problem(**_gemm(M, N, K, a, b, d, complex_=key[1] == "complex"))


def _tuned_spec(key):
    arch = key[0]
    p = _tuned_problem(key)
    if gett_f64_shape_class(p, arch, _CUS[arch]) != key[2]:
        raise SystemExit(f"{key}: the problem is not in its shape class")
    return tuned_spec(p, arch, num_cus=_CUS[arch]), arch


def _entry(idx: int):
    """(spec, arch, kernel): kernel is "gett" or "reduce" (split_k's second)."""
    if 0 <= idx < len(_CONFIGS):
        prob, kw, arch, *which = _CONFIGS[idx]
        spec = GettF64Spec(GettF64Problem(**prob), **kw)
        return spec, arch, (which[0] if which else "gett")
    j = idx - len(_CONFIGS)
    if 0 <= j < len(_TUNED):
        spec, arch = _tuned_spec(_TUNED[j])
        return spec, arch, "gett"
    j -= len(_TUNED)
    if 0 <= j < len(_F32_CONFIGS):
        prob, kw, arch = _F32_CONFIGS[j]
        return GettF64Spec(GettF64Problem(**prob), **kw), arch, "gett"
    raise SystemExit(f"unknown config index {idx}")


def _spec(idx: int):
    spec, arch, which = _entry(idx)
    return (spec, which), arch


def _build(sel, arch):
    spec, which = sel
    if which == "reduce":
        return build_gett_f64_reduce(spec, arch=arch)
    return build_gett_f64(spec, arch=arch)


def all_specs():
    """Every (spec, arch) the family compares, in index order."""
    return [_entry(i)[:2] for i in range(_count())]


def _count() -> int:
    return len(_CONFIGS) + len(_TUNED) + len(_F32_CONFIGS)


def _c_digits(d) -> str:
    if len(d) == 1:
        return f"dig1({d[0][0]}, {d[0][1]})"
    if len(d) == 2:
        return f"dig2({d[0][0]}, {d[0][1]}, {d[1][0]}, {d[1][1]})"
    raise ValueError("the C emitter spells one- or two-digit index maps")


def c_cases() -> str:
    """The switch cases of make_spec() in gett_f64_emit.c."""
    out = []
    b = lambda v: "true" if v else "false"  # noqa: E731
    for i in range(_count()):
        s, arch, which = _entry(i)
        p = s.problem
        lines = [f"    case {i}:", "        memset(&p, 0, sizeof p);"]
        lines += [f"        p.M = {p.M};", f"        p.N = {p.N};"]
        lines += [f"        p.K = {p.K};", f"        p.L = {p.L};"]
        for name in ("a_m", "a_k", "a_l", "b_n", "b_k", "b_l", "d_m", "d_n", "d_l"):
            lines.append(f"        p.{name} = {_c_digits(getattr(p, name))};")
        if p.c_m is not None:
            lines.append("        p.has_c = true;")
            for name in ("c_m", "c_n", "c_l"):
                lines.append(f"        p.{name} = {_c_digits(getattr(p, name))};")
        lines += [
            f"        p.complex_ = {b(p.complex_)};",
            f"        p.conj_a = {b(p.conj_a)};",
            f"        p.conj_b = {b(p.conj_b)};",
        ]
        if p.dtype == "f32":
            lines.append("        p.f32 = true;")
        lines += [
            "        *s = rocke_gett_f64_spec_default(&p);",
        ]
        for name in (
            "tile_m",
            "tile_n",
            "tile_k",
            "warps_m",
            "warps_n",
            "group_m",
            "xcds",
            "lds_swizzle_mode",
            "lds_stages",
            "prefetch_depth",
            "complex_accs",
            "k_stagger",
            "k_block",
            "split_k",
        ):
            lines.append(f"        s->{name} = {getattr(s, name)};")
        lines.append(f"        s->lds_pad = {-1 if s.lds_pad is None else s.lds_pad};")
        lines.append(f"        s->waves_per_eu = {s.waves_per_eu or 0};")
        for name in (
            "lds_swizzle",
            "bilinear",
            "prefetch",
            "sched_fence",
            "unroll2",
            "runtime_sizes",
        ):
            lines.append(f"        s->{name} = {b(getattr(s, name))};")
        lines.append(f"        *reduce = {1 if which == 'reduce' else 0};")
        lines.append(f'        return "{arch}";')
        out.append("\n".join(lines))
    return "\n".join(out) + "\n"


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--c-cases":
        sys.stdout.write(c_cases())
        return 0
    return run_emit(
        _spec,
        _build,
        usage="usage: gett_f64_emit.py <config_index> | --c-cases\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
