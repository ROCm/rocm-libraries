# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emulated-f32 GEMM routes on gfx950 and gfx942: complex f32 (cf32, one fused
4M kernel) and real f32 (bf16-split emulation).

Problems (RCR after mode merging, K contiguous on both inputs):

* cf32 (:func:`plan_cf32_gemm`), every tensor interleaved complex f32:
  ``D[M, Nc] = alpha * opA(A)[M, Kc] . opB(B)[Nc, Kc]^T (+ beta * C)``, ``op`` =
  identity or CONJ;
* f32 (:func:`plan_f32_gemm`): ``D[M, N] = alpha * A[M, K] . B[N, K]^T (+ beta * C)``.

The kernel is a universal-GEMM body with the operand prologue
(:mod:`rocke.instances.common.gemm_prologue`): the f32 pair view reads A and B
as f32 and splits every value into bf16 parts (``x6``: f32 accuracy class,
``x3``: TF32 class). For cf32 it reads B as the real embedding ``B'[2 Nc, 2 Kc]``
(row ``2n`` = ``(Bre, -Bim)``, row ``2n + 1`` = ``(Bim, Bre)``, built on the fly
from complex row ``n``), so the kernel's real ``M x 2Nc x 4Kc`` GEMM writes the
interleaved complex result directly.

Each target has its own named bodies and body rule (``CF32_BODY_RULE``).
gfx950 (rule ``"waves"``), by the number of 256 x 256 output tiles of the
kernel's GEMM (waves = tiles / CUs):

* many tiles: the row-interleave vector-epilogue body (``RIW``: 256 x 256 x 64,
  4 waves of 128 x 128 AGPR accumulators, 2-stage DMA ring, fragment prefetch +
  MFMA interleave, 16-byte non-temporal stores);
* a few waves of tiles, or a partly filled last wave: the same body as a
  persistent stream-K loop (``RIW_SK``), which fills every CU;
* fewer tiles than half the CUs: a cost model (``CF32_SMALL_COST``, fitted on
  measured grids) picks among the 128 x 128 three-stage ring as a stream-K
  loop (``Q3_SK``), the same ring data parallel with 8 waves (``C3``, real f32)
  and ``RIW_SK``: the small bodies for short K, ``RIW_SK`` for long K with
  enough tiles. ``alpha`` / bilinear run in the small bodies' scatter epilogue
  (the complex partner column is one DPP lane swap away; knob
  ``CF32_SMALL_EPI``);
* a kernel N that is not a multiple of 8 (the vector epilogue stores 8
  consecutive columns per lane): the cost model's best scatter-epilogue body,
  at any size (knob ``CF32_SCATTER_FALLBACK``).

gfx942 (rule ``"cost"``): the register-staged bodies (``vgpr_prefetch`` 4: one
LDS buffer, the global reads staged in registers one K tile ahead) of a body
set per route (``CF32_ROUTE_BODIES``, names ``G<M>x<N>x<K><variant>``: ``L3`` /
``L1`` the main-loop levers with a release window of 3 / 1 K atoms, ``U``
``vp_unroll``, ``S3`` ``lds_swizzle_l`` 3) with the scatter epilogue. The rule
(:func:`gemm_body_cost`) predicts each body's time ``a + b * ceil(tiles * L /
(CUs * occ)) * K`` (``K``: the problem's K, complex elements counted twice) with
per route coefficients (``CF32_COST``) and takes the least (equal: the first
name). The f32 recipes split with ``CF32_SPLIT_MODE`` (truncation for x6 / x9:
three truncated parts hold the 24 bits of an f32; rounding for x3 / x4). The
x6 recipe also lists the native f32 route there (as accurate, faster on small
problems): first below ``CF32_NATIVE_FIRST`` real multiply-adds. The
64-bit-address form is ``wide_addr`` on the route's ``CF32_WIDE_BODY`` (64-bit
batch and tile origins).

The bf16 GEMM dispatcher's own pick (the ``hb_sched`` 256 x 256 body) builds
with the prologue too, but the f32 pair view's split registers do not fit its
fragment schedule (scratch spills in the K loop), so it is not a candidate.

Launch adjustments: ragged M / N get the padded bounds (exact DMA resources on
the padded operand, vector-epilogue store guards); an operand past the 2 GiB DMA
range or a D past 2**31 elements gets ``wide_addr`` (64-bit tile origins,
non-persistent). The register-staged bodies take ``pad_m`` / ``pad_n`` (their
buffer ranges end at the operand) and ``pad_k`` for a ragged K, and
``wide_addr`` when their 32-bit form cannot address the operands. Every
adjusted spec is checked by ``is_valid_spec`` and ``launch_shape_error``;
:func:`gemm_launch_limits` gives a body's launch limits for callers that launch
it on extents padded to tile multiples (hipTensor's kernel pack).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...helpers.split_k import arch_num_cus
from ...instances.common.gemm_prologue import Prologue, attach_prologue
from ...instances.common.gemm_universal import (
    DTL_BUFFER_NUM_RECORDS,
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    launch_shape_error,
    streamk_workspace_bytes,
)

CF32_ARCHES = ("gfx950", "gfx942")
# the body rule of each target: "waves" (the thresholds below) or "cost"
# (:func:`gemm_body_cost`)
CF32_BODY_RULE: Dict[str, str] = {"gfx950": "waves", "gfx942": "cost"}
# gfx950's bodies (:func:`cf32_body`)
CF32_BODIES: Dict[str, Tuple[str, ...]] = {"gfx950": ("RIW", "RIW_SK", "C3", "Q3_SK")}

# The register-staged bodies (gfx942): tile M, N, K, waves M x N and the levers
# on the base trait of ``_vp_body`` (``vgpr_prefetch`` 4, one LDS buffer,
# swizzle row shift 0, 16 x 16 x 16 atoms; AGPR accumulators on the 256-row
# tiles). ``_LEV``: the main-loop levers of ``g5_plan.G5P_MAINLOOP`` with the
# release window ``vp_release_atoms`` (``L3``: 3; ``L1``: 1, the f32 pair view
# of a 64-deep K tile has two K atoms).
_LEV = dict(vp_tail_pack=3, vp_pair_reads=1, vp_mfma_first=1)
_VP_BODIES: Dict[str, Tuple[int, int, int, int, int, Dict[str, Any]]] = {
    "G256x256x64L3": (
        256,
        256,
        64,
        2,
        2,
        dict(_LEV, acc_agpr=True, vp_release_atoms=3),
    ),
    "G256x224x64L3": (
        256,
        224,
        64,
        2,
        2,
        dict(_LEV, acc_agpr=True, vp_release_atoms=3),
    ),
    "G256x224x64L1": (256, 224, 64, 2, 2, dict(_LEV, acc_agpr=True)),
    "G192x128x64L3": (192, 128, 64, 4, 2, dict(_LEV, vp_release_atoms=3)),
    "G192x128x64L1S3": (192, 128, 64, 4, 2, dict(_LEV, lds_swizzle_l=3)),
    "G192x128x64US3": (192, 128, 64, 4, 2, dict(vp_unroll=1, lds_swizzle_l=3)),
    "G128x128x128L3": (128, 128, 128, 2, 4, dict(_LEV, vp_release_atoms=3)),
    "G128x64x64": (128, 64, 64, 4, 2, {}),
    "G128x64x64U": (128, 64, 64, 4, 2, dict(vp_unroll=1)),
    "G64x64x128L3": (64, 64, 128, 2, 2, dict(_LEV, vp_release_atoms=3)),
    "G64x32x128U": (64, 32, 128, 4, 1, dict(vp_unroll=1)),
    "G32x32x128L3": (32, 32, 128, 2, 2, dict(_LEV, vp_release_atoms=3)),
    "G32x32x128U": (32, 32, 128, 2, 2, dict(vp_unroll=1)),
}
# The bodies of each route on gfx942 (measured body sets); x3 / x4 / x9 take
# x6's, cf32x3 / cf32x4 / cf32x9 cf32's (CF32_ROUTE_ALIAS).
CF32_ROUTE_BODIES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "gfx942": {
        "h16": (
            "G64x64x128L3",
            "G128x128x128L3",
            "G64x32x128U",
            "G192x128x64L3",
            "G256x256x64L3",
            "G32x32x128U",
            "G256x224x64L3",
        ),
        "x6": (
            "G64x64x128L3",
            "G32x32x128L3",
            "G128x128x128L3",
            "G192x128x64L1S3",
            "G256x224x64L1",
            "G128x64x64",
            "G64x32x128U",
        ),
        "cf32": (
            "G64x64x128L3",
            "G32x32x128L3",
            "G128x128x128L3",
            "G192x128x64US3",
            "G256x224x64L1",
            "G128x64x64U",
            "G64x32x128U",
        ),
    }
}
CF32_ROUTE_ALIAS: Dict[str, str] = {
    "x3": "x6",
    "x4": "x6",
    "x9": "x6",
    "cf32x3": "cf32",
    "cf32x4": "cf32",
    "cf32x9": "cf32",
}
# The 64-bit-address body of each route ("h16" or the f32 routes) of a target.
CF32_WIDE_BODY: Dict[str, Dict[str, str]] = {
    "gfx950": {"h16": "RIW", "f32": "RIW"},
    "gfx942": {"h16": "G256x224x64L3", "f32": "G256x224x64L1"},
}
# The operand split of the f32 recipes per route (Prologue.split_mode; "rne"
# when not listed): truncated parts on gfx942 for x6 / x9 (three truncated bf16
# parts hold an f32 exactly; the conversion without a rounding is the cheaper
# one there), rounded parts for x3 / x4 (two truncated parts would lose
# accuracy).
CF32_SPLIT_MODE: Dict[str, Dict[str, str]] = {
    "gfx942": {"x6": "trunc", "x9": "trunc", "cf32": "trunc", "cf32x9": "trunc"},
}
# The x6 recipe's native f32 route (the f32 strided GEMM: f32 products, at
# least as accurate as x6) per target: also a candidate of the x6 recipe ("x6"
# real, "cf32" complex), first in the rules' order below this many real
# multiply-adds (L M N K, complex 4 L M N K; L the batch count).
CF32_NATIVE_FIRST: Dict[str, Dict[str, float]] = {
    "gfx942": {"x6": 2.9e10, "cf32": 3.4e7}
}
# The "cost" rule: per route and body ``(a, b, occ)``: a body's predicted time
# ``a + b * ceil(tiles * L / (CUs * occ)) * K`` (relative units: each route's
# G64x64x128L3 has ``b`` = 1), ``K`` the problem's K (complex: 2 per element),
# fitted on measured gfx942 timings of the body sets (every arm checked against
# an f64 reference) and checked on held-out shapes.
CF32_COST: Dict[str, Dict[str, Tuple[float, float, int]]] = {
    "h16": {
        "G64x64x128L3": (787.652, 1.0, 1),
        "G128x128x128L3": (886.031, 2.41898, 1),
        "G64x32x128U": (802.0, 0.686454, 1),
        "G192x128x64L3": (1037.05, 3.39667, 1),
        "G256x256x64L3": (2169.68, 6.12695, 1),
        "G32x32x128U": (883.11, 0.417509, 1),
        "G256x224x64L3": (1953.16, 5.59897, 1),
    },
    "x6": {
        "G64x64x128L3": (163.696, 1.0, 1),
        "G32x32x128L3": (215.608, 0.37267, 1),
        "G128x128x128L3": (56.4924, 2.9503, 1),
        "G192x128x64L1S3": (-0.211177, 4.06212, 1),
        "G256x224x64L1": (248.237, 7.69944, 1),
        "G128x64x64": (94.9866, 1.75501, 1),
        "G64x32x128U": (204.381, 0.642756, 1),
    },
    "cf32": {
        "G64x64x128L3": (168.401, 1.0, 1),
        "G32x32x128L3": (216.253, 0.374793, 1),
        "G128x128x128L3": (38.5741, 2.94739, 1),
        "G192x128x64US3": (16.9452, 4.06542, 1),
        "G256x224x64L1": (233.056, 7.51937, 1),
        "G128x64x64U": (76.2714, 1.76291, 1),
        "G64x32x128U": (207.373, 0.649071, 1),
    },
}

# body selection thresholds, in waves = 256 x 256 output tiles / CU count
CF32_SK_ALWAYS_WAVES = 4.0  # below: stream-K (RIW_SK)
CF32_SK_MAX_WAVES = (
    32.0  # below, with a last wave filled at most CF32_SK_MAX_TAIL: RIW_SK
)
CF32_SK_MAX_TAIL = 0.65
CF32_SMALL_MAX_WAVES = 0.5  # below: the small-problem cost model (CF32_SMALL_COST)
# the small bodies (Q3_SK / C3) apply ``alpha`` / bilinear in their scatter
# epilogue (False: alpha / bilinear go to RIW_SK, the vector-epilogue body)
CF32_SMALL_EPI = True
# Below CF32_SMALL_MAX_WAVES the body comes from a cost model (False: the
# small body for every K): per body ``cost = a + b * w + c * f`` in relative
# units, with ``w`` the per-CU work of the body's padded tile grid (MACs / 1e6
# / CUs, stream-K) or its wave count x tile work (C3, data parallel) and ``f``
# the stream-K fix-up proxy (CUs sharing one tile - 1, times the tile area /
# 1e4), for the gfx950 CU count. Fitted (least squares) on measured gfx950
# grids of the three bodies, scale and bilinear, cf32 and f32, and checked on
# held-out random shapes (the K-blind small body loses at long K). ``alpha``
# uses the scale set.
CF32_SMALL_MODEL = True
# The vector-epilogue bodies (RIW / RIW_SK) store 8 consecutive columns per
# lane and refuse a kernel N that is not a multiple of 8; such problems take
# the best scatter-epilogue body of the cost model instead, at any size (False:
# refused as before).
CF32_SCATTER_FALLBACK = True
CF32_SMALL_COST = {
    ("cplx", "scale"): {
        "Q3_SK": (25.068, 1.0000, 1.4436),
        "RIW_SK": (43.322, 0.7182, 0.2278),
    },
    ("cplx", "bilinear"): {
        "Q3_SK": (27.977, 1.0000, 1.0972),
        "RIW_SK": (56.645, 0.7087, 0.2023),
    },
    ("real", "scale"): {
        "C3": (5.926, 1.0666, 0.0000),
        "Q3_SK": (25.351, 1.0000, 1.4313),
        "RIW_SK": (39.995, 0.7260, 0.3672),
    },
    ("real", "bilinear"): {
        "C3": (9.642, 1.0631, 0.0000),
        "Q3_SK": (29.168, 1.0000, 1.1775),
        "RIW_SK": (51.362, 0.7280, 0.3757),
    },
}


def _ring(**kw: Any) -> TraitSpec:
    t = dict(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        direct_to_lds=True,
        dtl_prefetch=True,
        lds_swizzle=True,
        dtl_cache_a=0,
        dtl_cache_b=0,
        chiplet_swizzle=True,
        chiplet_wgm=8,
        chiplet_chunk_size=64,
        dtl_stages=2,
        interwave_setprio=True,
    )
    t.update(kw)
    return TraitSpec(**t)


def _tile(tm: int, tn: int, tk: int, wm: int, wn: int) -> TileSpec:
    return TileSpec(
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        warp_m=wm,
        warp_n=wn,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )


_RI = dict(
    lds_swizzle=False,
    interwave_setprio=False,
    dtl_stages=2,
    acc_agpr=True,
    dtl_frag_prefetch=True,
    mfma_interleave=True,
    dtl_hoist=True,
    lds_row_interleave=True,
    rilv_vec_epilogue=True,
    rve_store_nt=True,
    ilv_merge=True,
    ilv_barrier_after=2,
    chiplet_wgm=16,
    chiplet_chunk_size=8,
)
_ILV = dict(
    dtl_frag_prefetch=True,
    mfma_interleave=True,
    ilv_merge=True,
    ilv_barrier_after=2,
    lds_swizzle_r=1,
    dtl_hoist=True,
)


def _vp_body(name: str) -> Tuple[TileSpec, TraitSpec, int]:
    """A register-staged body of ``_VP_BODIES``."""
    tm, tn, tk, wm, wn, lev = _VP_BODIES[name]
    tile = TileSpec(
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        warp_m=wm,
        warp_n=wn,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
    )
    t: Dict[str, Any] = dict(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        ab_double_buffer=False,
        chiplet_swizzle=True,
        lds_swizzle=True,
        lds_swizzle_r=0,
        vgpr_prefetch=4,
    )
    t.update(lev)
    return tile, TraitSpec(**t), 64 * wm * wn


def cf32_body(name: str, arch: str = "gfx950") -> Tuple[TileSpec, TraitSpec, int]:
    """``(tile, trait, block_size)`` of a named body of the emulated-f32 routes
    (gfx950: ``CF32_BODIES``; gfx942: the register-staged ``G*`` bodies)."""
    if name in _VP_BODIES:
        return _vp_body(name)
    cus = arch_num_cus(arch)
    if name == "RIW":
        return _tile(256, 256, 64, 2, 2), _ring(**_RI), 256
    if name == "RIW_SK":
        return (
            _tile(256, 256, 64, 2, 2),
            _ring(
                **dict(_RI, chiplet_chunk_size=32), persistent_ctas=cus, streamk=True
            ),
            256,
        )
    if name == "Q3_SK":
        return (
            _tile(128, 128, 64, 2, 2),
            _ring(dtl_stages=3, persistent_ctas=cus, streamk=True, **_ILV),
            256,
        )
    if name == "C3":
        # 128 x 128 x 64, 8 waves of 64 x 32, 3-stage ring, data parallel
        return _tile(128, 128, 64, 2, 4), _ring(dtl_stages=3, **_ILV), 512
    raise ValueError(f"cf32: unknown body {name!r}")


def _waves(M: int, Nk: int, arch: str) -> float:
    return (-(-M // 256)) * (-(-Nk // 256)) / arch_num_cus(arch)


def _needs_wide_k(M: int, Nk: int, Kh: int, b_rows: int) -> bool:
    """An operand past the DMA buffer range or D past the i32 element offsets
    (kernel extents: ``Nk`` output columns, ``Kh`` 16-bit halves per row)."""
    return max(M, b_rows) * Kh * 2 > DTL_BUFFER_NUM_RECORDS or M * Nk > (1 << 31)


def _vp_needs_wide(body: str, M: int, Nk: int, Kh: int, b_rows: int) -> bool:
    """A register-staged body's 32-bit form cannot launch ``M x Nk x Kh``:
    operand bytes past the unsigned 32-bit offsets (signed with a ragged,
    padded tile) or D past 2**31 elements (``launch_shape_error``)."""
    tm, tn = _VP_BODIES[body][:2]
    lim = 1 << 31 if (M % tm or Nk % tn) else 1 << 32
    return max(M, b_rows) * Kh * 2 >= lim or M * Nk > (1 << 31)


def _small_body(
    M: int, Nk: int, Kh: int, epi: str, cplx: bool, scatter_only: bool = False
) -> Optional[str]:
    """The body below CF32_SMALL_MAX_WAVES (None: RIW_SK). ``Kh``: the
    kernel's K in 16-bit halves (cf32 4 Kc, f32 2 K). ``scatter_only``: the
    best body without the vector epilogue (never None)."""
    if epi and not CF32_SMALL_EPI and not scatter_only:
        return None
    if not CF32_SMALL_MODEL:
        # (real f32: the data-parallel 8-wave 128 x 128 body; cf32's doubled
        # N / K keep the stream-K 4-wave body ahead)
        return "Q3_SK" if cplx else "C3"
    cost = CF32_SMALL_COST[
        ("cplx" if cplx else "real", "bilinear" if epi == "bilinear" else "scale")
    ]
    names = [b for b in cost if not (scatter_only and b == "RIW_SK")]
    best = min(names, key=lambda b: _small_cost(cost[b], b, M, Nk, Kh))
    return None if best == "RIW_SK" else best


def _small_cost(coef, body: str, M: int, Nk: int, Kh: int) -> float:
    """Relative cost of a small-problem body (see CF32_SMALL_COST)."""
    t = 256 if body == "RIW_SK" else 128
    tiles = (-(-M // t)) * (-(-Nk // t))
    tile_work = t * t * Kh / 1e6
    if body == "C3":
        w = (-(-tiles // 256)) * tile_work
        f = 0.0
    else:
        w = tiles * tile_work / 256
        f = (max(1.0, 256.0 / tiles) - 1.0) * (t * t) / 1e4
    a, b, c = coef
    return a + b * w + c * f


def _body_for_k(
    M: int, Nk: int, wide: bool, epi: str, arch: str, cplx: bool = True, Kh: int = 0
) -> str:
    waves = _waves(M, Nk, arch)
    if wide:
        return "RIW"  # 64-bit tile origins need the non-persistent grid
    if Nk % 8 and CF32_SCATTER_FALLBACK and (CF32_SMALL_EPI or not epi):
        return _small_body(M, Nk, Kh, epi, cplx, scatter_only=True)
    if waves < CF32_SMALL_MAX_WAVES:
        small = _small_body(M, Nk, Kh, epi, cplx)
        if small is not None:
            return small
        return "RIW_SK"
    if waves < CF32_SK_ALWAYS_WAVES:
        return "RIW_SK"
    tail = waves - int(waves)
    if waves < CF32_SK_MAX_WAVES and 0 < tail <= CF32_SK_MAX_TAIL:
        return "RIW_SK"  # a partly filled last wave: stream-K spreads it
    return "RIW"


def route_name(cplx: bool, recipe: str = "x6") -> str:
    """The route of an f32 problem: the recipe (real) or cf32 / cf32<recipe>."""
    if cplx:
        return "cf32" if recipe == "x6" else "cf32" + recipe
    return recipe


def route_bodies(arch: str, route: str) -> Tuple[str, ...]:
    """The bodies of a route on a "cost" rule target (``CF32_ROUTE_BODIES``)."""
    sets = CF32_ROUTE_BODIES[arch]
    return sets[route] if route in sets else sets[CF32_ROUTE_ALIAS[route]]


def wide_body(arch: str, route: str) -> str:
    """The 64-bit-address body of a route (``CF32_WIDE_BODY``)."""
    return CF32_WIDE_BODY[arch]["h16" if route == "h16" else "f32"]


def split_mode(arch: str, route: str) -> str:
    """The operand split of an f32 route (``CF32_SPLIT_MODE``; "rne" default)."""
    return CF32_SPLIT_MODE.get(arch, {}).get(route, "rne")


def native_first(
    M: int, N: int, K: int, *, cplx: bool, arch: str, batch: int = 1
) -> Optional[bool]:
    """The x6 recipe's native f32 route on ``arch`` (``CF32_NATIVE_FIRST``):
    None when the recipe does not list it there, else whether the rules put it
    ahead of the GEMM bodies (fewer real multiply-adds of all ``batch`` GEMMs
    than the threshold).
    hipTensor's planner applies the same rule from its pack's route table."""
    lim = CF32_NATIVE_FIRST.get(arch, {}).get("cf32" if cplx else "x6")
    if lim is None:
        return None
    return int(batch) * int(M) * int(N) * int(K) * (4 if cplx else 1) < lim


def gemm_body_coef(body: str, route: str = "x6") -> Optional[Tuple[float, float, int]]:
    """``(a, b, occ)`` of ``body`` in the "cost" rule for ``route`` (None: none)."""
    return CF32_COST.get(CF32_ROUTE_ALIAS.get(route, route), {}).get(body)


def gemm_body_cost(
    body: str,
    M: int,
    Nk: int,
    Kk: int,
    *,
    route: str = "x6",
    batch: int = 1,
    arch: str = "gfx942",
    cus: Optional[int] = None,
) -> float:
    """The "cost" rule's predicted time of ``body`` (relative units; ``inf``
    without coefficients): ``a + b * ceil(tiles * batch / (CUs * occ)) * Kk``
    with ``tiles`` the body's output tiles of the kernel extents ``M x Nk``
    (cf32: ``Nk`` = 2 N) and ``Kk`` the problem's K (complex: 2 per element).
    hipTensor's planner computes the same value from its pack's route table."""
    coef = gemm_body_coef(body, route)
    if coef is None:
        return float("inf")
    a, b, occ = coef
    tile = cf32_body(body, arch)[0]
    n = arch_num_cus(arch) if cus is None else int(cus)
    tiles = (-(-int(M) // tile.tile_m)) * (-(-int(Nk) // tile.tile_n)) * int(batch)
    waves = -(-tiles // (max(1, n) * max(1, int(occ))))
    return a + b * waves * int(Kk)


def gemm_bodies_by_cost(
    M: int,
    Nk: int,
    Kk: int,
    *,
    route: str = "x6",
    batch: int = 1,
    arch: str = "gfx942",
    cus: Optional[int] = None,
    bodies: Optional[Sequence[str]] = None,
) -> List[str]:
    """``bodies`` (default: the route's, :func:`route_bodies`) cheapest first
    by :func:`gemm_body_cost` (equal costs: by name, as hipTensor's planner)."""
    names = list(route_bodies(arch, route) if bodies is None else bodies)
    cost = {
        b: gemm_body_cost(b, M, Nk, Kk, route=route, batch=batch, arch=arch, cus=cus)
        for b in names
    }
    return sorted(names, key=lambda b: (cost[b], b))


def _choose(
    M: int,
    Nk: int,
    Kh: int,
    b_rows: int,
    epi: str,
    arch: str,
    cplx: bool,
    recipe: str = "x6",
    body: Optional[str] = None,
) -> Tuple[str, bool]:
    """``(body, wide)`` of a problem (``body``: a forced body)."""
    if CF32_BODY_RULE.get(arch) == "cost":
        route = route_name(cplx, recipe)
        name = body or gemm_bodies_by_cost(M, Nk, Kh // 2, route=route, arch=arch)[0]
        if name in _VP_BODIES and _vp_needs_wide(name, M, Nk, Kh, b_rows):
            return (body or wide_body(arch, route)), True
        return name, False
    wide = _needs_wide_k(M, Nk, Kh, b_rows)
    return (body or _body_for_k(M, Nk, wide, epi, arch, cplx=cplx, Kh=Kh)), wide


def _adjust_k(
    spec: UniversalGemmSpec, M: int, Nk: int, wide: bool
) -> UniversalGemmSpec:
    """Launch adjustments of a body for kernel extents ``M x Nk``."""
    t, tr = spec.tile, spec.trait
    upd: Dict[str, Any] = {}
    rm, rn = M % t.tile_m != 0, Nk % t.tile_n != 0
    if tr.vgpr_prefetch:
        # the register-staged body: buffer ranges end at the operand (pad_m /
        # pad_n bound the ragged tiles); 64-bit batch and tile origins
        if rm or rn:
            upd.update(pad_m=tr.pad_m or rm, pad_n=tr.pad_n or rn)
        if wide:
            upd["wide_addr"] = True
        return replace(spec, trait=replace(tr, **upd)) if upd else spec
    if rm or rn:
        upd.update(
            pad_m=tr.pad_m or rm,
            pad_n=tr.pad_n or rn,
            dtl_tight_rsrc=True,
            dtl_tight_ragged_only=True,
        )
        if tr.dtl_soff0:
            upd["dtl_soff0"] = False
        if tr.rilv_vec_epilogue and not tr.rve_pad:
            upd["rve_pad"] = 1
    if wide:
        upd.update(wide_addr=True, persistent_ctas=0, streamk=False)
        if tr.rve_pad == 2:
            upd["rve_pad"] = 1
        if not (tr.dtl_soff0 or tr.dtl_tight_rsrc or upd.get("dtl_tight_rsrc")):
            upd["dtl_soff0"] = True
    return replace(spec, trait=replace(tr, **upd)) if upd else spec


def cf32_body_for(
    M: int, Nc: int, Kc: int, *, arch: str = "gfx950", epi: str = "", recipe: str = "x6"
) -> str:
    """The body of :func:`cf32_body` the cf32 route picks for this problem."""
    M, Nc, Kc = int(M), int(Nc), int(Kc)
    return _choose(M, 2 * Nc, 4 * Kc, Nc, epi, arch, True, recipe)[0]


def f32_body_for(
    M: int, N: int, K: int, *, arch: str = "gfx950", epi: str = "", recipe: str = "x6"
) -> str:
    """The body of :func:`cf32_body` the real f32 route picks for this problem."""
    M, N, K = int(M), int(N), int(K)
    return _choose(M, N, 2 * K, N, epi, arch, False, recipe)[0]


# HIP's grid.y / grid.z limit
_GRID_YZ_MAX = 65535


def gemm_launch_limits(spec: UniversalGemmSpec) -> Dict[str, int]:
    """The launch limits of a body launched on extents padded to tile multiples
    (no ``pad_m`` / ``pad_n``), as hipTensor's kernel pack records them:

    * ``limit_operand_bytes``: every operand (all batches of a batched launch)
      below this many bytes (the 32-bit forms);
    * ``limit_c_elements``: C (all batches) at most this many elements;
    * ``limit_tile_operand_bytes`` / ``limit_c_tile_elements``: the
      64-bit-address form (``wide_addr``): one tile's operand rows (a complex B
      tile: ``tile_n / 2`` rows) and one tile's C rows;
    * ``limit_grid_yz``: rows of tiles (grid y) and batches (grid z).

    The ring bodies: the DMA buffer range and i32 element offsets. The
    register-staged bodies (``vgpr_prefetch``): unsigned 32-bit byte offsets of
    each operand from its base (i32 batch element offsets, doubled) and C
    within 2**31 elements; ``wide_addr`` checks a tile's span only (see
    ``launch_shape_error`` / ``batched_launch_error``)."""
    if spec.trait.vgpr_prefetch:
        return dict(
            limit_operand_bytes=1 << 32,
            limit_c_elements=1 << 31,
            limit_tile_operand_bytes=DTL_BUFFER_NUM_RECORDS,
            limit_c_tile_elements=1 << 31,
            limit_grid_yz=_GRID_YZ_MAX,
        )
    return dict(
        limit_operand_bytes=DTL_BUFFER_NUM_RECORDS,
        limit_c_elements=(1 << 31) - 1,
        limit_tile_operand_bytes=DTL_BUFFER_NUM_RECORDS,
        limit_c_tile_elements=1 << 31,
        limit_grid_yz=_GRID_YZ_MAX,
    )


@dataclass
class F32xPlan:
    """A planned emulated-f32 launch (real or complex): the spec (prologue
    attached), its kernel, grid and the stream-K workspace size (0 when none).
    ``M`` / ``N`` / ``K`` are the problem's extents (complex extents for cf32);
    :attr:`dims` gives the kernel's runtime extents."""

    M: int
    N: int
    K: int
    cplx: bool
    body: str
    spec: UniversalGemmSpec
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    workspace_bytes: int
    arch: str = "gfx950"
    _kernel: Any = None

    @property
    def Nc(self) -> int:
        """cf32: the complex N extent (= ``N``)."""
        return self.N

    @property
    def Kc(self) -> int:
        """cf32: the complex K extent (= ``K``)."""
        return self.K

    @property
    def kernel(self) -> Any:
        if self._kernel is None:
            self._kernel = build_universal_gemm(self.spec, self.arch)
        return self._kernel

    @property
    def dims(self) -> Dict[str, int]:
        """The kernel's runtime GEMM extents (K in 16-bit halves of the f32 pair
        view; cf32: the real embedding's 2 N columns and 4 K halves)."""
        if self.cplx:
            return {"M": self.M, "N": 2 * self.N, "K": 4 * self.K}
        return {"M": self.M, "N": self.N, "K": 2 * self.K}

    def workspace_split(self) -> Tuple[int, int]:
        """``(flags_bytes, partials_bytes)`` of the stream-K workspace (SkFlags
        at offset 0, zeroed once; SkPart at ``flags_bytes``)."""
        return streamk_workspace_bytes(self.spec)

    def describe(self) -> str:
        tr = self.spec.trait
        bits = [self.body]
        for k in ("pad_m", "pad_n", "dtl_tight_rsrc", "wide_addr", "streamk"):
            if getattr(tr, k):
                bits.append(k)
        if tr.rve_pad:
            bits.append(f"rve_pad{tr.rve_pad}")
        return "+".join(bits)


Cf32Plan = F32xPlan


def _plan(
    M: int,
    N: int,
    K: int,
    *,
    cplx: bool,
    arch: str,
    conj_a: bool,
    conj_b: bool,
    recipe: str,
    epi: str,
    body: Optional[str],
) -> F32xPlan:
    what = "cf32" if cplx else "f32"
    if arch not in CF32_ARCHES:
        raise ValueError(f"{what} route: arch {arch!r} not in {CF32_ARCHES}")
    M, N, K = int(M), int(N), int(K)
    if min(M, N, K) < 1:
        raise ValueError(f"{what} route: empty problem {M}x{N}x{K}")
    Nk, Kh = (2 * N, 4 * K) if cplx else (N, 2 * K)
    name, wide = _choose(M, Nk, Kh, N, epi, arch, cplx, recipe, body)
    tile, trait, bs = cf32_body(name, arch)
    spec = UniversalGemmSpec(
        name=f"{what}_{name.lower()}",
        tile=tile,
        trait=trait,
        data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
        block_size=bs,
    )
    spec = _adjust_k(spec, M, Nk, wide)
    if spec.trait.vgpr_prefetch and Kh % tile.tile_k:
        # the register-staged body's K mask (the ring bodies have no K tail)
        spec = replace(spec, trait=replace(spec.trait, pad_k=True))
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"{what} route: {name} spec invalid for {M}x{N}x{K}: {why}")
    pro: Dict[str, Any] = dict(
        src="f32",
        recipe=recipe,
        out_f32=True,
        split_at="use",
        cplx_b=cplx,
        conj_a=conj_a,
        conj_b=conj_b,
        epi=epi,
        # packed residual subtract (v_pk_add_f32)
        psub=True,
    )
    sm = split_mode(arch, route_name(cplx, recipe))
    if sm != "rne":
        pro["split_mode"] = sm
    attach_prologue(spec, Prologue(**pro))
    err = launch_shape_error(spec, M, Nk, Kh)
    if err is not None:
        raise ValueError(f"{what} route: {name} cannot launch {M}x{N}x{K}: {err}")
    t = spec.tile
    tm, tn = -(-M // t.tile_m), -(-Nk // t.tile_n)
    tr = spec.trait
    if tr.streamk:
        grid = (int(tr.persistent_ctas), 1, 1)
    elif tr.persistent_ctas:
        grid = (min(int(tr.persistent_ctas), tm * tn), 1, 1)
    else:
        grid = (tn, tm, 1)
    fb, pb = streamk_workspace_bytes(spec)
    return F32xPlan(
        M=M,
        N=N,
        K=K,
        cplx=cplx,
        body=name,
        spec=spec,
        grid=grid,
        block=(spec.block_size, 1, 1),
        workspace_bytes=fb + pb,
        arch=arch,
    )


def plan_cf32_gemm(
    M: int,
    Nc: int,
    Kc: int,
    *,
    arch: str = "gfx950",
    conj_a: bool = False,
    conj_b: bool = False,
    recipe: str = "x6",
    epi: str = "",
    body: Optional[str] = None,
) -> F32xPlan:
    """Plan ``D[M, Nc] = alpha opA(A) opB(B)^T (+ beta C)`` (cf32, interleaved,
    A ``[M, Kc]`` and B ``[Nc, Kc]`` K-contiguous, D / C ``[M, Nc]``).

    Kernel params: ``A`` and ``B`` (the complex tensors as 16-bit views of
    ``4 Kc`` halves per row), ``C`` (D, f32 ``[M, 2 Nc]``), ``M``, ``N = 2 Nc``,
    ``K = 4 Kc``; ``alpha_re`` / ``alpha_im`` (``epi`` "alpha" / "bilinear"),
    ``beta_re`` / ``beta_im`` and ``Cin`` (f32 ``[M, 2 Nc]``, "bilinear");
    ``SkFlags`` / ``SkPart`` for a stream-K body (:meth:`F32xPlan.workspace_split`).
    Raises ``ValueError`` when the problem is outside every body's contract."""
    return _plan(
        M,
        Nc,
        Kc,
        cplx=True,
        arch=arch,
        conj_a=conj_a,
        conj_b=conj_b,
        recipe=recipe,
        epi=epi,
        body=body,
    )


def plan_f32_gemm(
    M: int,
    N: int,
    K: int,
    *,
    arch: str = "gfx950",
    recipe: str = "x6",
    epi: str = "",
    body: Optional[str] = None,
) -> F32xPlan:
    """Plan the real f32 GEMM ``D[M, N] = alpha A[M, K] B[N, K]^T (+ beta C)``
    with the bf16-split emulation (``x6``: f32 accuracy class, ``x3``: TF32
    class) on the same bodies and launch rules as :func:`plan_cf32_gemm`.

    Kernel params: ``A`` / ``B`` (the f32 tensors as 16-bit views of ``2 K``
    halves per row), ``C`` (D, f32), ``M``, ``N``, ``K = 2 K``; ``alpha_re``
    (``epi`` "alpha" / "bilinear"), ``beta_re`` and ``Cin`` ("bilinear");
    ``SkFlags`` / ``SkPart`` for a stream-K body."""
    return _plan(
        M,
        N,
        K,
        cplx=False,
        arch=arch,
        conj_a=False,
        conj_b=False,
        recipe=recipe,
        epi=epi,
        body=body,
    )
