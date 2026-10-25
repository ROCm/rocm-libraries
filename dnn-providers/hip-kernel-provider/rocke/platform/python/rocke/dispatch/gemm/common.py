# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Common helpers for GEMM-family dispatcher cases."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from typing import Optional

from ...core.arch import ArchTarget
from ...helpers.split_k import arch_num_cus, select_split_k
from ...instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
)
from ..core import OperatorRequest, normalize_selector


@dataclass(frozen=True)
class GemmRequest(OperatorRequest):
    """Normalized GEMM request shared by GEMM-family dispatchers.

    Phase 1 uses this for FP16 RCR UniversalGemm. Later GEMM-family cases can
    either extend this type or define more specialized request types beside
    their case module.
    """

    M: int
    N: int
    K: int
    arch: str
    op: str = "gemm"
    dtype: str = "fp16"
    layout: str = "RCR"
    trans_a: bool = False
    trans_b: bool = True
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = normalize_dtype(self.dtype)
        d["layout"] = self.layout.upper()
        d["algorithm"] = normalize_selector(self.algorithm)
        d["spec_id"] = normalize_selector(self.spec_id)
        return d

    def dims(self) -> dict[str, int]:
        return {"M": int(self.M), "N": int(self.N), "K": int(self.K)}


GEMM_DIM_VOCABULARY = ("M", "N", "K")


def normalize_dtype(dtype: str) -> str:
    d = dtype.lower()
    if d in ("f16", "half"):
        return "fp16"
    return d


def basic_gemm_request_errors(req: OperatorRequest) -> list[str]:
    """Common shape/op checks shared by GEMM-family dispatchers."""
    if not isinstance(req, GemmRequest):
        return [f"expected GemmRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "gemm":
        errors.append(f"unsupported op {req.op!r}")
    for field in ("M", "N", "K"):
        if int(getattr(req, field)) <= 0:
            errors.append(f"{field} must be positive")
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    return errors


def rcr_request_errors(req: OperatorRequest, *, dtype: str) -> list[str]:
    """Shared RCR-layout request validation parametrized by element dtype.

    Used by every RCR UniversalGemm family module (fp16, bf16, ...) so the
    layout/transpose checks and the single-dtype gate live in one place. The
    family passes the canonical dtype it implements (e.g. ``"fp16"``, ``"bf16"``).
    """
    errors = basic_gemm_request_errors(req)
    if errors:
        return errors
    assert isinstance(req, GemmRequest)
    if normalize_dtype(req.dtype) != dtype:
        errors.append(
            f"unsupported dtype {req.dtype!r}; this family supports {dtype} only"
        )
    if req.layout.upper() != "RCR":
        errors.append(f"unsupported layout {req.layout!r}; RCR only")
    if req.trans_a or not req.trans_b:
        errors.append("RCR expects A row-major and B logically transposed")
    return errors


def apply_split_k(req: GemmRequest, spec: UniversalGemmSpec) -> UniversalGemmSpec:
    """Return ``spec`` with a split-K degree chosen for ``req`` on its arch.

    Split-K is engaged only for skinny / tall-N decode shapes whose base grid
    leaves the CU-rich CDNA device idle, and only on the MFMA (CDNA) family the
    kernel's atomic-add epilogue supports. For any shape that already fills the
    device -- and on RDNA, where the split-K epilogue is not wired -- the chosen
    degree is ``1`` and this returns the spec **unchanged** (so the default /
    square-GEMM path stays byte-identical). The ``ROCKE_GEMM_SPLIT_K`` env flag
    overrides the heuristic (see :mod:`rocke.helpers.split_k`).
    """
    target = ArchTarget.from_gfx(req.arch)
    if target.family != "cdna":
        return spec
    t = spec.tile
    decision = select_split_k(
        M=req.M,
        N=req.N,
        K=req.K,
        tile_m=t.tile_m,
        tile_n=t.tile_n,
        tile_k=t.tile_k,
    )
    if decision.split_k <= 1:
        return spec
    return replace(spec, trait=replace(spec.trait, split_k=decision.split_k))


# ---------------------------------------------------------------------------
# DTL ping-pong candidate geometry, shared by the fp16 / bf16 RCR registries
# (see examples/gfx950/gemm_perf_square_warpspec): a 256x256 block of 16 waves
# (4x4, 64x64 per wave) whose A/B tiles
# are DMA'd straight to LDS (buffer_load ... lds), ping-ponged with a depth-1
# prefetch, read through the element-granular XOR swizzle, with A and B cached
# in L2 (CACHE_ALL) and the chiplet-aware grid swizzle.
#   * gfx950 (160 KiB LDS, 4-dword DTL): 256x256x64, 16x16x32 atoms -> 128 KiB.
#   * gfx942 (64 KiB LDS, 1-dword DTL):  256x256x32, 16x16x16 atoms ->  64 KiB.
# ---------------------------------------------------------------------------

# Arches the ping-pong candidate is dispatched on: those with a geometry
# (``dtl_pingpong_tile_trait``) AND a cost model calibrated on that hardware
# (``DTL_PINGPONG_COST_MODELS``), every calibration point checked against an
# f64 full-output reference before and after timing.
DTL_PINGPONG_ARCHES = ("gfx942", "gfx950")
# Arches with a ping-pong geometry (sweeps / explicit selection).
DTL_PINGPONG_GEOMETRY_ARCHES = ("gfx942", "gfx950")

# Ring body levers of the dispatched candidate, per arch (each checked on
# device against a full-output f64 reference): the alias-scoped two-stage DTL ring (``dtl_stages=2``: per-stage A/B LDS
# allocations + ``lds_alias_scopes``, which removes the compiler-inserted
# ``s_waitcnt vmcnt(0)`` in front of every K tile's first ds_read) with the
# MFMA-cluster priority bookends (``interwave_setprio``). An arch absent here
# keeps the ping-pong body.
DTL_PINGPONG_RING: dict = {
    "gfx950": {"dtl_stages": 2, "interwave_setprio": True},
}


def dtl_pingpong_tile_trait(
    arch: str, *, chiplet_chunk_size: int = 64, ring: Optional[bool] = None
) -> tuple[TileSpec, TraitSpec]:
    """The ping-pong tile and trait for ``arch`` (gfx942 / gfx950).

    ``ring``: ``None`` (default) applies the arch's :data:`DTL_PINGPONG_RING`
    levers, ``False`` gives the ping-pong body, ``True`` forces the gfx950 ring
    levers on any arch with a geometry."""
    if arch == "gfx950":
        tile_k, wtk = 64, 32
    elif arch == "gfx942":
        tile_k, wtk = 32, 16
    else:
        raise ValueError(f"no DTL ping-pong GEMM geometry for {arch!r}")
    tile = TileSpec(
        tile_m=256,
        tile_n=256,
        tile_k=tile_k,
        warp_m=4,
        warp_n=4,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=wtk,
    )
    trait = TraitSpec(
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
        chiplet_chunk_size=chiplet_chunk_size,
    )
    if ring is None:
        levers = DTL_PINGPONG_RING.get(arch, {})
    else:
        levers = DTL_PINGPONG_RING["gfx950"] if ring else {}
    if levers:
        trait = replace(trait, **levers)
    return tile, trait


def dtl_pingpong_sweep_variants(
    spec: UniversalGemmSpec, arch: str
) -> tuple[UniversalGemmSpec, ...]:
    """The ping-pong candidate's sweep neighbourhood (selected spec first).

    Chiplet chunk 32 / off and the cshuffle epilogue (valid on gfx950, where the
    LDS gate counts the C tile aliased onto A/B; over the 64 KiB gfx942 budget,
    so dropped there), so a registry sweep sees the relevant variants instead of
    the single auto pick. Every variant is validity-checked.
    """
    from ...instances.common.gemm_universal import is_valid_spec

    tr = spec.trait
    out = [spec]
    variants = [
        replace(tr, chiplet_chunk_size=32),
        replace(tr, chiplet_swizzle=False),
        replace(tr, epilogue="cshuffle"),
    ]
    if tr.dtl_stages >= 2:
        # ring neighbourhood: the ping-pong body, the persistent overlapped
        # epilogue (one workgroup per CU), and the in-stream barrier
        variants += [
            replace(tr, dtl_stages=0, interwave_setprio=False),
            replace(tr, persistent_ctas=arch_num_cus(arch), chiplet_chunk_size=32),
            replace(tr, dtl_stream_barrier=True, interwave_setprio=False),
            # the persistent ring with the DTL resource hoist (short-K
            # shapes) and iglp_opt(1) (large-K shapes); see
            # examples/gfx950/gemm_perf_square_warpspec/DTL_RING_SHAPES_CASE_STUDY.md
            replace(
                tr,
                persistent_ctas=arch_num_cus(arch),
                chiplet_chunk_size=32,
                dtl_hoist=True,
            ),
            replace(tr, iglp_opt=1),
            # narrower chiplet super-tile groups (wide long-K grids; a sweep
            # variant here, the big-grid rule below selects it)
            replace(tr, iglp_opt=1, chiplet_wgm=4),
        ]
    for t2 in variants:
        if t2 == tr:
            continue
        cand = replace(spec, trait=t2)
        if is_valid_spec(cand, arch=arch)[0]:
            out.append(cand)
    if tr.dtl_stages >= 2 and arch == "gfx950":
        # the 4-wave AGPR interleave body (2x2 waves of 128x128, tile_k 32,
        # 3-stage ring) with chiplet wgm 8 (mid-size grids) and wgm 4 (very
        # large grids)
        ilv_tile = replace(spec.tile, tile_k=32, warp_m=2, warp_n=2)
        ilv = replace(
            tr,
            dtl_stages=3,
            dtl_frag_prefetch=True,
            acc_agpr=True,
            mfma_interleave=True,
            ilv_merge=True,
            ilv_barrier_after=2,
            lds_swizzle_r=1,
            dtl_hoist=True,
            chiplet_chunk_size=64,
        )
        for wgm in (8, 4):
            cand = replace(
                spec,
                tile=ilv_tile,
                trait=replace(ilv, chiplet_wgm=wgm),
                block_size=4 * spec.wave_size,
            )
            if is_valid_spec(cand, arch=arch)[0]:
                out.append(cand)
    if tr.dtl_stages >= 2 and arch in RILV_ARCHES:
        # row-interleave body: both K-tile schedules on the grid mappings the
        # dispatcher selects for some shape class
        grids = (
            dict(chiplet_wgm=16, chiplet_chunk_size=8, persistent_ctas=0),
            dict(
                chiplet_wgm=16, chiplet_chunk_size=8, persistent_ctas=arch_num_cus(arch)
            ),
            dict(
                chiplet_wgm=8, chiplet_chunk_size=16, persistent_ctas=arch_num_cus(arch)
            ),
            dict(
                chiplet_wgm=8, chiplet_chunk_size=32, persistent_ctas=arch_num_cus(arch)
            ),
            dict(chiplet_wgm=8, chiplet_chunk_size=64, persistent_ctas=0),
        )
        for hb, sia in ((False, False), (True, False), (True, True)):
            b_tile, b_trait = rilv_tile_trait(arch, hb, sia=sia)
            for g in grids:
                cand = replace(
                    spec,
                    tile=b_tile,
                    trait=replace(b_trait, **g),
                    block_size=4 * spec.wave_size,
                )
                if cand not in out and is_valid_spec(cand, arch=arch)[0]:
                    out.append(cand)
        # the hand-ordered (asm_loop) body on the same grids and on the
        # chunk-4 long-K grid
        a_tile, a_base = rilv_tile_trait(arch, False)
        a_trait = rilv_asm_loop_trait(a_base, False)
        for g in grids + (
            dict(chiplet_wgm=16, chiplet_chunk_size=4, persistent_ctas=0),
        ):
            cand = replace(
                spec,
                tile=a_tile,
                trait=replace(a_trait, **g),
                block_size=4 * spec.wave_size,
            )
            if cand not in out and is_valid_spec(cand, arch=arch)[0]:
                out.append(cand)
        # the dispatched tuned body (dtl_soff0 + dtl_addr_spread + hb_dtl_delay
        # + kloop_unroll on hb_sched) on the same grids, both unrolls
        for ku in (RILV_KLOOP_UNROLL_SHORT, RILV_KLOOP_UNROLL_LONG):
            x_trait = rilv_tuned_trait(a_base, kloop_unroll=ku)
            for g in grids:
                cand = replace(
                    spec,
                    tile=a_tile,
                    trait=replace(x_trait, **g),
                    block_size=4 * spec.wave_size,
                )
                if cand not in out and is_valid_spec(cand, arch=arch)[0]:
                    out.append(cand)
    return tuple(out)


# ---------------------------------------------------------------------------
# DTL-ring candidates for the shapes the 256x256 ping-pong / ring candidate
# serves badly or not at all (gfx950). The rules and thresholds below were
# calibrated on gfx950 timings (every kernel checked against a full-output f64
# reference) and then checked on held-out shapes; see
# examples/gfx950/gemm_perf_square_warpspec/DTL_RING_SHAPES_CASE_STUDY.md.
#   * a small-tile ring (128x128x64, 8 waves of 64x32, 2-stage ring + MFMA
#     priority bookends + DTL resource hoist) when the 256x256 tiles fill the
#     last wave poorly (exactly one wave of 256x256 tiles stays with the
#     aligned ring); when its tiles fit one wave (one workgroup per CU) the
#     3-stage fragment-prefetch interleave form of it, which hides the DMA
#     latency no co-resident workgroup can; for K % 64 == 32 a 128x128x32 /
#     256x128x32 interleave ring;
#   * a pad ring (the 16-wave persistent ring, P = #CUs, chiplet chunk 32)
#     for ragged M / N.
# Both carry pad_m / pad_n with exact DTL buffer bounds (``dtl_tight_rsrc``)
# and the full-tile epilogue fast path (``pad_fast_tile``), so any M / N with
# K % tile_k == 0 is admitted by the shape gate.
#
# K gates. Where the aligned default candidate can take the
# shape (M, N, K multiples of its tile) the small ring is held to the same K
# floor as the aligned ring (``DTL_RING_ADMIT_RULES`` k_lo), and the
# K % 64 == 32 bodies to their own floors (no ring body pays at the smallest
# K). Ragged shapes have no other candidate, so they are not K-gated; at small
# K on a moderate grid the small ring takes them instead of the pad ring.
# ---------------------------------------------------------------------------
DTL_RING_SMALL_ARCHES = ("gfx950",)
DTL_RING_PAD_ARCHES = ("gfx950",)
# the small-tile ring takes over below this last-wave fill of 256x256 tiles
DTL_RING_SMALL_MAX_FILL = 0.75
# ... and only from half a wave of its own tiles on (smaller problems are
# latency-bound and were not calibrated: they keep the cshuffle default)
DTL_RING_SMALL_MIN_TILES_PER_CU = 0.5
# K % 64 == 32: the 128x128x32 body on up to this many waves of 256x128 tiles,
# else the 256x128x32 AGPR body; their K floors (aligned shapes)
DTL_RING_SMALL_K32_C128_MAX_WAVES = 2
DTL_RING_SMALL_K32_MIN_K = 96
DTL_RING_SMALL_K32_AGPR_MIN_K = 160
# ragged shapes: the small ring instead of the pad ring up to this K and below
# this many 256x256 output tiles
DTL_RING_SMALL_RAGGED_MAX_K = 512
DTL_RING_SMALL_RAGGED_MAX_TILES = 1280
# the K % 64 == 0 small ring as the 3-stage fragment-prefetch interleave body
# when its 128x128 tiles fit one wave (one workgroup per CU) and K >= this
DTL_RING_SMALL_S3_MIN_K = 128
DTL_RING_SMALL_S3_SINGLE_WAVE = True

# Very large grids (>= DTL_BIG_GRID_TILES 256x256 tiles; the rule was checked
# on shapes held out when it was written). Chiplet wgm 4 (narrower super-tile
# groups) pays at small K, on very large tile counts at any K, once the A + B
# footprint passes about three quarters of the MALL, and on wide grids from a
# moderate K on. There the aligned candidate takes the 16-wave ring with wgm 4
# + iglp_opt(1) while A + B fit the MALL, and the persistent 4-wave AGPR
# interleave body beyond it and at small K; ragged shapes take the AGPR
# interleave pad body, on a slightly lower tile count than the aligned regime.
# Elsewhere (moderate tile count, a small footprint) the 16-wave wgm-8 ring
# stays. The effect is not L2 reuse (the L2 hit rate drops with wgm 4): the
# narrower groups spread the concurrent DTL streams over the XCDs, which
# shortens the DTL latency per K tile.
DTL_BIG_GRID_TILES = 9000
DTL_BIG_GRID_ALL_K_TILES = 30000
DTL_BIG_GRID_FOOTPRINT_BYTES = 192 << 20
DTL_BIG_GRID_MALL_BYTES = 256 << 20
DTL_BIG_GRID_SMALL_K = 128
DTL_BIG_GRID_WIDE_TILES = 20000
DTL_BIG_GRID_RAGGED_WIDE_TILES = 12000
DTL_BIG_GRID_WIDE_MIN_K = 512


def _tiles(M: int, N: int, tm: int, tn: int) -> int:
    return -(-int(M) // tm) * -(-int(N) // tn)


def dtl_big_grid_body(req: GemmRequest, *, ragged: bool = False) -> Optional[str]:
    """The big-grid body of the 256x256 ring candidates for ``req``:
    ``"agpr_ilv"`` (persistent AGPR interleave body, wgm 4), ``"ring_w4"`` (16-wave ring, wgm 4 + iglp_opt(1);
    aligned only) or ``None`` (the candidate's own ring body). See the
    :data:`DTL_BIG_GRID_TILES` block."""
    if req.arch != "gfx950":
        return None
    t = _tiles(req.M, req.N, 256, 256)
    if t < DTL_BIG_GRID_TILES:
        return None
    K = int(req.K)
    footprint = (int(req.M) + int(req.N)) * K * 2
    if K <= DTL_BIG_GRID_SMALL_K:
        return "agpr_ilv"
    wide = DTL_BIG_GRID_RAGGED_WIDE_TILES if ragged else DTL_BIG_GRID_WIDE_TILES
    if not (
        t >= DTL_BIG_GRID_ALL_K_TILES
        or footprint > DTL_BIG_GRID_FOOTPRINT_BYTES
        or (t >= wide and K >= DTL_BIG_GRID_WIDE_MIN_K)
    ):
        return None
    if ragged or footprint > DTL_BIG_GRID_MALL_BYTES:
        return "agpr_ilv"
    return "ring_w4"


def dtl_big_grid(req: GemmRequest) -> bool:
    """gfx950 problem whose aligned big-grid body is the persistent AGPR
    interleave body
    (:func:`dtl_big_grid_body`)."""
    return dtl_big_grid_body(req) == "agpr_ilv"


def agpr_ilv_big_grid_tile_trait(arch: str, **extra) -> tuple[TileSpec, TraitSpec]:
    """The AGPR interleave body (2x2 waves of 128x128, 16x16x32 atoms, tile_k 32, 3-stage ring,
    fragment prefetch + merged MFMA interleave, barrier after 2 MFMAs, swizzle
    R = 1, DTL resource hoist, AGPR accumulators) persistent on every CU with
    chiplet wgm 4 / chunk 64."""
    tile = TileSpec(
        tile_m=256,
        tile_n=256,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    _, base = dtl_pingpong_tile_trait(arch)
    trait = replace(
        base,
        chiplet_wgm=4,
        chiplet_chunk_size=64,
        dtl_stages=3,
        interwave_setprio=True,
        dtl_frag_prefetch=True,
        acc_agpr=True,
        mfma_interleave=True,
        ilv_merge=True,
        ilv_barrier_after=2,
        lds_swizzle_r=1,
        dtl_hoist=True,
        persistent_ctas=arch_num_cus(arch),
        **extra,
    )
    return tile, trait


# ---------------------------------------------------------------------------
# Row-interleave ("rilv") body: the 4-wave AGPR body (2x2 waves of 128x128,
# 16x16x32 atoms, tile_k 64, alias-scoped 2-stage DTL ring, fragment prefetch,
# DTL resource hoist) with an LDS row interleave in hipBLASLt's layout
# (``lds_row_interleave``: unswizzled 128-byte rows, one base VGPR per fragment
# stream, no VALU in the K loop) and the 16-byte non-temporal vector epilogue
# straight from the accumulators (``rilv_vec_epilogue`` + ``rve_store_nt``).
# Two K-tile schedules, picked by K:
#   * K >= RILV_HB_MIN_K, or >= RILV_HB_BIG_TILES output tiles with
#     K >= RILV_HB_BIG_MIN_K: a slot schedule in hipBLASLt's order
#     (``hb_sched``: one fragment read every 2 MFMAs, release / acquire
#     barriers), chiplet wgm 16 / chunk 8 (long K / big grids);
#   * otherwise the merged MFMA interleave (``ilv_merge``), persistent on every
#     CU with chiplet chunk 32 for K <= RILV_PERSIST_MAX_K, else chiplet wgm 16
#     / chunk 8.
# Aligned problems only (the vector epilogue has no M / N bounds path; padded
# shapes take ``rilv_pad_tile_trait`` or the ring bodies) with at least
# RILV_MIN_TILES 256x256 output tiles and K >= RILV_MIN_K.
#
# Threshold provenance: the RILV_* / SMALL_TILE_SHORTK_* K and tile-count
# thresholds were calibrated on gfx950 timings of a fixed shape set (every
# kernel checked against a full-output f64 reference); several sit on K values
# or tile counts of that set and were not validated on held-out shapes between
# them. They are policy defaults, not derived bounds: re-calibrate them when
# the shape population changes (dsl_docs gemm-optimization-rocke.md, section
# "Row-interleave dispatch classes").
# ---------------------------------------------------------------------------
RILV_ARCHES = ("gfx950",)
RILV_HB_MIN_K = 12288
RILV_HB_BIG_TILES = 4096
RILV_HB_BIG_MIN_K = 4096
RILV_PERSIST_MAX_K = 2304
# at least one full wave of 256x256 tiles; smaller grids keep the ring bodies
RILV_MIN_TILES = 256
# smallest calibrated K; below it the short-K / ring bodies keep the shape
RILV_MIN_K = 768
# a 32-byte LDS pad per KiB (``rilv_pad``, the hipBLASLt LDS layout). With the
# 16-byte pad every ds_read_b128 takes LDS bank-conflict cycles on gfx950; with
# 32 bytes the reads are conflict free.
RILV_LDS_PAD = 32
# The sia K-tile schedule (``sia_sched`` + ``sia_plr`` + ``sia_dtl_spread``:
# operand-grouped next-atom reads every 2 MFMAs, one release, the acquire
# RILV_SIA_ACQ_GAP MFMAs into the last atom, DTL issues on the free MFMA
# slots) as an alternative to ``hb_sched`` on the long-K class. It is used by
# the ``"sia"`` RILV_BODY (bf16 only) and by the ragged long-K pad class when
# RILV_PAD_HB_TUNED is off.
RILV_SIA_DTYPES = ("bf16",)
RILV_SIA_ACQ_GAP = 12


def rilv_admits(req: GemmRequest) -> bool:
    """gfx950 problem the row-interleave body serves (aligned, enough tiles)."""
    if req.arch not in RILV_ARCHES:
        return False
    M, N, K = int(req.M), int(req.N), int(req.K)
    if M % 256 or N % 256 or K % 64 or K < RILV_MIN_K:
        return False
    return (M // 256) * (N // 256) >= RILV_MIN_TILES


def rilv_hb_schedule(M: int, N: int, K: int) -> bool:
    """True where the row-interleave body takes the ``hb_sched`` K-tile
    schedule (long K, or a big grid at moderate K; see the block comment)."""
    tiles = -(-int(M) // 256) * -(-int(N) // 256)
    return int(K) >= RILV_HB_MIN_K or (
        tiles >= RILV_HB_BIG_TILES and int(K) >= RILV_HB_BIG_MIN_K
    )


def rilv_tile_trait(
    arch: str, hb: bool, persistent: bool = True, sia: bool = False
) -> tuple[TileSpec, TraitSpec]:
    """The row-interleave tile / trait: the ``hb_sched`` body (``hb``; with
    ``sia`` the ``sia_sched`` schedule of the same class) or the
    merged-interleave body, persistent with chunk 32 (``persistent``) or with
    the wgm 16 / chunk 8 grid (see the block comment)."""
    tile = TileSpec(
        tile_m=256,
        tile_n=256,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    _, base = dtl_pingpong_tile_trait(arch)
    common = dict(
        lds_swizzle=False,
        interwave_setprio=False,
        dtl_stages=2,
        acc_agpr=True,
        dtl_frag_prefetch=True,
        mfma_interleave=True,
        dtl_hoist=True,
        lds_row_interleave=True,
        rilv_pad=RILV_LDS_PAD,
        rilv_vec_epilogue=True,
        rve_store_nt=True,
    )
    if hb and sia:
        trait = replace(
            base,
            chiplet_wgm=16,
            chiplet_chunk_size=8,
            sia_sched=True,
            sia_plr=True,
            sia_dtl_spread=True,
            sia_acq_gap=RILV_SIA_ACQ_GAP,
            **common,
        )
    elif hb:
        trait = replace(
            base,
            chiplet_wgm=16,
            chiplet_chunk_size=8,
            hb_sched=True,
            hb_read_every=2,
            **common,
        )
    elif persistent:
        trait = replace(
            base,
            chiplet_wgm=8,
            chiplet_chunk_size=32,
            ilv_merge=True,
            persistent_ctas=arch_num_cus(arch),
            **common,
        )
    else:
        trait = replace(
            base,
            chiplet_wgm=16,
            chiplet_chunk_size=8,
            ilv_merge=True,
            **common,
        )
    return tile, trait


def rilv_spec_tile_trait(req: GemmRequest) -> tuple[TileSpec, TraitSpec]:
    """The row-interleave tile / trait the dispatcher selects for an aligned
    ``req`` (:func:`rilv_body`: the tuned body by default; the other bodies
    stay selectable)."""
    hb = rilv_hb_schedule(req.M, req.N, req.K)
    persistent = int(req.K) <= RILV_PERSIST_MAX_K
    body = rilv_body()
    if body == "tuned":
        tile, trait = rilv_tuned_tile_trait(req.arch, req.M, req.N, req.K)
        if rilv_hb_loop() == "asm_loop":
            trait = rilv_asm_loop_tuned_trait(trait)
        return tile, trait
    if body == "asm_loop":
        tile, trait = rilv_tile_trait(req.arch, hb, persistent=persistent)
        return tile, rilv_asm_loop_trait(trait, hb)
    return rilv_tile_trait(
        req.arch,
        hb,
        persistent=persistent,
        sia=body == "sia" and req.dtype in RILV_SIA_DTYPES,
    )


# ---------------------------------------------------------------------------
# Tuned row-interleave body: one body for every aligned row-interleave class.
# The hb_sched K-tile schedule (one fragment read every 2 MFMAs, release /
# acquire barriers) on the row-interleave body with the conflict-free 32-byte
# LDS pad (``rilv_pad``), the DTL origin in the buffer resource base with an
# inline soffset of 0 (``dtl_soff0``: one VMEM issue cycle per DTL instead of
# two), the per-tile DTL address SALU spread over free MFMA slots
# (``dtl_addr_spread``), the first DTL one slot after the release
# (``hb_dtl_delay``) and an unrolled K loop (``kloop_unroll``). Grid:
# persistent on every CU (chiplet wgm 8 / chunk 32) when K <=
# RILV_PERSIST_MAX_K and the grid has more than RILV_PERSIST_MIN_WAVES waves
# of 256x256 tiles (the workgroup turnover the persistent loop saves pays only
# beyond a couple of waves); else non-persistent chiplet wgm 16 / chunk 8.
# kloop_unroll RILV_KLOOP_UNROLL_LONG for long K (> RILV_PERSIST_MAX_K),
# RILV_KLOOP_UNROLL_SHORT otherwise (a longer unroll leaves too long a tail at
# short K), and two main-loop copies per SIMD parity (``hb_skew``).
#
# Body selection (``rilv_body``; ROCKE_GEMM_RILV_BODY overrides RILV_BODY):
#   "tuned"    the body above (default);
#   "asm_loop" the hand-ordered asm-loop body (``rilv_asm_loop_trait``);
#   "sia"      the base body with the sia schedule on the hb class (bf16);
#   "base"     the base row-interleave body (``rilv_tile_trait``).
# The non-default bodies are retained as selectable alternatives; they are not
# dispatched by default.
# ---------------------------------------------------------------------------
RILV_BODY_ENV = "ROCKE_GEMM_RILV_BODY"
RILV_BODIES = ("tuned", "asm_loop", "sia", "base")
RILV_BODY = "tuned"
RILV_PERSIST_MIN_WAVES = 2
RILV_KLOOP_UNROLL_LONG = 4
RILV_KLOOP_UNROLL_SHORT = 2
# two main-loop copies per SIMD parity on the hb_sched body (the odd-SIMD copy
# reads / issues one MFMA later, so SIMD pairs do not hit LDS in the same
# slot; ``hb_skew``)
RILV_HB_SKEW = 1


def rilv_body() -> str:
    """The row-interleave body the dispatcher builds: ``ROCKE_GEMM_RILV_BODY``
    if set, else :data:`RILV_BODY`. An unknown value raises."""
    v = os.environ.get(RILV_BODY_ENV, "").strip().lower() or RILV_BODY
    if v not in RILV_BODIES:
        raise ValueError(f"{RILV_BODY_ENV}={v!r}: expected one of {RILV_BODIES}")
    return v


def rilv_persistent(M: int, N: int, K: int, arch: str) -> bool:
    """True where the tuned row-interleave body runs persistent (see the
    block)."""
    tiles = -(-int(M) // 256) * -(-int(N) // 256)
    return int(
        K
    ) <= RILV_PERSIST_MAX_K and tiles > RILV_PERSIST_MIN_WAVES * arch_num_cus(arch)


# ---------------------------------------------------------------------------
# Instruction economy on the tuned body: one explicit lgkmcnt(0) at the end of
# every K tile (LLVM then drops its per-MFMA lgkmcnt(N) waits in the tile;
# ``hb_tile_lgkm0``) and the per-tile DTL address steps placed after the
# tile's last fragment read (``das_late``; otherwise LLVM floats them into the
# read gaps between two MFMAs). Both combine with ``mfma_serpentine``.
# ---------------------------------------------------------------------------
RILV_HB_TILE_LGKM0 = 1
RILV_DAS_LATE = True


def rilv_tuned_trait(trait: TraitSpec, *, kloop_unroll: int) -> TraitSpec:
    """The tuned body on a row-interleave trait (grid fields kept), with the
    instruction-economy levers (``RILV_HB_TILE_LGKM0`` / ``RILV_DAS_LATE``)."""
    return replace(
        trait,
        ilv_merge=False,
        sia_sched=False,
        sia_plr=False,
        sia_dtl_spread=False,
        sia_acq_gap=3,
        hb_sched=True,
        hb_read_every=2,
        dtl_soff0=True,
        dtl_addr_spread=True,
        hb_dtl_delay=1,
        kloop_unroll=kloop_unroll,
        hb_skew=RILV_HB_SKEW,
        hb_tile_lgkm0=RILV_HB_TILE_LGKM0,
        das_late=RILV_DAS_LATE,
    )


# Optional (off by default): the long-K aligned class on a different chiplet
# group (``chiplet_wgm`` / ``chiplet_chunk_size``) for mid-size grids of
# 256x256 tiles; () = the wgm 16 / chunk 8 grid everywhere.
RILV_LONGK_GRID: tuple = ()
RILV_LONGK_MIN_K = 8192
RILV_LONGK_TILES = (2048, 6400)  # inclusive 256x256-tile range


def rilv_longk_grid(M: int, N: int, K: int) -> bool:
    """Whether the long-K aligned body takes ``RILV_LONGK_GRID``."""
    if not RILV_LONGK_GRID or int(K) < RILV_LONGK_MIN_K:
        return False
    tiles = (int(M) // 256) * (int(N) // 256)
    return RILV_LONGK_TILES[0] <= tiles <= RILV_LONGK_TILES[1]


def rilv_tuned_tile_trait(
    arch: str, M: int, N: int, K: int
) -> tuple[TileSpec, TraitSpec]:
    """Tile / trait of the tuned row-interleave body for an aligned M x N x K."""
    tile, base = rilv_tile_trait(arch, False, persistent=False)
    if rilv_persistent(M, N, K, arch):
        grid = dict(
            chiplet_wgm=8, chiplet_chunk_size=32, persistent_ctas=arch_num_cus(arch)
        )
    elif rilv_longk_grid(M, N, K):
        grid = dict(
            chiplet_wgm=RILV_LONGK_GRID[0],
            chiplet_chunk_size=RILV_LONGK_GRID[1],
            persistent_ctas=0,
        )
    else:
        grid = dict(chiplet_wgm=16, chiplet_chunk_size=8, persistent_ctas=0)
    ku = (
        RILV_KLOOP_UNROLL_LONG
        if int(K) > RILV_PERSIST_MAX_K
        else RILV_KLOOP_UNROLL_SHORT
    )
    trait = rilv_tuned_trait(replace(base, **grid), kloop_unroll=ku)
    if RILV_SERPENTINE:
        trait = replace(trait, mfma_serpentine=True)
    return tile, trait


# ---------------------------------------------------------------------------
# XCD-local stream-K tail on the tuned row-interleave body (opt-in helpers,
# not dispatched). Aligned long-K grids with many waves of 256x256 tiles and an
# almost empty last wave lose most of a wave to quantisation; the persistent
# stream-K loop (``streamk`` + ``sk_xcd_local``: the owner and partners of a
# split tile on one XCD, partials through that XCD's L2) spreads the tail over
# every CU. ``rilv_streamk_tail_admits`` is the narrow class where that paid on
# gfx950: K >= RILV_STREAMK_TAIL_MIN_K, at least RILV_STREAMK_TAIL_MIN_WAVES
# waves and a last wave filled to at most RILV_STREAMK_TAIL_MAX_FILL. Outside
# it the stream-K loop lost to the data-parallel body. The thresholds are fitted
# on a small set of shapes, the gain varied between nodes, and a held-out K
# below RILV_STREAMK_TAIL_MIN_K also gained, so the rule is not validated: the
# dispatcher does not call these helpers. A stream-K kernel also takes two
# workspace arguments (``streamk_workspace_bytes``: flags zeroed once, and
# partials) that the dispatch binding does not allocate; a caller that uses
# the helpers must allocate them.
# ---------------------------------------------------------------------------
RILV_STREAMK_TAIL_MIN_K = 12288
RILV_STREAMK_TAIL_MIN_WAVES = 12
RILV_STREAMK_TAIL_MAX_FILL = 0.25


def rilv_streamk_tail_admits(M: int, N: int, K: int, arch: str) -> bool:
    """True for the aligned long-K grids with a near-empty last wave that the
    XCD-local stream-K tail targets (see the block; not dispatched)."""
    if arch not in RILV_ARCHES:
        return False
    M, N, K = int(M), int(N), int(K)
    if M % 256 or N % 256 or K % 64 or K < RILV_STREAMK_TAIL_MIN_K:
        return False
    cus = arch_num_cus(arch)
    tiles = (M // 256) * (N // 256)
    if tiles < RILV_STREAMK_TAIL_MIN_WAVES * cus:
        return False
    last = tiles % cus  # tiles of the last (partial) wave; 0 = whole waves
    return 0 < last <= RILV_STREAMK_TAIL_MAX_FILL * cus


def rilv_streamk_tail_trait(trait: TraitSpec, arch: str) -> TraitSpec:
    """``trait`` (the tuned row-interleave body) as a persistent XCD-local
    stream-K loop on every CU. Stream-K requires kloop_unroll 1; hb_skew is
    off on the stream-K loop."""
    return replace(
        trait,
        persistent_ctas=arch_num_cus(arch),
        streamk=True,
        sk_xcd_local=True,
        kloop_unroll=1,
        hb_skew=0,
    )


# ---------------------------------------------------------------------------
# K-tile emission of the tuned body (``rilv_hb_loop``; ROCKE_GEMM_RILV_HB_LOOP
# overrides RILV_HB_LOOP): "ir" (default) emits the hb_sched K tile through the
# IR scheduler; "asm_loop" emits it as a hand-ordered loop in the Tensile order
# (``asm_loop`` = "tensile", the two SIMD-parity loop copies via
# ``asm_loop_skew``), optionally with the address-step spread / DTL delay
# native to the asm table (RILV_ASM_LOOP_ADDR_SPREAD) and a StaggerU start
# offset on non-persistent grids (RILV_ASM_LOOP_STAGGER_U). "asm_loop" and both
# options are retained, default-off alternatives: on gfx950 they did not beat
# the "ir" emission.
# ---------------------------------------------------------------------------
RILV_HB_LOOP_ENV = "ROCKE_GEMM_RILV_HB_LOOP"
RILV_HB_LOOPS = ("ir", "asm_loop")
RILV_HB_LOOP = "ir"
# The boustrophedon MFMA order (``mfma_serpentine``: odd A rows walk the B
# fragments backwards, so no MFMA switches both operands at a row change) on
# the tuned body, the pad bodies and the short-K merged-interleave body: the
# same work with less operand-path toggling per MFMA.
RILV_SERPENTINE = True
RILV_ASM_LOOP_ADDR_SPREAD = False  # dtl_addr_spread + hb_dtl_delay 1 in the asm table
RILV_ASM_LOOP_STAGGER_U = 0  # StaggerU on non-persistent asm-loop grids (0 = off)
RILV_ASM_LOOP_STAGGER_STRIDE = 1


def rilv_hb_loop() -> str:
    """K-tile emission of the tuned row-interleave body:
    ``ROCKE_GEMM_RILV_HB_LOOP`` if set, else :data:`RILV_HB_LOOP`. An unknown
    value raises."""
    v = os.environ.get(RILV_HB_LOOP_ENV, "").strip().lower() or RILV_HB_LOOP
    if v not in RILV_HB_LOOPS:
        raise ValueError(f"{RILV_HB_LOOP_ENV}={v!r}: expected one of {RILV_HB_LOOPS}")
    return v


def rilv_asm_loop_tuned_trait(trait: TraitSpec) -> TraitSpec:
    """The asm-loop emission of a tuned row-interleave trait (same grid and
    kloop_unroll; see the block above)."""
    trait = replace(
        trait,
        hb_skew=0,
        hb_read_every=1,
        asm_loop="tensile",
        asm_loop_skew=1,
        dtl_addr_spread=RILV_ASM_LOOP_ADDR_SPREAD,
        hb_dtl_delay=1 if RILV_ASM_LOOP_ADDR_SPREAD else 0,
        # the instruction-economy levers apply to the IR hb_sched loop only
        hb_tile_lgkm0=0,
        das_late=False,
    )
    if RILV_ASM_LOOP_STAGGER_U and not trait.persistent_ctas:
        trait = replace(
            trait,
            stagger_u=RILV_ASM_LOOP_STAGGER_U,
            stagger_u_stride=RILV_ASM_LOOP_STAGGER_STRIDE,
        )
    return trait


# ---------------------------------------------------------------------------
# Hand-ordered main loop (the "asm_loop" RILV_BODY): the base row-interleave
# body with every K tile emitted in the Tensile loop order (``asm_loop`` =
# "tensile": per-operand release AND acquire, 4 barriers per tile, every
# fragment read / DTL issue / wait pinned to its MFMA slot) and two loop copies
# per SIMD parity (``asm_loop_skew`` = 1: odd-SIMD waves read one MFMA later,
# so SIMD pairs do not hit the LDS / TA in the same slot). The long-K class
# uses chiplet chunk RILV_ASM_LOOP_HB_CHUNK.
# ---------------------------------------------------------------------------
RILV_ASM_LOOP_HB_CHUNK = 4


def rilv_asm_loop_trait(trait: TraitSpec, hb: bool) -> TraitSpec:
    """The asm-loop version of a row-interleave trait (same grid; the long-K
    class with chiplet chunk ``RILV_ASM_LOOP_HB_CHUNK``)."""
    trait = replace(
        trait,
        ilv_merge=False,
        hb_sched=True,
        hb_read_every=1,
        asm_loop="tensile",
        asm_loop_skew=1,
    )
    if hb:
        trait = replace(trait, chiplet_chunk_size=RILV_ASM_LOOP_HB_CHUNK)
    return trait


# ---------------------------------------------------------------------------
# The row-interleave body on two more shape classes:
#   * ragged N (or M): the vector epilogue's branch-free bounds path
#     (``rve_pad`` 2: buffer stores over C, out-of-range lanes dropped by the
#     range check) with pad + exact DTL bounds (``dtl_tight_rsrc``) and the
#     full-tile fast path, in place of the pad ring / AGPR interleave bodies;
#   * aligned short K (RILV_SHORTK_MIN_K <= K < RILV_MIN_K) with >= one wave of
#     256x256 tiles filling the last wave to >= RILV_SHORTK_MIN_FILL: the
#     merged interleave body, non-persistent chiplet wgm 16 / chunk 8, in place
#     of the small-tile ring / the 16-wave ring.
# ---------------------------------------------------------------------------
RILV_SHORTK_MIN_K = 384
RILV_SHORTK_MIN_FILL = 0.6
RILV_SHORTK_DTYPES = ("fp16", "bf16")


def rilv_shortk_admits(req: GemmRequest) -> bool:
    """gfx950 aligned short-K problem the row-interleave body serves (see the
    block)."""
    if req.arch not in RILV_ARCHES:
        return False
    if normalize_dtype(req.dtype) not in RILV_SHORTK_DTYPES:
        return False
    M, N, K = int(req.M), int(req.N), int(req.K)
    if M % 256 or N % 256 or K % 64 or not RILV_SHORTK_MIN_K <= K < RILV_MIN_K:
        return False
    if (M // 256) * (N // 256) < RILV_MIN_TILES:
        return False
    return wave_fill(M, N, 256, 256, req.arch) >= RILV_SHORTK_MIN_FILL


def rilv_shortk_tile_trait(req: GemmRequest) -> tuple[TileSpec, TraitSpec]:
    """The merged-interleave row-interleave body, non-persistent wgm 16 /
    chunk 8; the persistent tuned body where ``rilv_persistent`` holds (many
    waves of short tiles)."""
    if rilv_body() == "tuned" and rilv_persistent(req.M, req.N, req.K, req.arch):
        return rilv_tuned_tile_trait(req.arch, req.M, req.N, req.K)
    if (
        RILV_SHORTK_SMALL_TILE
        and int(req.M) % 128 == 0
        and int(req.N) % 192 == 0
        and wave_fill(req.M, req.N, 256, 256, req.arch)
        < RILV_SHORTK_SMALL_TILE_MAX_FILL
    ):
        # the small-tile 128x192 body (exact tiles)
        return small_tile_128x192_exact_tile_trait()
    tile, trait = rilv_tile_trait(req.arch, False, persistent=False)
    if RILV_SERPENTINE and rilv_body() == "tuned":
        # the serpentine MFMA order on the merged-interleave body too
        trait = replace(trait, mfma_serpentine=True)
    return tile, trait


# The short-K row-interleave class (non-persistent) takes the small-tile body
# (128x192x64, 4x2 waves, 2-stage DTL ring, no chiplet swizzle, exact tiles:
# dtl_soff0 + cshuffle epilogue) where its tiles divide M and N and the
# 256x256 tiles fill their waves poorly (below
# ``RILV_SHORTK_SMALL_TILE_MAX_FILL``; with better-filled 256x256 waves the
# bigger tile stays ahead), instead of the merged-interleave 256x256 body.
RILV_SHORTK_SMALL_TILE = True
RILV_SHORTK_SMALL_TILE_MAX_FILL = 0.7


def small_tile_128x192_exact_tile_trait() -> tuple[TileSpec, TraitSpec]:
    """The small-tile 128x192x64 body with exact tiles (see
    ``small_tile_shortk_tile_trait``)."""
    trait = replace(
        _ring_pad_trait(),
        chiplet_swizzle=False,
        pad_m=False,
        pad_n=False,
        dtl_tight_rsrc=False,
        pad_fast_tile=False,
        dtl_soff0=True,
        epilogue="cshuffle",
    )
    return _ring_tile(128, 192, 64, 4, 2), trait


def rilv_pad_admits(req: GemmRequest) -> bool:
    """gfx950 ragged-M/N problem for the padded row-interleave body: N % 8 == 0 (16-byte
    vector stores), K % 64 == 0, K >= RILV_MIN_K, >= RILV_MIN_TILES
    tiles filling the last wave to >= DTL_RING_SMALL_MAX_FILL (else the
    small-tile ring), and the 32-bit DTL / C buffer ranges."""
    if req.arch not in RILV_ARCHES:
        return False
    M, N, K = int(req.M), int(req.N), int(req.K)
    if not (M % 256 or N % 256) or N % 8 or K % 64 or K < RILV_MIN_K:
        return False
    if max(M, N) * K * 2 >= 2**31 or M * N * 2 >= 2**32:
        return False
    if -(-M // 256) * -(-N // 256) < RILV_MIN_TILES:
        return False
    return wave_fill(M, N, 256, 256, req.arch) >= DTL_RING_SMALL_MAX_FILL


def rilv_pad_tile_trait(req: GemmRequest) -> tuple[TileSpec, TraitSpec]:
    """The row-interleave pick for ``req`` with pad on the ragged
    dimension(s), exact DTL bounds, the full-tile fast path and the
    branch-free vector epilogue bounds (``rve_pad`` 2)."""
    M, N, K = int(req.M), int(req.N), int(req.K)
    hb = rilv_hb_schedule(M, N, K)
    tiles = -(-M // 256) * -(-N // 256)
    # persistent chunk 32 also on big grids with K up to the hb threshold
    persistent = K <= RILV_PERSIST_MAX_K or tiles >= RILV_HB_BIG_TILES
    tuned = rilv_body() == "tuned"
    if tuned:
        # the hb_sched body (persistent, or the sia schedule on the long-K
        # class); dtl_soff0 does not combine with the exact DTL bounds the pad
        # path needs (which already issue with soffset 0)
        tile, trait = rilv_tile_trait(req.arch, hb, persistent=persistent, sia=hb)
        if hb and RILV_PAD_HB_GRID:
            # the long-K pad class on the wgm 32 / small-chunk grid
            trait = replace(
                trait,
                chiplet_wgm=RILV_PAD_HB_GRID[0],
                chiplet_chunk_size=RILV_PAD_HB_GRID[1],
            )
        if hb and RILV_PAD_HB_TUNED:
            # the tuned body (hb_sched + its levers, exact bounds on the padded
            # operand only) instead of the sia schedule, non-persistent
            trait = replace(
                trait,
                sia_sched=False,
                sia_plr=False,
                sia_dtl_spread=False,
                sia_acq_gap=3,
                hb_sched=True,
                hb_read_every=2,
                dtl_addr_spread=True,
                dtl_tight_ragged_only=True,
                hb_dtl_delay=1,
                hb_skew=RILV_HB_SKEW,
                kloop_unroll=RILV_KLOOP_UNROLL_LONG,
            )
        if not hb:
            trait = replace(
                trait,
                ilv_merge=False,
                hb_sched=True,
                hb_read_every=2,
                chiplet_wgm=8 if persistent else 16,
                chiplet_chunk_size=32 if persistent else 8,
                persistent_ctas=arch_num_cus(req.arch) if persistent else 0,
            )
            if persistent and RILV_PAD_PERSIST_TUNED:
                # the tuned body's levers on the persistent pad class
                # (dtl_addr_spread also builds the exact bounds; exact bounds
                # only on the padded operand)
                trait = replace(
                    trait,
                    dtl_addr_spread=True,
                    dtl_tight_ragged_only=True,
                    hb_dtl_delay=1,
                    hb_skew=RILV_HB_SKEW,
                    kloop_unroll=(
                        RILV_KLOOP_UNROLL_LONG
                        if K > RILV_PERSIST_MAX_K
                        else RILV_KLOOP_UNROLL_SHORT
                    ),
                )
    else:
        tile, trait = rilv_tile_trait(req.arch, hb, persistent=persistent)
    trait = replace(
        trait,
        pad_m=bool(int(req.M) % 256),
        pad_n=bool(int(req.N) % 256),
        dtl_tight_rsrc=True,
        pad_fast_tile=tiles >= RILV_PAD_FAST_TILE_MIN_TILES,
        rve_pad=2,
    )
    if tuned and RILV_SERPENTINE:
        # the serpentine MFMA order on the pad bodies too (hb_sched / sia)
        trait = replace(trait, mfma_serpentine=True)
    if tuned and trait.hb_sched and not trait.asm_loop:
        # the instruction-economy levers on the hb_sched pad bodies
        # (``RILV_PAD_HB_TILE_LGKM0`` / ``RILV_PAD_DAS_LATE``, default off)
        if RILV_PAD_HB_TILE_LGKM0:
            trait = replace(trait, hb_tile_lgkm0=RILV_HB_TILE_LGKM0)
        if RILV_PAD_DAS_LATE and trait.dtl_addr_spread:
            trait = replace(trait, das_late=RILV_DAS_LATE)
    return tile, trait


# ---------------------------------------------------------------------------
# Ragged / pad row-interleave class:
#   * the long-K pad class (non-persistent) is grid bound: chiplet wgm 32 with
#     a small chunk (RILV_PAD_HB_GRID) instead of wgm 16 / chunk 8;
#   * the persistent pad class takes the tuned body's levers (dtl_addr_spread
#     with exact bounds on the padded operand only, hb_dtl_delay 1, hb_skew,
#     kloop_unroll; RILV_PAD_PERSIST_TUNED).
# ---------------------------------------------------------------------------
RILV_PAD_HB_GRID = (32, 2)  # (chiplet_wgm, chiplet_chunk_size); () = wgm 16 / chunk 8
RILV_PAD_PERSIST_TUNED = True
# long-K pad class: the tuned body (hb_sched + dtl_addr_spread +
# dtl_tight_ragged_only + hb_dtl_delay + hb_skew + kloop_unroll) instead of the
# sia schedule
RILV_PAD_HB_TUNED = True
# The instruction-economy levers on the hb_sched pad bodies, default off.
# ``das_late`` with the exact DTL bounds (``dtl_tight_rsrc`` /
# ``dtl_tight_ragged_only``) makes LLVM spill the accumulators to scratch in
# the main loop (much slower, results still exact), so it stays off there (as
# does ``hb_read_every`` 1, same failure); ``hb_tile_lgkm0`` alone does not
# pay on these bodies.
RILV_PAD_HB_TILE_LGKM0 = False
RILV_PAD_DAS_LATE = False

# the full-tile fast path (``pad_fast_tile``) from this many 256x256 tiles on
# (it pays on large ragged grids and costs on small ones; the threshold lies
# between the calibrated grid sizes)
RILV_PAD_FAST_TILE_MIN_TILES = 2048


def ring_w4_tile_trait(arch: str) -> tuple[TileSpec, TraitSpec]:
    """The aligned ring body with chiplet wgm 4 + iglp_opt(1) (its ISA does not
    depend on what the process compiled before)."""
    tile, trait = dtl_pingpong_tile_trait(arch)
    return tile, replace(trait, chiplet_wgm=4, iglp_opt=1)


def big_grid_tile_trait(req: GemmRequest) -> Optional[tuple[TileSpec, TraitSpec]]:
    """Tile and trait of the aligned candidate's big-grid body, or ``None``."""
    body = dtl_big_grid_body(req)
    if body == "agpr_ilv":
        return agpr_ilv_big_grid_tile_trait(req.arch)
    if body == "ring_w4":
        return ring_w4_tile_trait(req.arch)
    return None


def wave_fill(M: int, N: int, tile_m: int, tile_n: int, arch: str) -> float:
    """Fill of the last wave of ``tile_m x tile_n`` output tiles over the CUs
    (1.0 = whole waves; 0.53 = 1.05 waves rounded up to 2)."""
    tiles = _tiles(M, N, tile_m, tile_n)
    waves = tiles / arch_num_cus(arch)
    return waves / max(1, -(-tiles // arch_num_cus(arch)))


def _ring_pad_trait(**extra) -> TraitSpec:
    return TraitSpec(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        pad_m=True,
        pad_n=True,
        direct_to_lds=True,
        dtl_prefetch=True,
        lds_swizzle=True,
        dtl_cache_a=0,
        dtl_cache_b=0,
        chiplet_swizzle=True,
        chiplet_wgm=8,
        chiplet_chunk_size=32,
        dtl_stages=2,
        interwave_setprio=True,
        dtl_hoist=True,
        dtl_tight_rsrc=True,
        pad_fast_tile=True,
        **extra,
    )


_S3ILV = dict(
    dtl_stages=3,
    dtl_frag_prefetch=True,
    mfma_interleave=True,
    ilv_merge=True,
    ilv_barrier_after=2,
)


# ---------------------------------------------------------------------------
# Non-power-of-two small tiles for the small ring's short-K regime (few output
# tiles, short K): the DTL ring bodies at 192x128 / 128x192 (8 waves of 96x32 /
# 32x96) and 256x128x32 (4 waves) replace the 128x128 / 256x128 small-ring
# geometries where SMALL_TILE_SHORTK_MIN_K <= K <= SMALL_TILE_SHORTK_MAX_K and
# at most SMALL_TILE_SHORTK_MAX_WAVES waves of its own tiles:
#   * K % 64 == 0, tiles of 192x128 fit one wave: 192x128x64 2x4 waves, the
#     3-stage fragment-prefetch interleave (merged, no ilv_barrier_after);
#   * K % 64 == 0, more than one wave: 128x192x64 4x2 waves, 2-stage ring,
#     no chiplet swizzle, when the last wave of its tiles is at least
#     SMALL_TILE_SHORTK_MIN_FILL full;
#   * K % 64 == 32: 256x128x32 4x1 waves, AGPR 3-stage interleave, no
#     interwave_setprio;
# When the tile divides M and N the pad / dtl_tight_rsrc / pad_fast_tile
# bounds are dropped, the DTLs issue with soffset 0 (``dtl_soff0``) and C is
# written by the cshuffle (LDS-staged, wide-store) epilogue. Every body is a
# combination of existing levers (no new kernel code);
# ``SMALL_TILE_SHORTK = False`` restores the 128x128 / 256x128 geometries.
# ---------------------------------------------------------------------------
SMALL_TILE_SHORTK = True
SMALL_TILE_SHORTK_MIN_K = 256
SMALL_TILE_SHORTK_MAX_K = 768
SMALL_TILE_SHORTK_MAX_WAVES = 3
# more than one wave of 128x192 tiles: only with a last-wave fill >= this
SMALL_TILE_SHORTK_MIN_FILL = 0.75


def _ring_tile(tm: int, tn: int, tk: int, wm: int, wn: int) -> TileSpec:
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


def small_tile_shortk_tile_trait(
    arch: str, M: int, N: int, K: int, dtype: Optional[str] = None
) -> Optional[tuple[TileSpec, TraitSpec]]:
    """The short-K small-tile body for a small-ring problem, or ``None``
    (outside its calibrated K range, off, or not gfx950)."""
    if not SMALL_TILE_SHORTK or arch not in RILV_ARCHES:
        return None
    M, N, K = int(M), int(N), int(K)
    if not SMALL_TILE_SHORTK_MIN_K <= K <= SMALL_TILE_SHORTK_MAX_K:
        return None
    cus = arch_num_cus(arch)
    if K % 64 == 0:
        if _tiles(M, N, 192, 128) <= cus:
            if M % 192 or N % 128:
                if M % 128 == 0 and N % 192 == 0:
                    tile = _ring_tile(128, 192, 64, 4, 2)
                else:
                    tile = _ring_tile(192, 128, 64, 2, 4)
            else:
                tile = _ring_tile(192, 128, 64, 2, 4)
            trait = replace(_ring_pad_trait(), **dict(_S3ILV, ilv_barrier_after=0))
        else:
            if M % 128 or N % 192:
                tile = (
                    _ring_tile(192, 128, 64, 2, 4)
                    if M % 192 == 0 and N % 128 == 0
                    else _ring_tile(128, 192, 64, 4, 2)
                )
            else:
                tile = _ring_tile(128, 192, 64, 4, 2)
            trait = replace(_ring_pad_trait(), chiplet_swizzle=False)
            if (
                wave_fill(M, N, tile.tile_m, tile.tile_n, arch)
                < SMALL_TILE_SHORTK_MIN_FILL
            ):
                # a poorly filled second wave of 128x192 tiles does not pay
                # (checked on held-out shapes): keep the 128x128 small ring
                return None
    elif K % 32 == 0:
        tile = _ring_tile(256, 128, 32, 4, 1)
        trait = replace(
            _ring_pad_trait(),
            acc_agpr=True,
            interwave_setprio=False,
            **_S3ILV,
        )
    else:
        return None
    if _tiles(M, N, tile.tile_m, tile.tile_n) > SMALL_TILE_SHORTK_MAX_WAVES * cus:
        # calibrated for the few-tile class only
        return None
    if _tiles(M, N, tile.tile_m, tile.tile_n) < DTL_RING_SMALL_MIN_TILES_PER_CU * cus:
        # fewer than half a wave of its tiles: keep the 128x128 geometry (whose
        # smaller tiles still give the small ring half a wave)
        return None
    if M % tile.tile_m == 0 and N % tile.tile_n == 0:
        # exact tiles: no bounds, the DTLs issue with soffset 0 (dtl_soff0) and
        # C leaves through the cshuffle (LDS-staged, wide-store) epilogue
        trait = replace(
            trait,
            pad_m=False,
            pad_n=False,
            dtl_tight_rsrc=False,
            pad_fast_tile=False,
            dtl_soff0=True,
            epilogue="cshuffle",
        )
    return tile, trait


def dtl_ring_small_tile_trait(
    arch: str,
    K: Optional[int] = None,
    M: Optional[int] = None,
    N: Optional[int] = None,
    dtype: Optional[str] = None,
) -> tuple[TileSpec, TraitSpec]:
    """The small-tile ring: 128x128x64, 2x4 waves of 64x32 (16x16x32 atoms),
    2-stage ring -- as the 3-stage fragment-prefetch interleave body when its
    tiles fit one wave and K >= :data:`DTL_RING_SMALL_S3_MIN_K`. For
    K % 64 != 0: a 128x128x32 2x2-wave interleave ring on up to
    :data:`DTL_RING_SMALL_K32_C128_MAX_WAVES` waves of 256x128 tiles, else a
    256x128x32 4-wave body (2x2 waves of 128x64) with AGPR accumulators, a
    3-stage ring and the merged MFMA interleave."""
    if arch not in DTL_RING_SMALL_ARCHES:
        raise ValueError(f"no small-tile DTL ring geometry for {arch!r}")
    if K is not None and M is not None and N is not None:
        r3 = small_tile_shortk_tile_trait(arch, M, N, K, dtype)
        if r3 is not None:
            return r3
    if K is not None and int(K) % 64:
        poor = (
            M is not None
            and N is not None
            and _tiles(M, N, 256, 128)
            <= DTL_RING_SMALL_K32_C128_MAX_WAVES * arch_num_cus(arch)
        )
        if poor:
            tile = TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            )
            return tile, replace(_ring_pad_trait(), **_S3ILV)
        tile = TileSpec(
            tile_m=256,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
        )
        return tile, replace(_ring_pad_trait(), acc_agpr=True, **_S3ILV)
    tile = TileSpec(
        tile_m=128,
        tile_n=128,
        tile_k=64,
        warp_m=2,
        warp_n=4,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    if (
        DTL_RING_SMALL_S3_SINGLE_WAVE
        and K is not None
        and M is not None
        and N is not None
        and int(K) >= DTL_RING_SMALL_S3_MIN_K
        and _tiles(M, N, 128, 128) <= arch_num_cus(arch)
    ):
        return tile, replace(_ring_pad_trait(), **_S3ILV)
    return tile, _ring_pad_trait()


def dtl_ring_small_sweep_variants(
    spec: UniversalGemmSpec, arch: str
) -> tuple[UniversalGemmSpec, ...]:
    """The small ring's sweep neighbourhood (selected spec first): the 2-stage
    and the 3-stage interleave form of the 128x128x64 body, and for tile_k 32
    the other K % 64 == 32 body; every variant validity-checked."""
    from ...instances.common.gemm_universal import is_valid_spec

    t, tr = spec.tile, spec.trait
    base = replace(
        tr,
        dtl_stages=2,
        dtl_frag_prefetch=False,
        mfma_interleave=False,
        ilv_merge=False,
        ilv_barrier_after=0,
        acc_agpr=False,
    )
    cands = []
    if t.tile_k == 64:
        cands = [
            replace(spec, trait=base),
            replace(spec, trait=replace(base, **_S3ILV)),
        ]
    else:
        c128 = replace(t, tile_m=128, tile_n=128, warp_m=2, warp_n=2)
        c256 = replace(t, tile_m=256, tile_n=128, warp_m=2, warp_n=2)
        cands = [
            replace(spec, tile=c128, trait=replace(base, **_S3ILV)),
            replace(spec, tile=c256, trait=replace(base, acc_agpr=True, **_S3ILV)),
        ]
    out = [spec]
    for c in cands:
        if c != spec and c not in out and is_valid_spec(c, arch=arch)[0]:
            out.append(c)
    return tuple(out)


def dtl_ring_pad_tile_trait(
    arch: str, big_grid: bool = False
) -> tuple[TileSpec, TraitSpec]:
    """The pad ring: the 16-wave 256x256x64 ring, persistent on every CU; on a
    very large grid the AGPR interleave big-grid body
    (:func:`agpr_ilv_big_grid_tile_trait`) with the same pad + exact DTL bounds
    + full-tile fast path."""
    if arch not in DTL_RING_PAD_ARCHES:
        raise ValueError(f"no padded DTL ring geometry for {arch!r}")
    if big_grid:
        return agpr_ilv_big_grid_tile_trait(
            arch, pad_m=True, pad_n=True, dtl_tight_rsrc=True, pad_fast_tile=True
        )
    tile, _ = dtl_pingpong_tile_trait(arch)
    return tile, _ring_pad_trait(persistent_ctas=arch_num_cus(arch))


def _default_tile(req: GemmRequest) -> Optional[TileCostModel]:
    """The default (cshuffle) candidate's tile as calibrated for the dispatched
    ring body of ``req``'s (arch, dtype) (``DTL_PINGPONG_COST_MODELS``)."""
    models = DTL_PINGPONG_COST_MODELS.get((req.arch, normalize_dtype(req.dtype)))
    return None if models is None else models[1]


def _ring_split_k_reason(req: GemmRequest) -> Optional[str]:
    """Why a ring candidate must not take ``req`` because the default candidate
    engages split-K there (the decode regime no ring was calibrated in), or
    ``None``. Calls :func:`select_split_k` with the default candidate's tile, as
    :func:`dtl_pingpong_admit` does, instead of relying on a tile-count floor
    that happens to equal its gate."""
    d = _default_tile(req)
    if d is None:
        return f"no default-candidate calibration for {req.arch} {req.dtype}"
    sk = select_split_k(
        M=req.M, N=req.N, K=req.K, tile_m=d.tile_m, tile_n=d.tile_n, tile_k=d.tile_k
    )
    if sk.split_k > 1:
        return (
            f"the default candidate engages split-K {sk.split_k} for "
            f"{req.M}x{req.N}x{req.K} (decode regime; the ring is not calibrated there)"
        )
    return None


def _aligned_to_default(req: GemmRequest) -> bool:
    d = _default_tile(req)
    return (
        d is not None
        and int(req.M) % d.tile_m == 0
        and int(req.N) % d.tile_n == 0
        and int(req.K) % d.tile_k == 0
    )


def _ragged_small_k(req: GemmRequest) -> bool:
    """Ragged M / N at small K on a moderate grid: the small ring's regime even
    with a well-filled 256x256 wave."""
    return (
        not (int(req.M) % 256 == 0 and int(req.N) % 256 == 0)
        and int(req.K) <= DTL_RING_SMALL_RAGGED_MAX_K
        and _tiles(req.M, req.N, 256, 256) < DTL_RING_SMALL_RAGGED_MAX_TILES
    )


def dtl_ring_small_admit(req: GemmRequest, spec: UniversalGemmSpec) -> tuple[bool, str]:
    """Small-tile ring: the 256x256 tiles fill the last wave poorly (or a
    ragged small-K problem, :func:`_ragged_small_k`), its own tiles still give
    at least half a wave, the default candidate does not split K, and -- where
    the aligned default candidate can take the problem -- K is above the
    calibrated floor of the selected body."""
    if rilv_shortk_admits(req):
        return False, "the aligned short-K row-interleave body serves it"
    why = _ring_split_k_reason(req)
    if why is not None:
        return False, why
    fill = wave_fill(req.M, req.N, 256, 256, req.arch)
    ragged_small_k = _ragged_small_k(req)
    if fill >= DTL_RING_SMALL_MAX_FILL and not ragged_small_k:
        return False, (
            f"256x256 tiles fill the last wave to {fill:.2f} >= "
            f"{DTL_RING_SMALL_MAX_FILL} for {req.M}x{req.N} on {req.arch}"
        )
    t = spec.tile
    tiles = _tiles(req.M, req.N, t.tile_m, t.tile_n)
    need = DTL_RING_SMALL_MIN_TILES_PER_CU * arch_num_cus(req.arch)
    if tiles < need:
        return False, (
            f"{tiles} {t.tile_m}x{t.tile_n} tiles < {need:.0f} (half a wave): "
            "latency-bound, not calibrated"
        )
    if _aligned_to_default(req):
        if t.tile_k == 64:
            rule = DTL_RING_ADMIT_RULES.get((req.arch, normalize_dtype(req.dtype)))
            k_min = rule.k_lo if rule is not None else DTL_RING_SMALL_RAGGED_MAX_K
        elif spec.trait.acc_agpr:
            k_min = DTL_RING_SMALL_K32_AGPR_MIN_K
        else:
            k_min = DTL_RING_SMALL_K32_MIN_K
        if int(req.K) < k_min:
            return False, (
                f"K={req.K} < {k_min}: the {t.tile_m}x{t.tile_n}x{t.tile_k} small "
                f"ring loses to the default candidate there ({req.M}x{req.N} is "
                "aligned to it)"
            )
    return True, (
        f"ok (256x256 last-wave fill {fill:.2f}"
        + (", ragged small K)" if ragged_small_k else ")")
    )


def dtl_ring_pad_admit(req: GemmRequest, spec: UniversalGemmSpec) -> tuple[bool, str]:
    """Pad ring: ragged M / N (the aligned shapes belong to the ping-pong / ring
    candidate) with a well-filled last wave (else the small-tile ring), outside
    the ragged small-K regime of the small ring, and no split-K default."""
    t = spec.tile
    if req.M % t.tile_m == 0 and req.N % t.tile_n == 0:
        return False, (
            f"{req.M}x{req.N} is a multiple of {t.tile_m}x{t.tile_n}: the "
            "aligned DTL candidate covers it"
        )
    why = _ring_split_k_reason(req)
    if why is not None:
        return False, why
    fill = wave_fill(req.M, req.N, t.tile_m, t.tile_n, req.arch)
    if fill < DTL_RING_SMALL_MAX_FILL:
        return False, (
            f"{t.tile_m}x{t.tile_n} tiles fill the last wave to {fill:.2f} < "
            f"{DTL_RING_SMALL_MAX_FILL} (the small-tile ring's regime)"
        )
    if _ragged_small_k(req):
        return False, (
            f"K={req.K} <= {DTL_RING_SMALL_RAGGED_MAX_K} on fewer than "
            f"{DTL_RING_SMALL_RAGGED_MAX_TILES} 256x256 tiles: the small-tile "
            "ring's ragged small-K regime"
        )
    return True, "ok"


def persistent_grid(spec: UniversalGemmSpec, M: int, N: int) -> tuple[int, int, int]:
    """Launch grid of a ``persistent_ctas = P`` body: ``min(P, tiles)``
    workgroups on x. The kernel's flat workgroup id is ``y * tiles_n + x``, so
    x alone covers ids ``0 .. P-1``; workgroups ``>= P`` would only exit, so
    the full 2D grid is correct but launches (and reserves LDS for) thousands of
    empty workgroups."""
    t = spec.tile
    if spec.trait.streamk:
        # stream-K: the in-kernel partition spreads K tiles over all P
        # workgroups, so exactly P must run (and be co-resident)
        return (int(spec.trait.persistent_ctas), 1, 1)
    tiles = -(-int(M) // t.tile_m) * -(-int(N) // t.tile_n)
    return (max(1, min(int(spec.trait.persistent_ctas), tiles)), 1, 1)


def fills_device(req: GemmRequest, spec: UniversalGemmSpec) -> tuple[bool, str]:
    """Big-tile admission by device fill (kept as a selectable policy).

    Admits the ping-pong candidate only when ``ceil(M/tile_m) * ceil(N/tile_n)
    >= num_CUs``. On gfx950 this is both too strict (the candidate already pays
    well below one full wave) and too loose (it admits tiny-K shapes the
    candidate loses), so the default policy is :func:`dtl_pingpong_admit`.
    """
    t = spec.tile
    tiles = -(-int(req.M) // t.tile_m) * -(-int(req.N) // t.tile_n)
    cus = arch_num_cus(req.arch)
    if tiles < cus:
        return False, (
            f"{tiles} output tiles of {t.tile_m}x{t.tile_n} < {cus} CUs on "
            f"{req.arch} (big-tile candidate needs a full device wave)"
        )
    return True, "ok"


# ---------------------------------------------------------------------------
# Ping-pong candidate admission: per-CU wave-efficiency cost model.
#
# The ping-pong kernel (256x256 tile, 1 workgroup per CU) competes with the
# 128x128 cshuffle default (4x the tiles, several co-resident workgroups per
# CU). Which one is faster depends on how many tiles each CU gets, how well
# co-resident tiles overlap, how full the device is, and how much fixed
# per-tile work (prologue, epilogue) there is relative to K. For a candidate c:
#
#     T_c = ceil(M / tile_m) * ceil(N / tile_n)   output tiles
#     n_c = ceil(T_c / CUs)                         tiles on the busiest CU
#     u_c = min(1, T_c / CUs)                       fraction of CUs with a tile
#     t_c = a_c * n_c + b_c * K * n_c**theta_c * u_c**gamma_c
#
# ``a`` is the per-tile fixed cost, ``b`` the per-K cost of one tile alone on
# a CU, ``theta`` <= 1 how much co-resident tiles on one CU overlap and
# ``gamma`` the speed-up of a partially filled device (less shared-bandwidth
# contention). The coefficients are dimensionless relative costs (normalised so
# the default kernel's ``b`` is 1), least-squares fitted per (arch, dtype) to
# on-device timings (gfx950: squares including non-multiples of the CU count,
# a wide K range and skinny M / N; gfx942: fewer shapes over the same ranges;
# every point checked against an f64 full-output reference). The CU count is
# the static per-arch value (``arch_num_cus``), so gfx942 parts with fewer CUs
# use the full-device model. The candidate is admitted when
# ``t_F <= DTL_PINGPONG_COST_MARGIN * t_default``; the margin was chosen by
# leave-one-out cross-validation over that data (it absorbs the model's error
# in the small-K regime where the two kernels are close). Recalibrate when
# either kernel or the target changes. Arches without coefficients never admit
# the candidate.
#
# Split-K: the ping-pong candidate is selected WITHOUT split-K (its select skips
# ``apply_split_k``), and it is not admitted where the default candidate would
# engage split-K (the decode / GEMV regime the model was not calibrated in), so
# both sides of the comparison are the single-K-pass kernels that were calibrated.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileCostModel:
    """``t = a * n + b * K * n**theta * u**gamma`` (see the block comment)."""

    tile_m: int
    tile_n: int
    tile_k: int
    a: float
    b: float
    theta: float
    gamma: float

    def cost(self, M: int, N: int, K: int, num_cus: int) -> float:
        tiles = -(-int(M) // self.tile_m) * -(-int(N) // self.tile_n)
        n = -(-tiles // num_cus)
        u = min(1.0, tiles / num_cus)
        return self.a * n + self.b * int(K) * n**self.theta * u**self.gamma


# (arch, dtype, body) -> (256x256 candidate model, default-candidate model).
# The 256x256 candidate's body decides the coefficients (``dtl_pingpong_body``):
# ``"pingpong"`` is the DTL ping-pong body (``dtl_stages == 0``), ``"ring"``
# the alias-scoped DTL ring (``dtl_stages >= 2``, the gfx950 dispatch pick via
# ``DTL_PINGPONG_RING``). Both gfx950 bodies stay calibrated, so turning the
# ring off (``DTL_PINGPONG_RING`` without gfx950, or ``ring=False``) judges the
# ping-pong body with its own coefficients. gfx942 was calibrated on fewer
# shapes and without a held-out set; the gfx950 ping-pong rows were checked on
# held-out shapes. On gfx950 the ring's cost model is the fallback
# policy -- the threshold rule below predicts the faster kernel better there
# (the ring wins at any tile count once K is large, which the model's per-tile
# fixed cost does not capture).
DTL_PINGPONG_COST_MODELS_BY_BODY = {
    ("gfx942", "fp16", "pingpong"): (
        TileCostModel(256, 256, 32, a=281.1, b=2.2663, theta=0.95, gamma=0.30),
        TileCostModel(128, 128, 16, a=56.1, b=1.0, theta=1.00, gamma=0.0),
    ),
    ("gfx942", "bf16", "pingpong"): (
        TileCostModel(256, 256, 32, a=248.6, b=1.7940, theta=0.90, gamma=0.30),
        TileCostModel(128, 128, 16, a=47.6, b=1.0, theta=0.95, gamma=0.0),
    ),
    ("gfx950", "fp16", "pingpong"): (
        TileCostModel(256, 256, 64, a=422.5, b=1.6093, theta=0.90, gamma=0.20),
        TileCostModel(128, 128, 32, a=88.4, b=1.0, theta=0.80, gamma=0.0),
    ),
    ("gfx950", "bf16", "pingpong"): (
        TileCostModel(256, 256, 64, a=413.2, b=1.5272, theta=0.90, gamma=0.20),
        TileCostModel(128, 128, 32, a=102.3, b=1.0, theta=0.85, gamma=0.0),
    ),
    ("gfx950", "fp16", "ring"): (
        TileCostModel(256, 256, 64, a=399.6, b=1.1499, theta=0.95, gamma=0.15),
        TileCostModel(128, 128, 32, a=83.8, b=1.0, theta=0.75, gamma=0.0),
    ),
    ("gfx950", "bf16", "ring"): (
        TileCostModel(256, 256, 64, a=375.6, b=1.0786, theta=0.95, gamma=0.15),
        TileCostModel(128, 128, 32, a=91.3, b=1.0, theta=0.80, gamma=0.0),
    ),
}


def dtl_pingpong_body(spec: UniversalGemmSpec) -> str:
    """``"ring"`` for the alias-scoped DTL ring body (``dtl_stages >= 2``),
    ``"pingpong"`` for the DTL ping-pong body."""
    return "ring" if spec.trait.dtl_stages >= 2 else "pingpong"


def _dispatched_body(arch: str) -> str:
    return (
        "ring"
        if DTL_PINGPONG_RING.get(arch, {}).get("dtl_stages", 0) >= 2
        else "pingpong"
    )


# (arch, dtype) -> the models of the body each arch DISPATCHES (a read-only
# view of ``DTL_PINGPONG_COST_MODELS_BY_BODY``; kept for callers that ask about
# the dispatched candidate without a spec).
DTL_PINGPONG_COST_MODELS = {
    (a, d): m
    for (a, d, body), m in DTL_PINGPONG_COST_MODELS_BY_BODY.items()
    if body == _dispatched_body(a)
}

# Per-(arch, body) cost-model margin (chosen by leave-one-out over each fit's
# data): admit when ``t_F <= margin * t_default``.
DTL_PINGPONG_COST_MARGINS_BY_BODY = {
    ("gfx942", "pingpong"): 0.95,
    ("gfx950", "pingpong"): 0.95,
    ("gfx950", "ring"): 1.0,
}
# arch -> margin of the dispatched body (read-only view).
DTL_PINGPONG_COST_MARGINS = {
    a: m
    for (a, body), m in DTL_PINGPONG_COST_MARGINS_BY_BODY.items()
    if body == _dispatched_body(a)
}


@dataclass(frozen=True)
class RingAdmitRule:
    """Threshold admission of the 256x256 candidate: admit iff ``K >= k_hi``
    or (``K >= k_lo`` and ``ceil(M/256) * ceil(N/256) >= t_min``)."""

    k_lo: int
    k_hi: int
    t_min: int

    def admits(self, M: int, N: int, K: int, tile_m: int, tile_n: int) -> bool:
        tiles = -(-int(M) // tile_m) * -(-int(N) // tile_n)
        return int(K) >= self.k_hi or (int(K) >= self.k_lo and tiles >= self.t_min)


# (arch, dtype) -> threshold rule for the RING body (it is applied only when
# the candidate is the ring, ``dtl_pingpong_body == "ring"``; the ping-pong
# body always uses its cost model). gfx950: grid-searched on squares, a wide K
# range and skinny / few-tile shapes down to one output tile, every point
# checked against an f64 full-output reference; under leave-one-out it
# predicted the faster kernel better than the refitted cost model. ``t_min``
# is an absolute 256x256 tile count calibrated on a full gfx950 device; like
# every CU count in this dispatcher (``arch_num_cus`` is static), it is not
# recalibrated for partitioned modes (CPX / DPX) or other CU counts.
DTL_RING_ADMIT_RULES = {
    ("gfx950", "fp16"): RingAdmitRule(k_lo=128, k_hi=1024, t_min=100),
    ("gfx950", "bf16"): RingAdmitRule(k_lo=64, k_hi=768, t_min=144),
}

# Admission policies for the ping-pong candidate: "auto" (default: the threshold
# rule where one is calibrated for the (arch, dtype), else the cost model),
# "rule", "cost_model", "fills_device" (the previous one-wave rule), "always",
# "never". Selected by the ROCKE_GEMM_DTL_ADMIT environment variable (like
# ROCKE_GEMM_SPLIT_K, an explicit operator override; it never changes kernel
# validity).
DTL_PINGPONG_COST_MARGIN = 0.95  # default for an arch absent from the per-arch map

DTL_ADMIT_ENV = "ROCKE_GEMM_DTL_ADMIT"
DTL_ADMIT_POLICIES = ("auto", "rule", "cost_model", "fills_device", "always", "never")


def dtl_admit_policy() -> str:
    v = os.environ.get(DTL_ADMIT_ENV, "").strip().lower() or "auto"
    if v not in DTL_ADMIT_POLICIES:
        raise ValueError(
            f"{DTL_ADMIT_ENV}={v!r}; expected one of {', '.join(DTL_ADMIT_POLICIES)}"
        )
    return v


def dtl_pingpong_admit(req: GemmRequest, spec: UniversalGemmSpec) -> tuple[bool, str]:
    """Ping-pong candidate admission: the calibrated threshold rule (``auto`` /
    ``rule``) or the cost model says it beats the default candidate."""
    policy = dtl_admit_policy()
    if policy == "always":
        return True, "ok (policy always)"
    if policy == "auto" and rilv_shortk_admits(req):
        return True, "ok (row-interleave short-K body)"
    if policy == "never":
        return False, f"ping-pong admission disabled ({DTL_ADMIT_ENV}=never)"
    if policy == "fills_device":
        return fills_device(req, spec)
    key = (req.arch, normalize_dtype(req.dtype))
    rule = DTL_RING_ADMIT_RULES.get(key)
    if policy == "rule" and rule is None:
        return False, f"no ping-pong threshold rule for {req.arch} {req.dtype}"
    body = dtl_pingpong_body(spec)
    if policy == "rule" and body != "ring":
        return False, f"the threshold rule is calibrated for the ring body, not {body}"
    use_rule = rule is not None and body == "ring" and policy in ("auto", "rule")
    # the calibration (also of the rule: its split-K regime check uses the
    # default candidate's tile) exists only where a cost model does
    models = DTL_PINGPONG_COST_MODELS_BY_BODY.get(key + (body,))
    if models is None:
        return False, f"no {body} cost model for {req.arch} {req.dtype}"
    f_model, d_model = models
    t = spec.tile
    if (t.tile_m, t.tile_n) != (f_model.tile_m, f_model.tile_n):
        return False, (
            f"ping-pong cost model is calibrated for {f_model.tile_m}x{f_model.tile_n}, "
            f"spec tile is {t.tile_m}x{t.tile_n}"
        )
    sk = select_split_k(
        M=req.M,
        N=req.N,
        K=req.K,
        tile_m=d_model.tile_m,
        tile_n=d_model.tile_n,
        tile_k=d_model.tile_k,
    )
    if sk.split_k > 1:
        return False, (
            f"the default candidate engages split-K {sk.split_k} for "
            f"{req.M}x{req.N}x{req.K} (decode regime; ping-pong not calibrated there)"
        )
    if use_rule:
        if not rule.admits(req.M, req.N, req.K, t.tile_m, t.tile_n):
            return False, (
                f"threshold rule: K={req.K} < {rule.k_hi} and not (K >= {rule.k_lo} and "
                f">= {rule.t_min} output tiles) for {req.M}x{req.N}x{req.K} on {req.arch}"
            )
        return True, "ok (threshold rule)"
    cus = arch_num_cus(req.arch)
    margin = DTL_PINGPONG_COST_MARGINS_BY_BODY.get(
        (req.arch, body), DTL_PINGPONG_COST_MARGIN
    )
    t_f = f_model.cost(req.M, req.N, req.K, cus)
    t_d = d_model.cost(req.M, req.N, req.K, cus)
    if t_f > margin * t_d:
        return False, (
            f"cost model: ping-pong {t_f:.0f} > {margin} x default "
            f"{t_d:.0f} (relative units) for {req.M}x{req.N}x{req.K} on {req.arch}"
        )
    return True, "ok"
