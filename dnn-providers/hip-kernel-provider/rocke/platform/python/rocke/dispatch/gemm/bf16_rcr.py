# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""BF16 RCR UniversalGemm dispatcher case.

This is the second GEMM-family case and the worked template for adding more
dtypes/layouts. It reuses the same machinery as :mod:`fp16_rcr`:

* the dtype-/layout-generic request validation (:func:`rcr_request_errors`),
* a declared :class:`Capability` per candidate, whose explicit ``arches`` list
  keeps RDNA/WMMA candidates off CDNA arches and CDNA candidates off RDNA ones,
* the generic config predicate plus the builder's validity gate
  (:func:`universal_spec_supported` = :func:`gemm_config_supported`, which is
  already dtype-agnostic (it consults the per-arch MMA catalog), then
  ``is_valid_spec``).

The only material difference from the fp16 case is the MMA atom geometry of
the CDNA cshuffle candidate: it uses a 16x16 warp tile with a 4x4 warp grid
rather than fp16's 32x32 / 2x2. (This is a historical choice, not an atom gap:
the arch catalog has bf16 32x32x8 on gfx90a/gfx942 and 32x32x8 / 32x32x16 on
gfx950. The DTL ping-pong candidate shares fp16's 16x16 geometry.)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Sequence, Tuple

from ...core.arch import ArchTarget
from ...helpers.manifest import gemm_args_signature
from ...helpers.spec import ceil_div_grid
from ...instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    check_launch_shape,
)
from ..core import (
    CandidateRegistry,
    Capability,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    ShapeRange,
    stable_json_hash,
    selector_matches,
)
from .binding import gemm_rcr_binding
from .common import (
    DTL_PINGPONG_ARCHES,
    GEMM_DIM_VOCABULARY,
    GemmRequest,
    apply_split_k,
    dtl_pingpong_sweep_variants,
    dtl_pingpong_tile_trait,
    dtl_pingpong_admit,
    dtl_ring_pad_admit,
    dtl_ring_pad_tile_trait,
    dtl_ring_small_admit,
    dtl_ring_small_sweep_variants,
    dtl_ring_small_tile_trait,
    rilv_admits,
    rilv_spec_tile_trait,
    rilv_pad_admits,
    rilv_pad_tile_trait,
    rilv_shortk_admits,
    rilv_shortk_tile_trait,
    big_grid_tile_trait,
    dtl_big_grid_body,
    DTL_RING_PAD_ARCHES,
    DTL_RING_SMALL_ARCHES,
    persistent_grid,
    rcr_request_errors,
)
from .support import (
    request_shape_supported,
    universal_spec_supported,
)

_FAMILY = "gemm_bf16_rcr"
_ALGORITHM = "universal_gemm"
GEMM_BF16_RCR_ABI_VERSION = "hipkg-gemm-bf16-rcr/v1"


def _request_errors(req: OperatorRequest) -> list[str]:
    return rcr_request_errors(req, dtype="bf16")


def _make_spec(
    *,
    name: str,
    arch: str,
    tile: TileSpec,
    trait: TraitSpec,
) -> UniversalGemmSpec:
    wave = ArchTarget.from_gfx(arch).wave_size
    return UniversalGemmSpec(
        name=name,
        tile=tile,
        trait=trait,
        data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", layout="RCR"),
        wave_size=wave,
    )


def _spec_cdna_cshuffle(req: GemmRequest, name: str) -> UniversalGemmSpec:
    # The 128x128 cshuffle tile is built from a 16x16 atom with a 4x4 warp
    # grid (block_size = 4*4*64 = 1024); a historical choice -- bf16 32x32 atoms
    # do exist on CDNA (see the module docstring). tile_k
    # widens to the largest atom K available on the arch (32 on gfx950).
    tile_k = 32 if req.arch == "gfx950" else 16
    warp_tile_k = 32 if req.arch == "gfx950" else 16
    return _make_spec(
        name=name,
        arch=req.arch,
        tile=TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=tile_k,
            warp_m=4,
            warp_n=4,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=warp_tile_k,
        ),
        trait=TraitSpec(pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"),
    )


def _spec_cdna_dtl_pingpong(req: GemmRequest, name: str) -> UniversalGemmSpec:
    # The DTL ping-pong body (see ``common.dtl_pingpong_tile_trait``): 256x256
    # block, 16 waves, DTL + depth-1 prefetch + element swizzle + A/B CACHE_ALL
    # + chiplet swizzle (the ring form of it on gfx950).
    tile, trait = dtl_pingpong_tile_trait(req.arch)
    big = big_grid_tile_trait(req)
    if big is not None:
        # very large grids -> the persistent AGPR interleave body or the ring
        # with chiplet wgm 4 + iglp1 (common.dtl_big_grid_body)
        tile, trait = big
    if rilv_admits(req):
        # the 4-wave AGPR row-interleave body + nt vector epilogue
        # (``common.rilv_spec_tile_trait``)
        tile, trait = rilv_spec_tile_trait(req)
    elif rilv_shortk_admits(req):
        # aligned short K (``common.rilv_shortk_admits``)
        tile, trait = rilv_shortk_tile_trait(req)
    return _make_spec(name=name, arch=req.arch, tile=tile, trait=trait)


def _spec_cdna_dtl_ring_small(req: GemmRequest, name: str) -> UniversalGemmSpec:
    # 128x128x64 8-wave DTL ring with pad + exact DTL bounds
    # (``common.dtl_ring_small_tile_trait``) for poorly filled 256x256 waves.
    tile, trait = dtl_ring_small_tile_trait(
        req.arch, req.K, req.M, req.N, dtype=req.dtype
    )
    return _make_spec(name=name, arch=req.arch, tile=tile, trait=trait)


def _spec_cdna_dtl_ring_pad(req: GemmRequest, name: str) -> UniversalGemmSpec:
    # the 16-wave persistent ring with pad + exact DTL bounds
    # (``common.dtl_ring_pad_tile_trait``) for ragged M / N.
    tile, trait = dtl_ring_pad_tile_trait(
        req.arch, dtl_big_grid_body(req, ragged=True) == "agpr_ilv"
    )
    if rilv_pad_admits(req):
        # the row-interleave body with the vector epilogue's branch-free
        # bounds (``common.rilv_pad_tile_trait``)
        tile, trait = rilv_pad_tile_trait(req)
    return _make_spec(name=name, arch=req.arch, tile=tile, trait=trait)


def _spec_cdna_mem(req: GemmRequest, name: str) -> UniversalGemmSpec:
    return _make_spec(
        name=name,
        arch=req.arch,
        tile=TileSpec(
            tile_m=64,
            tile_n=128,
            tile_k=16 if req.arch == "gfx942" else 32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16 if req.arch == "gfx942" else 32,
        ),
        trait=TraitSpec(pipeline="mem", scheduler="intrawave", epilogue="default"),
    )


def _spec_rdna_wmma(req: GemmRequest, name: str) -> UniversalGemmSpec:
    return _make_spec(
        name=name,
        arch=req.arch,
        tile=TileSpec(
            tile_m=64,
            tile_n=32,
            tile_k=16,
            warp_m=2,
            warp_n=1,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline="mem", scheduler="intrawave", epilogue="default"),
    )


def _spec_rdna_wmma_small(req: GemmRequest, name: str) -> UniversalGemmSpec:
    return _make_spec(
        name=name,
        arch=req.arch,
        tile=TileSpec(
            tile_m=32,
            tile_n=32,
            tile_k=16,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline="mem", scheduler="intrawave", epilogue="default"),
    )


# Decode-shaped GEMMs (tiny M) leave a 128x128 / 64x128 tile's grid far below
# the device CU count. This small-M candidate uses the deep-K bf16 hero atom
# (16x16x32) with a 16-row tile and a padded shape so a 1-4 token M is legal; it
# is the carrier for the split-K heuristic, which fills the idle CUs by slicing
# K. Mirrors the proven streamk block-tile recipe.
_DECODE_TILE_N = 64
_DECODE_M_MAX = 32  # only compete for skinny (decode/GEMV) M


def _spec_cdna_decode(req: GemmRequest, name: str) -> UniversalGemmSpec:
    wt_m, wt_n, wt_k = 16, 16, 32
    warp_n = max(1, _DECODE_TILE_N // wt_n)
    while warp_n > 1 and (warp_n * 64 > 1024 or _DECODE_TILE_N % (warp_n * wt_n) != 0):
        warp_n //= 2
    return _make_spec(
        name=name,
        arch=req.arch,
        tile=TileSpec(
            tile_m=16,
            tile_n=_DECODE_TILE_N,
            tile_k=32,
            warp_m=1,
            warp_n=warp_n,
            warp_k=1,
            warp_tile_m=wt_m,
            warp_tile_n=wt_n,
            warp_tile_k=wt_k,
        ),
        trait=TraitSpec(
            pipeline="compv4",
            scheduler="intrawave",
            epilogue="default",
            pad_m=True,
            pad_n=True,
            pad_k=True,
        ),
    )


def _make_candidate(
    *,
    name: str,
    spec_id: str,
    priority: int,
    spec_fn: Callable[[GemmRequest, str], UniversalGemmSpec],
    arches: Tuple[str, ...],
    shapes: Tuple[ShapeRange, ...] = (),
    admit: Callable[[GemmRequest, UniversalGemmSpec], Tuple[bool, str]] | None = None,
    sweep_variants: (
        Callable[[UniversalGemmSpec, str], Sequence[UniversalGemmSpec]] | None
    ) = None,
    split_k: bool = True,
) -> KernelCandidate:
    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, GemmRequest)
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        spec = spec_fn(req, name)
        ok, why = universal_spec_supported(spec, arch=req.arch)
        if not ok:
            return False, why
        ok, why = request_shape_supported(req, spec)
        if not ok or admit is None:
            return ok, why
        return admit(req, spec)

    def select(req: OperatorRequest) -> UniversalGemmSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, GemmRequest)
        # Engage split-K for skinny/tall-N decode shapes that leave the device
        # idle; a no-op (returns the spec unchanged) for shapes that already
        # fill the device, keeping the default / square path byte-identical.
        spec = spec_fn(req, name)
        return apply_split_k(req, spec) if split_k else spec

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=_ALGORITHM,
        spec_id=spec_id,
        abi_version=GEMM_BF16_RCR_ABI_VERSION,
        priority=priority,
        capability=Capability(
            arches=arches, dtypes=("bf16",), layouts=("RCR",), shapes=shapes
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: gemm_args_signature(dtype="bf16"),
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (
            (
                tuple(sweep_variants(select(req), req.arch))
                if sweep_variants is not None
                else (select(req),)
            )
            if candidate.admits(req)[0]
            else ()
        ),
        build=build_universal_gemm,
        bind=lambda result, verify: gemm_rcr_binding(result, verify, dtype="bf16"),
    )
    return candidate


def _grid(spec: UniversalGemmSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    t = spec.tile
    assert isinstance(req, GemmRequest)
    # the launch shape must fit the body (request_shape_supported already
    # routes such requests elsewhere; this guards direct users of the spec)
    check_launch_shape(spec, req.M, req.N, req.K)
    if spec.trait.persistent_ctas:
        # persistent tile loop: launch only the P workgroups that own tiles
        return persistent_grid(spec, req.M, req.N)
    # Split-K adds a Z dimension of ``split_k`` K-slice CTAs per (m,n) tile;
    # split_k == 1 (default) collapses to the canonical 2D grid.
    return ceil_div_grid((req.N, t.tile_n), (req.M, t.tile_m), (spec.trait.split_k, 1))


# Explicit gfx targets rather than a cdna/rdna family label; see fp16_rcr for
# why family is the wrong gate. bf16's CDNA lists differ from fp16's: gfx90a has
# the 16x16x16 bf16 atom the cshuffle candidate needs, and the decode candidate's
# deep-K 16x16x32 bf16 atom exists only on gfx950.
_CDNA_MFMA_BF16 = ("gfx90a", "gfx942", "gfx950")
_CDNA_MEM_BF16 = ("gfx942", "gfx950")
_CDNA_DECODE_BF16 = ("gfx950",)
_RDNA_WMMA = ("gfx11-generic", "gfx1151", "gfx1201")

GEMM_BF16_REGISTRY = CandidateRegistry(
    _FAMILY,
    dim_vocabulary=GEMM_DIM_VOCABULARY,
    require_build=True,
    require_binding=True,
)
GEMM_BF16_REGISTRY.extend(
    (
        _make_candidate(
            name="universal_gemm_bf16_cdna_dtl_ring_small",
            spec_id="cdna_dtl_ring_small",
            priority=4,
            spec_fn=_spec_cdna_dtl_ring_small,
            arches=DTL_RING_SMALL_ARCHES,
            admit=dtl_ring_small_admit,
            sweep_variants=dtl_ring_small_sweep_variants,
            split_k=False,
        ),
        _make_candidate(
            name="universal_gemm_bf16_cdna_dtl_ring_pad",
            spec_id="cdna_dtl_ring_pad",
            priority=6,
            spec_fn=_spec_cdna_dtl_ring_pad,
            arches=DTL_RING_PAD_ARCHES,
            admit=dtl_ring_pad_admit,
            split_k=False,
        ),
        _make_candidate(
            name="universal_gemm_bf16_cdna_dtl_pingpong",
            spec_id="cdna_dtl_pingpong_256x256",
            priority=5,
            spec_fn=_spec_cdna_dtl_pingpong,
            arches=DTL_PINGPONG_ARCHES,
            admit=dtl_pingpong_admit,
            sweep_variants=dtl_pingpong_sweep_variants,
            # Selected without split-K: the admission model compares the
            # single-K-pass kernels it was calibrated on (see common.py).
            split_k=False,
        ),
        _make_candidate(
            name="universal_gemm_bf16_cdna_cshuffle",
            spec_id="cdna_cshuffle_default",
            priority=10,
            spec_fn=_spec_cdna_cshuffle,
            arches=_CDNA_MFMA_BF16,
        ),
        _make_candidate(
            name="universal_gemm_bf16_rdna_wmma",
            spec_id="rdna_wmma_default",
            priority=10,
            spec_fn=_spec_rdna_wmma,
            arches=_RDNA_WMMA,
        ),
        _make_candidate(
            name="universal_gemm_bf16_cdna_mem",
            spec_id="cdna_mem_64x128",
            priority=20,
            spec_fn=_spec_cdna_mem,
            arches=_CDNA_MEM_BF16,
        ),
        _make_candidate(
            name="universal_gemm_bf16_rdna_wmma_small",
            spec_id="rdna_wmma_32x32",
            priority=20,
            spec_fn=_spec_rdna_wmma_small,
            arches=_RDNA_WMMA,
        ),
        _make_candidate(
            name="universal_gemm_bf16_cdna_decode",
            spec_id="cdna_decode_16x64",
            priority=30,
            spec_fn=_spec_cdna_decode,
            arches=_CDNA_DECODE_BF16,
            # Only compete for skinny (decode/GEMV) M. This was an imperative
            # shape gate; as data it is visible to a coverage query.
            shapes=(ShapeRange("M", max=_DECODE_M_MAX),),
        ),
    )
)


def gemm_bf16_candidates() -> Tuple[KernelCandidate, ...]:
    return GEMM_BF16_REGISTRY.candidates()


def _kernel_id(
    req: GemmRequest, candidate: KernelCandidate, spec: UniversalGemmSpec
) -> KernelId:
    request_hash = stable_json_hash(req.normalized(), n=16)
    spec_hash = stable_json_hash(asdict(spec), n=16)
    return KernelId(
        op="gemm",
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=request_hash,
        spec_hash=spec_hash,
    )


def build_kernel(result: DispatchResult):
    """Deprecated in favour of ``result.build()``.

    Kept because benchmark harnesses and examples import it by name; it now
    delegates so there is one definition of how a selection becomes IR.
    """
    return result.build()


def registered_gemm_bf16_combos(
    req: OperatorRequest,
) -> Tuple[Tuple[KernelCandidate, UniversalGemmSpec], ...]:
    """Every registered BF16 RCR candidate that can launch ``req``.

    Probes opt-in variants and expands each candidate's ``sweep_space``.
    Production :func:`dispatch_gemm_bf16` is unchanged.
    """
    if _request_errors(req):
        return ()
    return GEMM_BF16_REGISTRY.combos(req)


def gemm_bf16_sweep_space(req: OperatorRequest) -> Sequence[UniversalGemmSpec]:
    """Bounded sweep space from all registered BF16 RCR candidates."""
    if _request_errors(req):
        return ()
    return GEMM_BF16_REGISTRY.sweep_space(req)


def dispatch_gemm_bf16_all(req: GemmRequest) -> Tuple[DispatchResult, ...]:
    """Every eligible BF16 RCR kernel for ``req``, including opt-in variants."""
    if _request_errors(req):
        return ()
    return GEMM_BF16_REGISTRY.dispatch_all(req, kernel_id=_kernel_id)


def dispatch_gemm_bf16(
    req: GemmRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select a registered BF16 RCR UniversalGemm candidate for ``req``."""
    candidate = GEMM_BF16_REGISTRY.select(req, ranker=ranker)
    spec = candidate.select_spec(req)
    kid = _kernel_id(req, candidate, spec)
    return DispatchResult(
        request=req,
        candidate=candidate,
        spec=spec,
        kernel_id=kid,
        grid=candidate.grid(spec, req),
        block=candidate.block(spec),
        signature=tuple(candidate.signature(spec)),
        explanation=(
            f"selected {candidate.name} for bf16 RCR GEMM on {req.arch}",
            f"algorithm={candidate.algorithm}",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )
