# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""FP16 RCR UniversalGemm dispatcher case."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Sequence, Tuple

from ...core.arch import ArchTarget
from ...helpers.spec import ceil_div_grid
from ...instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    check_launch_shape,
    launch_shape_error,
)
from ..core import (
    CandidateRegistry,
    Capability,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    stable_json_hash,
    ShapeRange,
    selector_matches,
)
from ...instances.common.gemm_mn_operand import GEMM_LAYOUTS
from . import selector as _selector
from .binding import gemm_rcr_binding
from .skinny import (
    RING_ANY_PRIORITY,
    make_selected_skinny_candidate,
    make_skinny_candidate,
    ring_small_any_admit,
    ring_small_any_spec,
)
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
    native_layout_spec,
    universal_fallback_tile_trait,
    UNIVERSAL_FALLBACK,
    UNIVERSAL_FALLBACK_ARCHES,
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
    G1_F16_DECODE,
    G1_PAD_FALLBACK_ARCHES,
    g1_adapt_spec,
    g1_gemm_signature,
    g1_sweep_filter,
    g1_decode_tile_trait,
    g1_pad_fallback_admit,
    g1_pad_tile_trait,
    g1_ring_any_admit,
    g1_ring_any_retile,
    g1_ring_any_uses_pad_ring,
    G5_VGPR_ARCHES,
    native_any_admit,
    ring_any_first_admit,
    G5_VGPR_PRIORITY,
    g5_vgpr_admit,
    g5_vgpr_tile_trait,
    g5_vgpr_sweep_tile_traits,
    _g5_tile_trait,
)
from .support import (
    request_shape_supported,
    universal_spec_supported,
)

_FAMILY = "gemm_fp16_rcr"
_ALGORITHM = "universal_gemm"
GEMM_FP16_RCR_ABI_VERSION = "hipkg-gemm-fp16-rcr/v1"


def _request_errors(req: OperatorRequest) -> list[str]:
    return rcr_request_errors(req, dtype="fp16")


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
        data=DataSpec(dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", layout="RCR"),
        wave_size=wave,
    )


def _spec_cdna_cshuffle(req: GemmRequest, name: str) -> UniversalGemmSpec:
    if req.arch == "gfx942":
        return _make_spec(
            name=name,
            arch=req.arch,
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=16,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=8,
            ),
            trait=TraitSpec(
                pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"
            ),
        )
    return _make_spec(
        name=name,
        arch=req.arch,
        tile=TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
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


def _spec_cdna_vgpr(req: GemmRequest, name: str) -> UniversalGemmSpec:
    # the gfx942 register-staged prefetch body
    # (``common.g5_vgpr_tile_trait``); the admit hook refuses requests it does
    # not apply to, so the fallback geometry only feeds the support checks.
    tt = g5_vgpr_tile_trait(req)
    tile, trait = tt if tt is not None else _g5_tile_trait("256x256x64", False, False)
    return _make_spec(name=name, arch=req.arch, tile=tile, trait=trait)


def _sweep_cdna_vgpr(req: GemmRequest) -> Tuple[UniversalGemmSpec, ...]:
    # the cost model's cheapest plans (the pick first); a
    # non-RCR request gets each plan's native form
    out = []
    for t, tr in g5_vgpr_sweep_tile_traits(req):
        s = _make_spec(name="universal_gemm_cdna_vgpr", arch=req.arch, tile=t, trait=tr)
        if str(getattr(req, "layout", "RCR")).upper() != "RCR":
            s, _ = native_layout_spec(req, s)
            if s is None:
                continue
        out.append(s)
    return tuple(out)


def _spec_cdna_dtl_ring_small(req: GemmRequest, name: str) -> UniversalGemmSpec:
    # 128x128x64 8-wave DTL ring with pad + exact DTL bounds
    # (``common.dtl_ring_small_tile_trait``) for poorly filled 256x256 waves.
    tile, trait = dtl_ring_small_tile_trait(
        req.arch, req.K, req.M, req.N, dtype=req.dtype
    )
    return _make_spec(name=name, arch=req.arch, tile=tile, trait=trait)


def _spec_cdna_ring_small_any(req: GemmRequest, name: str) -> UniversalGemmSpec:
    # the small ring with pads / few-tile split-K for shapes
    # its own admission does not take (``skinny.ring_small_any_spec``)
    return ring_small_any_spec(req, _spec_cdna_dtl_ring_small(req, name))


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


def _spec_cdna_dtl_ring_any(req: GemmRequest, name: str) -> UniversalGemmSpec:
    # the ring bodies past their admission rules
    if g1_ring_any_uses_pad_ring(req):
        return _spec_cdna_dtl_ring_pad(req, name)
    return g1_ring_any_retile(req, _spec_cdna_dtl_ring_small(req, name))


def _spec_cdna_pad(req: GemmRequest, name: str) -> UniversalGemmSpec:
    # the padded fallback (``common.g1_pad_tile_trait``)
    tile, trait = g1_pad_tile_trait(req)
    return _make_spec(name=name, arch=req.arch, tile=tile, trait=trait)


def _spec_cdna_decode(req: GemmRequest, name: str) -> UniversalGemmSpec:
    # the bf16 decode body for f16 (``common.g1_decode_tile_trait``)
    tile, trait = g1_decode_tile_trait(req.arch, "fp16")
    return _make_spec(name=name, arch=req.arch, tile=tile, trait=trait)


def _spec_cdna_universal_fallback(req: GemmRequest, name: str) -> UniversalGemmSpec:
    # the universal padded fallback (common.UNIVERSAL_FALLBACK)
    tile, trait = universal_fallback_tile_trait(req.arch, req.M, req.N, req.K)
    return _make_spec(name=name, arch=req.arch, tile=tile, trait=trait)


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
    sweep_req: Callable[[GemmRequest], Sequence[UniversalGemmSpec]] | None = None,
    layouts: Tuple[str, ...] = ("RCR",),
) -> KernelCandidate:
    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, GemmRequest)
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        spec = g1_adapt_spec(req, spec_fn(req, name))
        # native NN / TN / TT: the RCR body with the
        # request's layout; admission stays the RCR candidate's shape rule
        nspec, why = native_layout_spec(req, spec)
        if nspec is None:
            return False, why
        ok, why = universal_spec_supported(nspec, arch=req.arch)
        if not ok:
            return False, why
        ok, why = request_shape_supported(req, nspec)
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
        spec = g1_adapt_spec(req, spec_fn(req, name))
        if split_k:
            sk_spec = apply_split_k(req, spec)
            # a split variant the launch rules refuse keeps the admitted spec
            if sk_spec is not spec and not launch_shape_error(
                sk_spec, req.M, req.N, req.K
            ):
                nsk, _ = native_layout_spec(req, sk_spec)
                if nsk is not None:
                    return nsk
        nspec, why = native_layout_spec(req, spec)
        if nspec is None:  # pragma: no cover - support() already refused
            raise ValueError(f"{name}: {why}")
        return nspec

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=_ALGORITHM,
        spec_id=spec_id,
        abi_version=GEMM_FP16_RCR_ABI_VERSION,
        priority=priority,
        capability=Capability(
            arches=arches, dtypes=("fp16",), layouts=layouts, shapes=shapes
        ),
        _supports=support,
        select_spec=select,
        signature=lambda spec: g1_gemm_signature(spec, "fp16"),
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (
            (
                tuple(sweep_req(req))
                if sweep_req is not None
                else (
                    g1_sweep_filter(req, sweep_variants(select(req), req.arch))
                    if sweep_variants is not None
                    else (select(req),)
                )
            )
            if candidate.admits(req)[0]
            else ()
        ),
        build=build_universal_gemm,
        bind=lambda result, verify: gemm_rcr_binding(result, verify, dtype="fp16"),
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


# Explicit gfx targets rather than a cdna/rdna family label. Family does not
# imply wave size -- gfx1250 is cdna at wave32 -- so a family gate would admit a
# wave32 target into these wave64 MFMA candidates. An arch absent from a list is
# a target the candidate was never built or run against.
_CDNA_MFMA_FP16 = ("gfx942", "gfx950")
_RDNA_WMMA = ("gfx11-generic", "gfx1151", "gfx1201")

GEMM_FP16_REGISTRY = CandidateRegistry(
    _FAMILY,
    dim_vocabulary=GEMM_DIM_VOCABULARY,
    require_build=True,
    require_binding=True,
)
GEMM_FP16_REGISTRY.extend(
    (
        # register-direct split-K body for skinny / decode /
        # small few-tile problems (``skinny.SKINNY_REGION``); fp16 and bf16
        make_skinny_candidate(
            name="universal_gemm_fp16_cdna_skinny",
            family=_FAMILY,
            abi_version=GEMM_FP16_RCR_ABI_VERSION,
            dtype="fp16",
        ),
        _make_candidate(
            name="universal_gemm_fp16_cdna_ring_small_any",
            spec_id="cdna_ring_small_any",
            priority=RING_ANY_PRIORITY,
            spec_fn=_spec_cdna_ring_small_any,
            arches=DTL_RING_SMALL_ARCHES,
            admit=native_any_admit(ring_small_any_admit),
            split_k=False,
            layouts=GEMM_LAYOUTS,
        ),
        make_skinny_candidate(
            name="universal_gemm_fp16_cdna_skinny_fallback",
            family=_FAMILY,
            abi_version=GEMM_FP16_RCR_ABI_VERSION,
            dtype="fp16",
            fallback=True,
        ),
        _make_candidate(
            name="universal_gemm_fp16_cdna_vgpr",
            spec_id="cdna_vgpr",
            priority=G5_VGPR_PRIORITY,
            spec_fn=_spec_cdna_vgpr,
            arches=G5_VGPR_ARCHES,
            admit=g5_vgpr_admit,
            split_k=False,
            sweep_req=_sweep_cdna_vgpr,
            # NN / TN / TT natively (common.R2_VP_MN)
            layouts=GEMM_LAYOUTS,
        ),
        _make_candidate(
            name="universal_gemm_fp16_cdna_dtl_ring_small",
            spec_id="cdna_dtl_ring_small",
            priority=4,
            spec_fn=_spec_cdna_dtl_ring_small,
            arches=DTL_RING_SMALL_ARCHES,
            admit=dtl_ring_small_admit,
            sweep_variants=dtl_ring_small_sweep_variants,
            split_k=False,
            layouts=GEMM_LAYOUTS,
        ),
        _make_candidate(
            name="universal_gemm_fp16_cdna_dtl_ring_pad",
            spec_id="cdna_dtl_ring_pad",
            priority=6,
            spec_fn=_spec_cdna_dtl_ring_pad,
            arches=DTL_RING_PAD_ARCHES,
            admit=dtl_ring_pad_admit,
            split_k=False,
            layouts=GEMM_LAYOUTS,
        ),
        _make_candidate(
            name="universal_gemm_fp16_cdna_dtl_pingpong",
            spec_id="cdna_dtl_pingpong_256x256",
            priority=5,
            spec_fn=_spec_cdna_dtl_pingpong,
            arches=DTL_PINGPONG_ARCHES,
            admit=dtl_pingpong_admit,
            sweep_variants=dtl_pingpong_sweep_variants,
            # Selected without split-K: the admission model compares the
            # single-K-pass kernels it was calibrated on (see common.py).
            split_k=False,
            layouts=GEMM_LAYOUTS,
        ),
        _make_candidate(
            name="universal_gemm_fp16_cdna_cshuffle",
            spec_id="cdna_cshuffle_default",
            priority=10,
            spec_fn=_spec_cdna_cshuffle,
            arches=_CDNA_MFMA_FP16,
            layouts=GEMM_LAYOUTS,
        ),
        _make_candidate(
            name="universal_gemm_fp16_rdna_wmma",
            spec_id="rdna_wmma_default",
            priority=10,
            spec_fn=_spec_rdna_wmma,
            arches=_RDNA_WMMA,
        ),
        _make_candidate(
            name="universal_gemm_fp16_cdna_mem",
            spec_id="cdna_mem_64x128",
            priority=20,
            spec_fn=_spec_cdna_mem,
            arches=_CDNA_MFMA_FP16,
            layouts=GEMM_LAYOUTS,
        ),
        _make_candidate(
            name="universal_gemm_fp16_rdna_wmma_small",
            spec_id="rdna_wmma_32x32",
            priority=20,
            spec_fn=_spec_rdna_wmma_small,
            arches=_RDNA_WMMA,
        ),
        _make_candidate(
            name="universal_gemm_fp16_cdna_universal_fallback",
            spec_id="cdna_universal_fallback",
            priority=100,
            spec_fn=_spec_cdna_universal_fallback,
            arches=UNIVERSAL_FALLBACK_ARCHES if UNIVERSAL_FALLBACK else (),
            split_k=False,
            layouts=GEMM_LAYOUTS,
        ),
    )
)
if G1_F16_DECODE:
    GEMM_FP16_REGISTRY.extend(
        (
            _make_candidate(
                name="universal_gemm_fp16_cdna_decode",
                spec_id="cdna_decode_16x64",
                priority=30,
                spec_fn=_spec_cdna_decode,
                arches=_CDNA_MFMA_FP16,
                shapes=(ShapeRange("M", max=32),),
            ),
        )
    )
GEMM_FP16_REGISTRY.extend(
    (
        # where ring_small_any (priority 9) and ring_any both admit, ring_any is
        # usually the faster body: the same ring_any body one priority ahead,
        # admitted only there (MERGE_RING_ANY_FIRST)
        _make_candidate(
            name="universal_gemm_fp16_cdna_dtl_ring_any_first",
            spec_id="cdna_dtl_ring_any_first",
            priority=RING_ANY_PRIORITY - 1,
            spec_fn=_spec_cdna_dtl_ring_any,
            arches=DTL_RING_SMALL_ARCHES,
            admit=native_any_admit(
                ring_any_first_admit(
                    g1_ring_any_admit,
                    lambda: GEMM_FP16_REGISTRY.get(
                        "universal_gemm_fp16_cdna_ring_small_any"
                    ),
                )
            ),
            split_k=False,
            layouts=GEMM_LAYOUTS,
        ),
        _make_candidate(
            name="universal_gemm_fp16_cdna_dtl_ring_any",
            spec_id="cdna_dtl_ring_any",
            priority=35,
            spec_fn=_spec_cdna_dtl_ring_any,
            arches=DTL_RING_SMALL_ARCHES,
            admit=native_any_admit(g1_ring_any_admit),
            split_k=False,
            layouts=GEMM_LAYOUTS,
        ),
        _make_candidate(
            name="universal_gemm_fp16_cdna_pad",
            spec_id="cdna_pad_fallback",
            priority=40,
            spec_fn=_spec_cdna_pad,
            arches=G1_PAD_FALLBACK_ARCHES,
            admit=g1_pad_fallback_admit,
        ),
    )
)


# The learned selector's pick when it is not a registered candidate's own spec
# (a sweep variant, a one-step trait mutation, or a library tiling; see
# ``selector.py``). Opt-in: priority selection never sees it; ``dispatch_gemm_fp16``
# reaches it through ``selector.dispatch_override``, and a request pinned to
# ``spec_id="cdna_selected"`` reselects the same spec. Its sweep space is the
# selector's top-k (the autotune-lite shortlist).
_SELECTED_NAME = "universal_gemm_fp16_cdna_selected"


def _make_selected_candidate() -> KernelCandidate:
    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        return _selector.selected_supports(req, GEMM_FP16_REGISTRY)

    def select(req: OperatorRequest) -> UniversalGemmSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{_SELECTED_NAME} does not support request: {why}")
        return _selector.selected_spec(req, GEMM_FP16_REGISTRY)

    candidate = KernelCandidate(
        name=_SELECTED_NAME,
        family=_FAMILY,
        algorithm=_ALGORITHM,
        spec_id="cdna_selected",
        abi_version=GEMM_FP16_RCR_ABI_VERSION,
        priority=1,
        capability=Capability(
            arches=("gfx942", "gfx950"), dtypes=("fp16",), layouts=GEMM_LAYOUTS
        ),
        _supports=support,
        select_spec=select,
        signature=lambda spec: g1_gemm_signature(spec, "fp16"),
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (
            _selector.selected_sweep(req, GEMM_FP16_REGISTRY)
            if candidate.admits(req)[0]
            else ()
        ),
        build=build_universal_gemm,
        bind=lambda result, verify: gemm_rcr_binding(result, verify, dtype="fp16"),
        opt_in=True,
    )
    return candidate


GEMM_FP16_REGISTRY.register(_make_selected_candidate())
# the selector's gemm_skinny pick (another split-K degree / lever of a
# registered skinny configuration; ``selector.SELECTOR_SKINNY``), opt-in
_SELECTED_SKINNY_NAME = "universal_gemm_fp16_cdna_selected_skinny"
GEMM_FP16_REGISTRY.register(
    make_selected_skinny_candidate(
        name=_SELECTED_SKINNY_NAME,
        family=_FAMILY,
        abi_version=GEMM_FP16_RCR_ABI_VERSION,
        dtype="fp16",
        registry=lambda: GEMM_FP16_REGISTRY,
    )
)


def gemm_fp16_candidates() -> Tuple[KernelCandidate, ...]:
    return GEMM_FP16_REGISTRY.candidates()


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


def registered_gemm_fp16_combos(
    req: OperatorRequest,
) -> Tuple[Tuple[KernelCandidate, UniversalGemmSpec], ...]:
    """Every registered FP16 RCR candidate that can launch ``req``.

    Probes opt-in variants and expands each candidate's ``sweep_space``.
    Production :func:`dispatch_gemm_fp16` is unchanged.
    """
    if _request_errors(req):
        return ()
    return GEMM_FP16_REGISTRY.combos(req)


def gemm_fp16_sweep_space(req: OperatorRequest) -> Sequence[UniversalGemmSpec]:
    """Bounded sweep space from all registered FP16 RCR candidates."""
    if _request_errors(req):
        return ()
    return GEMM_FP16_REGISTRY.sweep_space(req)


def dispatch_gemm_fp16_all(req: GemmRequest) -> Tuple[DispatchResult, ...]:
    """Every eligible FP16 RCR kernel for ``req``, including opt-in variants."""
    if _request_errors(req):
        return ()
    return GEMM_FP16_REGISTRY.dispatch_all(req, kernel_id=_kernel_id)


def dispatch_gemm_fp16(
    req: GemmRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select a registered FP16 RCR UniversalGemm candidate for ``req``."""
    candidate = GEMM_FP16_REGISTRY.select(req, ranker=ranker)
    spec = candidate.select_spec(req)
    why: Tuple[str, ...] = ()
    if ranker is None:
        # the learned selector (``selector.py``; ROCKE_GEMM_SELECTOR=0 = rules only)
        candidate, spec, why = _selector.dispatch_override(
            req,
            GEMM_FP16_REGISTRY,
            candidate,
            spec,
            selected_name=_SELECTED_NAME,
            selected_skinny_name=_SELECTED_SKINNY_NAME,
        )
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
            f"selected {candidate.name} for fp16 RCR GEMM on {req.arch}",
            f"algorithm={candidate.algorithm}",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
            *why,
        ),
    )
