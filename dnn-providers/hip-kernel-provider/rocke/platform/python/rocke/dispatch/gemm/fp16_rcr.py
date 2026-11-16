# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""FP16 RCR UniversalGemm dispatcher case."""

from __future__ import annotations

from dataclasses import asdict
from functools import partial
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
    GEMM_DIM_VOCABULARY,
    RDNA_WMMA_GEMM_ARCHES,
    GemmRequest,
    apply_split_k,
    rcr_request_errors,
)
from .direct_a import gfx1201_direct_a_candidates
from .support import (
    gemm_config_supported,
    request_shape_supported,
    support_query_from_universal_spec,
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


def _spec_gfx1201_compv4(
    req: GemmRequest,
    name: str,
    *,
    tile: TileSpec,
    allow_1wg: bool = False,
    store_after_mma: bool = True,
    group_m: int = 4,
    k_stagger: int = 8,
    k_stagger_step: int = 4,
    global_addressing: str = "auto",
) -> UniversalGemmSpec:
    return _make_spec(
        name=name,
        arch=req.arch,
        tile=tile,
        trait=TraitSpec(
            pipeline="compv4",
            scheduler="intrawave",
            epilogue="default",
            lds_swizzle=True,
            lds_db_allow_1wg=allow_1wg,
            lds_store_after_mma=store_after_mma,
            group_m=group_m,
            k_stagger=k_stagger,
            k_stagger_step=k_stagger_step,
            global_addressing=global_addressing,
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
        ok, why = gemm_config_supported(
            support_query_from_universal_spec(spec, arch=req.arch)
        )
        if not ok:
            return False, why
        return request_shape_supported(req, spec)

    def select(req: OperatorRequest) -> UniversalGemmSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, GemmRequest)
        # Engage split-K for skinny/tall-N decode shapes that leave the device
        # idle; a no-op (returns the spec unchanged) for shapes that already
        # fill the device, keeping the default / square path byte-identical.
        return apply_split_k(req, spec_fn(req, name))

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=_ALGORITHM,
        spec_id=spec_id,
        abi_version=GEMM_FP16_RCR_ABI_VERSION,
        priority=priority,
        capability=Capability(
            arches=arches, dtypes=("fp16",), layouts=("RCR",), shapes=shapes
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: gemm_args_signature(),
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=build_universal_gemm,
        bind=lambda result, verify: gemm_rcr_binding(result, verify, dtype="fp16"),
    )
    return candidate


def _grid(spec: UniversalGemmSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    t = spec.tile
    assert isinstance(req, GemmRequest)
    # Split-K adds a Z dimension of ``split_k`` K-slice CTAs per (m,n) tile;
    # split_k == 1 (default) collapses to the canonical 2D grid.
    return ceil_div_grid((req.N, t.tile_n), (req.M, t.tile_m), (spec.trait.split_k, 1))


# Explicit gfx targets rather than a cdna/rdna family label. Family does not
# imply wave size -- gfx1250 is cdna at wave32 -- so a family gate would admit a
# wave32 target into these wave64 MFMA candidates. A candidate admits a target
# when its list names the target or its generic (the row arch_specs.json points
# the target at); any other target is one the candidate was never built or run
# against.
_CDNA_MFMA_FP16 = ("gfx942", "gfx950")

# The measured gfx1201 palette is deliberately not a gfx12-family claim.
# Bounds are output elements and reduction elements; tile divisibility remains
# the common request-shape predicate. No host probe or timing enters selection.
_GFX1201_ARCHES = ("gfx1201",)
_GFX1201_SMALL_MAX_OUTPUT = 3 << 18
_GFX1201_MEDIUM_MAX_OUTPUT = 1 << 20
_GFX1201_LARGE_MIN_OUTPUT = 1 << 22
_GFX1201_LARGE_MAX_OUTPUT = (1 << 24) - 1
_GFX1201_MEDIUM_SHAPES = (
    ShapeRange(
        "output_elements",
        min=_GFX1201_SMALL_MAX_OUTPUT + 1,
        max=_GFX1201_MEDIUM_MAX_OUTPUT,
    ),
    ShapeRange("K", min=64),
)
_GFX1201_LARGE_SHAPES = (
    ShapeRange(
        "output_elements",
        min=_GFX1201_LARGE_MIN_OUTPUT + 1,
        max=_GFX1201_LARGE_MAX_OUTPUT,
    ),
)
_GFX1201_SMALL = partial(
    _spec_gfx1201_compv4, tile=TileSpec(64, 64, 64, 2, 2, 1, 16, 16, 16)
)
_GFX1201_SMALL_LOW_K = partial(_GFX1201_SMALL, group_m=8, k_stagger=0, k_stagger_step=1)
_GFX1201_DEFAULT = partial(
    _spec_gfx1201_compv4, tile=TileSpec(128, 128, 32, 2, 2, 1, 16, 16, 16)
)
_GFX1201_MEDIUM = partial(
    _spec_gfx1201_compv4,
    tile=TileSpec(128, 128, 64, 2, 4, 1, 16, 16, 16),
    allow_1wg=True,
)
_GFX1201_MEDIUM_LOW_K = partial(
    _GFX1201_MEDIUM, group_m=8, k_stagger=0, k_stagger_step=1
)
_GFX1201_WIDE = partial(
    _spec_gfx1201_compv4,
    tile=TileSpec(128, 256, 32, 2, 4, 1, 16, 16, 16),
    allow_1wg=True,
    group_m=8,
    k_stagger=0,
    k_stagger_step=1,
)
_GFX1201_LONG_K = partial(
    _spec_gfx1201_compv4,
    tile=TileSpec(256, 128, 32, 4, 2, 1, 16, 16, 16),
    allow_1wg=True,
)
# Measured: the two-path kernel is slower than legacy addressing on this tile
# and on short_k32, so both keep the legacy loads.
_GFX1201_K16 = partial(
    _spec_gfx1201_compv4,
    tile=TileSpec(128, 128, 16, 2, 2, 1, 16, 16, 16),
    global_addressing="legacy",
)
_GFX1201_FRINGE_M = partial(
    _spec_gfx1201_compv4, tile=TileSpec(32, 64, 64, 1, 2, 1, 16, 16, 16)
)
_GFX1201_FRINGE_N = partial(
    _spec_gfx1201_compv4, tile=TileSpec(64, 32, 64, 2, 1, 1, 16, 16, 16)
)
_GFX1201_FRINGE = partial(
    _spec_gfx1201_compv4, tile=TileSpec(32, 32, 64, 1, 1, 1, 16, 16, 16)
)
_GFX1201_SHORT = partial(
    _spec_gfx1201_compv4, tile=TileSpec(32, 32, 16, 1, 1, 1, 16, 16, 16)
)

_DIRECT_A, _DIRECT_A_FALLBACK = gfx1201_direct_a_candidates(
    dtype="fp16",
    family=_FAMILY,
    algorithm=_ALGORITHM,
    abi_version=GEMM_FP16_RCR_ABI_VERSION,
    request_errors=_request_errors,
    grid=_grid,
    class_priority=0,
    fallback_priority=90,
)

GEMM_FP16_REGISTRY = CandidateRegistry(
    _FAMILY,
    dim_vocabulary=GEMM_DIM_VOCABULARY,
    require_build=True,
    require_binding=True,
)
GEMM_FP16_REGISTRY.extend(
    (
        _make_candidate(
            name="universal_gemm_fp16_cdna_cshuffle",
            spec_id="cdna_cshuffle_default",
            priority=10,
            spec_fn=_spec_cdna_cshuffle,
            arches=_CDNA_MFMA_FP16,
        ),
        # The measured gfx1201 direct-A classes come first (see direct_a.py).
        *_DIRECT_A,
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_short_k32",
            spec_id="gfx1201_compv4_short_k32",
            priority=1,
            spec_fn=partial(
                _spec_gfx1201_compv4,
                tile=TileSpec(128, 128, 32, 2, 4, 1, 16, 16, 16),
                store_after_mma=False,
                global_addressing="legacy",
            ),
            arches=_GFX1201_ARCHES,
            shapes=(ShapeRange("K", max=32),),
        ),
        # K a multiple of 8192 is a 16 KiB row stride: rows alias in the
        # caches, so the K-32 tiles and direct A reads are the slow choices
        # there. Bands: the 128x128x64 tile with
        # grouped-M 8 from 2^21 outputs, the same tile with the stagger above
        # 2^18, then 64x64x64 (grouped-M 8 above 2^18, where 128 does not
        # divide M and N).
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_long_stride",
            spec_id="gfx1201_compv4_long_stride",
            priority=2,
            spec_fn=_GFX1201_MEDIUM_LOW_K,
            arches=_GFX1201_ARCHES,
            shapes=(
                ShapeRange("output_elements", min=1 << 21),
                ShapeRange("K", multiple_of=8192),
            ),
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_long_stride_medium",
            spec_id="gfx1201_compv4_long_stride_medium",
            priority=2,
            spec_fn=_GFX1201_MEDIUM,
            arches=_GFX1201_ARCHES,
            shapes=(
                ShapeRange("output_elements", min=(1 << 18) + 1),
                ShapeRange("K", multiple_of=8192),
            ),
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_long_stride_small_grouped",
            spec_id="gfx1201_compv4_long_stride_small_grouped",
            priority=2,
            spec_fn=_GFX1201_SMALL_LOW_K,
            arches=_GFX1201_ARCHES,
            shapes=(
                ShapeRange("output_elements", min=(1 << 18) + 1),
                ShapeRange("K", multiple_of=8192),
            ),
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_long_stride_small",
            spec_id="gfx1201_compv4_long_stride_small",
            priority=2,
            spec_fn=_GFX1201_SMALL,
            arches=_GFX1201_ARCHES,
            shapes=(
                ShapeRange("output_elements", max=1 << 18),
                ShapeRange("K", multiple_of=8192),
            ),
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_small",
            spec_id="gfx1201_compv4_small",
            priority=3,
            spec_fn=_GFX1201_SMALL,
            arches=_GFX1201_ARCHES,
            shapes=(ShapeRange("output_elements", max=_GFX1201_SMALL_MAX_OUTPUT),),
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_medium_low_k",
            spec_id="gfx1201_compv4_medium_low_k",
            priority=3,
            spec_fn=_GFX1201_MEDIUM_LOW_K,
            arches=_GFX1201_ARCHES,
            shapes=_GFX1201_MEDIUM_SHAPES + (ShapeRange("K", min=1024, max=2047),),
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_medium",
            spec_id="gfx1201_compv4_medium",
            priority=4,
            spec_fn=_GFX1201_MEDIUM,
            arches=_GFX1201_ARCHES,
            shapes=_GFX1201_MEDIUM_SHAPES,
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_large_long_k",
            spec_id="gfx1201_compv4_large_long_k",
            priority=5,
            spec_fn=_GFX1201_LONG_K,
            arches=_GFX1201_ARCHES,
            shapes=_GFX1201_LARGE_SHAPES + (ShapeRange("K", min=4096, max=8191),),
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_large_wide",
            spec_id="gfx1201_compv4_large_wide",
            priority=6,
            spec_fn=_GFX1201_WIDE,
            arches=_GFX1201_ARCHES,
            shapes=_GFX1201_LARGE_SHAPES
            + (
                ShapeRange("M", max=2048),
                ShapeRange("N", min=4096),
                ShapeRange("K", min=2048, max=4095),
            ),
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_default",
            spec_id="gfx1201_compv4_default",
            priority=7,
            spec_fn=_GFX1201_DEFAULT,
            arches=_GFX1201_ARCHES,
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_aligned64",
            spec_id="gfx1201_compv4_aligned64",
            priority=8,
            spec_fn=_GFX1201_SMALL,
            arches=_GFX1201_ARCHES,
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_k16",
            spec_id="gfx1201_compv4_k16",
            priority=9,
            spec_fn=_GFX1201_K16,
            arches=_GFX1201_ARCHES,
            shapes=(ShapeRange("K", min=32),),
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_fringe_m",
            spec_id="gfx1201_compv4_fringe_m",
            priority=10,
            spec_fn=_GFX1201_FRINGE_M,
            arches=_GFX1201_ARCHES,
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_fringe_n",
            spec_id="gfx1201_compv4_fringe_n",
            priority=11,
            spec_fn=_GFX1201_FRINGE_N,
            arches=_GFX1201_ARCHES,
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_fringe",
            spec_id="gfx1201_compv4_fringe",
            priority=12,
            spec_fn=_GFX1201_FRINGE,
            arches=_GFX1201_ARCHES,
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1201_unaligned_k16",
            spec_id="gfx1201_compv4_unaligned_k16",
            priority=13,
            spec_fn=_GFX1201_SHORT,
            arches=_GFX1201_ARCHES,
            shapes=(ShapeRange("K", min=32),),
        ),
        _make_candidate(
            name="universal_gemm_fp16_rdna_wmma",
            spec_id="rdna_wmma_default",
            priority=30,
            spec_fn=_spec_rdna_wmma,
            arches=RDNA_WMMA_GEMM_ARCHES,
        ),
        _make_candidate(
            name="universal_gemm_fp16_cdna_mem",
            spec_id="cdna_mem_64x128",
            priority=20,
            spec_fn=_spec_cdna_mem,
            arches=_CDNA_MFMA_FP16,
        ),
        _make_candidate(
            name="universal_gemm_fp16_rdna_wmma_small",
            spec_id="rdna_wmma_32x32",
            priority=40,
            spec_fn=_spec_rdna_wmma_small,
            arches=RDNA_WMMA_GEMM_ARCHES,
        ),
        # Last resort on gfx1201: M or N that no other tile divides. Partial
        # tiles make these correct for any M and N; they are not tuned.
        _DIRECT_A_FALLBACK,
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
        ),
    )
