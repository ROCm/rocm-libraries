# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""FP16 RCR UniversalGemm dispatcher case."""

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
)
from .binding import gemm_rcr_binding
from .common import (
    GEMM_DIM_VOCABULARY,
    GemmRequest,
    apply_split_k,
    rcr_request_errors,
    selector_matches,
)
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


def _make_candidate(
    *,
    name: str,
    spec_id: str,
    priority: int,
    spec_fn: Callable[[GemmRequest, str], UniversalGemmSpec],
    arches: Tuple[str, ...],
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
        capability=Capability(arches=arches, dtypes=("fp16",), layouts=("RCR",)),
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
        _make_candidate(
            name="universal_gemm_fp16_cdna_cshuffle",
            spec_id="cdna_cshuffle_default",
            priority=10,
            spec_fn=_spec_cdna_cshuffle,
            arches=_CDNA_MFMA_FP16,
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
        ),
        _make_candidate(
            name="universal_gemm_fp16_rdna_wmma_small",
            spec_id="rdna_wmma_32x32",
            priority=20,
            spec_fn=_spec_rdna_wmma_small,
            arches=_RDNA_WMMA,
        ),
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


def gemm_fp16_sweep_space(req: OperatorRequest) -> Sequence[UniversalGemmSpec]:
    """Bounded sweep space from all registered FP16 RCR candidates."""
    if _request_errors(req):
        return ()
    specs: list[UniversalGemmSpec] = []
    seen = set()
    for candidate in GEMM_FP16_REGISTRY.supported(req):
        spec = candidate.select_spec(req)
        spec_hash = stable_json_hash(asdict(spec), n=16)
        if spec_hash not in seen:
            seen.add(spec_hash)
            specs.append(spec)
    return tuple(specs)


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
