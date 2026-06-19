# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""BF16 RCR UniversalGemm dispatcher case.

This is the second GEMM-family case and the worked template for adding more
dtypes/layouts. It reuses the same machinery as :mod:`fp16_rcr`:

* the dtype-/layout-generic request validation (:func:`rcr_request_errors`),
* the arch-family gate (:func:`arch_family_supported`) that keeps RDNA/WMMA
  candidates off CDNA arches and CDNA candidates off RDNA arches,
* the generic config predicate (:func:`gemm_config_supported`), which is already
  dtype-agnostic (it consults the per-arch MMA catalog).

The only material difference from the fp16 case is the MMA atom geometry: bf16
on CDNA has no 32x32 atom (only 16x16x16 / 16x16x32 on gfx950, 16x16x16 on
gfx942), so the CDNA cshuffle candidate uses a 16x16x16 warp tile with a 4x4
warp grid rather than fp16's 32x32 / 2x2.
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
)
from ..core import (
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    stable_json_hash,
)
from .common import (
    GemmRequest,
    arch_family_supported,
    rcr_request_errors,
    selector_matches,
)
from .support import (
    gemm_config_supported,
    request_shape_supported,
    support_query_from_universal_spec,
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
    # bf16 CDNA has no 32x32 atom, so the 128x128 cshuffle tile is built from a
    # 16x16x16 atom with a 4x4 warp grid (block_size = 4*4*64 = 1024). tile_k
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
    arch_family: str,
) -> KernelCandidate:
    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, GemmRequest)
        ok, why = arch_family_supported(req, arch_family)
        if not ok:
            return False, why
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
        ok, why = support(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, GemmRequest)
        return spec_fn(req, name)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=_ALGORITHM,
        spec_id=spec_id,
        abi_version=GEMM_BF16_RCR_ABI_VERSION,
        priority=priority,
        supports=support,
        select_spec=select,
        signature=lambda _spec: gemm_args_signature(),
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if support(req)[0] else (),
    )
    return candidate


def _grid(spec: UniversalGemmSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    t = spec.tile
    assert isinstance(req, GemmRequest)
    return ceil_div_grid((req.N, t.tile_n), (req.M, t.tile_m))


GEMM_BF16_REGISTRY = CandidateRegistry(_FAMILY)
GEMM_BF16_REGISTRY.extend(
    (
        _make_candidate(
            name="universal_gemm_bf16_cdna_cshuffle",
            spec_id="cdna_cshuffle_default",
            priority=10,
            spec_fn=_spec_cdna_cshuffle,
            arch_family="cdna",
        ),
        _make_candidate(
            name="universal_gemm_bf16_rdna_wmma",
            spec_id="rdna_wmma_default",
            priority=10,
            spec_fn=_spec_rdna_wmma,
            arch_family="rdna",
        ),
        _make_candidate(
            name="universal_gemm_bf16_cdna_mem",
            spec_id="cdna_mem_64x128",
            priority=20,
            spec_fn=_spec_cdna_mem,
            arch_family="cdna",
        ),
        _make_candidate(
            name="universal_gemm_bf16_rdna_wmma_small",
            spec_id="rdna_wmma_32x32",
            priority=20,
            spec_fn=_spec_rdna_wmma_small,
            arch_family="rdna",
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
    return build_universal_gemm(result.spec, arch=result.request.arch)


def gemm_bf16_sweep_space(req: OperatorRequest) -> Sequence[UniversalGemmSpec]:
    """Bounded sweep space from all registered BF16 RCR candidates."""
    if _request_errors(req):
        return ()
    specs: list[UniversalGemmSpec] = []
    seen = set()
    for candidate in GEMM_BF16_REGISTRY.supported(req):
        spec = candidate.select_spec(req)
        spec_hash = stable_json_hash(asdict(spec), n=16)
        if spec_hash not in seen:
            seen.add(spec_hash)
            specs.append(spec)
    return tuple(specs)


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
