# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""FP8 / BF8 RCR block-scale GEMM dispatcher case.

The kernel body is :mod:`rocke.instances.common.block_scale_gemm`, which has
shipped since the CK Tile ``38_block_scale_gemm`` port but was reachable only by
constructing a spec by hand. This module puts it behind the dispatcher so a
framework request selects it the way it already selects the fp16 / bf16
UniversalGemm cases.

Scope is the shape a decoding LLM actually issues. vLLM's dynamic fp8 linear
layers call ``torch._scaled_mm`` with A row-major ``(M, K)``, B a weight matrix
row-major ``(N, K)``, and a single f32 scale per side -- so the candidate
registered here is RCR, per-tensor scaled, bf16 out. Block-granular scales (the
DeepSeek-style ``(1, 1, 128)`` grouping the kernel was written for) are a
separate candidate rather than a mode of this one, because they change what the
scale operands mean; see :class:`Fp8GemmRequest`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence, Tuple

from ...instances.common.block_scale_gemm import (
    BlockScaleGemmSpec,
    block_scale_gemm_grid,
    block_scale_gemm_signature,
    build_block_scale_gemm,
    is_valid_spec,
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
)
from .binding_fp8 import block_scale_gemm_fp8_binding
from .common import (
    GEMM_DIM_VOCABULARY,
    GemmRequest,
    normalize_selector,
    rcr_request_errors,
    selector_matches,
)

_FAMILY = "gemm_fp8_rcr"
_ALGORITHM = "block_scale_gemm"
GEMM_FP8_RCR_ABI_VERSION = "hipkg-gemm-fp8-rcr/v1"

_DTYPES = ("fp8e4m3", "bf8e5m2")
# The fp8 / bf8 16x16x32 MFMA atom the body emits ships from gfx940 onward, and
# the kernel is arch-neutral across these two. An arch absent from this list is
# one the candidate was never built or run against.
_CDNA_MFMA_FP8 = ("gfx942", "gfx950")

# What the body requires of the shape, as data so it answers coverage questions
# without a request: M and N must fill whole 16x16 output tiles (v1 has no
# partial-tile path) and K must be a whole number of 32-deep MFMA atoms.
_TILE = 16
_ATOM_K = 32

SCALE_PER_TENSOR = "per_tensor"
SCALE_BLOCK = "block"


@dataclass(frozen=True)
class Fp8GemmRequest(GemmRequest):
    """A GEMM request that also says how its scale operands are laid out.

    Scale granularity is carried as a *feature* rather than a dimension because
    it changes what the scale pointers mean, not how large they are. A candidate
    reading one scale per tensor cannot serve a request supplying one per
    128-deep K block; making that a feature means the registry rejects the pair
    outright instead of selecting a kernel that would quietly read the wrong
    element.
    """

    dtype: str = "fp8e4m3"
    scale_mode: str = SCALE_PER_TENSOR

    def features(self) -> frozenset[str]:
        return frozenset({f"scale:{normalize_selector(self.scale_mode)}"})


def _request_errors(req: OperatorRequest) -> list[str]:
    return rcr_request_errors(req, dtype=_DTYPES)


def _spec_per_tensor(req: GemmRequest, name: str) -> BlockScaleGemmSpec:
    """Per-tensor scaled spec for ``req``.

    One scale for the whole of A and one for the whole of B is the degenerate
    block grouping ``(M, N, K)``: the body's scale index arithmetic then folds
    to element 0 on both sides, so per-tensor scaling needs no separate code
    path in the kernel.
    """
    from .common import normalize_dtype

    return BlockScaleGemmSpec(
        name=name,
        M=req.M,
        N=req.N,
        K=req.K,
        quant_mode="abquant",
        mantissa_dtype=normalize_dtype(req.dtype),
        group_size_mnk=(req.M, req.N, req.K),
        layout="RCR",
        dtype_c="bf16",
    )


def _make_candidate(
    *,
    name: str,
    spec_id: str,
    priority: int,
    spec_fn: Callable[[GemmRequest, str], BlockScaleGemmSpec],
    scale_feature: str,
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
        return is_valid_spec(spec_fn(req, name), arch=req.arch)

    def select(req: OperatorRequest) -> BlockScaleGemmSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, GemmRequest)
        return spec_fn(req, name)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=_ALGORITHM,
        spec_id=spec_id,
        abi_version=GEMM_FP8_RCR_ABI_VERSION,
        priority=priority,
        capability=Capability(
            arches=arches,
            dtypes=_DTYPES,
            layouts=("RCR",),
            shapes=(
                ShapeRange(dims=frozenset({"M", "N"}), min=_TILE, multiple_of=_TILE),
                ShapeRange(dims="K", min=_ATOM_K, multiple_of=_ATOM_K),
            ),
            supports_features=frozenset({scale_feature}),
            requires_features=frozenset({scale_feature}),
        ),
        _supports=support,
        select_spec=select,
        signature=block_scale_gemm_signature,
        grid=lambda spec, _req: block_scale_gemm_grid(spec),
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=build_block_scale_gemm,
        bind=block_scale_gemm_fp8_binding,
    )
    return candidate


GEMM_FP8_REGISTRY = CandidateRegistry(
    _FAMILY,
    dim_vocabulary=GEMM_DIM_VOCABULARY,
    require_build=True,
    require_binding=True,
)
GEMM_FP8_REGISTRY.extend(
    (
        _make_candidate(
            name="block_scale_gemm_fp8_rcr_per_tensor",
            spec_id="cdna_mfma_16x16_per_tensor",
            priority=10,
            spec_fn=_spec_per_tensor,
            scale_feature=f"scale:{SCALE_PER_TENSOR}",
            arches=_CDNA_MFMA_FP8,
        ),
    )
)


def gemm_fp8_candidates() -> Tuple[KernelCandidate, ...]:
    return GEMM_FP8_REGISTRY.candidates()


def _kernel_id(
    req: GemmRequest, candidate: KernelCandidate, spec: BlockScaleGemmSpec
) -> KernelId:
    return KernelId(
        op="gemm",
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=stable_json_hash(req.normalized(), n=16),
        spec_hash=stable_json_hash(asdict(spec), n=16),
    )


def gemm_fp8_sweep_space(req: OperatorRequest) -> Sequence[BlockScaleGemmSpec]:
    """Bounded sweep space from all registered fp8 RCR candidates."""
    if _request_errors(req):
        return ()
    specs: list[BlockScaleGemmSpec] = []
    seen = set()
    for candidate in GEMM_FP8_REGISTRY.supported(req):
        spec = candidate.select_spec(req)
        spec_hash = stable_json_hash(asdict(spec), n=16)
        if spec_hash not in seen:
            seen.add(spec_hash)
            specs.append(spec)
    return tuple(specs)


def dispatch_gemm_fp8(
    req: GemmRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select a registered fp8 / bf8 RCR block-scale candidate for ``req``."""
    candidate = GEMM_FP8_REGISTRY.select(req, ranker=ranker)
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
            f"selected {candidate.name} for {spec.mantissa_dtype} RCR GEMM "
            f"on {req.arch}",
            f"algorithm={candidate.algorithm}",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )
