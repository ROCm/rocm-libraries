# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatch for local rank-staged MoE reduction epilogues.

This family selects only the single-device arithmetic kernels. Collective
transport remains an explicit caller responsibility; a supported request means
that its rank-major staging layout can be reduced, not that rocKE can move data
between devices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ...instances.common.moe_rank_reduce import (
    MoeRankReduceRMSNormSpec,
    MoeRankReduceScatterSpec,
    build_moe_rank_reduce_rmsnorm,
    build_moe_rank_reduce_scatter,
    is_valid_rmsnorm_spec,
    is_valid_scatter_spec,
    moe_rank_reduce_rmsnorm_grid,
    moe_rank_reduce_rmsnorm_signature,
    moe_rank_reduce_scatter_grid,
    moe_rank_reduce_scatter_signature,
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

_FAMILY = "moe_rank_reduce"
MOE_RANK_REDUCE_ABI_VERSION = "hipkg-moe-rank-reduce/v1"
_SUPPORTED_ARCHES = ("gfx950",)
_SUPPORTED_DTYPES = ("f16", "bf16")
_OPERATIONS = ("rmsnorm", "scatter")


@dataclass(frozen=True)
class MoeRankReduceRequest(OperatorRequest):
    """Normalized request for one local rank-staged reduction epilogue."""

    rows: int
    width: int
    world_size: int
    arch: str
    operation: str
    rank: int = 0
    dtype: str = "bf16"
    fp32_internal: bool = False
    op: str = _FAMILY
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        values = asdict(self)
        values["dtype"] = _normalize_dtype(self.dtype)
        values["operation"] = self.operation.strip().lower()
        return values

    def dims(self) -> dict[str, int]:
        return {
            "rows": int(self.rows),
            "width": int(self.width),
            "world_size": int(self.world_size),
            "rank": int(self.rank),
        }


MOE_RANK_REDUCE_DIM_VOCABULARY = ("rows", "width", "world_size", "rank")


def _normalize_dtype(dtype: str) -> str:
    value = dtype.strip().lower()
    return "f16" if value in ("fp16", "half") else value


def _select_geometry(n_per_block: int) -> tuple[int, int]:
    for block_size in (256, 128, 64):
        for vec in (8, 4, 2):
            if n_per_block % (block_size * vec) == 0:
                return block_size, vec
    raise ValueError(
        f"no rank-reduce geometry for n_per_block={n_per_block}; "
        "it must divide block_size*vec for block_size in {64,128,256} "
        "and vec in {2,4,8}"
    )


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, MoeRankReduceRequest):
        return [f"expected MoeRankReduceRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != _FAMILY:
        errors.append(f"unsupported op {req.op!r}")
    operation = req.operation.strip().lower()
    if operation not in _OPERATIONS:
        errors.append(f"operation must be one of {_OPERATIONS}, got {req.operation!r}")
    if req.rows <= 0:
        errors.append("rows must be positive")
    if req.width <= 0:
        errors.append("width must be positive")
    if req.world_size <= 0:
        errors.append("world_size must be positive")
    if not 0 <= req.rank < max(req.world_size, 1):
        errors.append("rank must be in [0, world_size)")
    if operation == "scatter" and req.world_size > 0 and req.width % req.world_size:
        errors.append("scatter width must be divisible by world_size")
    if _normalize_dtype(req.dtype) not in _SUPPORTED_DTYPES:
        errors.append(f"unsupported dtype {req.dtype!r}")
    return errors


def _selector_matches(
    req: MoeRankReduceRequest, candidate: KernelCandidate
) -> tuple[bool, str]:
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"


def _rmsnorm_spec(req: MoeRankReduceRequest) -> MoeRankReduceRMSNormSpec:
    block_size, vec = _select_geometry(req.width)
    return MoeRankReduceRMSNormSpec(
        width=req.width,
        world_size=req.world_size,
        dtype=_normalize_dtype(req.dtype),
        block_size=block_size,
        vec=vec,
        fp32_internal=req.fp32_internal,
    )


def _scatter_spec(req: MoeRankReduceRequest) -> MoeRankReduceScatterSpec:
    shard_width = req.width // req.world_size
    block_size, vec = _select_geometry(shard_width)
    return MoeRankReduceScatterSpec(
        width=req.width,
        world_size=req.world_size,
        dtype=_normalize_dtype(req.dtype),
        block_size=block_size,
        vec=vec,
    )


def _build(spec, arch: str):
    if isinstance(spec, MoeRankReduceRMSNormSpec):
        return build_moe_rank_reduce_rmsnorm(spec, arch)
    if isinstance(spec, MoeRankReduceScatterSpec):
        return build_moe_rank_reduce_scatter(spec, arch)
    raise TypeError(f"unsupported rank-reduce spec {type(spec).__name__}")


def _signature(spec):
    if isinstance(spec, MoeRankReduceRMSNormSpec):
        return moe_rank_reduce_rmsnorm_signature(spec)
    return moe_rank_reduce_scatter_signature(spec)


def _grid(spec, req: MoeRankReduceRequest) -> tuple[int, int, int]:
    if isinstance(spec, MoeRankReduceRMSNormSpec):
        return moe_rank_reduce_rmsnorm_grid(req.rows, spec)
    return moe_rank_reduce_scatter_grid(req.rows, spec)


def _make_candidate(operation: str, priority: int = 10) -> KernelCandidate:
    spec_fn = _rmsnorm_spec if operation == "rmsnorm" else _scatter_spec
    valid_fn = (
        is_valid_rmsnorm_spec if operation == "rmsnorm" else is_valid_scatter_spec
    )

    def support(req: OperatorRequest) -> tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, MoeRankReduceRequest)
        if req.operation.strip().lower() != operation:
            return False, f"operation {req.operation!r} != {operation!r}"
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        try:
            spec = spec_fn(req)
        except (ValueError, ZeroDivisionError) as error:
            return False, str(error)
        return valid_fn(spec, req.arch)

    def select(req: OperatorRequest):
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{candidate.name} does not support request: {why}")
        assert isinstance(req, MoeRankReduceRequest)
        return spec_fn(req)

    candidate = KernelCandidate(
        name=f"moe_rank_reduce_{operation}",
        family=_FAMILY,
        algorithm=operation,
        spec_id=operation,
        abi_version=MOE_RANK_REDUCE_ABI_VERSION,
        priority=priority,
        capability=Capability(
            arches=_SUPPORTED_ARCHES,
            dtypes=_SUPPORTED_DTYPES,
        ),
        _supports=support,
        select_spec=select,
        signature=_signature,
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=_build,
    )
    return candidate


MOE_RANK_REDUCE_REGISTRY = CandidateRegistry(
    _FAMILY,
    dim_vocabulary=MOE_RANK_REDUCE_DIM_VOCABULARY,
    require_build=True,
)
MOE_RANK_REDUCE_REGISTRY.extend(
    (_make_candidate("rmsnorm"), _make_candidate("scatter"))
)


def moe_rank_reduce_candidates() -> tuple[KernelCandidate, ...]:
    return MOE_RANK_REDUCE_REGISTRY.candidates()


def _spec_struct(spec) -> dict[str, object]:
    values: dict[str, object] = {
        "operation": (
            "rmsnorm" if isinstance(spec, MoeRankReduceRMSNormSpec) else "scatter"
        ),
        "width": spec.width,
        "world_size": spec.world_size,
        "dtype": spec.dtype,
        "block_size": spec.block_size,
        "vec": spec.vec,
    }
    if isinstance(spec, MoeRankReduceRMSNormSpec):
        values["fp32_internal"] = spec.fp32_internal
    return values


def dispatch_moe_rank_reduce(
    req: MoeRankReduceRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select one local rank-staged reduction epilogue."""

    candidate = MOE_RANK_REDUCE_REGISTRY.select(req, ranker=ranker)
    spec = candidate.select_spec(req)
    request_hash = stable_json_hash(req.normalized(), n=16)
    spec_hash = stable_json_hash(_spec_struct(spec), n=16)
    kernel_id = KernelId(
        op=_FAMILY,
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=request_hash,
        spec_hash=spec_hash,
    )
    return DispatchResult(
        request=req,
        candidate=candidate,
        spec=spec,
        kernel_id=kernel_id,
        grid=candidate.grid(spec, req),
        block=candidate.block(spec),
        signature=tuple(candidate.signature(spec)),
        explanation=(
            f"selected {candidate.name} for local rank-major staging",
            "collective transport is caller-owned",
            f"spec_hash={spec_hash}",
            f"request_hash={request_hash}",
        ),
    )


__all__ = [
    "MOE_RANK_REDUCE_ABI_VERSION",
    "MOE_RANK_REDUCE_REGISTRY",
    "MoeRankReduceRequest",
    "dispatch_moe_rank_reduce",
    "moe_rank_reduce_candidates",
]
