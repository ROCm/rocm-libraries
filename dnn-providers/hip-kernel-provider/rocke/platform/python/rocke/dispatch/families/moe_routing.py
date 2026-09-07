# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatch for fused MoE routing and compact active-expert packing."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ...instances.common.moe_topk_active_pack import (
    MoeTopkActivePackSpec,
    build_moe_topk_active_pack,
    is_valid_spec,
    moe_topk_active_pack_grid,
    moe_topk_active_pack_signature,
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

_FAMILY = "moe_routing"
MOE_ROUTING_ABI_VERSION = "hipkg-moe-routing/v1"


@dataclass(frozen=True)
class MoeRoutingRequest(OperatorRequest):
    """Normalized small-batch MoE routing request."""

    tokens: int
    experts: int
    topk: int
    arch: str
    num_expert_groups: int = 1
    topk_groups: int = 1
    tile_m: int = 16
    renormalize: bool = True
    dtype: str = "f32"
    op: str = _FAMILY
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        values = asdict(self)
        values["dtype"] = self.dtype.strip().lower()
        return values

    def dims(self) -> dict[str, int]:
        return {
            "tokens": self.tokens,
            "experts": self.experts,
            "topk": self.topk,
            "num_expert_groups": self.num_expert_groups,
            "topk_groups": self.topk_groups,
        }


MOE_ROUTING_DIM_VOCABULARY = (
    "tokens",
    "experts",
    "topk",
    "num_expert_groups",
    "topk_groups",
)


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, MoeRoutingRequest):
        return [f"expected MoeRoutingRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != _FAMILY:
        errors.append(f"unsupported op {req.op!r}")
    if req.tokens <= 0 or req.experts <= 0 or req.topk <= 0:
        errors.append("tokens, experts, and topk must be positive")
    if req.topk > req.experts:
        errors.append("topk must be <= experts")
    if req.dtype.strip().lower() != "f32":
        errors.append("the routing logits contract requires f32")
    return errors


def _selector_matches(req: MoeRoutingRequest, candidate: KernelCandidate):
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"


def _spec(req: MoeRoutingRequest) -> MoeTopkActivePackSpec:
    return MoeTopkActivePackSpec(
        tokens=req.tokens,
        experts=req.experts,
        topk=req.topk,
        tile_m=req.tile_m,
        num_expert_groups=req.num_expert_groups,
        topk_groups=req.topk_groups,
        renormalize=req.renormalize,
    )


def _support(req: OperatorRequest) -> tuple[bool, str]:
    errors = _request_errors(req)
    if errors:
        return False, "; ".join(errors)
    assert isinstance(req, MoeRoutingRequest)
    ok, why = _selector_matches(req, _CANDIDATE)
    if not ok:
        return False, why
    return is_valid_spec(_spec(req), req.arch)


def _select(req: OperatorRequest) -> MoeTopkActivePackSpec:
    ok, why = _CANDIDATE.admits(req)
    if not ok:
        raise ValueError(f"{_CANDIDATE.name} does not support request: {why}")
    assert isinstance(req, MoeRoutingRequest)
    return _spec(req)


_CANDIDATE = KernelCandidate(
    name="moe_topk_active_pack",
    family=_FAMILY,
    algorithm="topk_active_pack",
    spec_id="topk_active_pack",
    abi_version=MOE_ROUTING_ABI_VERSION,
    priority=10,
    capability=Capability(arches=("gfx950",), dtypes=("f32",)),
    _supports=_support,
    select_spec=_select,
    signature=moe_topk_active_pack_signature,
    grid=lambda spec, _req: moe_topk_active_pack_grid(spec),
    block=lambda spec: (spec.block_size, 1, 1),
    sweep_space=lambda req: (_select(req),) if _CANDIDATE.admits(req)[0] else (),
    build=build_moe_topk_active_pack,
)

MOE_ROUTING_REGISTRY = CandidateRegistry(
    _FAMILY, dim_vocabulary=MOE_ROUTING_DIM_VOCABULARY, require_build=True
)
MOE_ROUTING_REGISTRY.register(_CANDIDATE)


def moe_routing_candidates() -> tuple[KernelCandidate, ...]:
    return MOE_ROUTING_REGISTRY.candidates()


def dispatch_moe_routing(
    req: MoeRoutingRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select the fused correction-biased top-k active-pack kernel."""

    candidate = MOE_ROUTING_REGISTRY.select(req, ranker=ranker)
    spec = candidate.select_spec(req)
    request_hash = stable_json_hash(req.normalized(), n=16)
    spec_hash = stable_json_hash(
        {
            "tokens": spec.tokens,
            "experts": spec.experts,
            "topk": spec.topk,
            "tile_m": spec.tile_m,
            "groups": (spec.num_expert_groups, spec.topk_groups),
            "renormalize": spec.renormalize,
        },
        n=16,
    )
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
            "selected fused top-k, active-block scan, and compact metadata pack",
            f"static downstream block capacity={spec.max_blocks}",
            f"spec_hash={spec_hash}",
            f"request_hash={request_hash}",
        ),
    )


__all__ = [
    "MOE_ROUTING_ABI_VERSION",
    "MOE_ROUTING_REGISTRY",
    "MoeRoutingRequest",
    "dispatch_moe_routing",
    "moe_routing_candidates",
]
