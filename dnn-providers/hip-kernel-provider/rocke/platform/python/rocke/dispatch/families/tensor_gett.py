# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tensor contraction (GETT) dispatcher family, f16 / bf16 on gfx950.

A request carries one :class:`rocke.instances.common.tensor.TensorGettSpec`
problem (mode lists, extents, strides, dtype, ``bilinear``) and the target.
One candidate is registered:

* ``tensor_gett_t32s3`` -- the native GETT (one launch, operands read in place,
  permuted alpha / beta epilogue) with ``tile_k`` 32 and three direct-to-LDS
  stages, chosen from the plan alone. Problems the GETT planner cannot map
  (batch / Hadamard modes, an operand whose fastest free run is odd, K modes
  that do not split into the K tile) are not admitted; a caller falls back to
  a permute + GEMM composition for those.

The candidate's ``sweep_space`` is its knob neighbourhood (K tile, stage
count, persistent grid, paired 16-byte epilogue stores), for the tuning
engine to search. Python engine only (no C++ builder).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace

from ...instances.common.tensor import (
    TENSOR_GETT_ARCHS,
    TENSOR_GETT_DTYPES,
    TensorGettSpec,
    build_tensor_gett,
    is_valid_spec,
    tensor_gett_grid,
    tensor_gett_signature,
)
from ..core import (
    CandidateRegistry,
    Capability,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    selector_matches,
    stable_json_hash,
)

_FAMILY = "tensor_gett"
TENSOR_GETT_ABI_VERSION = "rocke-tensor-gett/v1"


@dataclass(frozen=True)
class TensorGettRequest(OperatorRequest):
    """One f16 / bf16 GETT launch. ``problem`` carries the modes, extents,
    strides, dtype and ``knobs.bilinear``; its body levers are ignored (the
    candidate picks them)."""

    problem: TensorGettSpec
    arch: str
    op: str = "contraction"
    algorithm: str = "auto"
    spec_id: str = "auto"

    @property
    def dtype(self) -> str:
        return "fp16" if self.problem.dtype in ("f16", "fp16") else self.problem.dtype

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = self.dtype
        return d

    def dims(self) -> dict[str, int]:
        ext = dict(self.problem.extents)
        p = self.problem
        size = lambda ms: _prod(ext[m] for m in ms)
        k_modes = [m for m in p.modes_a if m in p.modes_b and m not in p.modes_d]
        m_modes = [m for m in p.modes_d if m in p.modes_a]
        n_modes = [m for m in p.modes_d if m in p.modes_b and m not in p.modes_a]
        return {"M": size(m_modes), "N": size(n_modes), "K": size(k_modes)}


def _prod(xs) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


TENSOR_GETT_DIM_VOCABULARY = ("M", "N", "K")


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, TensorGettRequest):
        return [f"expected TensorGettRequest, got {type(req).__name__}"]
    errors = []
    if req.op != "contraction":
        errors.append(f"unsupported op {req.op!r}")
    if req.dtype not in TENSOR_GETT_DTYPES:
        errors.append(f"unsupported dtype {req.dtype!r}")
    return errors


def _spec_t32s3(req: TensorGettRequest) -> TensorGettSpec:
    return replace(
        req.problem,
        dtype=req.dtype,
        tile_k=32,
        dtl_stages=3,
        persistent_ctas=0,
        knobs=replace(req.problem.knobs, bilinear=req.problem.bilinear),
    )


def _neighbourhood(spec: TensorGettSpec) -> tuple[TensorGettSpec, ...]:
    """The default spec, then one-lever moves the tuner starts from."""
    out = [spec]
    for kw in (
        {"dtl_stages": 2},
        {"dtl_stages": 4},
        {"tile_k": 64},
        {"persistent_ctas": 256},
        {"knobs": replace(spec.knobs, epi_pair=True)},
    ):
        cand = replace(spec, **kw)
        if cand != spec and cand not in out:
            out.append(cand)
    return tuple(out)


def _make_candidate(*, name, spec_id, priority, spec_fn) -> KernelCandidate:
    def support(req: OperatorRequest) -> tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, TensorGettRequest)
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        return is_valid_spec(spec_fn(req), arch=req.arch)

    def select(req: OperatorRequest) -> TensorGettSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, TensorGettRequest)
        return spec_fn(req)

    def sweep(req: OperatorRequest):
        if not candidate.admits(req)[0]:
            return ()
        assert isinstance(req, TensorGettRequest)
        return tuple(
            s
            for s in _neighbourhood(spec_fn(req))
            if is_valid_spec(s, arch=req.arch)[0]
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm="gett",
        spec_id=spec_id,
        abi_version=TENSOR_GETT_ABI_VERSION,
        priority=priority,
        capability=Capability(arches=TENSOR_GETT_ARCHS, dtypes=TENSOR_GETT_DTYPES),
        _supports=support,
        select_spec=select,
        signature=tensor_gett_signature,
        grid=lambda spec, _req: tensor_gett_grid(spec),
        block=lambda spec: (256, 1, 1),
        sweep_space=sweep,
        build=lambda spec, arch: build_tensor_gett(spec, arch=arch),
    )
    return candidate


TENSOR_GETT_REGISTRY = CandidateRegistry(
    _FAMILY, dim_vocabulary=TENSOR_GETT_DIM_VOCABULARY, require_build=True
)
TENSOR_GETT_REGISTRY.extend(
    (
        _make_candidate(
            name="tensor_gett_t32s3",
            spec_id="t32s3",
            priority=10,
            spec_fn=_spec_t32s3,
        ),
    )
)


def tensor_gett_candidates() -> tuple[KernelCandidate, ...]:
    return TENSOR_GETT_REGISTRY.candidates()


def _kernel_id(req: TensorGettRequest, candidate: KernelCandidate, spec) -> KernelId:
    return KernelId(
        op="contraction",
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=stable_json_hash(req.normalized(), n=16),
        spec_hash=stable_json_hash(asdict(spec), n=16),
    )


def tensor_gett_sweep_space(req: OperatorRequest) -> Sequence[TensorGettSpec]:
    if _request_errors(req):
        return ()
    return TENSOR_GETT_REGISTRY.sweep_space(req)


def dispatch_tensor_gett_all(req: TensorGettRequest) -> tuple[DispatchResult, ...]:
    if _request_errors(req):
        return ()
    return TENSOR_GETT_REGISTRY.dispatch_all(req, kernel_id=_kernel_id)


def dispatch_tensor_gett(
    req: TensorGettRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select a registered tensor GETT candidate for ``req``."""
    candidate = TENSOR_GETT_REGISTRY.select(req, ranker=ranker)
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
            f"selected {candidate.name} for {req.dtype} GETT on {req.arch}",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )


__all__ = [
    "TENSOR_GETT_REGISTRY",
    "TensorGettRequest",
    "dispatch_tensor_gett",
    "dispatch_tensor_gett_all",
    "tensor_gett_candidates",
    "tensor_gett_sweep_space",
]
