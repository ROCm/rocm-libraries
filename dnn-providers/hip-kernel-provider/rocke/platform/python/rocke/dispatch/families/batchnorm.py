# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""BatchNorm2D dispatcher family.

Its own family, not a third ``kind`` folded into
:mod:`rocke.dispatch.families.norm`: ``norm.py``'s ``layernorm``/``rmsnorm``
reduce *within one row* (one CTA per row, grid ``(rows, 1, 1)``, dim
vocabulary ``rows``/``cols``). BatchNorm reduces *across* ``N*H*W`` *per
channel* -- one CTA per channel, grid ``(c, 1, 1)``, dim vocabulary
``n``/``c``/``h``/``w``. Grafting that onto ``norm.py`` would mean diverging
its ``dims()``/``_grid()``/candidate shape for one kind, which is sharing the
file, not the abstraction. See
:mod:`rocke.instances.common.batchnorm2d`'s module docstring for the full
rationale.

Only one kind backs this family (:mod:`rocke.instances.common.batchnorm2d`),
so unlike ``norm.py`` there is no ``kind`` candidate axis: ``training`` is a
request field selecting one of two code paths in the *same* spec/kernel, not
a different instance module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence, Tuple

from ...core.arch import ArchTarget, arches_with_own_facts
from ...instances.common import batchnorm2d as _bn
from ..core import (
    Capability,
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    stable_json_hash,
    selector_matches,
)
from .norm import _BLOCK_SIZES, _VECS

_FAMILY = "batchnorm2d"
BATCHNORM_ABI_VERSION = "hipkg-batchnorm2d/v1"


@dataclass(frozen=True)
class BatchNormRequest(OperatorRequest):
    """BatchNorm2D forward request: an ``(N, C, H, W)`` NCHW activation."""

    n: int
    c: int
    h: int
    w: int
    arch: str
    training: bool = True
    op: str = "batchnorm"
    dtype: str = "fp16"
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = _bn_dtype(self.dtype)
        return d

    def dims(self) -> dict[str, int]:
        return {
            "n": int(self.n),
            "c": int(self.c),
            "h": int(self.h),
            "w": int(self.w),
        }


BATCHNORM_DIM_VOCABULARY = ("n", "c", "h", "w")

# Same arch-agnostic reasoning as norm.py: pure elementwise + LDS-tree
# reduction (training) or pure elementwise stream (inference), no MFMA atom,
# so every candidate declares every known target and the per-arch LDS /
# max-threads checks in ``is_valid_spec`` do the narrowing.
_BATCHNORM_ARCHES = arches_with_own_facts()


def _bn_dtype(dtype: str) -> str:
    d = dtype.lower()
    if d == "fp16":
        return "f16"
    return d


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, BatchNormRequest):
        return [f"expected BatchNormRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "batchnorm":
        errors.append(f"unsupported op {req.op!r}")
    for field in ("n", "c", "h", "w"):
        if int(getattr(req, field)) <= 0:
            errors.append(f"{field} must be positive")
    if _bn_dtype(req.dtype) not in ("f16", "bf16"):
        errors.append(f"unsupported dtype {req.dtype!r}; f16/bf16 only")
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    return errors


def _make_spec(req: BatchNormRequest, *, block_size: int, vec: int) -> _bn.BatchNorm2DSpec:
    dtype = _bn_dtype(req.dtype)
    wave = ArchTarget.from_gfx(req.arch).wave_size
    return _bn.BatchNorm2DSpec(
        n=int(req.n),
        c=int(req.c),
        h=int(req.h),
        w=int(req.w),
        block_size=block_size,
        vec=vec,
        dtype=dtype,
        training=bool(req.training),
        wave_size=wave,
    )


def _make_candidate(*, block_size: int, vec: int, priority: int) -> KernelCandidate:
    spec_id = f"batchnorm_b{block_size}_v{vec}"
    name = f"batchnorm2d_{spec_id}"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, BatchNormRequest)
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        spec = _make_spec(req, block_size=block_size, vec=vec)
        return _bn.is_valid_spec(spec, arch=req.arch)

    def select(req: OperatorRequest) -> _bn.BatchNorm2DSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, BatchNormRequest)
        return _make_spec(req, block_size=block_size, vec=vec)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm="batchnorm",
        spec_id=spec_id,
        abi_version=BATCHNORM_ABI_VERSION,
        priority=priority,
        capability=Capability(arches=_BATCHNORM_ARCHES, dtypes=("f16", "bf16")),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=_build,
    )
    return candidate


def _build(spec: _bn.BatchNorm2DSpec, _arch: str):
    return _bn.build_batchnorm2d(spec)


def _grid(spec: _bn.BatchNorm2DSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    # One CTA per channel.
    assert isinstance(req, BatchNormRequest)
    return (int(req.c), 1, 1)


def _build_registry() -> CandidateRegistry:
    reg = CandidateRegistry(
        _FAMILY, dim_vocabulary=BATCHNORM_DIM_VOCABULARY, require_build=True
    )
    # Priority order = largest block_size first, then widest vec, so the
    # highest-throughput candidate that the arch accepts wins under ``auto``.
    cands = []
    prio = 10
    for bs in _BLOCK_SIZES:
        for v in _VECS:
            cands.append(_make_candidate(block_size=bs, vec=v, priority=prio))
            prio += 1
    reg.extend(cands)
    return reg


BATCHNORM_REGISTRY = _build_registry()


def batchnorm_candidates() -> Tuple[KernelCandidate, ...]:
    return BATCHNORM_REGISTRY.candidates()


def _kernel_id(req: BatchNormRequest, candidate: KernelCandidate, spec) -> KernelId:
    request_hash = stable_json_hash(req.normalized(), n=16)
    spec_hash = stable_json_hash(asdict(spec), n=16)
    return KernelId(
        op="batchnorm",
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=request_hash,
        spec_hash=spec_hash,
    )


def registered_batchnorm_combos(
    req: OperatorRequest,
) -> Tuple[Tuple[KernelCandidate, object], ...]:
    """Every registered batchnorm candidate that can launch ``req``.

    Probes opt-in variants and expands each candidate's ``sweep_space``.
    Production :func:`dispatch_batchnorm` is unchanged.
    """
    if _request_errors(req):
        return ()
    return BATCHNORM_REGISTRY.combos(req)


def batchnorm_sweep_space(req: OperatorRequest) -> Sequence[object]:
    if _request_errors(req):
        return ()
    return BATCHNORM_REGISTRY.sweep_space(req)


def dispatch_batchnorm_all(req: BatchNormRequest) -> Tuple[DispatchResult, ...]:
    """Every eligible batchnorm2d kernel for ``req``, including opt-in variants."""
    if _request_errors(req):
        return ()
    return BATCHNORM_REGISTRY.dispatch_all(req, kernel_id=_kernel_id)


def dispatch_batchnorm(
    req: BatchNormRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select a registered batchnorm2d candidate for ``req``."""
    candidate = BATCHNORM_REGISTRY.select(req, ranker=ranker)
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
            f"selected {candidate.name} for batchnorm "
            f"(training={req.training}) on {req.arch}",
            f"algorithm={candidate.algorithm}",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )
