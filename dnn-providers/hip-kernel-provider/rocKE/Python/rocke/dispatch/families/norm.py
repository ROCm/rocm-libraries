# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Normalization (layernorm / rmsnorm) dispatcher family.

Worked implementation mirroring :mod:`rocke.dispatch.gemm.bf16_rcr`. Two
``kind`` values are supported: ``rmsnorm`` (backed by
:mod:`rocke.instances.common.rmsnorm2d`) and ``layernorm`` (backed by
:mod:`rocke.instances.common.layernorm2d`).

Norm kernels are NOT MMA-bound: each launches one CTA per row and uses an
LDS-tree reduction, so the support predicate is occupancy / vector-width driven
(``validate_io`` + LDS-capacity + max-threads-per-block from
:class:`rocke.core.arch.ArchTarget`) rather than a GEMM tile/atom check. The
instances' own ``is_valid_spec`` is the source of truth for that predicate; this
family wraps it.

These kernels are arch-family agnostic (they run on both CDNA and RDNA), so
there is no ``arch_family`` gate -- only the per-arch LDS / thread checks the
instance validators already encode.

Selection: a request carries ``(rows, cols)``. ``cols`` is the per-row width
``N``; ``n_per_block`` is fixed to ``cols`` (one CTA spans the whole row). The
candidate set is the cross product of block_size and vec; ``auto`` picks the
highest-priority (largest block_size, then widest vec) candidate whose
``is_valid_spec`` accepts the request on the target arch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence, Tuple

from ...core.arch import ArchTarget
from ...instances.common import layernorm2d as _ln
from ...instances.common import rmsnorm2d as _rms
from ..core import (
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    stable_json_hash,
)

_FAMILY = "norm2d"
NORM_ABI_VERSION = "hipkg-norm2d/v1"

# Allowed knob grids, mirroring IOSpecRule's allowed_block_sizes / allowed_vecs.
_BLOCK_SIZES = (1024, 512, 256, 128, 64)
_VECS = (8, 4, 2)
_KINDS = ("rmsnorm", "layernorm")


@dataclass(frozen=True)
class NormRequest(OperatorRequest):
    """Normalized row-wise normalization request (layernorm / rmsnorm)."""

    rows: int
    cols: int
    arch: str
    kind: str = "rmsnorm"  # "rmsnorm" | "layernorm"
    op: str = "norm"
    dtype: str = "fp16"
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = _norm_dtype(self.dtype)
        d["kind"] = self.kind.lower()
        return d


def _norm_dtype(dtype: str) -> str:
    d = dtype.lower()
    if d == "fp16":
        return "f16"
    return d


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, NormRequest):
        return [f"expected NormRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "norm":
        errors.append(f"unsupported op {req.op!r}")
    for field in ("rows", "cols"):
        if int(getattr(req, field)) <= 0:
            errors.append(f"{field} must be positive")
    if req.kind.lower() not in _KINDS:
        errors.append(f"unsupported norm kind {req.kind!r}; one of {_KINDS}")
    if _norm_dtype(req.dtype) not in ("f16", "bf16"):
        errors.append(f"unsupported dtype {req.dtype!r}; f16/bf16 only")
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    return errors


def _selector_matches(req: NormRequest, candidate: KernelCandidate) -> Tuple[bool, str]:
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"


def _make_spec(req: NormRequest, *, block_size: int, vec: int):
    """Build the instance spec for one (kind, block_size, vec) candidate."""
    dtype = _norm_dtype(req.dtype)
    wave = ArchTarget.from_gfx(req.arch).wave_size
    if req.kind.lower() == "rmsnorm":
        return _rms.RMSNorm2DSpec(
            n_per_block=int(req.cols),
            block_size=block_size,
            vec=vec,
            dtype=dtype,
            wave_size=wave,
        )
    return _ln.LayerNorm2DSpec(
        n_per_block=int(req.cols),
        block_size=block_size,
        vec=vec,
        dtype=dtype,
        wave_size=wave,
    )


def _is_valid(req: NormRequest, spec) -> Tuple[bool, str]:
    if req.kind.lower() == "rmsnorm":
        return _rms.is_valid_spec(spec, arch=req.arch)
    return _ln.is_valid_spec(spec, arch=req.arch)


def _make_candidate(
    *, kind: str, block_size: int, vec: int, priority: int
) -> KernelCandidate:
    spec_id = f"{kind}_b{block_size}_v{vec}"
    name = f"norm2d_{spec_id}"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, NormRequest)
        if req.kind.lower() != kind:
            return False, f"candidate kind {kind!r} != request kind {req.kind!r}"
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        spec = _make_spec(req, block_size=block_size, vec=vec)
        return _is_valid(req, spec)

    def select(req: OperatorRequest):
        ok, why = support(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, NormRequest)
        return _make_spec(req, block_size=block_size, vec=vec)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=kind,
        spec_id=spec_id,
        abi_version=NORM_ABI_VERSION,
        priority=priority,
        supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if support(req)[0] else (),
    )
    return candidate


def _grid(spec, req: OperatorRequest) -> Tuple[int, int, int]:
    # One CTA per row.
    assert isinstance(req, NormRequest)
    return (int(req.rows), 1, 1)


def _build_registry() -> CandidateRegistry:
    reg = CandidateRegistry(_FAMILY)
    # Priority order = largest block_size first, then widest vec, so the
    # highest-throughput candidate that the arch accepts wins under ``auto``.
    cands = []
    prio = 10
    for bs in _BLOCK_SIZES:
        for v in _VECS:
            for kind in _KINDS:
                cands.append(
                    _make_candidate(kind=kind, block_size=bs, vec=v, priority=prio)
                )
            prio += 1
    reg.extend(cands)
    return reg


NORM_REGISTRY = _build_registry()


def norm_candidates() -> Tuple[KernelCandidate, ...]:
    return NORM_REGISTRY.candidates()


def _kernel_id(req: NormRequest, candidate: KernelCandidate, spec) -> KernelId:
    request_hash = stable_json_hash(req.normalized(), n=16)
    spec_hash = stable_json_hash(asdict(spec), n=16)
    return KernelId(
        op="norm",
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=request_hash,
        spec_hash=spec_hash,
    )


def norm_sweep_space(req: OperatorRequest) -> Sequence[object]:
    if _request_errors(req):
        return ()
    specs = []
    seen = set()
    for candidate in NORM_REGISTRY.supported(req):
        spec = candidate.select_spec(req)
        h = stable_json_hash(asdict(spec), n=16)
        if h not in seen:
            seen.add(h)
            specs.append(spec)
    return tuple(specs)


def dispatch_norm(req: NormRequest, *, ranker: Ranker | None = None) -> DispatchResult:
    """Select a registered norm2d candidate for ``req``."""
    candidate = NORM_REGISTRY.select(req, ranker=ranker)
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
            f"selected {candidate.name} for {req.kind} norm on {req.arch}",
            f"algorithm={candidate.algorithm}",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )
