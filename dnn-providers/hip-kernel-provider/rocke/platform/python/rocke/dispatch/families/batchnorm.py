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

# ``channels_per_block`` candidate axis: 1 (default, byte-identical to the
# original one-CTA-per-channel emission) plus a few power-of-two groupings.
# ``1`` is always the highest-priority value (see ``_cpb_rank``) -- this is a
# default-OFF lever, exposed for measurement via ``registered_batchnorm_combos``
# / ``batchnorm_sweep_space`` / ``dispatch_batchnorm_all``, never auto-selected
# in production unless a caller pins its ``spec_id`` explicitly. It exists for
# shapes whose grid (``C`` CTAs) is small enough that launch overhead
# dominates -- see ``instances/common/batchnorm2d.py``'s module docstring.
_CHANNELS_PER_BLOCK = (1, 2, 4, 8)

# ``n_groups`` candidate axis: 1 (default, byte-identical to the original
# per-sample N-loop emission) plus a few power-of-two splits. ``1`` is always
# the highest-priority value (see ``_build_registry``) -- this is a
# default-OFF lever (valid for both ``training=True`` and ``training=False``
# requests -- see ``instances/common/batchnorm2d.py``'s
# ``build_batchnorm2d``/``is_valid_spec`` for the training-mode Welford merge),
# exposed the same way as ``_CHANNELS_PER_BLOCK`` for measurement, never
# auto-selected in production unless a caller pins its ``spec_id`` explicitly.
# See ``instances/common/batchnorm2d.py``'s module docstring for the shapes it
# targets (``H*W`` small enough to structurally cap ``block_size``).
_N_GROUPS = (1, 2, 4, 8)

# ``fast_reduce`` candidate axis: False (default, byte-identical to the
# original flat block_size-wide LDS Welford tree) and True (wave-shuffle-
# first Welford merge). ``False`` is always the highest-priority value (see
# ``_build_registry``) -- this is a default-OFF, training-only lever
# (``is_valid_spec`` rejects ``fast_reduce=True`` combined with
# ``training=False``), exposed the same way as ``_CHANNELS_PER_BLOCK`` /
# ``_N_GROUPS`` for measurement, never auto-selected in production unless a
# caller pins its ``spec_id`` explicitly. See
# ``instances/common/batchnorm2d.py``'s module docstring for what it trades
# (fewer LDS barriers in the per-channel reduction) and
# ``SUPPORT_MATRIX.md`` for the measured win.
_FAST_REDUCE = (False, True)


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


def _make_spec(
    req: BatchNormRequest,
    *,
    block_size: int,
    vec: int,
    channels_per_block: int = 1,
    n_groups: int = 1,
    fast_reduce: bool = False,
) -> _bn.BatchNorm2DSpec:
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
        channels_per_block=channels_per_block,
        n_groups=n_groups,
        fast_reduce=fast_reduce,
        wave_size=wave,
    )


def _make_candidate(
    *,
    block_size: int,
    vec: int,
    channels_per_block: int,
    n_groups: int,
    fast_reduce: bool,
    priority: int,
) -> KernelCandidate:
    # Matches BatchNorm2DSpec.kernel_name()'s convention: no suffix for the
    # default channels_per_block=1 / n_groups=1 / fast_reduce=False, so this
    # spec_id is stable for every pre-existing caller/test that pins it.
    cpb_suffix = f"_cpb{channels_per_block}" if channels_per_block != 1 else ""
    ng_suffix = f"_ng{n_groups}" if n_groups != 1 else ""
    fr_suffix = "_fr" if fast_reduce else ""
    spec_id = f"batchnorm_b{block_size}_v{vec}{cpb_suffix}{ng_suffix}{fr_suffix}"
    name = f"batchnorm2d_{spec_id}"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, BatchNormRequest)
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        spec = _make_spec(
            req,
            block_size=block_size,
            vec=vec,
            channels_per_block=channels_per_block,
            n_groups=n_groups,
            fast_reduce=fast_reduce,
        )
        return _bn.is_valid_spec(spec, arch=req.arch)

    def select(req: OperatorRequest) -> _bn.BatchNorm2DSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, BatchNormRequest)
        return _make_spec(
            req,
            block_size=block_size,
            vec=vec,
            channels_per_block=channels_per_block,
            n_groups=n_groups,
            fast_reduce=fast_reduce,
        )

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
    # One CTA per channels_per_block channels (1, the default, is one CTA
    # per channel).
    assert isinstance(req, BatchNormRequest)
    return (int(req.c) // spec.channels_per_block, 1, 1)


def _build_registry() -> CandidateRegistry:
    reg = CandidateRegistry(
        _FAMILY, dim_vocabulary=BATCHNORM_DIM_VOCABULARY, require_build=True
    )
    # Priority order = channels_per_block=1, n_groups=1 first (unchanged
    # production behavior; every cpb>1 / ng>1 candidate is a strictly
    # lower-priority, default-off lever -- see _CHANNELS_PER_BLOCK's and
    # _N_GROUPS's docstrings), then largest block_size, then widest vec
    # within a block_size tier, so the highest-throughput cpb=1/ng=1
    # candidate that the arch accepts still wins under ``auto`` exactly as
    # before either lever was added. ``n_groups>1`` candidates that don't fit
    # a request (e.g. non-divisible H*W, see is_valid_spec) are simply
    # rejected by ``is_valid_spec`` at dispatch time (no explicit skip needed
    # here); this applies the same way to both ``training=True`` and
    # ``training=False`` requests.
    cands = []
    for cpb_rank, cpb in enumerate(_CHANNELS_PER_BLOCK):
        for ng_rank, ng in enumerate(_N_GROUPS):
            for fr_rank, fr in enumerate(_FAST_REDUCE):
                prio = 10 + cpb_rank * 100000 + ng_rank * 1000 + fr_rank * 100
                for bs in _BLOCK_SIZES:
                    for v in _VECS:
                        cands.append(
                            _make_candidate(
                                block_size=bs,
                                vec=v,
                                channels_per_block=cpb,
                                n_groups=ng,
                                fast_reduce=fr,
                                priority=prio,
                            )
                        )
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


def _spec_has_tail(req: BatchNormRequest, spec: _bn.BatchNorm2DSpec) -> bool:
    """Would this (block_size, vec[, n_groups]) need batchnorm2d's HW
    scalar tail?

    Both ``n_groups == 1`` (scoped to ``block_size``) and ``n_groups > 1``
    (scoped to ``hw_threads = block_size // n_groups``) now accept a
    non-exactly-dividing HW via an in-kernel masked tail (see
    ``is_valid_spec``'s relaxation for both cases) -- this is the
    per-request signal :func:`_default_ranker` uses to keep picking an
    exact-fit candidate whenever one exists, falling back to a
    tail-needing one only when every same-tier candidate needs one.
    """
    hw = int(req.h) * int(req.w)
    hw_threads = spec.block_size // spec.n_groups
    chunk = hw_threads * spec.vec
    return hw % chunk != 0


def _default_ranker(
    req: OperatorRequest, candidates: Sequence[KernelCandidate]
) -> Sequence[KernelCandidate]:
    assert isinstance(req, BatchNormRequest)

    def key(c: KernelCandidate) -> Tuple[int, bool, int]:
        spec = c.select_spec(req)
        # The (cpb, ng, fr) tier is encoded in the hundreds+ digits of
        # ``priority`` (see ``_build_registry``); flooring to the nearest
        # 100 isolates that tier from the +0..14 (block_size, vec) offset
        # within it, so tail-free-vs-tail only reorders candidates inside
        # one tier, never promotes a worse tier ahead of a better one.
        tier = (c.priority // 100) * 100
        return (tier, _spec_has_tail(req, spec), c.priority)

    return sorted(candidates, key=key)


def dispatch_batchnorm(
    req: BatchNormRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select a registered batchnorm2d candidate for ``req``."""
    candidate = BATCHNORM_REGISTRY.select(req, ranker=ranker or _default_ranker)
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
