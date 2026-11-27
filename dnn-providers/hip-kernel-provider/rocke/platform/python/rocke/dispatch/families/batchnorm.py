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
    DimRelation,
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
    backward: bool = False

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = _bn_dtype(self.dtype)
        if not self.backward:
            d.pop("backward")
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
    if d == "fp32":
        return "f32"
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
    if _bn_dtype(req.dtype) not in ("f16", "bf16", "f32"):
        errors.append(f"unsupported dtype {req.dtype!r}; f16/bf16/f32 only")
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
        capability=Capability(arches=_BATCHNORM_ARCHES, dtypes=("f16", "bf16", "f32")),
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


# ===================================================================== #
# Dynamic (AOT-generic) candidates -- capability-admission scaffolding
# ===================================================================== #
#
# conv's AOT instances (library/dispatch/grouped_convolution.py's _admits) are
# served by one compiled binary per tuning config that *admits* any runtime
# problem its declared capability covers, instead of one build per exact
# shape. instances/common/batchnorm2d.py's ``dynamic=True`` kernels
# (BatchNorm2DSpec.dynamic) now do the same at the kernel-body level: N/H/W
# are runtime kernargs, and channels_per_block/n_groups generalize via a
# dispatch-time "does this request's C/N divide evenly" check rather than a
# build-time one (see is_valid_spec's dynamic branch).
#
# This section is that dispatch-time check, built on the SAME Capability /
# DimRelation data the framework already has (core.py) -- no bespoke
# "_admits" needed, since DimRelation's "multiple_of" op already expresses
# exactly "this candidate's C must divide the request's C" as data.
#
# Deliberately NOT wired into BATCHNORM_REGISTRY / dispatch_batchnorm:
# dynamic_batchnorm_candidates_for() is gated by
# ENABLE_DYNAMIC_BATCHNORM_DISPATCH (False) and returns () unconditionally
# while that flag is off, so production dispatch is completely unaffected.
# The admission machinery (capability relations + support()) is real and
# exercisable directly (flip the flag, or call the candidates' own
# .admits()) -- this is scaffolding ahead of actually routing production
# traffic through it, not a placeholder stub.

ENABLE_DYNAMIC_BATCHNORM_DISPATCH = False


def _make_dynamic_candidate(
    *,
    block_size: int,
    vec: int,
    channels_per_block: int,
    n_groups: int,
    fast_reduce: bool,
    priority: int,
) -> KernelCandidate:
    cpb_suffix = f"_cpb{channels_per_block}" if channels_per_block != 1 else ""
    ng_suffix = f"_ng{n_groups}" if n_groups != 1 else ""
    fr_suffix = "_fr" if fast_reduce else ""
    spec_id = f"batchnorm_dyn_b{block_size}_v{vec}{cpb_suffix}{ng_suffix}{fr_suffix}"
    name = f"batchnorm2d_{spec_id}"

    def _placeholder_spec(req: BatchNormRequest) -> _bn.BatchNorm2DSpec:
        # n/c/h/w are capacity placeholders, never read in-kernel in
        # dynamic mode (see BatchNorm2DSpec.dynamic's docstring) -- the
        # real runtime N/C/H/W come from the request at launch time, not
        # from this spec. c=channels_per_block is the smallest value that
        # satisfies channels_per_block's own "c positive and (when not
        # dynamic) divisible by cpb" shape here.
        dtype = _bn_dtype(req.dtype)
        wave = ArchTarget.from_gfx(req.arch).wave_size
        return _bn.BatchNorm2DSpec(
            n=max(n_groups, 1),
            c=max(channels_per_block, 1),
            h=1,
            w=1,
            block_size=block_size,
            vec=vec,
            dtype=dtype,
            training=bool(req.training),
            save_mean_invstd=bool(req.training),
            channels_per_block=channels_per_block,
            n_groups=n_groups,
            fast_reduce=fast_reduce,
            wave_size=wave,
            dynamic=True,
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, BatchNormRequest)
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        spec = _placeholder_spec(req)
        return _bn.is_valid_spec(spec, arch=req.arch)

    def select(req: OperatorRequest) -> _bn.BatchNorm2DSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, BatchNormRequest)
        return _placeholder_spec(req)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm="batchnorm",
        spec_id=spec_id,
        abi_version=BATCHNORM_ABI_VERSION,
        priority=priority,
        capability=Capability(
            arches=_BATCHNORM_ARCHES,
            dtypes=("f16", "bf16", "f32"),
            relations=(
                DimRelation(lhs="c", op="multiple_of", rhs=channels_per_block),
                DimRelation(lhs="n", op="multiple_of", rhs=n_groups),
            ),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=_build,
        opt_in=True,
    )
    return candidate


def _build_dynamic_candidates() -> Tuple[KernelCandidate, ...]:
    cands = []
    for cpb_rank, cpb in enumerate(_CHANNELS_PER_BLOCK):
        for ng_rank, ng in enumerate(_N_GROUPS):
            for fr_rank, fr in enumerate(_FAST_REDUCE):
                prio = 10 + cpb_rank * 100000 + ng_rank * 1000 + fr_rank * 100
                for bs in _BLOCK_SIZES:
                    for v in _VECS:
                        cands.append(
                            _make_dynamic_candidate(
                                block_size=bs,
                                vec=v,
                                channels_per_block=cpb,
                                n_groups=ng,
                                fast_reduce=fr,
                                priority=prio,
                            )
                        )
                        prio += 1
    return tuple(cands)


_DYNAMIC_BATCHNORM_CANDIDATES = _build_dynamic_candidates()


def dynamic_batchnorm_candidates() -> Tuple[KernelCandidate, ...]:
    """Every registered dynamic (AOT-generic) batchnorm2d candidate,
    unfiltered by any request -- for introspection and direct testing of
    the admission machinery regardless of :data:`ENABLE_DYNAMIC_BATCHNORM_DISPATCH`.
    """
    return _DYNAMIC_BATCHNORM_CANDIDATES + _FLAT_INFER_CANDIDATES


def dynamic_batchnorm_candidates_for(
    req: BatchNormRequest,
) -> Tuple[KernelCandidate, ...]:
    """Dynamic candidates admitting ``req``, gated by
    :data:`ENABLE_DYNAMIC_BATCHNORM_DISPATCH`.

    Returns ``()`` unconditionally while that flag is ``False`` (its
    default): the dynamic kernel path has not yet been validated/benchmarked
    enough to route real dispatch traffic through it, so this is
    intentionally inert in production today, not an oversight. Flip the
    flag (or call a candidate's own ``.admits(req)``) to exercise the real
    admission logic directly.
    """
    if not ENABLE_DYNAMIC_BATCHNORM_DISPATCH:
        return ()
    if _request_errors(req):
        return ()
    return tuple(c for c in dynamic_batchnorm_candidates() if c.admits(req)[0])


# --- flat elementwise inference candidates (extra dynamic candidates) ---

_FLAT_BLOCK_SIZES = (1024, 512, 256, 128)
_FLAT_VECS = (8, 4, 2, 1)
_FLAT_UNROLLS = (1, 2, 4)


def _make_flat_infer_candidate(
    *, block_size: int, vec: int, unroll: int, aligned: bool, priority: int
) -> KernelCandidate:
    al = "_al" if aligned else ""
    spec_id = f"batchnorm_flat_b{block_size}_v{vec}_u{unroll}{al}"
    name = f"batchnorm2d_{spec_id}"

    def _spec(req: BatchNormRequest) -> _bn.BatchNorm2DFlatInferSpec:
        return _bn.BatchNorm2DFlatInferSpec(
            block_size=block_size,
            vec=vec,
            unroll=unroll,
            aligned=aligned,
            dtype=_bn_dtype(req.dtype),
            wave_size=ArchTarget.from_gfx(req.arch).wave_size,
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, BatchNormRequest)
        if req.training:
            return False, "flat kernel is inference-only"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        spec = _spec(req)
        ok, why = _bn.is_valid_flat_infer_spec(spec, arch=req.arch)
        if not ok:
            return False, why
        return _bn.flat_infer_admits(
            spec, int(req.n), int(req.c), int(req.h), int(req.w)
        )

    def select(req: OperatorRequest) -> _bn.BatchNorm2DFlatInferSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, BatchNormRequest)
        return _spec(req)

    def grid(spec, req: OperatorRequest) -> Tuple[int, int, int]:
        assert isinstance(req, BatchNormRequest)
        return _bn.batchnorm2d_flat_infer_grid(req.n, req.c, req.h, req.w, spec)

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
        grid=grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=lambda spec, _arch: _bn.build_batchnorm2d_flat_infer(spec),
        opt_in=True,
    )
    return candidate


def _build_flat_infer_candidates() -> Tuple[KernelCandidate, ...]:
    cands = []
    prio = 900000
    for aligned in (True, False):
        for vec in _FLAT_VECS:
            if vec == 1 and not aligned:
                continue
            for unroll in _FLAT_UNROLLS:
                for bs in _FLAT_BLOCK_SIZES:
                    cands.append(
                        _make_flat_infer_candidate(
                            block_size=bs,
                            vec=vec,
                            unroll=unroll,
                            aligned=aligned,
                            priority=prio,
                        )
                    )
                    prio += 1
    return tuple(cands)


_FLAT_INFER_CANDIDATES = _build_flat_infer_candidates()


def flat_infer_batchnorm_candidates() -> Tuple[KernelCandidate, ...]:
    """Every flat elementwise inference candidate, unfiltered by request."""
    return _FLAT_INFER_CANDIDATES


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


# ===================================================================== #
# Dynamic cross-CTA split-reduction candidates (training forward)
# ===================================================================== #
#
# A split candidate is two kernel launches (reduce + apply) sharing one
# caller-owned workspace, so it does not fit ``KernelCandidate``'s
# one-kernel / one-grid shape and lives next to, not inside, the registry
# above. One candidate = one compiled kernel pair; the split geometry
# (batch / spatial split counts) is a runtime launch plan, enumerated per
# request by :meth:`BatchNormSplitCandidate.plans`.

_SPLIT_BLOCK_SIZES = (512, 256, 128)
# f16/bf16 admit vec 8/4, f32 admits vec 4/2 (see ``is_valid_split_dyn_spec``).
_SPLIT_VECS = (8, 4, 2)
_SPLIT_SN_REQ = (1, 2, 4, 8, 16, 32, 64)
_SPLIT_SH_REQ = (1, 2, 4, 8, 16, 32)
# Launch-grid bounds: enough CTAs to fill the device, and a hard ceiling.
_SPLIT_MIN_CTAS = 256
_SPLIT_MAX_CTAS = 16384


@dataclass(frozen=True)
class BatchNormSplitCandidate:
    block_size: int
    vec: int
    fast_reduce: bool
    backward: bool = False

    @property
    def spec_id(self) -> str:
        fr = "_fr" if self.fast_reduce else ""
        bwd = "bwd_" if self.backward else ""
        return f"batchnorm_{bwd}splitdyn_b{self.block_size}_v{self.vec}{fr}"

    @property
    def name(self) -> str:
        return f"batchnorm2d_{self.spec_id}"

    def spec(self, dtype: str, arch: str) -> _bn.BatchNorm2DSplitDynSpec:
        cls = (
            _bn.BatchNorm2DBwdSplitSpec
            if self.backward
            else _bn.BatchNorm2DSplitDynSpec
        )
        return cls(
            block_size=self.block_size,
            vec=self.vec,
            dtype=_bn_dtype(dtype),
            fast_reduce=self.fast_reduce,
            save_mean_invstd=True,
            wave_size=ArchTarget.from_gfx(arch).wave_size,
        )

    def build_reduce(self, spec, _arch: str = "gfx950"):
        if self.backward:
            return _bn.build_batchnorm2d_bwd_split_reduce(spec)
        return _bn.build_batchnorm2d_split_dyn_reduce(spec)

    def build_apply(self, spec, _arch: str = "gfx950"):
        if self.backward:
            return _bn.build_batchnorm2d_bwd_split_apply(spec)
        return _bn.build_batchnorm2d_split_dyn_apply(spec)

    def workspace_floats(self, c: int, plan) -> int:
        if self.backward:
            return _bn.batchnorm2d_bwd_split_workspace_floats(c, plan)
        return _bn.batchnorm2d_split_dyn_workspace_floats(c, plan)

    def plans(self, req: BatchNormRequest) -> Tuple[_bn.BatchNorm2DSplitPlan, ...]:
        """Distinct valid split geometries for ``req`` whose total CTA count
        (``C * splits``) lies in ``[_SPLIT_MIN_CTAS, _SPLIT_MAX_CTAS]``."""
        errors = _request_errors(req)
        if errors or bool(req.backward) != self.backward:
            return ()
        if not self.backward and not req.training:
            return ()
        spec = self.spec(req.dtype, req.arch)
        if not _bn.is_valid_split_dyn_spec(spec, req.arch)[0]:
            return ()
        out = []
        for sn_req in _SPLIT_SN_REQ:
            for sh_req in _SPLIT_SH_REQ:
                plan = _bn.batchnorm2d_split_plan(
                    spec,
                    int(req.n),
                    int(req.h),
                    int(req.w),
                    sn_req=sn_req,
                    sh_req=sh_req,
                )
                if plan is None or plan in out:
                    continue
                ctas = int(req.c) * plan.splits
                if _SPLIT_MIN_CTAS <= ctas <= _SPLIT_MAX_CTAS:
                    out.append(plan)
        return tuple(out)

    def admits(self, req: BatchNormRequest) -> Tuple[bool, str]:
        if not self.plans(req):
            return False, "no valid split plan"
        return True, ""


def _build_split_candidates() -> Tuple[BatchNormSplitCandidate, ...]:
    return tuple(
        BatchNormSplitCandidate(bs, v, fr)
        for bs in _SPLIT_BLOCK_SIZES
        for v in _SPLIT_VECS
        for fr in (False, True)
    )


_DYNAMIC_SPLIT_CANDIDATES = _build_split_candidates()


def dynamic_batchnorm_split_candidates() -> Tuple[BatchNormSplitCandidate, ...]:
    """Every dynamic split-reduction candidate, unfiltered by any request."""
    return _DYNAMIC_SPLIT_CANDIDATES


_DYNAMIC_BWD_SPLIT_CANDIDATES = tuple(
    BatchNormSplitCandidate(bs, v, fr, backward=True)
    for bs in _SPLIT_BLOCK_SIZES
    for v in _SPLIT_VECS
    for fr in (False, True)
)


def dynamic_batchnorm_bwd_split_candidates() -> Tuple[BatchNormSplitCandidate, ...]:
    """Every dynamic split backward candidate (two launches), unfiltered."""
    return _DYNAMIC_BWD_SPLIT_CANDIDATES


# ===================================================================== #
# Dynamic single-launch backward candidates
# ===================================================================== #
#
# One CTA per ``channels_per_block`` channels; ``n_groups`` splits the CTA
# over samples (small H*W).  Opt-in, never auto-selected; requests carry
# ``backward=True``.

_BWD_BLOCK_SIZES = (1024, 512, 256, 128)
_BWD_VECS = (8, 4, 2)
# (channels_per_block, n_groups) variants.
_BWD_VARIANTS = ((1, 1), (1, 2), (1, 4), (1, 8), (2, 1), (4, 1), (8, 1))


def _make_bwd_candidate(
    *,
    block_size: int,
    vec: int,
    channels_per_block: int,
    n_groups: int,
    fast_reduce: bool,
    priority: int,
) -> KernelCandidate:
    cpb_suffix = f"_cpb{channels_per_block}" if channels_per_block != 1 else ""
    ng_suffix = f"_ng{n_groups}" if n_groups != 1 else ""
    fr_suffix = "_fr" if fast_reduce else ""
    spec_id = (
        f"batchnorm_bwd_dyn_b{block_size}_v{vec}{cpb_suffix}{ng_suffix}{fr_suffix}"
    )
    name = f"batchnorm2d_{spec_id}"

    def _spec(req: BatchNormRequest) -> _bn.BatchNorm2DBwdSpec:
        return _bn.BatchNorm2DBwdSpec(
            block_size=block_size,
            vec=vec,
            dtype=_bn_dtype(req.dtype),
            channels_per_block=channels_per_block,
            n_groups=n_groups,
            fast_reduce=fast_reduce,
            wave_size=ArchTarget.from_gfx(req.arch).wave_size,
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, BatchNormRequest)
        if not req.backward:
            return False, "backward kernel needs a backward request"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        return _bn.is_valid_bwd_spec(_spec(req), arch=req.arch)

    def select(req: OperatorRequest) -> _bn.BatchNorm2DBwdSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, BatchNormRequest)
        return _spec(req)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm="batchnorm_bwd",
        spec_id=spec_id,
        abi_version=BATCHNORM_ABI_VERSION,
        priority=priority,
        capability=Capability(
            arches=_BATCHNORM_ARCHES,
            dtypes=("f16", "bf16", "f32"),
            relations=(
                DimRelation(lhs="c", op="multiple_of", rhs=channels_per_block),
                DimRelation(lhs="n", op="multiple_of", rhs=n_groups),
            ),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=lambda spec, req: _bn.batchnorm2d_bwd_grid(int(req.c), spec),
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=lambda spec, _arch: _bn.build_batchnorm2d_bwd_dynamic(spec),
        opt_in=True,
    )
    return candidate


def _build_bwd_candidates() -> Tuple[KernelCandidate, ...]:
    cands = []
    prio = 10
    for var_rank, (cpb, ng) in enumerate(_BWD_VARIANTS):
        for fr in (False, True):
            for bs in _BWD_BLOCK_SIZES:
                for v in _BWD_VECS:
                    cands.append(
                        _make_bwd_candidate(
                            block_size=bs,
                            vec=v,
                            channels_per_block=cpb,
                            n_groups=ng,
                            fast_reduce=fr,
                            priority=prio + var_rank * 1000,
                        )
                    )
                    prio += 1
    return tuple(cands)


_DYNAMIC_BWD_CANDIDATES = _build_bwd_candidates()


def dynamic_batchnorm_bwd_candidates() -> Tuple[KernelCandidate, ...]:
    """Every single-launch dynamic backward candidate, unfiltered by request."""
    return _DYNAMIC_BWD_CANDIDATES


# ===================================================================== #
# NHWC (channels-last) dynamic candidates: infer, training, backward
# ===================================================================== #
#
# Opt-in measurement candidates, outside the NCHW registry (own request
# vocabulary: plain ``n, c, h, w`` ints).  ``variant`` is ``"infer"`` (one
# streaming kernel), ``"fused"`` (single launch, one CTA per channel tile) or
# ``"split"`` (reduce + apply launches).

# Candidate geometries ``(block, tx, vec, unroll)`` per mode, chosen by a
# measured sweep over block {256, 512, 1024} x tx {2..64} x vec {2, 4, 8} x
# unroll {2, 4, 8, 16}; the single-launch fused variant is not part of the
# default set (its builder stays available).
_NHWC_GEOMS = {
    "infer": (
        (256, 4, 8, 2),
        (256, 8, 8, 2),
        (256, 16, 8, 4),
        (256, 32, 8, 2),
        (256, 8, 8, 4),
        (256, 32, 8, 8),
        (256, 64, 4, 2),
        (256, 32, 8, 4),
        (256, 4, 8, 4),
        (256, 8, 8, 8),
        (256, 32, 8, 16),
        (256, 64, 8, 4),
    ),
    "train": (
        (256, 8, 4, 4),
        (1024, 16, 4, 4),
        (1024, 8, 4, 4),
        (1024, 32, 4, 16),
        (512, 8, 8, 16),
        (1024, 32, 2, 4),
        (1024, 16, 2, 4),
        (512, 4, 8, 16),
        (1024, 32, 8, 16),
        (1024, 16, 4, 16),
        (512, 16, 8, 16),
        (1024, 8, 2, 4),
        (1024, 32, 2, 16),
        (512, 32, 2, 16),
    ),
    "bwd": (
        (512, 16, 2, 4),
        (1024, 16, 4, 4),
        (1024, 32, 4, 8),
        (1024, 8, 4, 4),
        (1024, 32, 2, 4),
        (1024, 64, 4, 4),
        (1024, 32, 2, 16),
        (512, 16, 4, 8),
        (1024, 16, 2, 4),
        (512, 16, 8, 8),
        (512, 8, 8, 8),
        (1024, 32, 2, 8),
        (1024, 32, 2, 2),
        (1024, 8, 2, 8),
    ),
}
_NHWC_VARIANTS = {"infer": ("infer",), "train": ("split",), "bwd": ("split",)}
_NHWC_KINDS = {"infer": ("infer",), "fused": ("fused",), "split": ("reduce", "apply")}


@dataclass(frozen=True)
class BatchNormNhwcCandidate:
    mode: str
    variant: str
    block_size: int
    tx: int
    vec: int
    unroll: int = 4

    @property
    def spec_id(self) -> str:
        return (
            f"batchnorm_nhwc_{self.mode}_{self.variant}_b{self.block_size}"
            f"_tx{self.tx}_v{self.vec}_u{self.unroll}"
        )

    @property
    def name(self) -> str:
        return f"batchnorm2d_{self.spec_id}"

    @property
    def kinds(self) -> Tuple[str, ...]:
        return _NHWC_KINDS[self.variant]

    def spec(self, dtype: str, arch: str) -> _bn.BatchNorm2DNhwcSpec:
        return _bn.BatchNorm2DNhwcSpec(
            mode=self.mode,
            block_size=self.block_size,
            tx=self.tx,
            vec=self.vec,
            unroll=self.unroll,
            dtype=_bn_dtype(dtype),
            wave_size=ArchTarget.from_gfx(arch).wave_size,
        )

    def build(self, kind: str, spec=None, arch: str = "gfx950"):
        return _bn.build_batchnorm2d_nhwc(spec or self.spec("fp16", arch), kind)

    def workspace_floats(self, c: int, plan) -> int:
        return (
            _bn.batchnorm2d_nhwc_workspace_floats(c, plan)
            if self.variant == "split"
            else 0
        )

    def plans(
        self, n: int, c: int, h: int, w: int, dtype: str, arch: str
    ) -> Tuple[_bn.BatchNorm2DNhwcPlan, ...]:
        spec = self.spec(dtype, arch)
        if not _bn.is_valid_nhwc_spec(spec, arch)[0]:
            return ()
        return _bn.batchnorm2d_nhwc_plans(spec, self.variant, n, c, h, w)


def dynamic_batchnorm_nhwc_candidates(mode: str) -> Tuple[BatchNormNhwcCandidate, ...]:
    """Every NHWC candidate of ``mode`` (``"infer"`` / ``"train"`` / ``"bwd"``), unfiltered."""
    return tuple(
        BatchNormNhwcCandidate(mode, var, bs, tx, v, u)
        for var in _NHWC_VARIANTS[mode]
        for bs, tx, v, u in _NHWC_GEOMS[mode]
    )


# ===================================================================== #
# Read-once (persistent grid, cross-CTA exchange) candidates
# ===================================================================== #
#
# Opt-in, never auto-selected; admitted only for big tensors where a per-row
# measurement says they win. The launch plan needs the co-resident CTA capacity
# of the compiled kernel (``max_resident``), which only the runtime can query.

from ...instances.common import batchnorm2d_ro as _ro  # noqa: E402

RO_REGS = (2, 4, 8, 16, 32)
RO_BLOCKS = (256, 512)
_RO_SN_REQ = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
_RO_SH_REQ = (1, 2, 4, 8, 16, 32)

# Admission rule of the NCHW / NCDHW read-once plans: f16 / f32 data, an x tensor of at least
# ``RO_MIN_BYTES`` and a per-channel split of ``RO_MIN_SPLIT <= sn * sh <= RO_MAX_SPLIT`` work
# items.  ``rule=False`` on ``plans`` gives the full sweep space.
RO_MIN_BYTES = 8_000_000
RO_MIN_SPLIT = 2
RO_MAX_SPLIT = 16
_RO_ELEM_BYTES = {"fp16": 2, "f16": 2, "fp32": 4, "f32": 4}


@dataclass(frozen=True)
class BatchNormRoCandidate:
    """NCHW / NCDHW read-once training forward or backward."""

    block_size: int
    vec: int
    regs: int
    fast_reduce: bool
    backward: bool = False

    @property
    def name(self) -> str:
        fr = "_fr" if self.fast_reduce else ""
        kind = "bwd" if self.backward else "train"
        return f"batchnorm2d_ro_{kind}_b{self.block_size}_v{self.vec}_r{self.regs}{fr}"

    def spec(self, dtype: str, arch: str = "gfx950") -> _ro.BatchNorm2DRoSpec:
        return _ro.BatchNorm2DRoSpec(
            block_size=self.block_size,
            vec=self.vec,
            regs=self.regs,
            dtype={"fp16": "f16", "fp32": "f32"}.get(dtype, dtype),
            fast_reduce=self.fast_reduce,
        )

    def build(self, spec):
        return (
            _ro.build_batchnorm2d_ro_bwd(spec)
            if self.backward
            else _ro.build_batchnorm2d_ro_train(spec)
        )

    def plans(
        self,
        dtype: str,
        n: int,
        c: int,
        h: int,
        w: int,
        max_resident: int,
        rule: bool = True,
    ):
        if rule and (
            dtype not in _RO_ELEM_BYTES
            or n * c * h * w * _RO_ELEM_BYTES[dtype] < RO_MIN_BYTES
        ):
            return []
        spec = self.spec(dtype)
        if not _ro.is_valid_ro_spec(spec)[0]:
            return []
        out = []
        for sn_req in _RO_SN_REQ:
            for sh_req in _RO_SH_REQ:
                p = _ro.batchnorm2d_ro_plan(
                    spec,
                    n,
                    c,
                    h,
                    w,
                    sn_req=sn_req,
                    sh_req=sh_req,
                    max_resident=max_resident,
                )
                if p is None or p in out:
                    continue
                if rule and not RO_MIN_SPLIT <= p.sn * p.sh <= RO_MAX_SPLIT:
                    continue
                out.append(p)
        return out


def dynamic_batchnorm_ro_candidates(
    backward: bool, dtype: str = "f16"
) -> Tuple[BatchNormRoCandidate, ...]:
    vecs = (4, 2) if dtype in ("f32", "fp32") else (8, 4)
    return tuple(
        BatchNormRoCandidate(bs, v, r, False, backward)
        for bs in RO_BLOCKS
        for v in vecs
        for r in RO_REGS
    )


# ===================================================================== #
# Multi-channel-per-wave small-tensor candidates (NCHW, single launch)
# ===================================================================== #
#
# Opt-in measurement candidates: training forward and backward, one launch,
# the whole tensor of a channel group held in registers.  Shape-dependent
# geometry (samples per wave, slots per lane) is part of the candidate, so a
# candidate admits only the shapes it was sized for.

_MCW_WAVES = (1, 2, 4, 8, 16)
_MCW_CPWS = (1, 2, 4, 8)
_MCW_NPWS = (1, 2, 4, 8, 16, 32, 64, 128)
_MCW_SLOTS = (1, 2, 4, 8, 16)
_MCW_VECS = (1, 2, 4, 8)

# Shipped multi-channel-per-wave set: a greedy cover of the full space, keyed by (dtype, mode).
# Entries are the candidate ``spec_id`` suffixes after ``batchnorm_mcw_<mode>_``.  The full space
# (``dynamic_batchnorm_mcw_candidates``) stays reachable for sweeps.
_MCW_SHIPPED = {
    ("bf16", "bwd"): (
        "w8_cp1_n2_s1_v8",
        "w4_cp1_n8_s1_v8",
    ),
    ("bf16", "train"): (
        "w16_cp1_n1_s1_v8",
        "w4_cp1_n8_s1_v8",
    ),
    ("fp16", "bwd"): (
        "w16_cp1_n8_s1_v4",
        "w4_cp1_n4_s1_v4",
        "w8_cp1_n16_s1",
        "w1_cp1_n8_s1_v4",
        "w8_cp4_n2_s4",
        "w8_cp1_n2_s8",
        "w16_cp2_n8_s2",
        "w4_cp8_n1_s16_v2",
        "w2_cp1_n1_s1_v4",
        "w8_cp2_n16_s1_v2",
        "w8_cp2_n2_s4_v2",
        "w1_cp2_n8_s2",
        "w16_cp4_n8_s1_v4",
        "w1_cp1_n2_s8",
        "w1_cp1_n2_s1_v2",
        "w16_cp1_n1_s1",
        "w2_cp1_n2_s1_v8",
        "w8_cp4_n16_s1",
        "w8_cp4_n2_s2_v8",
        "w8_cp1_n2_s1_v8",
        "w1_cp4_n4_s1",
    ),
    ("fp16", "train"): (
        "w2_cp1_n8_s1_v4",
        "w16_cp1_n8_s1_v4",
        "w8_cp1_n16_s1",
        "w2_cp2_n8_s2",
        "w4_cp1_n4_s8",
        "w2_cp1_n1_s1_v8",
        "w1_cp1_n8_s1_v4",
        "w8_cp2_n16_s1_v2",
        "w4_cp2_n2_s2",
        "w2_cp1_n2_s1_v8",
        "w16_cp4_n1_s4_v4",
        "w16_cp2_n8_s2",
        "w2_cp1_n1_s2_v2",
        "w8_cp2_n16_s1_v8",
        "w4_cp1_n4_s1",
        "w16_cp2_n8_s2_v4",
        "w8_cp4_n2_s2_v8",
        "w4_cp4_n1_s2_v8",
        "w8_cp8_n16_s1_v8",
        "w8_cp4_n16_s1",
        "w1_cp1_n2_s1_v2",
        "w2_cp4_n4_s4_v4",
        "w16_cp2_n8_s1_v8",
        "w4_cp1_n4_s1_v8",
        "w8_cp4_n16_s1_v4",
    ),
    ("fp32", "bwd"): (
        "w8_cp1_n2_s2_v4",
        "w16_cp1_n1_s16",
        "w1_cp2_n2_s16",
        "w1_cp1_n2_s16",
    ),
    ("fp32", "train"): (
        "w8_cp1_n2_s2_v4",
        "w16_cp1_n1_s16",
        "w1_cp4_n2_s8_v4",
        "w1_cp1_n2_s16",
    ),
}


@dataclass(frozen=True)
class BatchNormMcwCandidate:
    mode: str
    waves: int
    cpw: int
    npw: int
    slots: int
    vec: int = 1

    variant = "fused"
    kinds = ("fused",)

    @property
    def spec_id(self) -> str:
        return (
            f"batchnorm_mcw_{self.mode}_w{self.waves}_cp{self.cpw}_n{self.npw}_s{self.slots}"
            + (f"_v{self.vec}" if self.vec != 1 else "")
        )

    @property
    def name(self) -> str:
        return f"batchnorm2d_{self.spec_id}"

    @property
    def block_size(self) -> int:
        return self.waves * 64

    def spec(self, dtype: str, arch: str) -> _bn.BatchNorm2DMcwSpec:
        return _bn.BatchNorm2DMcwSpec(
            mode=self.mode,
            waves=self.waves,
            cpw=self.cpw,
            npw=self.npw,
            slots=self.slots,
            vec=self.vec,
            dtype=_bn_dtype(dtype),
            wave_size=ArchTarget.from_gfx(arch).wave_size,
        )

    def build(self, kind: str, spec=None, arch: str = "gfx950"):
        return _bn.build_batchnorm2d_mcw(spec or self.spec("fp16", arch))

    def admits(self, n: int, c: int, h: int, w: int, dtype: str, arch: str) -> bool:
        spec = self.spec(dtype, arch)
        return (
            _bn.is_valid_mcw_spec(spec, arch)[0] and _bn.mcw_admits(spec, n, c, h, w)[0]
        )

    def shipped(self, dtype: str) -> bool:
        """True when this candidate is part of the shipped set of ``dtype``."""
        prefix = f"batchnorm2d_batchnorm_mcw_{self.mode}_"
        return self.name[len(prefix) :] in _MCW_SHIPPED.get((dtype, self.mode), ())


def dynamic_batchnorm_mcw_candidates(mode: str) -> Tuple[BatchNormMcwCandidate, ...]:
    """Every multi-channel-per-wave candidate of ``mode`` (``"train"`` / ``"bwd"``) within the register budget."""
    out = []
    for w in _MCW_WAVES:
        for cpw in _MCW_CPWS:
            for npw in _MCW_NPWS:
                for s in _MCW_SLOTS:
                    for v in _MCW_VECS:
                        cand = BatchNormMcwCandidate(mode, w, cpw, npw, s, v)
                        if _bn.is_valid_mcw_spec(cand.spec("fp16", "gfx950"))[0]:
                            out.append(cand)
    return tuple(out)


def shipped_batchnorm_mcw_candidates(
    mode: str, dtype: str
) -> Tuple[BatchNormMcwCandidate, ...]:
    """The shipped multi-channel-per-wave candidates of ``mode`` / ``dtype``."""
    return tuple(c for c in dynamic_batchnorm_mcw_candidates(mode) if c.shipped(dtype))
