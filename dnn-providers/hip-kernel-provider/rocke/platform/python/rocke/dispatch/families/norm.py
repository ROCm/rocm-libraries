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

These kernels are arch agnostic (they run on both CDNA and RDNA, at both wave
sizes), so every candidate declares every known target and the narrowing is done
by the per-arch LDS / thread checks the instance validators already encode.

Selection: a request carries ``(rows, cols)``. ``cols`` is the per-row width
``N``; ``n_per_block`` is fixed to ``cols`` (one CTA spans the whole row). The
candidate set is the cross product of block_size and vec; ``auto`` picks the
highest-priority candidate whose ``is_valid_spec`` accepts the request on the
target arch.

Priority order (see :data:`_VEC_PRIORITY` / :data:`_BLOCK_SIZE_PRIORITY`):
widest ``vec`` first, then ``block_size`` in sweet-spot order
``256, 128, 512, 64, 1024``. This replaced a "largest block_size first, then
widest vec" rule that measured worse on every one of 8 representative
shapes checked (both kinds, both dtypes, rows/cols from 256x256 to
4096x8192): a full-occupancy CTA (``block_size=1024``) consistently loses to
a narrower one, because the extra LDS a bigger block's Welford buffers need
lowers the number of CTAs resident per CU. ``block_size=256``/``128`` won or
tied-within-noise everywhere tested; the previous rule's own pick
(``block_size=1024`` or the largest legal size) was never the fastest
candidate measured and was up to ~1.48x slower (rows=2048, cols=1024, f16:
``b512_v2`` vs ``b64_v8``). ``block_size=64`` alone is not a safe
replacement either -- it was ~7.7% slower than ``b256_v8`` at
rows=4096, cols=8192 -- hence the explicit sweet-spot order rather than
"smallest first".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence, Tuple

from ...core.arch import ArchTarget, arches_with_own_facts
from ...instances.common import layernorm2d as _ln
from ...instances.common import layernorm2d_bwd as _lnb
from ...instances.common import rmsnorm2d as _rms
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

_FAMILY = "norm2d"
NORM_ABI_VERSION = "hipkg-norm2d/v1"

# Allowed knob grids, mirroring IOSpecRule's allowed_block_sizes / allowed_vecs.
_BLOCK_SIZES = (1024, 512, 256, 128, 64)
_VECS = (8, 4, 2)
_KINDS = ("rmsnorm", "layernorm")

# ``fast_reduce`` candidate axis (layernorm only -- rmsnorm has no such spec
# field): False (default, the original flat block_size-wide LDS Welford
# tree) and True (wave-shuffle-first Welford merge, fewer barriers; see
# ``instances/common/layernorm2d.py``'s ``welford_block_reduce_stable_wave_prologue``
# usage). ``False`` is always the highest-priority value (see
# ``_build_registry``/``_build_dynamic_candidates``) -- a default-OFF lever,
# exposed the same way ``dispatch/families/batchnorm.py``'s ``_FAST_REDUCE``
# is, for AOT-sweep measurement; never auto-selected in production over the
# ``False`` candidate unless a caller pins its ``spec_id`` explicitly.
_FAST_REDUCE = (False, True)

# ``cheap_reduce`` candidate axis (layernorm only, same shape as
# ``_FAST_REDUCE``): False (default, numerically-stable Welford) and True
# (MIOpen-style naive ``E[X^2] - E[X]^2`` two-pass reduction -- see
# ``LayerNorm2DSpec``'s docstring for the stability trade-off). Mutually
# exclusive with ``fast_reduce`` (``is_valid_spec`` rejects both set); the
# candidate loop below skips that combination rather than registering an
# always-invalid candidate. ``False`` is always the highest-priority value,
# same default-OFF-lever contract as ``_FAST_REDUCE``.
_CHEAP_REDUCE = (False, True)

# Static selection priority: lower wins. Vec dominates (wider vector loads
# consistently measured faster or tied); within a vec tier, block_size is
# ordered by measured sweet spot rather than monotonically by size -- see the
# module docstring for the measurements this replaced.
_VEC_PRIORITY = {8: 0, 4: 1, 2: 2}
_BLOCK_SIZE_PRIORITY = {256: 0, 128: 1, 512: 2, 64: 3, 1024: 4}


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

    def dims(self) -> dict[str, int]:
        return {"rows": int(self.rows), "cols": int(self.cols)}


NORM_DIM_VOCABULARY = ("rows", "cols")

# Norm kernels are genuinely portable: they derive ``wave_size`` from the target
# and use an LDS-tree reduction rather than an MMA atom, so one candidate serves
# both wave sizes. This is the documented exception to the wave-size consistency
# invariant, which exists to catch MMA kernels that bake a wave size into their
# geometry. The per-arch LDS / max-threads checks in the instance validators are
# what actually narrow this list per request. Each generic target stands for the
# processors whose rows point at it (``listed_arch_for``), so listing them again
# would only build the generic's kernels once more per member.
_NORM_ARCHES = arches_with_own_facts()


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


def _make_spec(
    req: NormRequest, *, block_size: int, vec: int,
    fast_reduce: bool = False, cheap_reduce: bool = False,
):
    """Build the instance spec for one (kind, block_size, vec) candidate.

    ``fast_reduce``/``cheap_reduce`` only apply to ``layernorm``
    (``RMSNorm2DSpec`` has no such fields); callers must not pass them for
    ``kind="rmsnorm"``.
    """
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
        fast_reduce=fast_reduce,
        cheap_reduce=cheap_reduce,
    )


def _is_valid(req: NormRequest, spec) -> Tuple[bool, str]:
    if req.kind.lower() == "rmsnorm":
        return _rms.is_valid_spec(spec, arch=req.arch)
    return _ln.is_valid_spec(spec, arch=req.arch)


def _make_candidate(
    *, kind: str, block_size: int, vec: int, priority: int,
    fast_reduce: bool = False, cheap_reduce: bool = False,
) -> KernelCandidate:
    fr_suffix = "_fr" if fast_reduce else ""
    cr_suffix = "_cr" if cheap_reduce else ""
    spec_id = f"{kind}_b{block_size}_v{vec}{fr_suffix}{cr_suffix}"
    name = f"norm2d_{spec_id}"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, NormRequest)
        if req.kind.lower() != kind:
            return False, f"candidate kind {kind!r} != request kind {req.kind!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        spec = _make_spec(
            req, block_size=block_size, vec=vec,
            fast_reduce=fast_reduce, cheap_reduce=cheap_reduce,
        )
        return _is_valid(req, spec)

    def select(req: OperatorRequest):
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, NormRequest)
        return _make_spec(
            req, block_size=block_size, vec=vec,
            fast_reduce=fast_reduce, cheap_reduce=cheap_reduce,
        )

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=kind,
        spec_id=spec_id,
        abi_version=NORM_ABI_VERSION,
        priority=priority,
        capability=Capability(arches=_NORM_ARCHES, dtypes=("f16", "bf16")),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=_build,
    )
    return candidate


def _build(spec, _arch: str):
    """Both norm builders take only a spec.

    Norm derives its wave size from the spec rather than the target, which is
    the same property that lets one candidate declare every arch (section 10),
    so there is no arch to pass down.
    """
    if isinstance(spec, _rms.RMSNorm2DSpec):
        return _rms.build_rmsnorm2d(spec)
    return _ln.build_layernorm2d(spec)


def _grid(spec, req: OperatorRequest) -> Tuple[int, int, int]:
    # One CTA per row.
    assert isinstance(req, NormRequest)
    return (int(req.rows), 1, 1)


def _candidate_priority(*, block_size: int, vec: int) -> int:
    """Lower wins. Vec tier dominates; see module docstring for the
    measurements behind this order and :data:`_VEC_PRIORITY` /
    :data:`_BLOCK_SIZE_PRIORITY` for the tables."""
    return _VEC_PRIORITY[vec] * 10 + _BLOCK_SIZE_PRIORITY[block_size]


def _build_registry() -> CandidateRegistry:
    reg = CandidateRegistry(
        _FAMILY, dim_vocabulary=NORM_DIM_VOCABULARY, require_build=True
    )
    cands = []
    for bs in _BLOCK_SIZES:
        for v in _VECS:
            prio = _candidate_priority(block_size=bs, vec=v)
            for kind in _KINDS:
                if kind == "layernorm":
                    # fast_reduce / cheap_reduce are layernorm-only,
                    # default-OFF levers (see their docstrings): (False,
                    # False) keeps exactly the pre-existing
                    # priority/behavior, every other combination is
                    # appended at strictly lower priority so it never wins
                    # "auto", but is still reachable by the AOT sweep
                    # (registered_norm_combos/sweep_space) and by a caller
                    # pinning its spec_id. fast_reduce=True and
                    # cheap_reduce=True together selects the wave-prologue
                    # pair reduction (block_lds_reduce_pair_with_wave_prologue,
                    # see LayerNorm2DSpec's docstring) -- valid since that
                    # helper now exists, so every combination is registered.
                    for fr_rank, fr in enumerate(_FAST_REDUCE):
                        for cr_rank, cr in enumerate(_CHEAP_REDUCE):
                            cands.append(
                                _make_candidate(
                                    kind=kind, block_size=bs, vec=v,
                                    priority=prio + fr_rank * 10000 + cr_rank * 20000,
                                    fast_reduce=fr, cheap_reduce=cr,
                                )
                            )
                else:
                    cands.append(
                        _make_candidate(kind=kind, block_size=bs, vec=v, priority=prio)
                    )
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


def registered_norm_combos(
    req: OperatorRequest,
) -> Tuple[Tuple[KernelCandidate, object], ...]:
    """Every registered norm candidate that can launch ``req``.

    Probes opt-in variants and expands each candidate's ``sweep_space``.
    Production :func:`dispatch_norm` is unchanged.
    """
    if _request_errors(req):
        return ()
    return NORM_REGISTRY.combos(req)


def norm_sweep_space(req: OperatorRequest) -> Sequence[object]:
    if _request_errors(req):
        return ()
    return NORM_REGISTRY.sweep_space(req)


def dispatch_norm_all(req: NormRequest) -> Tuple[DispatchResult, ...]:
    """Every eligible norm2d kernel for ``req``, including opt-in variants."""
    if _request_errors(req):
        return ()
    return NORM_REGISTRY.dispatch_all(req, kernel_id=_kernel_id)


# ===================================================================== #
# Dynamic (AOT-generic) candidates -- capability-admission scaffolding
# ===================================================================== #
#
# Same pattern as dispatch/families/batchnorm.py's dynamic-candidate
# section (see its module comment for the full rationale): layernorm2d's
# ``dynamic=True`` kernels (LayerNorm2DSpec.dynamic) make the row width a
# runtime kernarg instead of a compile-time shape, so one compiled binary
# admits any ``cols`` -- no DimRelation needed here (unlike batchnorm's
# channels_per_block/n_groups), since the dynamic layernorm2d kernel body
# enforces no capacity ceiling on ``cols`` at all (see
# _build_layernorm2d_dynamic's docstring). ``rmsnorm`` has no dynamic
# variant yet (RMSNorm2DSpec has no ``dynamic`` field), so only
# ``kind="layernorm"`` dynamic candidates exist.
#
# Deliberately NOT wired into NORM_REGISTRY / dispatch_norm:
# dynamic_norm_candidates_for() is gated by ENABLE_DYNAMIC_NORM_DISPATCH
# (False) and returns () unconditionally while that flag is off, so
# production dispatch is completely unaffected.

ENABLE_DYNAMIC_NORM_DISPATCH = False


def _make_dynamic_candidate(
    *, block_size: int, vec: int, priority: int,
    fast_reduce: bool = False, cheap_reduce: bool = False,
) -> KernelCandidate:
    fr_suffix = "_fr" if fast_reduce else ""
    cr_suffix = "_cr" if cheap_reduce else ""
    spec_id = f"layernorm_dyn_b{block_size}_v{vec}{fr_suffix}{cr_suffix}"
    name = f"norm2d_{spec_id}"

    def _placeholder_spec(req: NormRequest) -> _ln.LayerNorm2DSpec:
        # n_per_block is a capacity placeholder: the dynamic kernel body
        # never enforces it as a ceiling on the real runtime ``cols`` (see
        # _build_layernorm2d_dynamic's docstring), so any positive multiple
        # of block_size*vec here is fine -- block_size*vec itself is the
        # smallest legal value.
        dtype = _norm_dtype(req.dtype)
        wave = ArchTarget.from_gfx(req.arch).wave_size
        return _ln.LayerNorm2DSpec(
            n_per_block=block_size * vec,
            block_size=block_size,
            vec=vec,
            dtype=dtype,
            wave_size=wave,
            dynamic=True,
            fast_reduce=fast_reduce,
            cheap_reduce=cheap_reduce,
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, NormRequest)
        if req.kind.lower() != "layernorm":
            return False, f"candidate kind 'layernorm' != request kind {req.kind!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        spec = _placeholder_spec(req)
        return _ln.is_valid_spec(spec, arch=req.arch)

    def select(req: OperatorRequest) -> _ln.LayerNorm2DSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, NormRequest)
        return _placeholder_spec(req)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm="layernorm",
        spec_id=spec_id,
        abi_version=NORM_ABI_VERSION,
        priority=priority,
        capability=Capability(arches=_NORM_ARCHES, dtypes=("f16", "bf16")),
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
    for bs in _BLOCK_SIZES:
        for v in _VECS:
            prio = _candidate_priority(block_size=bs, vec=v)
            for fr_rank, fr in enumerate(_FAST_REDUCE):
                for cr_rank, cr in enumerate(_CHEAP_REDUCE):
                    cands.append(
                        _make_dynamic_candidate(
                            block_size=bs, vec=v,
                            priority=prio + fr_rank * 10000 + cr_rank * 20000,
                            fast_reduce=fr, cheap_reduce=cr,
                        )
                    )
    return tuple(cands)


_DYNAMIC_NORM_CANDIDATES = _build_dynamic_candidates()


def dynamic_norm_candidates() -> Tuple[KernelCandidate, ...]:
    """Every registered dynamic (AOT-generic) layernorm2d candidate,
    unfiltered by any request -- for introspection and direct testing of
    the admission machinery regardless of :data:`ENABLE_DYNAMIC_NORM_DISPATCH`.
    """
    return _DYNAMIC_NORM_CANDIDATES


def dynamic_norm_candidates_for(req: NormRequest) -> Tuple[KernelCandidate, ...]:
    """Dynamic candidates admitting ``req``, gated by
    :data:`ENABLE_DYNAMIC_NORM_DISPATCH`.

    Returns ``()`` unconditionally while that flag is ``False`` (its
    default) -- see :func:`rocke.dispatch.families.batchnorm.dynamic_batchnorm_candidates_for`
    for the identical rationale.
    """
    if not ENABLE_DYNAMIC_NORM_DISPATCH:
        return ()
    if _request_errors(req):
        return ()
    return tuple(c for c in _DYNAMIC_NORM_CANDIDATES if c.admits(req)[0])


# ===================================================================== #
# LayerNorm2D backward (AOT-generic) candidates
# ===================================================================== #
#
# instances/common/layernorm2d_bwd.py is three shape-free kernels (``dx``,
# ``dwdb``, ``dwdb_reduce``) that together make one MIOpen-style backward
# (``MIOpenDriver layernorm* -F 0``). Each part is tuned independently --
# the parts run back to back and share no on-chip state -- so each gets its
# own candidate list over its own knobs, keyed by ``LayerNorm2DBwdSpec.part``.
# Like the forward's dynamic candidates these are opt-in (never selected by
# dispatch_norm) and live outside NORM_REGISTRY; the AOT sweep queries them
# directly through dynamic_norm_bwd_candidates().
#
# ``dwdb`` / ``dwdb_reduce`` vector-access ``vec`` columns, so they admit a
# request only when ``cols % vec == 0``; ``dx`` has a masked tail and admits
# any ``cols``. ``grid`` reports the ``P == 1`` (no workspace) launch; the
# host picks the runtime row-split ``P`` (see
# layernorm2d_bwd.default_row_splits) and launches ``dwdb`` on
# ``(grid_x, P, 1)`` plus ``dwdb_reduce`` when ``P > 1``.

_BWD_UNROLLS = _lnb.UNROLLS


def _make_bwd_candidate(
    *, part: str, block_size: int, vec: int, priority: int,
    fast_reduce: bool = False, unroll: int = 4,
) -> KernelCandidate:
    u_suffix = f"_u{unroll}" if part != "dx" else ""
    fr_suffix = "_fr" if fast_reduce else ""
    spec_id = f"layernorm_bwd_{part}_b{block_size}_v{vec}{u_suffix}{fr_suffix}"
    name = f"norm2d_{spec_id}"

    def _spec(req: NormRequest) -> _lnb.LayerNorm2DBwdSpec:
        return _lnb.LayerNorm2DBwdSpec(
            part=part,
            block_size=block_size,
            vec=vec,
            dtype=_norm_dtype(req.dtype),
            wave_size=ArchTarget.from_gfx(req.arch).wave_size,
            fast_reduce=fast_reduce,
            unroll=unroll,
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, NormRequest)
        if req.kind.lower() != "layernorm":
            return False, f"candidate kind 'layernorm' != request kind {req.kind!r}"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        if part != "dx" and int(req.cols) % vec != 0:
            return False, f"{part} needs cols % vec == 0 (cols={req.cols}, vec={vec})"
        return _lnb.is_valid_spec(_spec(req), arch=req.arch)

    def select(req: OperatorRequest) -> _lnb.LayerNorm2DBwdSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, NormRequest)
        return _spec(req)

    def grid(spec, req: OperatorRequest) -> Tuple[int, int, int]:
        assert isinstance(req, NormRequest)
        return _lnb.layernorm2d_bwd_grid(spec, rows=int(req.rows), cols=int(req.cols))

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm="layernorm_bwd",
        spec_id=spec_id,
        abi_version=NORM_ABI_VERSION,
        priority=priority,
        capability=Capability(arches=_NORM_ARCHES, dtypes=("f16", "bf16")),
        _supports=support,
        select_spec=select,
        signature=lambda spec: _lnb.layernorm2d_bwd_signature(spec),
        grid=grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=lambda spec, _arch: _lnb.build_layernorm2d_bwd(spec),
        opt_in=True,
    )
    return candidate


def _build_bwd_candidates() -> Tuple[KernelCandidate, ...]:
    cands = []
    for bs in _BLOCK_SIZES:
        for v in _VECS:
            prio = _candidate_priority(block_size=bs, vec=v)
            for fr_rank, fr in enumerate(_FAST_REDUCE):
                cands.append(
                    _make_bwd_candidate(
                        part="dx", block_size=bs, vec=v,
                        priority=prio + fr_rank * 10000, fast_reduce=fr,
                    )
                )
            for part in ("dwdb", "dwdb_reduce"):
                for u_rank, u in enumerate(_BWD_UNROLLS):
                    cands.append(
                        _make_bwd_candidate(
                            part=part, block_size=bs, vec=v,
                            priority=prio + u_rank * 10000, unroll=u,
                        )
                    )
    return tuple(cands)


_DYNAMIC_NORM_BWD_CANDIDATES = _build_bwd_candidates()


def dynamic_norm_bwd_candidates(part: str | None = None) -> Tuple[KernelCandidate, ...]:
    """Every LayerNorm2D backward candidate (optionally only one ``part``),
    unfiltered by any request."""
    if part is None:
        return _DYNAMIC_NORM_BWD_CANDIDATES
    return tuple(
        c for c in _DYNAMIC_NORM_BWD_CANDIDATES if c.spec_id.startswith(f"layernorm_bwd_{part}_b")
    )


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
