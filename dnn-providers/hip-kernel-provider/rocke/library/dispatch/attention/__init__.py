# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Attention / FMHA dispatcher family (path-level selection).

Worked implementation mirroring :mod:`rocke.dispatch.gemm.bf16_rcr`, backed by
:mod:`kernels.common.attention_unified` (the unified tiled FMHA emitter).

This module owns only the assembly: the registry, the entry points, and the
re-exports that make ``dispatch.attention`` one import for callers. What each
candidate *is* lives in the arch module that owns it. See :mod:`.common` for
the scope of the dispatch decision and what it deliberately defers.

Registration is explicit rather than an import side effect, so the registry
contents are a readable list, a test can assemble a registry from a subset of
arch modules, and adding an arch touches exactly one line here.
"""

from __future__ import annotations

from typing import Iterator, Sequence, Tuple

from rocke.core.arch import ArchTarget
from rocke.dispatch.core import (
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    make_kernel_id,
    spec_identity,
    stable_json_hash,
)

from . import generic, gfx942, gfx942_tuning, gfx950, gfx950_tuning, gfx1250
from .common import (
    ATTENTION_ABI_VERSION,
    ATTENTION_DIM_VOCABULARY,
    ATTENTION_FEATURES,
    UNIFIED_BLOCK_SIZES,
    UNIFIED_HEAD_SIZES,
    AttentionMaskType,
    AttentionRequest,
    AttentionSpec,
    AttentionTuningSpec,
    FAMILY,
    _device_num_cus,
    _problem,
    _request_errors,
    _resolve_num_cus,
    _selector_matches,
)

_FAMILY = FAMILY

ATTENTION_ROUTE_REGISTRY = CandidateRegistry(
    _FAMILY, dim_vocabulary=ATTENTION_DIM_VOCABULARY
)
ATTENTION_EXECUTION_REGISTRY = CandidateRegistry(
    _FAMILY,
    dim_vocabulary=ATTENTION_DIM_VOCABULARY,
    require_build=True,
    require_torch_binding=True,
)
generic.register(ATTENTION_ROUTE_REGISTRY)
gfx942.register_route(ATTENTION_ROUTE_REGISTRY)
gfx950.register_route(ATTENTION_ROUTE_REGISTRY)
gfx1250.register(ATTENTION_ROUTE_REGISTRY)
gfx942_tuning.register(ATTENTION_ROUTE_REGISTRY)
gfx950_tuning.register(ATTENTION_ROUTE_REGISTRY)
gfx942.register_execution(ATTENTION_EXECUTION_REGISTRY)
gfx950.register_execution(ATTENTION_EXECUTION_REGISTRY)
gfx1250.register(ATTENTION_EXECUTION_REGISTRY)
gfx942_tuning.register(ATTENTION_EXECUTION_REGISTRY)
gfx950_tuning.register(ATTENTION_EXECUTION_REGISTRY)
# Compatibility alias: production auto-dispatch and candidate listing.
ATTENTION_REGISTRY = ATTENTION_ROUTE_REGISTRY


def attention_candidates() -> Tuple[KernelCandidate, ...]:
    return ATTENTION_REGISTRY.candidates()


def attention_execution_candidates() -> Tuple[KernelCandidate, ...]:
    return ATTENTION_EXECUTION_REGISTRY.candidates()


def _attention_selector_ok(req: OperatorRequest, candidate: KernelCandidate) -> bool:
    """Default algorithm/spec_id filter plus the gfx950 dense family-id alias."""
    assert isinstance(req, AttentionRequest)
    wanted = req.algorithm.strip().lower()
    sid = req.spec_id.strip().lower()
    family = gfx950.GFX950_DENSE_FAMILY_SPEC_ID
    if wanted not in ("auto", candidate.algorithm):
        return False
    if sid not in ("auto", candidate.spec_id) and not (
        sid == family
        and candidate.algorithm == "attention_dense"
        and req.arch == "gfx950"
    ):
        return False
    return True


def iter_registered_attention_combos(
    req: AttentionRequest,
    *,
    candidate_prefix: str = "",
    tuning_id_prefix: str = "",
) -> Iterator[Tuple[KernelCandidate, object]]:
    """Yield each executable ``(candidate, spec)`` that can launch ``req``.

    Delegates the opt-in probe and ``sweep_space`` expansion to
    :meth:`CandidateRegistry.iter_combos` on :data:`ATTENTION_EXECUTION_REGISTRY`.
    Routing-only unified path labels are not executable and are omitted.
    ``req.algorithm`` still filters when it is not ``auto``. ``spec_id``
    likewise, except the gfx950 dense family id admits every dense variant.
    """
    if not isinstance(req, AttentionRequest):
        raise TypeError(f"expected AttentionRequest, got {type(req).__name__}")
    for candidate, spec in ATTENTION_EXECUTION_REGISTRY.iter_combos(
        req,
        candidate_prefix=candidate_prefix,
        selector_ok=_attention_selector_ok,
    ):
        if (
            candidate.algorithm == "unified_tuning"
            and tuning_id_prefix
            and not getattr(spec, "tuning_id", "").startswith(tuning_id_prefix)
        ):
            continue
        yield candidate, spec


def registered_attention_combos(
    req: AttentionRequest,
    *,
    candidate_prefix: str = "",
    tuning_id_prefix: str = "",
) -> Tuple[Tuple[KernelCandidate, object], ...]:
    """Materialized :func:`iter_registered_attention_combos`."""
    return tuple(
        iter_registered_attention_combos(
            req,
            candidate_prefix=candidate_prefix,
            tuning_id_prefix=tuning_id_prefix,
        )
    )


def attention_dispatch_result(
    req: AttentionRequest, candidate: KernelCandidate, spec: object
) -> DispatchResult:
    """Wrap an already-selected executable ``(candidate, spec)`` as a result."""
    return DispatchResult(
        request=req,
        candidate=candidate,
        spec=spec,
        kernel_id=_kernel_id(req, candidate, spec),
        grid=candidate.grid(spec, req),
        block=candidate.block(spec),
        signature=tuple(candidate.signature(spec)),
        explanation=(
            f"sweep {candidate.name} ({candidate.algorithm}) on {req.arch}",
            f"algorithm={candidate.algorithm}",
            f"spec_id={candidate.spec_id}",
        ),
    )


def iter_dispatch_attention_all(
    req: AttentionRequest,
    *,
    candidate_prefix: str = "",
    tuning_id_prefix: str = "",
) -> Iterator[DispatchResult]:
    """Yield each :func:`attention_dispatch_result` for ``req``."""
    if _request_errors(req):
        return
    for candidate, spec in iter_registered_attention_combos(
        req,
        candidate_prefix=candidate_prefix,
        tuning_id_prefix=tuning_id_prefix,
    ):
        yield attention_dispatch_result(req, candidate, spec)


def dense_spec_for_request(req: AttentionRequest):
    """Return the concrete dense spec selected by an explicit ``req.arch``."""
    if not isinstance(req, AttentionRequest):
        raise TypeError(f"expected AttentionRequest, got {type(req).__name__}")
    arch = req.arch.strip() if isinstance(req.arch, str) else ""
    if not arch:
        raise ValueError("attention dense dispatch requires an explicit arch")
    try:
        gfx = ArchTarget.from_gfx(arch).gfx
    except KeyError as exc:
        raise ValueError(f"unsupported attention dense arch {arch!r}") from exc

    factories = {
        "gfx942": gfx942.dense_spec_for_request,
        "gfx950": gfx950.dense_spec_for_request,
    }
    try:
        factory = factories[gfx]
    except KeyError as exc:
        raise ValueError(
            f"attention dense has no spec factory for arch {gfx!r}"
        ) from exc
    return factory(req)


def _kernel_id(
    req: AttentionRequest, candidate: KernelCandidate, spec: object
) -> KernelId:
    return make_kernel_id(req, candidate, spec, op="attention")


def attention_sweep_space(
    req: OperatorRequest,
    *,
    candidate_prefix: str = "",
    tuning_id_prefix: str = "",
) -> Sequence[object]:
    """Every concrete engine/configuration available to a sweep.

    Unlike normal dispatch, this deliberately probes opt-in executable
    candidates and expands ``candidate.sweep_space``. Production
    ``algorithm='auto'`` selection remains on ``ATTENTION_ROUTE_REGISTRY.supported``
    and never sees tuning candidates.
    """
    if _request_errors(req):
        return ()
    assert isinstance(req, AttentionRequest)
    specs = []
    seen = set()
    for _candidate, spec in registered_attention_combos(
        req,
        candidate_prefix=candidate_prefix,
        tuning_id_prefix=tuning_id_prefix,
    ):
        # This API feeds the unified paged-attention harness, whose input ABI
        # requires a path-bearing 2D/3D spec. Dense and WMMA remain available
        # through registered_attention_combos / dispatch_attention_all.
        if not hasattr(spec, "path"):
            continue
        h = spec_identity(spec)
        if h not in seen:
            seen.add(h)
            specs.append(spec)
    return tuple(specs)


def dispatch_attention_all(
    req: AttentionRequest,
    *,
    candidate_prefix: str = "",
    tuning_id_prefix: str = "",
) -> Tuple[DispatchResult, ...]:
    """Every eligible attention kernel for ``req``, including opt-in variants.

    Uses :func:`registered_attention_combos` so dense candidates keep their
    standalone dense spec. Production :func:`dispatch_attention` is unchanged.
    """
    return tuple(
        iter_dispatch_attention_all(
            req,
            candidate_prefix=candidate_prefix,
            tuning_id_prefix=tuning_id_prefix,
        )
    )


def priority_ranker(
    request: OperatorRequest, candidates: Sequence[KernelCandidate]
) -> Sequence[KernelCandidate]:
    """Honor registered ``(priority, name)`` order (identity over ``supported``).

    ``dispatch_attention`` defaults to :func:`attention_ranker` so gfx950 dense
    variants follow the historical auto policy. Pass this ranker explicitly
    to ignore that policy and take the first registered name.
    """
    return candidates


def attention_ranker(
    request: OperatorRequest, candidates: Sequence[KernelCandidate]
) -> Sequence[KernelCandidate]:
    """Default engine-level ranker.

    gfx950 dense variants cannot be ranked by ``(priority, name)`` alone: the
    production name is persist+wide-DMA, which is wrong for short seq / D=64 /
    SWA. Prefer the auto-policy variant; every other candidate keeps registered
    ``(priority, name)`` order (same as :func:`priority_ranker` on gfx942).
    """
    return gfx950.rank_dense_variants(request, candidates)


def dispatch_attention(
    req: AttentionRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select the unified attention kernel PATH for ``req``.

    Returns the 2D-tiled or 3D split-KV path (a pure function of the problem),
    gated by the native-backend coverage predicate. The CTA geometry is left to
    the instance builder (see :mod:`.common` -- deferred from parity).

    ``ranker`` is the engine-level selection seam; when omitted,
    :func:`attention_ranker` is used so gfx950 dense auto-policy is preserved.
    """
    candidate = ATTENTION_REGISTRY.select(req, ranker=ranker or attention_ranker)
    spec = candidate.select_spec(req)
    kid = _kernel_id(req, candidate, spec)
    # Standalone kernels (gfx1250 WMMA) return their builder's spec, which has
    # no `path` -- selecting one *is* the decision, with nothing left to route.
    path = getattr(spec, "path", "")
    selected = f"{path} path" if path else candidate.algorithm
    return DispatchResult(
        request=req,
        candidate=candidate,
        spec=spec,
        kernel_id=kid,
        grid=candidate.grid(spec, req),
        block=candidate.block(spec),
        signature=tuple(candidate.signature(spec)),
        explanation=(
            f"selected {candidate.name} ({selected}) on {req.arch}",
            f"algorithm={candidate.algorithm}",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )


__all__ = [
    "ATTENTION_ABI_VERSION",
    "ATTENTION_DIM_VOCABULARY",
    "ATTENTION_EXECUTION_REGISTRY",
    "ATTENTION_FEATURES",
    "ATTENTION_REGISTRY",
    "ATTENTION_ROUTE_REGISTRY",
    "UNIFIED_BLOCK_SIZES",
    "UNIFIED_HEAD_SIZES",
    "AttentionMaskType",
    "AttentionRequest",
    "AttentionSpec",
    "AttentionTuningSpec",
    "attention_candidates",
    "attention_execution_candidates",
    "attention_ranker",
    "attention_dispatch_result",
    "attention_sweep_space",
    "dense_spec_for_request",
    "dispatch_attention",
    "dispatch_attention_all",
    "iter_dispatch_attention_all",
    "iter_registered_attention_combos",
    "priority_ranker",
    "registered_attention_combos",
]
