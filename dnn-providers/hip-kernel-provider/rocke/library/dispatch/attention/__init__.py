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

from dataclasses import asdict
from typing import Sequence, Tuple

from rocke.dispatch.core import (
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    stable_json_hash,
)

from . import generic, gfx942, gfx950, gfx1250
from .common import (
    ATTENTION_ABI_VERSION,
    ATTENTION_DIM_VOCABULARY,
    ATTENTION_FEATURES,
    UNIFIED_BLOCK_SIZES,
    UNIFIED_HEAD_SIZES,
    AttentionRequest,
    AttentionSpec,
    FAMILY,
    _device_num_cus,
    _problem,
    _request_errors,
    _resolve_num_cus,
    _selector_matches,
)
from .gfx950 import dense_spec_for_request

_FAMILY = FAMILY

ATTENTION_REGISTRY = CandidateRegistry(_FAMILY, dim_vocabulary=ATTENTION_DIM_VOCABULARY)
for _module in (generic, gfx942, gfx950, gfx1250):
    _module.register(ATTENTION_REGISTRY)


def attention_candidates() -> Tuple[KernelCandidate, ...]:
    return ATTENTION_REGISTRY.candidates()


def _kernel_id(
    req: AttentionRequest, candidate: KernelCandidate, spec: AttentionSpec
) -> KernelId:
    request_hash = stable_json_hash(req.normalized(), n=16)
    spec_hash = stable_json_hash(asdict(spec), n=16)
    return KernelId(
        op="attention",
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=request_hash,
        spec_hash=spec_hash,
    )


def attention_sweep_space(req: OperatorRequest) -> Sequence[AttentionSpec]:
    if _request_errors(req):
        return ()
    specs = []
    seen = set()
    for candidate in ATTENTION_REGISTRY.supported(req):
        spec = candidate.select_spec(req)
        h = stable_json_hash(asdict(spec), n=16)
        if h not in seen:
            seen.add(h)
            specs.append(spec)
    return tuple(specs)


def priority_ranker(
    request: OperatorRequest, candidates: Sequence[KernelCandidate]
) -> Sequence[KernelCandidate]:
    """Default engine-level ranker: honor registered ``(priority, name)`` order.

    ``CandidateRegistry.supported`` already returns candidates sorted ascending by
    ``(priority, name)``, so this is an identity pass -- the explicit default that
    ``dispatch_attention`` applies when no ranker is supplied. It exists as a named
    seam: a heuristic ranker (engine-level selection driven by problem metadata)
    is a drop-in replacement that reorders these same candidates best-first.
    """
    return candidates


def dispatch_attention(
    req: AttentionRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select the unified attention kernel PATH for ``req``.

    Returns the 2D-tiled or 3D split-KV path (a pure function of the problem),
    gated by the native-backend coverage predicate. The CTA geometry is left to
    the instance builder (see :mod:`.common` -- deferred from parity).

    ``ranker`` is the engine-level selection seam; when omitted, the registered
    priority order is used via :func:`priority_ranker` (behavior-preserving).
    """
    candidate = ATTENTION_REGISTRY.select(req, ranker=ranker or priority_ranker)
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
    "ATTENTION_FEATURES",
    "ATTENTION_REGISTRY",
    "UNIFIED_BLOCK_SIZES",
    "UNIFIED_HEAD_SIZES",
    "AttentionRequest",
    "AttentionSpec",
    "attention_candidates",
    "attention_sweep_space",
    "dense_spec_for_request",
    "dispatch_attention",
    "priority_ranker",
]
