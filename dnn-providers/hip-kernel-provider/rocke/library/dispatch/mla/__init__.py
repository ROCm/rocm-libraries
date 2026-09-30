# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Multi-head latent attention dispatcher family.

A *sibling* of :mod:`dispatch.attention`, not an extension of it -- see
:mod:`.common` for why the two registries cannot be merged.

This module owns only the assembly: the registry, the entry points, and the
re-exports that make ``dispatch.mla`` one import for callers. What each
candidate *is* lives in the arch module that owns it.

The registry sets ``require_build=True``. Attention leaves it off because its
unified candidates select a *path* and hand the CTA geometry to a downstream
builder; every MLA candidate, by contrast, names a concrete builder, so the
ratchet is free to turn on now and stops a later candidate from rejoining the
selectable-but-not-compilable set by omission.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

from rocke.dispatch.core import (
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    make_kernel_id,
)

from . import gfx942
from .common import (
    FAMILY,
    MLA_ABI_VERSION,
    MLA_DIM_VOCABULARY,
    MLA_FEATURES,
    MLARequest,
    _request_errors,
    num_q_blocks_for,
)

_FAMILY = FAMILY

MLA_REGISTRY = CandidateRegistry(
    _FAMILY, dim_vocabulary=MLA_DIM_VOCABULARY, require_build=True
)
for _module in (gfx942,):
    _module.register(MLA_REGISTRY)


def mla_candidates() -> Tuple[KernelCandidate, ...]:
    return MLA_REGISTRY.candidates()


def _kernel_id(req: MLARequest, candidate: KernelCandidate, spec: Any) -> KernelId:
    # ``req.op`` rather than a literal: this family already anticipates a
    # second op (decode-absorb), and a hard-coded "mla_prefill_fwd" would give
    # two different ops the same selection key.
    return make_kernel_id(req, candidate, spec, op=req.op)


def registered_mla_combos(
    req: OperatorRequest,
) -> Tuple[Tuple[KernelCandidate, Any], ...]:
    """Every registered MLA ``(candidate, spec)`` that can launch ``req``."""
    if _request_errors(req):
        return ()
    return MLA_REGISTRY.combos(req)


def mla_sweep_space(req: OperatorRequest) -> Sequence[Any]:
    if _request_errors(req):
        return ()
    return MLA_REGISTRY.sweep_space(req)


def dispatch_mla_all(req: MLARequest) -> Tuple[DispatchResult, ...]:
    """Every eligible MLA kernel for ``req``, one result per combo."""
    if _request_errors(req):
        return ()
    return MLA_REGISTRY.dispatch_all(req, kernel_id=_kernel_id)


def dispatch_mla(req: MLARequest, *, ranker: Ranker | None = None) -> DispatchResult:
    """Select the MLA kernel for ``req``.

    Unlike :func:`dispatch.attention.dispatch_attention`, the returned grid and
    signature are the real ones: an MLA candidate names a concrete builder, so
    there is no downstream geometry decision left to defer.
    """
    candidate = MLA_REGISTRY.select(req, ranker=ranker)
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
            f"selected {candidate.name} ({candidate.algorithm}) on {req.arch}",
            f"algorithm={candidate.algorithm}",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )


__all__ = [
    "FAMILY",
    "MLA_ABI_VERSION",
    "MLA_DIM_VOCABULARY",
    "MLA_FEATURES",
    "MLA_REGISTRY",
    "MLARequest",
    "dispatch_mla",
    "dispatch_mla_all",
    "mla_candidates",
    "mla_sweep_space",
    "num_q_blocks_for",
    "registered_mla_combos",
]
