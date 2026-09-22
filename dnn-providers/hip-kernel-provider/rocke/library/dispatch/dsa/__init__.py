# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""DSA lightning-indexer dispatcher family (path-level selection).

Backed by the gfx942 lightning-indexer kernel. This module owns only the
assembly -- the registry, the entry point, and the re-exports that make
``dispatch.dsa`` one import for callers. What the candidate is lives in the arch
module that owns it; see :mod:`.common` for the scope of the dispatch decision.

Its own registry, not the unified attention registry: a DSA geometry must not
become selectable for a standard attention request as a side effect, exactly as
MLA takes its own registry.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence, Tuple

from rocke.dispatch.core import (
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    make_kernel_id,
    stable_json_hash,
)

from . import gfx942
from .common import (
    FAMILY,
    DSA_INDEXER_ABI_VERSION,
    DSA_INDEXER_DIM_VOCABULARY,
    DSA_INDEXER_FEATURES,
    IndexerRequest,
    _request_errors,
)

# ``require_build=True``: the candidate maps 1:1 to an emitted kernel with a real
# builder, so a candidate that cannot compile has no reason to be selectable.
DSA_INDEXER_REGISTRY = CandidateRegistry(
    FAMILY, dim_vocabulary=DSA_INDEXER_DIM_VOCABULARY, require_build=True
)
for _module in (gfx942,):
    _module.register(DSA_INDEXER_REGISTRY)


def indexer_candidates() -> Tuple[KernelCandidate, ...]:
    return DSA_INDEXER_REGISTRY.candidates()


def _kernel_id(req: IndexerRequest, candidate: KernelCandidate, spec: Any) -> KernelId:
    return make_kernel_id(req, candidate, spec, op="lightning_indexer")


def indexer_sweep_space(req: OperatorRequest) -> Sequence[Any]:
    if _request_errors(req):
        return ()
    specs = []
    seen = set()
    for candidate in DSA_INDEXER_REGISTRY.supported(req):
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

    ``CandidateRegistry.supported`` already returns candidates sorted ascending
    by ``(priority, name)``, so this is an identity pass. It exists as a named
    seam for the MFMA/fp8 candidates that will join the family later.
    """
    return candidates


def dispatch_lightning_indexer(
    req: IndexerRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select the lightning-indexer kernel for ``req``."""
    candidate = DSA_INDEXER_REGISTRY.select(req, ranker=ranker or priority_ranker)
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
            f"seqlen_q={req.seqlen_q} seqlen_k={req.seqlen_k}",
            f"n_index_heads={req.n_index_heads} index_head_dim={req.index_head_dim}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )


__all__ = [
    "DSA_INDEXER_ABI_VERSION",
    "DSA_INDEXER_DIM_VOCABULARY",
    "DSA_INDEXER_FEATURES",
    "DSA_INDEXER_REGISTRY",
    "IndexerRequest",
    "dispatch_lightning_indexer",
    "indexer_candidates",
    "indexer_sweep_space",
    "priority_ranker",
]
