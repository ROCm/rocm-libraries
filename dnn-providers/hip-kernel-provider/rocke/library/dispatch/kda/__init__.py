# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Chunkwise Kimi Delta Attention dispatcher family (path-level selection).

Backed by :mod:`kernels.gfx942.kda_chunkwise`. This module owns only the
assembly -- the registry, the entry point, and the re-exports that make
``dispatch.kda`` one import for callers. What each candidate *is* lives in the
arch module that owns it. See :mod:`.common` for the scope of the dispatch
decision and what it defers.

Registration is explicit rather than an import side effect, so the registry
contents are a readable list, a test can assemble a registry from a subset of
arch modules, and adding an arch touches exactly one line here.
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
    stable_json_hash,
)

from . import gfx942
from .common import (
    FAMILY,
    KDA_ABI_VERSION,
    KDA_CHUNK_SIZES,
    KDA_DIM_VOCABULARY,
    KDA_DTYPES,
    KDA_FEATURES,
    KDA_PARTITION_HEAD_V,
    KdaRequest,
    _request_errors,
    _selector_matches,
)

_FAMILY = FAMILY

# ``require_build=True``: every candidate here maps 1:1 to an emitted kernel
# with a real builder, so a candidate that cannot compile has no reason to be
# selectable. ``require_binding`` stays off -- see :mod:`.common`.
KDA_REGISTRY = CandidateRegistry(
    _FAMILY, dim_vocabulary=KDA_DIM_VOCABULARY, require_build=True
)
for _module in (gfx942,):
    _module.register(KDA_REGISTRY)


def kda_candidates() -> Tuple[KernelCandidate, ...]:
    return KDA_REGISTRY.candidates()


def _kernel_id(req: KdaRequest, candidate: KernelCandidate, spec: Any) -> KernelId:
    request_hash = stable_json_hash(req.normalized(), n=16)
    spec_hash = stable_json_hash(asdict(spec), n=16)
    return KernelId(
        op="kda",
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=request_hash,
        spec_hash=spec_hash,
    )


def kda_sweep_space(req: OperatorRequest) -> Sequence[Any]:
    if _request_errors(req):
        return ()
    specs = []
    seen = set()
    for candidate in KDA_REGISTRY.supported(req):
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
    by ``(priority, name)``, so this is an identity pass -- the explicit default
    ``dispatch_kda`` applies when no ranker is supplied. It exists as a named
    seam for the measured fused/split ranker that :mod:`.common` defers.
    """
    return candidates


def dispatch_kda(req: KdaRequest, *, ranker: Ranker | None = None) -> DispatchResult:
    """Select the chunkwise KDA kernel for ``req``.

    Returns the fused prefill kernel unless the request names a split-path
    half by ``algorithm`` / ``spec_id``.
    """
    candidate = KDA_REGISTRY.select(req, ranker=ranker or priority_ranker)
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
            f"v_partitions={req.v_partitions} workgroups={req.workgroups}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )


__all__ = [
    "KDA_ABI_VERSION",
    "KDA_CHUNK_SIZES",
    "KDA_DIM_VOCABULARY",
    "KDA_DTYPES",
    "KDA_FEATURES",
    "KDA_PARTITION_HEAD_V",
    "KDA_REGISTRY",
    "KdaRequest",
    "dispatch_kda",
    "kda_candidates",
    "kda_sweep_space",
    "priority_ranker",
]
