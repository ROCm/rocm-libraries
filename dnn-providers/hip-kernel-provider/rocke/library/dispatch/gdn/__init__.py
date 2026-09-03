# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GDN decode dispatcher: registry assembly and the public entry point.

Importing this package is what registers the family's candidates, so it must be
reachable from whatever imports the dispatch tree -- a family absent from the
registry is unreachable no matter how complete its kernel is.
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

from . import gfx950
from .common import (
    FAMILY,
    GDN_ABI_VERSION,
    GDN_DIM_VOCABULARY,
    GdnDecodeRequest,
    normalize_dtype,
    request_errors,
)

_ARCH_MODULES = (gfx950,)

GDN_REGISTRY = CandidateRegistry(
    FAMILY, dim_vocabulary=GDN_DIM_VOCABULARY, require_build=True
)
for _module in _ARCH_MODULES:
    _module.register(GDN_REGISTRY)


def gdn_candidates() -> Tuple[KernelCandidate, ...]:
    return GDN_REGISTRY.candidates()


def _kernel_id(
    req: GdnDecodeRequest, candidate: KernelCandidate, spec: Any
) -> KernelId:
    return KernelId(
        op="gdn_decode",
        family=FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=stable_json_hash(req.normalized(), n=16),
        spec_hash=stable_json_hash(asdict(spec), n=16),
    )


def gdn_sweep_space(req: OperatorRequest) -> Sequence[Any]:
    """Every distinct spec any candidate would build for ``req``.

    Under ``auto`` the tuning table admits exactly one candidate, so a tuner
    that wants the whole tile space must ask per ``spec_id``; this returns what
    is reachable for the request as given.
    """
    if request_errors(req):
        return ()
    specs = []
    seen = set()
    for candidate in GDN_REGISTRY.supported(req):
        spec = candidate.select_spec(req)
        digest = stable_json_hash(asdict(spec), n=16)
        if digest not in seen:
            seen.add(digest)
            specs.append(spec)
    return tuple(specs)


def dispatch_gdn_decode(
    req: GdnDecodeRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select a registered GDN decode candidate for ``req``."""
    candidate = GDN_REGISTRY.select(req, ranker=ranker)
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
            f"selected {candidate.name} for batch {req.batch} on {req.arch}",
            f"tile=(nw={spec.num_warps}, wtk={spec.warp_threads_k}, "
            f"bpv={spec.blocks_per_v_dim})",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
        ),
    )


__all__ = [
    "FAMILY",
    "GDN_ABI_VERSION",
    "GDN_DIM_VOCABULARY",
    "GDN_REGISTRY",
    "GdnDecodeRequest",
    "dispatch_gdn_decode",
    "gdn_candidates",
    "gdn_sweep_space",
    "normalize_dtype",
]
