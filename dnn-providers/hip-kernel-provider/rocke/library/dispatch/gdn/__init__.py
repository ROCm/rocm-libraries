# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GDN dispatcher: registry assembly and the public entry points.

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
    make_dispatch_result,
    make_kernel_id,
    pin_to_spec,
    stable_json_hash,
)

from . import gfx950
from . import prefill_gfx950
from .common import (
    FAMILY,
    GDN_ABI_VERSION,
    GDN_DIM_VOCABULARY,
    GdnDecodeRequest,
    GdnDecodeTunedSpec,
    normalize_dtype,
    request_errors,
)
from .prefill_common import (
    FAMILY_PREFILL,
    GDN_PREFILL_ABI_VERSION,
    GDN_PREFILL_DIM_VOCABULARY,
    GdnPrefillRequest,
)

_ARCH_MODULES = (gfx950,)

GDN_REGISTRY = CandidateRegistry(
    FAMILY, dim_vocabulary=GDN_DIM_VOCABULARY, require_build=True
)
for _module in _ARCH_MODULES:
    _module.register(GDN_REGISTRY)

GDN_PREFILL_REGISTRY = CandidateRegistry(
    FAMILY_PREFILL, dim_vocabulary=GDN_PREFILL_DIM_VOCABULARY, require_build=True
)
for _module in (prefill_gfx950,):
    _module.register(GDN_PREFILL_REGISTRY)


def gdn_candidates() -> Tuple[KernelCandidate, ...]:
    return GDN_REGISTRY.candidates()


def _kernel_id(
    req: GdnDecodeRequest, candidate: KernelCandidate, spec: Any
) -> KernelId:
    return make_kernel_id(req, candidate, spec, op="gdn_decode")


def dispatch_gdn_decode_all(req: GdnDecodeRequest) -> Tuple[DispatchResult, ...]:
    """One result per registered (tile x canonical knob set) legal for ``req``,
    at the current sweep level (``waves_per_eu`` differs by level; see
    ``axes.py``).

    Tiles come in registry priority order, and each tile's base configuration
    (every knob at its default, what ``auto`` runs on that tile) comes first
    among its configurations. Each result's request is pinned to its spec
    (``spec_id``, ``tuning_id``, ``tuning_knobs``), so dispatching it again
    selects exactly that kernel. :func:`dispatch_gdn_decode_bases` keeps only
    the base configurations.
    """
    if request_errors(req):
        return ()
    return GDN_REGISTRY.dispatch_all(req, kernel_id=_kernel_id)


def dispatch_gdn_decode_bases(req: GdnDecodeRequest) -> Tuple[DispatchResult, ...]:
    """Each legal registered tile's base configuration (no knobs: what ``auto``
    runs on that tile), one result per tile in registry priority order. What a
    caller that compares tiles wants, rather than every configuration
    :func:`dispatch_gdn_decode_all` lists."""
    return tuple(
        result for result in dispatch_gdn_decode_all(req) if not result.spec.knobs
    )


def gdn_sweep_space(req: OperatorRequest) -> Sequence[Any]:
    """Every distinct tuned spec admitted by the registry for ``req``."""
    if not isinstance(req, GdnDecodeRequest):
        return ()
    return tuple(result.spec for result in dispatch_gdn_decode_all(req))


def dispatch_gdn_decode(
    req: GdnDecodeRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select a registered GDN decode configuration for ``req``.

    ``auto`` returns the static tile's base configuration (every knob at its
    default). A ``spec_id`` pin (with optional ``tuning_id`` /
    ``tuning_knobs``) returns exactly that configuration or raises
    :class:`rocke.dispatch.core.PinRefused` with the candidate's reason; it
    never falls back to another kernel. The result's request is pinned to the
    selected spec, so it replays the same kernel; ``result.spec`` is a
    :class:`GdnDecodeTunedSpec`, whose ``kernel_spec`` is what the builder and
    launcher read.
    """
    candidate = GDN_REGISTRY.select(req, ranker=ranker)
    spec = candidate.select_spec(req)
    request = pin_to_spec(req, candidate, spec)
    kernel = spec.kernel_spec
    return make_dispatch_result(
        request,
        candidate,
        spec,
        kernel_id=_kernel_id(request, candidate, spec),
        headline=(
            f"selected {candidate.name} for batch {req.batch} on {req.arch}: "
            f"tile=(nw={kernel.num_warps}, wtk={kernel.warp_threads_k}, "
            f"bpv={kernel.blocks_per_v_dim})"
        ),
    )


def gdn_prefill_candidates() -> Tuple[KernelCandidate, ...]:
    return GDN_PREFILL_REGISTRY.candidates()


def _prefill_kernel_id(
    req: GdnPrefillRequest, candidate: KernelCandidate, spec: Any
) -> KernelId:
    return make_kernel_id(req, candidate, spec, op="gdn_prefill")


def dispatch_gdn_prefill(
    req: GdnPrefillRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select a GDN prefill split-half for ``req``.

    GDN prefill is a two-launch split path with no fused default, so the caller
    pins ``algorithm="chunk_prep"`` then ``"chunk_scan"``; ``auto`` is rejected
    rather than resolved to one half of a two-launch path.
    """
    if (
        req.algorithm.strip().lower() == "auto"
        and req.spec_id.strip().lower() == "auto"
    ):
        raise ValueError(
            "GDN prefill has no fused default: dispatch algorithm='chunk_prep' "
            "then algorithm='chunk_scan' (there is no single-kernel GDN prefill)"
        )
    candidate = GDN_PREFILL_REGISTRY.select(req, ranker=ranker)
    spec = candidate.select_spec(req)
    kid = _prefill_kernel_id(req, candidate, spec)
    return DispatchResult(
        request=req,
        candidate=candidate,
        spec=spec,
        kernel_id=kid,
        grid=candidate.grid(spec, req),
        block=candidate.block(spec),
        signature=tuple(candidate.signature(spec)),
        explanation=(
            f"selected {candidate.name} for BH={req.batch_heads} on {req.arch}",
            f"value_splits={getattr(spec, 'value_splits', 1)}, "
            f"block_size={spec.tile.block_size}, chunk={spec.tile.chunk}",
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
    "GdnDecodeTunedSpec",
    "dispatch_gdn_decode",
    "dispatch_gdn_decode_all",
    "dispatch_gdn_decode_bases",
    "gdn_candidates",
    "gdn_sweep_space",
    "normalize_dtype",
    "FAMILY_PREFILL",
    "GDN_PREFILL_ABI_VERSION",
    "GDN_PREFILL_DIM_VOCABULARY",
    "GDN_PREFILL_REGISTRY",
    "GdnPrefillRequest",
    "dispatch_gdn_prefill",
    "gdn_prefill_candidates",
]
