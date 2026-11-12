# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SDPA dispatcher family: the hipDNN-facing request contract.

:class:`SdpaRequest` represents every field of a hipDNN SDPA forward node;
:func:`from_hipdnn` builds one from the node's flatbuffer JSON. Candidates gate
on :meth:`SdpaRequest.features` through the shared ``Capability`` contract.

SDPA registers no kernel of its own, so :data:`SDPA_REGISTRY` is an empty
``CandidateRegistry``; SDPA requests are served by the ``dispatch.attention``
unified and dense candidates through the gates in :mod:`.bind_unified` and
:mod:`.bind_dense`. :func:`registered_sdpa_combos`, :func:`sdpa_sweep_space`
and :func:`dispatch_sdpa_all` are the same wrappers every family has (see
``rocke.dispatch.families.norm``): each short-circuits on a request error, then
returns what the registry holds plus the attention combos of the family an SDPA
gate admits the request to -- nothing for a request both gates decline. The ABI version
arrives with the first candidate, whose ``KernelCandidate.abi_version`` carries
it.

Those combos and dispatch results describe the attention kernels; they do
not run an SDPA request. A result's ``bind_torch()`` is the attention binding:
it takes attention-format operands and defaults the scale to ``1/sqrt(d)``,
where SDPA's default is 1.0. To run an SDPA request on its own tensors, use
:func:`sdpa_unified_bind` or :func:`sdpa_dense_bind`, which translate the
operands and pass the request's scale.

See ``KERNEL_AUTHORING.md`` sections E and F for the request-contract and
operand-routing rules this module follows, and its attention appendix
("Route a family over ``dispatch.attention``") for the binding-layer checklist
this package's gates (:mod:`.bind_unified`, :mod:`.bind_dense`) follow
specifically, since SDPA routes to another family's candidates instead of
registering its own.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

from rocke.dispatch.core import (
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    make_kernel_id,
)

from dispatch.attention import dispatch_attention_all, registered_attention_combos
from dispatch.attention.unified_rules import TUNING_ALGORITHM

from .bind_dense import sdpa_dense_admits, sdpa_dense_bind, sdpa_dense_candidates
from .bind_shared import to_attention_request
from .bind_unified import (
    sdpa_unified_admits,
    sdpa_unified_bind,
    sdpa_unified_candidates,
)
from .common import (
    SDPA_DIM_VOCABULARY,
    SDPA_DTYPES,
    SDPA_FEATURES,
    Dtype,
    SdpaBias,
    SdpaDropout,
    SdpaFp8,
    SdpaPaged,
    SdpaRequest,
    SdpaTensor,
)
from .hipdnn import HIPDNN_DTYPES, from_hipdnn, from_hipdnn_graph

SDPA_REGISTRY = CandidateRegistry("sdpa", dim_vocabulary=SDPA_DIM_VOCABULARY)


def _request_errors(req: OperatorRequest) -> list[str]:
    """Why ``req`` cannot reach the registry; empty when it can.

    An ``SdpaRequest`` is validated at construction, so the remaining error is
    a request of another family.
    """
    if not isinstance(req, SdpaRequest):
        return [f"expected SdpaRequest, got {type(req).__name__}"]
    if req.op != "sdpa_fwd":
        return [f"unsupported op {req.op!r}"]
    return []


def _kernel_id(req: SdpaRequest, candidate: KernelCandidate, spec: Any) -> KernelId:
    return make_kernel_id(req, candidate, spec, op="sdpa_fwd")


def _serves(req: SdpaRequest):
    """Which attention ``(candidate, spec)`` combos serve ``req``: the unified
    tuning variants when the unified gate admits it, and the dense candidates
    the dense gate names. ``None`` when both gates decline it.

    A tuning variant's own precheck (``dispatch.attention.candidate._tuning_precheck``)
    already gates LSE per its path -- both tiled 2D and split-KV (3D) write it
    on gfx942/gfx950 -- so nothing here re-decides which path an LSE request
    may reach; that would only drop combos the variant's own gate already
    admitted (see ``dispatch.sdpa.bind_unified._unified_supported``, which
    trusts the same per-candidate gate for the production route).
    """
    unified = bool(sdpa_unified_candidates(req))
    dense = frozenset(sdpa_dense_candidates(req))
    if not unified and not dense:
        return None

    def serves(candidate: KernelCandidate, spec: Any) -> bool:
        if candidate.algorithm == TUNING_ALGORITHM:
            return unified
        return candidate.name in dense

    return serves


def _attention_combos(req: SdpaRequest) -> Tuple[Tuple[KernelCandidate, Any], ...]:
    """The attention combos an SDPA gate admits ``req`` to."""
    serves = _serves(req)
    if serves is None:
        return ()
    return tuple(
        (c, spec)
        for c, spec in registered_attention_combos(to_attention_request(req))
        if serves(c, spec)
    )


def registered_sdpa_combos(
    req: OperatorRequest,
) -> Tuple[Tuple[KernelCandidate, Any], ...]:
    """Every registered SDPA candidate that can launch ``req``, with its specs."""
    if _request_errors(req):
        return ()
    return SDPA_REGISTRY.combos(req) + _attention_combos(req)


def sdpa_sweep_space(req: OperatorRequest) -> Sequence[Any]:
    if _request_errors(req):
        return ()
    return tuple(SDPA_REGISTRY.sweep_space(req)) + tuple(
        spec for _, spec in _attention_combos(req)
    )


def dispatch_sdpa_all(req: OperatorRequest) -> Tuple[DispatchResult, ...]:
    """Every eligible SDPA kernel for ``req``, including opt-in variants.

    For ranking and triage. Each attention result binds an
    ``AttentionRequest``, so its ``bind_torch()`` does not take the SDPA
    tensors or carry the request's scale; launch an SDPA request through
    :func:`sdpa_unified_bind` or :func:`sdpa_dense_bind` instead.
    """
    if _request_errors(req):
        return ()
    serves = _serves(req)
    attention = (
        ()
        if serves is None
        else tuple(
            r
            for r in dispatch_attention_all(to_attention_request(req))
            if serves(r.candidate, r.spec)
        )
    )
    return SDPA_REGISTRY.dispatch_all(req, kernel_id=_kernel_id) + attention


__all__ = [
    "HIPDNN_DTYPES",
    "SDPA_DIM_VOCABULARY",
    "SDPA_DTYPES",
    "SDPA_FEATURES",
    "SDPA_REGISTRY",
    "Dtype",
    "SdpaBias",
    "SdpaDropout",
    "SdpaFp8",
    "SdpaPaged",
    "SdpaRequest",
    "SdpaTensor",
    "dispatch_sdpa_all",
    "from_hipdnn",
    "from_hipdnn_graph",
    "registered_sdpa_combos",
    "sdpa_dense_admits",
    "sdpa_dense_bind",
    "sdpa_sweep_space",
    "sdpa_unified_admits",
    "sdpa_unified_bind",
]
