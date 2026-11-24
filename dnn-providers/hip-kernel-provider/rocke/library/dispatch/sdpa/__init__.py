# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SDPA dispatcher family: the hipDNN-facing request contract.

:class:`SdpaRequest` represents every field of a hipDNN SDPA forward node;
:func:`from_hipdnn` builds one from the node's flatbuffer JSON. Candidates gate
on :meth:`SdpaRequest.features` through the shared ``Capability`` contract.

No SDPA kernel is registered yet (kernel wiring is a follow-up -- see
``KERNEL_AUTHORING.md``), so :data:`SDPA_REGISTRY` is an empty
``CandidateRegistry``. :func:`registered_sdpa_combos`, :func:`sdpa_sweep_space`
and :func:`dispatch_sdpa_all` are the same wrappers every family has (see
``rocke.dispatch.families.norm``): each short-circuits on a request error, then
delegates to the registry, so a candidate registered there is reachable through
them without changing this module. The ABI version arrives with the first
candidate, whose ``KernelCandidate.abi_version`` carries it.
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


def registered_sdpa_combos(
    req: OperatorRequest,
) -> Tuple[Tuple[KernelCandidate, Any], ...]:
    """Every registered SDPA candidate that can launch ``req``, with its specs."""
    if _request_errors(req):
        return ()
    return SDPA_REGISTRY.combos(req)


def sdpa_sweep_space(req: OperatorRequest) -> Sequence[Any]:
    if _request_errors(req):
        return ()
    return SDPA_REGISTRY.sweep_space(req)


def dispatch_sdpa_all(req: OperatorRequest) -> Tuple[DispatchResult, ...]:
    """Every eligible SDPA kernel for ``req``, including opt-in variants."""
    if _request_errors(req):
        return ()
    return SDPA_REGISTRY.dispatch_all(req, kernel_id=_kernel_id)


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
    "sdpa_sweep_space",
]
