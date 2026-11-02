# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SDPA dispatcher family: the hipDNN-facing request contract.

:class:`SdpaRequest` represents every field of a hipDNN SDPA forward node;
:func:`from_hipdnn` builds one from the node's flatbuffer JSON. Candidates gate
on :meth:`SdpaRequest.features` through the shared ``Capability`` contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rocke.dispatch.core import DispatchResult, KernelCandidate

from dispatch.attention import (
    attention_sweep_space,
    dispatch_attention_all,
    registered_attention_combos,
)

from .bind_shared import to_attention_request
from .common import (
    SDPA_ABI_VERSION,
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
    _request_errors,
)
from .hipdnn import HIPDNN_DTYPES, from_hipdnn, from_hipdnn_graph


def registered_sdpa_combos(
    req: SdpaRequest,
) -> tuple[tuple[KernelCandidate, Any], ...]:
    """Every registered unified or dense ``(candidate, spec)`` that can launch ``req``.

    Delegates to the existing ``dispatch.attention`` registry through the
    same ``to_attention_request`` translation the unified and dense bindings
    use: SDPA registers no candidate, knob or registry of its own, so there is
    nothing here beyond that translation.
    """
    if _request_errors(req):
        return ()
    return registered_attention_combos(to_attention_request(req))


def sdpa_sweep_space(req: SdpaRequest) -> Sequence[Any]:
    """Every admissible spec for ``req``, opt-in (dense) candidates included."""
    if _request_errors(req):
        return ()
    return attention_sweep_space(to_attention_request(req))


def dispatch_sdpa_all(req: SdpaRequest) -> tuple[DispatchResult, ...]:
    """Every eligible unified or dense kernel for ``req``, one result per combo."""
    if _request_errors(req):
        return ()
    return dispatch_attention_all(to_attention_request(req))


__all__ = [
    "HIPDNN_DTYPES",
    "SDPA_ABI_VERSION",
    "SDPA_DIM_VOCABULARY",
    "SDPA_DTYPES",
    "SDPA_FEATURES",
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
