# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SDPA dispatcher family: the hipDNN-facing request contract.

:class:`SdpaRequest` represents every field of a hipDNN SDPA forward node;
:func:`from_hipdnn` builds one from the node's flatbuffer JSON. Candidates gate
on :meth:`SdpaRequest.features` through the shared ``Capability`` contract.

No SDPA kernel family is registered against this contract yet (kernel wiring
is a follow-up -- see ``KERNEL_AUTHORING.md``), so :func:`registered_sdpa_combos`,
:func:`sdpa_sweep_space`, and :func:`dispatch_sdpa_all` are the delegation
points that follow-up plugs a real ``CandidateRegistry`` into. Each accepts
only a validated :class:`SdpaRequest` (``__post_init__`` already rejects a
malformed one) and, until that registry exists, always report nothing to run.
"""

from __future__ import annotations

from typing import Tuple

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
)
from .hipdnn import HIPDNN_DTYPES, from_hipdnn, from_hipdnn_graph


def registered_sdpa_combos(req: SdpaRequest) -> Tuple[Tuple[object, object], ...]:
    """Every executable ``(candidate, spec)`` pair registered for ``req``.

    Request errors short-circuit before any registry delegation: an
    ``SdpaRequest`` is already validated at construction
    (:meth:`SdpaRequest._validate`), so the only error this raises is a wrong
    argument type. No SDPA kernel family is registered against this contract
    yet, so this always returns an empty tuple.
    """
    if not isinstance(req, SdpaRequest):
        raise TypeError(f"expected SdpaRequest, got {type(req).__name__}")
    return ()


def sdpa_sweep_space(req: SdpaRequest) -> Tuple[object, ...]:
    """Every concrete engine/configuration :func:`registered_sdpa_combos`
    makes available to a sweep for ``req``."""
    return tuple(spec for _candidate, spec in registered_sdpa_combos(req))


def dispatch_sdpa_all(req: SdpaRequest) -> Tuple[object, ...]:
    """Every eligible dispatch result from :func:`registered_sdpa_combos` for
    ``req``."""
    return tuple(registered_sdpa_combos(req))


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
