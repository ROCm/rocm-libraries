# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SDPA dispatcher family: the hipDNN-facing request contract.

:class:`SdpaRequest` represents every field of a hipDNN SDPA forward node;
:func:`from_hipdnn` builds one from the node's flatbuffer JSON. Candidates gate
on :meth:`SdpaRequest.features` through the shared ``Capability`` contract.
"""

from __future__ import annotations

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
    "from_hipdnn",
    "from_hipdnn_graph",
]
