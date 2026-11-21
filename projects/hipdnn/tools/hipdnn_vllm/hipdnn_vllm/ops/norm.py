# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Python entry points for ``torch.ops.hipdnn_vllm.rms_norm``.

The op itself raises on a decline (message decoded by
:func:`hipdnn_vllm.declines.parse_op_error`). Callers that must not raise, such
as a vLLM IR provider's ``supports_args``, ask :func:`rms_norm_decline` first:
it builds (and caches) the plan for the signature without executing.
"""

from __future__ import annotations

import torch

from .. import _bootstrap


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    _bootstrap.load()
    return torch.ops.hipdnn_vllm.rms_norm(x, weight, float(eps))


def rms_norm_decline(x: torch.Tensor, weight: torch.Tensor, eps: float) -> str:
    """``""`` when hipDNN runs this signature, else the decline reason."""
    _bootstrap.load()
    from .. import _C

    return _C.rms_norm_probe(x, weight, float(eps))
