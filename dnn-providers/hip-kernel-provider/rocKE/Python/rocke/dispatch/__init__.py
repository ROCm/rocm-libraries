# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CK DSL dispatcher surface.

The dispatcher started with FP16 RCR GEMM only; it now also implements BF16 RCR
GEMM (the worked template for further dtypes/layouts) and carries documented
scaffolds for the remaining operator families (conv, attention, moe, norm) in
:mod:`rocke.dispatch.families`. The basic request/result contract
(``OperatorRequest`` / ``DispatchResult`` / ``CandidateRegistry``) is shared by
all families.
"""

from __future__ import annotations

from .core import (
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
)
from .families import (
    AttentionRequest,
    ConvRequest,
    MoeRequest,
    NormRequest,
    dispatch_attention,
    dispatch_conv,
    dispatch_moe,
    dispatch_norm,
)
from .gemm import (
    GemmRequest,
    dispatch_gemm_bf16,
    dispatch_gemm_fp16,
    gemm_bf16_candidates,
    gemm_bf16_sweep_space,
    gemm_fp16_candidates,
    gemm_fp16_sweep_space,
)

__all__ = [
    "DispatchResult",
    "CandidateRegistry",
    "GemmRequest",
    "KernelCandidate",
    "KernelId",
    "OperatorRequest",
    "dispatch_gemm_fp16",
    "dispatch_gemm_bf16",
    "gemm_fp16_candidates",
    "gemm_bf16_candidates",
    "gemm_fp16_sweep_space",
    "gemm_bf16_sweep_space",
    # operator families
    "ConvRequest",
    "AttentionRequest",
    "MoeRequest",
    "NormRequest",
    "dispatch_conv",
    "dispatch_attention",
    "dispatch_moe",
    "dispatch_norm",
]
