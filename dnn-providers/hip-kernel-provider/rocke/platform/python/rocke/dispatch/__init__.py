# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocKE dispatcher surface.

The dispatcher started with FP16 RCR GEMM only; it now also implements BF16 RCR
GEMM (the worked template for further dtypes/layouts) and carries documented
scaffolds for the remaining operator families (moe, norm) in
:mod:`rocke.dispatch.families`. The basic request/result contract
(``OperatorRequest`` / ``DispatchResult`` / ``CandidateRegistry``) is shared by
all families.
"""

from __future__ import annotations

from .core import (
    Capability,
    CandidateRegistry,
    DimRelation,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    ShapeRange,
    opt_in_probe,
    spec_identity,
)
from .families import (
    MoeRequest,
    NormRequest,
    dispatch_moe,
    dispatch_moe_all,
    dispatch_norm,
    dispatch_norm_all,
    moe_sweep_space,
    norm_sweep_space,
    registered_moe_combos,
    registered_norm_combos,
)
from .gemm import (
    GemmRequest,
    dispatch_gemm_bf16,
    dispatch_gemm_bf16_all,
    dispatch_gemm_fp16,
    dispatch_gemm_fp16_all,
    gemm_bf16_candidates,
    gemm_bf16_sweep_space,
    gemm_fp16_candidates,
    gemm_fp16_sweep_space,
    registered_gemm_bf16_combos,
    registered_gemm_fp16_combos,
)

__all__ = [
    "Capability",
    "DimRelation",
    "DispatchResult",
    "CandidateRegistry",
    "GemmRequest",
    "KernelCandidate",
    "KernelId",
    "OperatorRequest",
    "ShapeRange",
    "opt_in_probe",
    "spec_identity",
    "dispatch_gemm_fp16",
    "dispatch_gemm_fp16_all",
    "dispatch_gemm_bf16",
    "dispatch_gemm_bf16_all",
    "gemm_fp16_candidates",
    "gemm_bf16_candidates",
    "gemm_fp16_sweep_space",
    "gemm_bf16_sweep_space",
    "registered_gemm_fp16_combos",
    "registered_gemm_bf16_combos",
    # operator families
    "MoeRequest",
    "NormRequest",
    "dispatch_moe",
    "dispatch_moe_all",
    "dispatch_norm",
    "dispatch_norm_all",
    "moe_sweep_space",
    "norm_sweep_space",
    "registered_moe_combos",
    "registered_norm_combos",
]
