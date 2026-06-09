# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Minimal CK DSL dispatcher surface.

Phase 1 intentionally starts with FP16 RCR GEMM only. The public entry point is
``dispatch_gemm_fp16``; broader operator families can register later without
changing the basic request/result contract.
"""

from __future__ import annotations

from .core import (
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
)
from .gemm import (
    GemmRequest,
    dispatch_gemm_fp16,
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
    "gemm_fp16_candidates",
    "gemm_fp16_sweep_space",
]
