# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GEMM-family dispatcher package.

The package is organized by case so future GEMM variants can reuse common
request/shape helpers without growing one monolithic module:

* ``common.py``: GEMM-family request and validation helpers.
* ``fp16_rcr.py``: phase-1 UniversalGemm FP16 RCR candidates.
"""

from __future__ import annotations

from .common import GemmRequest
from .fp16_rcr import (
    GEMM_FP16_RCR_ABI_VERSION,
    GEMM_FP16_REGISTRY,
    build_kernel,
    dispatch_gemm_fp16,
    gemm_fp16_candidates,
    gemm_fp16_sweep_space,
)

__all__ = [
    "GEMM_FP16_RCR_ABI_VERSION",
    "GEMM_FP16_REGISTRY",
    "GemmRequest",
    "build_kernel",
    "dispatch_gemm_fp16",
    "gemm_fp16_candidates",
    "gemm_fp16_sweep_space",
]
