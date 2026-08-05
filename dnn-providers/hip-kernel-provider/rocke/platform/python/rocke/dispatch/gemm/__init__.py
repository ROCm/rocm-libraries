# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GEMM-family dispatcher package.

The package is organized by case so future GEMM variants can reuse common
request/shape helpers without growing one monolithic module:

* ``common.py``: GEMM-family request and validation helpers (dtype/layout
  generic), including the dimension vocabulary capabilities constrain.
* ``fp16_rcr.py``: UniversalGemm FP16 RCR candidates (the first case).
* ``bf16_rcr.py``: UniversalGemm BF16 RCR candidates (the worked template for
  adding further dtypes/layouts).
* ``fp8_rcr.py``: block-scale FP8 / BF8 RCR candidates. A different kernel body
  from the two above -- UniversalGemm takes one dtype for A, B, and C and has no
  scale operands, so a scaled low-bit GEMM cannot be expressed as one of its
  specs.
"""

from __future__ import annotations

from .bf16_rcr import (
    GEMM_BF16_RCR_ABI_VERSION,
    GEMM_BF16_REGISTRY,
    dispatch_gemm_bf16,
    gemm_bf16_candidates,
    gemm_bf16_sweep_space,
)
from .bf16_rcr import build_kernel as build_kernel_bf16
from .common import GemmRequest
from .fp8_rcr import (
    GEMM_FP8_RCR_ABI_VERSION,
    GEMM_FP8_REGISTRY,
    Fp8GemmRequest,
    dispatch_gemm_fp8,
    gemm_fp8_candidates,
    gemm_fp8_sweep_space,
)
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
    "GEMM_BF16_RCR_ABI_VERSION",
    "GEMM_BF16_REGISTRY",
    "GEMM_FP8_RCR_ABI_VERSION",
    "GEMM_FP8_REGISTRY",
    "Fp8GemmRequest",
    "GemmRequest",
    "build_kernel",
    "build_kernel_bf16",
    "dispatch_gemm_fp16",
    "dispatch_gemm_bf16",
    "dispatch_gemm_fp8",
    "gemm_fp16_candidates",
    "gemm_bf16_candidates",
    "gemm_fp8_candidates",
    "gemm_fp16_sweep_space",
    "gemm_bf16_sweep_space",
    "gemm_fp8_sweep_space",
]
