# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Former name of :mod:`.mx_gemm`, kept for existing callers."""

from __future__ import annotations

from .mx_gemm import (
    MMA_SHAPES,
    OUT_DTYPES,
    TILE_K,
    TILE_MN,
    WAVE_GROUPS,
    IntOps,
    LdsPad,
    derived_lds_pads,
    is_valid_spec,
    validate,
)
from .mx_gemm import MxGemmLdsLayout as MxFp4GemmLdsLayout
from .mx_gemm import MxGemmSpec as MxFp4GemmSpec
from .mx_gemm import build_mx_gemm as build_mxfp4_gemm
from .mx_gemm import mx_gemm_grid as mxfp4_gemm_grid
from .mx_gemm import mx_gemm_lds_layout as mxfp4_gemm_lds_layout
from .mx_gemm import mx_gemm_signature as mxfp4_gemm_signature

__all__ = [
    "MMA_SHAPES",
    "OUT_DTYPES",
    "TILE_K",
    "TILE_MN",
    "WAVE_GROUPS",
    "IntOps",
    "LdsPad",
    "MxFp4GemmLdsLayout",
    "MxFp4GemmSpec",
    "build_mxfp4_gemm",
    "derived_lds_pads",
    "is_valid_spec",
    "mxfp4_gemm_grid",
    "mxfp4_gemm_lds_layout",
    "mxfp4_gemm_signature",
    "validate",
]
