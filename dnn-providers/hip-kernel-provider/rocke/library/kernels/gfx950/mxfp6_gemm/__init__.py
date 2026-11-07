# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 MXFP6 x MXFP6 GEMM (OCP MX E2M3 elements, E8M0 block-32 scales, bf16 out).

``gemm``: spec (every performance choice is a defaulted knob) and builder;
``layout``: host-side canonical data, scale repack, references and LDS maps.
Python engine only (``tile.mfma_scale_mx``).
"""

from .gemm import MUTANTS, Mxfp6GemmSpec, build_mxfp6_gemm, descriptor, grid_of, kernel_name, validate_spec

__all__ = [
    "MUTANTS",
    "Mxfp6GemmSpec",
    "build_mxfp6_gemm",
    "descriptor",
    "grid_of",
    "kernel_name",
    "validate_spec",
]
