# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DEPRECATED: rocke.instances.common.conv_implicit_gemm_wgrad_two_stage has moved to kernels.common.conv_implicit_gemm_wgrad_two_stage.

This stub re-exports all symbols for backwards compatibility.
Import from kernels.common.conv_implicit_gemm_wgrad_two_stage instead.
"""
import warnings as _w

_w.warn(
    "rocke.instances.common.conv_implicit_gemm_wgrad_two_stage is deprecated; "
    "import from kernels.common.conv_implicit_gemm_wgrad_two_stage",
    DeprecationWarning,
    stacklevel=2,
)
from kernels.common.conv_implicit_gemm_wgrad_two_stage import *  # noqa: F401,F403
