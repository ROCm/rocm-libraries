# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DEPRECATED: rocke.instances.common.conv_implicit_gemm_dgrad has moved to kernels.common.conv_implicit_gemm_dgrad.

This stub re-exports all symbols for backwards compatibility.
Import from kernels.common.conv_implicit_gemm_dgrad instead.
"""
import warnings as _w

_w.warn(
    "rocke.instances.common.conv_implicit_gemm_dgrad is deprecated; "
    "import from kernels.common.conv_implicit_gemm_dgrad",
    DeprecationWarning,
    stacklevel=2,
)
from kernels.common.conv_implicit_gemm_dgrad import *  # noqa: F401,F403
