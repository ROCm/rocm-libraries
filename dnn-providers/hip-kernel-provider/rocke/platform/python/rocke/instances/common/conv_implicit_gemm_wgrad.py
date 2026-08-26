# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DEPRECATED: conv_implicit_gemm_wgrad has moved to library/kernels/common/conv_implicit_gemm_wgrad.py.

This stub re-exports all symbols for backwards compatibility.
Import from kernels.common.conv_implicit_gemm_wgrad instead.
"""
import warnings as _w
_w.warn(
    "rocke.instances.common.conv_implicit_gemm_wgrad is deprecated; "
    "import from kernels.common.conv_implicit_gemm_wgrad",
    DeprecationWarning,
    stacklevel=2,
)
from kernels.common.conv_implicit_gemm_wgrad import *  # noqa: F401,F403
