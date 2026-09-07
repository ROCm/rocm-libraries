# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DEPRECATED: rocke.instances.common._conv_implicit_gemm_common has moved to kernels.common._conv_implicit_gemm_common.

This stub re-exports all symbols for backwards compatibility.
Import from kernels.common._conv_implicit_gemm_common instead.
"""
import warnings as _w

_w.warn(
    "rocke.instances.common._conv_implicit_gemm_common is deprecated; "
    "import from kernels.common._conv_implicit_gemm_common",
    DeprecationWarning,
    stacklevel=2,
)
from kernels.common._conv_implicit_gemm_common import *  # noqa: F401,F403
