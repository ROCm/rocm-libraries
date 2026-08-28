# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DEPRECATED: deep_fused_conv_pool has moved to library/kernels/common/deep_fused_conv_pool.py.

This stub re-exports all symbols for backwards compatibility.
Import from kernels.common.deep_fused_conv_pool instead.
"""
import warnings as _w

_w.warn(
    "rocke.instances.common.deep_fused_conv_pool is deprecated; "
    "import from kernels.common.deep_fused_conv_pool",
    DeprecationWarning,
    stacklevel=2,
)
from kernels.common.deep_fused_conv_pool import *  # noqa: F401,F403
