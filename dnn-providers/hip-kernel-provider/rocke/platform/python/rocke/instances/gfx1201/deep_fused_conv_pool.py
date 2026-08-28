# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DEPRECATED: deep_fused_conv_pool has moved to library/kernels/gfx1201/deep_fused_conv_pool.py."""
import warnings as _w

_w.warn(
    "rocke.instances.gfx1201.deep_fused_conv_pool is deprecated; "
    "import from kernels.gfx1201.deep_fused_conv_pool",
    DeprecationWarning,
    stacklevel=2,
)
from kernels.gfx1201.deep_fused_conv_pool import *  # noqa: F401,F403
