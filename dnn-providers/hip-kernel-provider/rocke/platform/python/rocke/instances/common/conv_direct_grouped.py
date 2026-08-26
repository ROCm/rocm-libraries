# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DEPRECATED: conv_direct_grouped has moved to library/kernels/common/conv_direct_grouped.py.

This stub re-exports all symbols for backwards compatibility.
Import from kernels.common.conv_direct_grouped instead.
"""
import warnings as _w
_w.warn(
    "rocke.instances.common.conv_direct_grouped is deprecated; "
    "import from kernels.common.conv_direct_grouped",
    DeprecationWarning,
    stacklevel=2,
)
from kernels.common.conv_direct_grouped import *  # noqa: F401,F403
