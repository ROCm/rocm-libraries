# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DEPRECATED: img2col has moved to library/kernels/common/img2col.py.

This stub re-exports all symbols for backwards compatibility.
Import from kernels.common.img2col instead.
"""
import warnings as _w
_w.warn(
    "rocke.instances.common.img2col is deprecated; "
    "import from kernels.common.img2col",
    DeprecationWarning,
    stacklevel=2,
)
from kernels.common.img2col import *  # noqa: F401,F403
