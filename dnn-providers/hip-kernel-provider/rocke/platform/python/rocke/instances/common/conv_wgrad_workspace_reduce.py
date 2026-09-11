# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DEPRECATED: rocke.instances.common.conv_wgrad_workspace_reduce has moved to kernels.common.conv_wgrad_workspace_reduce.

This stub re-exports all symbols for backwards compatibility.
Import from kernels.common.conv_wgrad_workspace_reduce instead.
"""
import warnings as _w

_w.warn(
    "rocke.instances.common.conv_wgrad_workspace_reduce is deprecated; "
    "import from kernels.common.conv_wgrad_workspace_reduce",
    DeprecationWarning,
    stacklevel=2,
)
from kernels.common.conv_wgrad_workspace_reduce import *  # noqa: F401,F403
