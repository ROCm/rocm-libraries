# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Low-level Python bindings for the hipBLASLt GEMM API."""

from . import _core

__all__ = ["_core"]
__version__ = _core.__version__
