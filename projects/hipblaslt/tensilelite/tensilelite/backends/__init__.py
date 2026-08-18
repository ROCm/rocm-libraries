# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from .base import OptimizationBackend, BackendFactory
from .tensile_backend import TensileBackend

# Register backends
BackendFactory.register("tensile", TensileBackend)

__all__ = ["OptimizationBackend", "BackendFactory", "TensileBackend"]
