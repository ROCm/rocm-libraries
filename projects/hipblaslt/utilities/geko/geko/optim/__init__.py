# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GEMM Kernel Optimization module.

This module provides the integration framework for GEMM kernel optimization.
It includes optimization execution and progress-tracking utilities for kernel
optimization workflows. Architecture-specific optimization configuration is
now produced by geko.config_generator and consumed directly by
geko.optim.optim.configure.

Key components:
- optim: Main optimization workflow
- utils: Progress tracking and optimization status utilities

The optimization process works with Tensile configs generated from hipBLASLt logs
to generate high-performance GEMM kernels.
"""

from geko.optim.optim import *
from geko.optim import utils
