################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""GEKO - GEMM Kernel Optimization Framework.

A comprehensive framework for optimizing General Matrix Multiply (GEMM) kernels.
Provides automated workflow for hipBLASLt kernel optimization,
benchmarking, and integration.

Modules:
    bench: Benchmark execution and performance analysis.
    optim: Optimization execution and configuration generation.
    search: Dense benchmarks to find the best kernel for a given GEMM.
    library: Solution library management and merging operations.
    utils: Common utilities and helper functions.
    constants: Data type definitions and field mappings.
    schemas: Structured data schemas for GEMM optimization workflows.
"""

import geko.bench
import geko.optim
import geko.utils
import geko.library
import geko.search
import logging

logger = logging.getLogger("GEKO")

FORMAT = "%(name)s:%(levelname)s [%(module)s:%(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def _set_log_level(verbose: int) -> None:
    if verbose <= 0:
        logger.setLevel(logging.WARNING)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
