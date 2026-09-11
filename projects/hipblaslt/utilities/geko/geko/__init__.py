# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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
