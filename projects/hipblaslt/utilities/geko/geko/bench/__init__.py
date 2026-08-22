# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Benchmark execution and performance analysis module.

This module is a thin wrapper around the hipblaslt-bench binary. It does not
implement benchmarking itself; it invokes the binary via subprocess,
parses its output, and provides utilities for reference-vs-tuned comparisons.

Provides functionality for:
- Running hipBLASLt benchmarks.
- Parsing benchmark output files.
- Comparing performance between reference and optimized kernels.
- Log file processing and analysis.
"""

from .bench import *
from . import utils
from . import log
