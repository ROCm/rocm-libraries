# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Benchmark execution and performance analysis module.

Provides functionality for:
- Running hipBLASLt benchmarks.
- Parsing benchmark output files.
- Comparing performance between reference and optimized kernels.
- Log file processing and analysis.
"""

from .bench import *
from . import utils
from . import log
