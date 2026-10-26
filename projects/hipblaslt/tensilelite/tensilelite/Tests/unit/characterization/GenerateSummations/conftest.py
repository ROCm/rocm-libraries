################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
# Local conftest for GenerateSummations characterization tests.
#
# GenerateSummations.py does a module-level ``import pandas as pd``. pandas is NOT
# installed in the CPU-only characterization container (numpy IS — 2.x), so the
# module cannot be imported without it. We inject a MagicMock ONLY for pandas so
# the module imports and its pandas-free helpers (createLibraryForBenchmark) can be
# characterized. We deliberately do NOT touch numpy: numpy is a real, installed
# dependency used across the suite, and globally replacing it with a mock would
# corrupt other tests sharing this xdist worker process.
################################################################################
import sys
from unittest.mock import MagicMock

# pandas only — never numpy.
if "pandas" not in sys.modules:
    sys.modules["pandas"] = MagicMock()
