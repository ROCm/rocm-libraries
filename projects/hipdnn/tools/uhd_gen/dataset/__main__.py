#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Entry point for `python -m uhd_gen.dataset`.

Separate from `python -m uhd_gen`, whose `__main__` imports the training stack at module
scope; publishing needs only pandas and a Parquet engine.
"""
from __future__ import annotations

import sys

from .publish import main

if __name__ == "__main__":
    sys.exit(main())
