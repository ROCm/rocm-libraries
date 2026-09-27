#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Entry point for `python -m uhd_gen.dataset`.

Its own entry point rather than a subcommand of `python -m uhd_gen`, which is what the rest of
the pipeline is reached through. Publishing a dataset is arithmetic over a declaration and a row:
it needs pandas and a Parquet engine, and nothing else (see requirements.txt beside this file).
`uhd_gen/__main__.py` imports numpy, lightgbm and flatbuffers at module scope because every other
stage genuinely needs them -- routing this one through it would put the whole training stack
between a sweep machine and the dataset it collected.

Importing `uhd_gen.dataset` costs only `uhd_gen/__init__.py`, which is stdlib.
"""
from __future__ import annotations

import sys

from .publish import main

if __name__ == "__main__":
    sys.exit(main())
