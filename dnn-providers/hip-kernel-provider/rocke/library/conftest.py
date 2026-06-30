# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pytest root config for the rocke/library tree (pytest invoked from library root).

Inserts both package roots at the front of sys.path so imports resolve
without an external PYTHONPATH.  Paths are derived from this file's location
(relative), making the tree copy-safe into another repo.

parents[N] math (from rocke/library/conftest.py):
  parents[0] = rocke/library          <- library source root
  parents[1] = rocke
  parents[1] / platform / Python      <- platform package root
"""

import sys
from pathlib import Path

try:
    # Normal case: loaded by pytest, __file__ is set.
    #   parent of library/conftest.py -> rocke/library (library root)
    _LIB_ROOT = Path(__file__).resolve().parent
except NameError:
    # exec() context: CWD must be rocke/library.
    _LIB_ROOT = Path.cwd().resolve()

_PLATFORM_ROOT = _LIB_ROOT.parent / "platform" / "Python"

for _root in (_LIB_ROOT, _PLATFORM_ROOT):
    _s = str(_root)
    if _s not in sys.path:
        sys.path.insert(0, _s)
