# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Pytest root config for the rocKE library test tree. Puts BOTH the library
# package root (exposing `kernels`/`builders`/`dispatch`) and the platform
# Python engine root (exposing `rocke`) on sys.path so the attention tests
# resolve without an external PYTHONPATH. The library legally depends on the
# platform SDK (one-way rule: library -> platform); the reverse is forbidden.
#
# Paths are derived from this file's location so the tree stays copy-able
# verbatim, and are probed against BOTH layouts this tree runs in:
#
#   source tree     : rocke/library/tests/conftest.py
#     kernels -> parent (rocke/library) ; rocke -> rocke/platform/python
#   installed CI lane: <prefix>/tests/library/conftest.py  (packages co-located)
#     kernels -> this dir                ; rocke -> <prefix> (also on PYTHONPATH)

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# `kernels`/`builders`/`dispatch`: co-located with this conftest in the installed
# lane, or one level up (rocke/library) in the source tree.
for _cand in (_HERE, _HERE.parent):
    if (_cand / "kernels").is_dir():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

# `rocke` (platform engine): at <prefix>/rocke in the installed lane, or
# rocke/platform/python in the source tree (both share the parents[1] base:
# installed parents[1] = <prefix>; source parents[1] = the rocke root).
for _cand in (_HERE.parents[1], _HERE.parents[1] / "platform" / "python"):
    if (_cand / "rocke").is_dir():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break
