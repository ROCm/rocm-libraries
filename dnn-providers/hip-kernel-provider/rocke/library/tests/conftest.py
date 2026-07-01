# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Pytest root config for the rocKE library test tree. Puts BOTH the library
# source root (rocke/library, exposing `kernels`/`builders`/`dispatch`) and the
# platform Python engine root (rocke/platform/Python, exposing `rocke`) on
# sys.path so the attention tests resolve without an external PYTHONPATH. The
# library legally depends on the platform SDK (one-way rule: library -> platform);
# the reverse is forbidden. Paths are derived from this file's location so the
# tree stays copy-able verbatim into another repo.
#
# parents[1] -> rocke/library
# parents[2] -> rocke

import sys
from pathlib import Path

_LIBROOT = Path(__file__).resolve().parents[1]  # tests -> rocke/library
if str(_LIBROOT) not in sys.path:
    sys.path.insert(0, str(_LIBROOT))

_PYROOT = Path(__file__).resolve().parents[2] / "platform" / "Python"
if str(_PYROOT) not in sys.path:
    sys.path.insert(0, str(_PYROOT))
