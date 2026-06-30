# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Pytest root config for the rocKE engine test tree. Puts the Python engine
# package root (rocKE/Python) on sys.path so `import rocke` resolves without an
# external PYTHONPATH. Path is derived from this file's location (relative), so
# the tree stays copy-able verbatim into another repo.

import sys
from pathlib import Path

_ROCKE = Path(__file__).resolve().parents[1]  # tests -> rocKE
_PYROOT = _ROCKE / "Python"
if str(_PYROOT) not in sys.path:
    sys.path.insert(0, str(_PYROOT))
