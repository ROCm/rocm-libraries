# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Pytest root config for the rocKE engine test tree. Puts the Python engine
# package root (rocke/platform/Python) on sys.path so `import rocke` resolves
# without an external PYTHONPATH. Platform tests MUST NOT depend on the library
# (kernels/builders/dispatch): the attention test surface lives under
# rocke/library/tests with its own conftest. Paths are derived from this file's
# location (relative), so the tree stays copy-able verbatim into another repo.
#
# parents[1] -> rocke/platform  (rocKE root)

import sys
from pathlib import Path

_ROCKE = Path(__file__).resolve().parents[1]  # tests -> rocke/platform
_PYROOT = _ROCKE / "Python"
if str(_PYROOT) not in sys.path:
    sys.path.insert(0, str(_PYROOT))
