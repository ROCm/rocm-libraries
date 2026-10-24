# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Make the ``pymosaic`` package importable from the source tree without
requiring an editable install.

Pytest's rootdir is the repo root (via ``pytest.ini``).
The standalone JIT package lives under ``python/pymosaic/pymosaic/``. We
insert ``python/pymosaic/`` at the front of ``sys.path`` so
``import pymosaic`` resolves to the source files.

The source path is inserted at the *front* of ``sys.path``, so it takes
precedence over any installed (including editable) ``pymosaic`` — tests
always run against the working copy. (If the path is already present it
isn't re-added, leaving its existing priority intact.)
"""

import sys
from pathlib import Path

_PYMOSAIC_DIR = Path(__file__).resolve().parent.parent / "pymosaic"
if str(_PYMOSAIC_DIR) not in sys.path:
    sys.path.insert(0, str(_PYMOSAIC_DIR))
