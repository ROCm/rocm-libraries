# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Self-bootstrap ``sys.path`` so this suite runs standalone.

The provider-owned generation tests need BOTH:

  * ``projects/composablekernel/python``  -> the ``ck_dsl`` DSL package
  * ``dnn-providers/ck-dsl-provider/python`` -> the ``ck_dsl_provider``
    package under test

Neither is installed in the POC checkout, so we compute both paths
relative to this file (mirroring the path computation the relocated 2c
test used) and prepend them to ``sys.path`` at collection time. This lets
the suite be invoked directly::

    .venv/bin/python -m pytest dnn-providers/ck-dsl-provider/python/tests

with no PYTHONPATH plumbing required from the caller.

This file deliberately does NOT import torch (nor anything that imports
torch): the whole point of relocating the coverage here is that it runs
with torch absent.
"""

from __future__ import annotations

import os
import sys

# tests/ -> python/ -> ck-dsl-provider/ -> dnn-providers/ -> <repo root>
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROVIDER_PYTHON = os.path.normpath(os.path.join(_THIS_DIR, ".."))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
_CK_DSL_PYTHON = os.path.normpath(
    os.path.join(_REPO_ROOT, "projects", "composablekernel", "python")
)


def _prepend(path: str) -> None:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


# ck_dsl first so ``import ck_dsl`` from inside the provider resolves to
# this checkout, then the provider package itself.
_prepend(_CK_DSL_PYTHON)
_prepend(_PROVIDER_PYTHON)
