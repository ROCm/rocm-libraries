# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Filesystem locators for non-package rocke assets (data / utility trees).

This module holds the *only* path derivations left in the rocke Python tree.
Everything else resolves through ordinary imports: the ``rocke`` package and the
library packages (``kernels`` / ``builders`` / ``dispatch``) are editable-installed
into the build/dev environment, so no script touches ``sys.path`` to find them
(see rocke/BUILDING.md).

Some assets -- notably ``dsl_docs/`` and the loose ``_ua_shape_utils`` benchmark
helper -- live under the platform *source* root, outside the importable ``rocke``
package, so they cannot be reached with ``importlib.resources`` and need an
explicit locator.  Each locator honours an environment override so the trees can
be relocated (e.g. consumed out of the source checkout).
"""

import os
from pathlib import Path

# This file: rocke/platform/python/rocke/assets.py
#   parents[0] = rocke   parents[1] = python   parents[2] = platform
_PLATFORM_ROOT = Path(__file__).resolve().parents[2]


def platform_root() -> Path:
    """The ``rocke/platform`` source root (parent of ``python/``, ``dsl_docs/``, ``cpp/``).

    Override with ``ROCKE_PLATFORM_ROOT``.
    """
    override = os.environ.get("ROCKE_PLATFORM_ROOT")
    return Path(override) if override else _PLATFORM_ROOT


def dsl_docs_dir() -> Path:
    """The ``dsl_docs`` documentation + utilities tree.

    Override with ``ROCKE_DSL_DOCS``.
    """
    override = os.environ.get("ROCKE_DSL_DOCS")
    return Path(override) if override else platform_root() / "dsl_docs"


def shape_utils_dir() -> Path:
    """Directory containing the loose ``_ua_shape_utils`` benchmark helper module."""
    return dsl_docs_dir() / "optimization" / "utilities" / "tools" / "stage1_benchmark"
