################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Shared path resolution for characterization suites.

Several suites address shipped source/data files as ``Tensile/...`` paths
relative to the process CWD. The tox ``unit``/``coverage-unit`` environments run
from the package root, so those resolve; the installed-artifact test tree
(``share/hipblaslt/tensilelite``, run by TheRock) is driven from the CI
workspace root, where the same relative paths raise ``FileNotFoundError``.

Resolve such paths CWD-first (preserving current behavior) and fall back to a
location anchored at the installed ``Tensile`` package, so the suites work
regardless of the process CWD.

This is *not* a test module (no ``test_`` prefix, not collected).
"""

from pathlib import Path


def tensile_tree_root() -> Path:
    """Return the directory that contains the installed ``Tensile`` package."""
    import Tensile

    return Path(Tensile.__file__).resolve().parent.parent


def resolve_tensile_path(relpath) -> Path:
    """Resolve a ``Tensile/...``-relative path, CWD-first then package-anchored.

    Absolute paths and paths that exist relative to the CWD are returned as-is.
    Otherwise the path is re-anchored at :func:`tensile_tree_root`; if that
    exists it is returned, else the original path is returned so callers raise
    the familiar ``FileNotFoundError``.
    """
    p = Path(relpath)
    if p.is_absolute() or p.exists():
        return p
    anchored = tensile_tree_root() / relpath
    return anchored if anchored.exists() else p
