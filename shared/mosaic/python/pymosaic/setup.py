# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build shim for pymosaic.

All static metadata lives in ``pyproject.toml``. The one dynamic step
that needs code is *vendoring the mosaic C++ headers* into the package
tree so the JIT can find ``<mosaic/*.hpp>`` after a wheel install (when
the surrounding repo is no longer on disk).

Source of truth for the headers is the repo-root ``include/`` directory.
We copy ``include/mosaic`` into ``pymosaic/_include/mosaic`` on every
build. When building from an sdist that already carries ``_include``
(the repo is not present), we keep the bundled copy.

The vendored copy is intentionally git-ignored — it is a build artifact,
not source.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup

_HERE = Path(__file__).parent.resolve()
_PKG_INCLUDE = _HERE / "pymosaic" / "_include"


def _vendor_headers() -> None:
    # python/pymosaic -> python -> <repo root>/include
    repo_include = _HERE.parent.parent / "include"
    src = repo_include / "mosaic"

    if src.is_dir():
        dst = _PKG_INCLUDE / "mosaic"
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        return

    # No repo checkout next to us: rely on a pre-vendored copy (sdist).
    if not (_PKG_INCLUDE / "mosaic").is_dir():
        raise SystemExit(
            "pymosaic build error: could not locate the mosaic C++ headers.\n"
            f"  expected at: {src}\n"
            f"  or vendored: {_PKG_INCLUDE / 'mosaic'}\n"
            "Install from the full repo (the headers live in <repo>/include/mosaic)."
        )


_vendor_headers()

setup()
