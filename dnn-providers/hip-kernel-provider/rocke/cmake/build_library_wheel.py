#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build a rocke wheel from a temporary copy of its source directory.

setuptools builds in place and writes build/ and *.egg-info next to the
sources. Run directly on the checkout, that output dirties a shared or
read-only source tree, can collide between concurrent builds, matches the
CONFIGURE_DEPENDS globs in rocke/CMakeLists.txt (forcing a CMake re-run on the
next build), and lets stale files in build/lib (e.g. a deleted module) leak
into the wheel. Building from a fresh copy in a unique temporary directory
avoids all of these; the directory is removed even if the build fails.

Used for both the library and the platform wheel.

Usage: build_library_wheel.py <source-dir> <wheel-dir> [<entry> ...]

With no <entry>, the whole source directory is copied. Otherwise only the
named top-level entries (files or directories) are copied, which keeps the
platform wheel from copying the C++ engine, docs and tests it does not ship.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Top-level entries of library/ that never belong in the wheel: in-place build
# output left by older builds or editable installs, and the (unpackaged) tests.
_SKIP_TOP_LEVEL = {"build", "tests"}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    src = Path(sys.argv[1]).resolve()
    wheel_dir = sys.argv[2]
    entries = sys.argv[3:]

    def ignore(directory: str, names: list[str]) -> set[str]:
        skip = {n for n in names if n.endswith(".egg-info")}
        if Path(directory).resolve() == src:
            skip |= {n for n in names if n in _SKIP_TOP_LEVEL}
        return skip

    with tempfile.TemporaryDirectory(prefix="rocke-wheel-") as tmp:
        stage = Path(tmp) / src.name
        if entries:
            stage.mkdir()
            for entry in entries:
                source = src / entry
                if source.is_dir():
                    shutil.copytree(source, stage / entry, ignore=ignore)
                else:
                    shutil.copy2(source, stage / entry)
        else:
            shutil.copytree(src, stage, ignore=ignore)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                wheel_dir,
                str(stage),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
