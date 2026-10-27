#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build the rocke-library wheel from a temporary copy of library/.

setuptools builds in place and writes build/lib/**/*.py next to the sources.
Run directly on library/, that output matches the CONFIGURE_DEPENDS glob in
rocke/CMakeLists.txt and forces a CMake re-run on the next build, and stale
files in build/lib (e.g. a deleted module) can leak into the wheel. Building
from a fresh copy in a unique temporary directory avoids both; the directory
is removed even if the build fails.

Usage: build_library_wheel.py <library-dir> <wheel-dir>
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
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    src = Path(sys.argv[1]).resolve()
    wheel_dir = sys.argv[2]

    def ignore(directory: str, names: list[str]) -> set[str]:
        if Path(directory).resolve() != src:
            return set()
        return {n for n in names if n in _SKIP_TOP_LEVEL or n.endswith(".egg-info")}

    with tempfile.TemporaryDirectory(prefix="rocke-library-wheel-") as tmp:
        stage = Path(tmp) / "library"
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
