# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithExtension(build_py):
    """Copy pre-built nanobind extension into the package before building."""

    def run(self):
        super().run()

        ext_dir = os.environ.get("HIPDNN_EXT_DIR", "")
        if not ext_dir:
            return

        ext_path = Path(ext_dir)
        extensions = glob.glob(str(ext_path / "hipdnn_frontend_python*"))
        if not extensions:
            raise RuntimeError(
                f"No hipdnn_frontend_python extension found in {ext_path}"
            )

        pkg_dir = Path(self.build_lib) / "hipdnn_frontend"
        pkg_dir.mkdir(parents=True, exist_ok=True)

        for ext in extensions:
            shutil.copy2(ext, pkg_dir / Path(ext).name)


setup(cmdclass={"build_py": BuildPyWithExtension})
