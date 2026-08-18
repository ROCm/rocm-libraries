# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import runpy
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


_metadata = runpy.run_path(str(Path(__file__).resolve().parents[1] / "release_metadata.py"))
_version = _metadata["distribution_version"](os.environ.get("ROCM_PATH", "/opt/rocm"))


class CleanBuildPy(build_py):
    """Prevent stale package files in build/lib from leaking into wheels."""

    def run(self):
        build_lib = Path(self.build_lib)
        if build_lib.exists():
            shutil.rmtree(build_lib)
        super().run()


setup(
    version=_version,
    install_requires=[f"tensilelite=={_version}"],
    cmdclass={"build_py": CleanBuildPy},
)
