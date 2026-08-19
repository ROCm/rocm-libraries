# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import runpy
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


_metadata = runpy.run_path(str(Path(__file__).resolve().parents[1] / "release_metadata.py"))


def _build_rocm_version() -> str:
    value = os.environ.get("TENSILELITE_ROCM_VERSION")
    if not value:
        raise RuntimeError(
            "TENSILELITE_ROCM_VERSION is required to build a TensileLite wheel. "
            "Use the CMake or Invoke build frontend, or supply the selected SDK identity explicitly."
        )
    return value


_version = _metadata["distribution_version"](_build_rocm_version())


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
