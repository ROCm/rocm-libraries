# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import runpy
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py

_metadata = runpy.run_path(str(Path(__file__).with_name("release_metadata.py")))


def _build_rocm_version() -> str:
    value = os.environ.get("TENSILELITE_ROCM_VERSION")
    if not value:
        raise RuntimeError(
            "TENSILELITE_ROCM_VERSION=X.Y.Z is required to build a TensileLite wheel. "
            "Use the CMake or Invoke build frontend, or supply the selected SDK base version explicitly."
        )
    return value


class CleanBuildPy(build_py):
    """
    Regenerate ``build/lib`` so stale files cannot leak into a wheel.

    Setuptools does not prune files left by earlier package layouts or exclusions.
    """

    def run(self):
        build_lib = Path(self.build_lib)
        if build_lib.exists():
            shutil.rmtree(build_lib)
        super().run()


setup(
    version=_metadata["distribution_version"](_build_rocm_version()),
    install_requires=[
        "packaging",
        "pyyaml",
        "msgpack",
        "joblib>=1.4.0",
        "filelock",
        "numpy",
        "rocisa",
    ],
    cmdclass={"build_py": CleanBuildPy},
)
