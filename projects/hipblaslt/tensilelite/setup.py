# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import runpy
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py

_metadata = runpy.run_path(str(Path(__file__).with_name("release_metadata.py")))


class CleanBuildPy(build_py):
    """Prevent ignored/stale packages in build/lib from leaking into wheels."""

    def run(self):
        build_lib = Path(self.build_lib)
        if build_lib.exists():
            shutil.rmtree(build_lib)
        super().run()


setup(
    version=_metadata["distribution_version"]("5.0.0"),
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
