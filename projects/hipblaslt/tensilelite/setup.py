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
    """
    Return the ROCm identity encoded in the TensileLite wheel.

    CMake and Invoke pass ``TENSILELITE_ROCM_VERSION`` as the authoritative
    selected build identity, including TheRock's package identity.
    Tox needs a narrow bootstrap fallback while it installs the package,
    so it reads the selected ``ROCM_PATH/.info/version``.
    Other direct ``setup.py`` calls fail rather than silently tag a wheel
    from an ambient ROCm installation.
    """
    explicit_rocm_version = os.environ.get("TENSILELITE_ROCM_VERSION")
    if not explicit_rocm_version:
        raise RuntimeError(
            "TENSILELITE_ROCM_VERSION=X.Y.Z is required to build a TensileLite wheel. "
            "Use the CMake or Invoke build frontend, or supply the selected SDK base version explicitly."
        )
    return explicit_rocm_version


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
