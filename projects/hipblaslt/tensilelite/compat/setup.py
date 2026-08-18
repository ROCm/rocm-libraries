# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from email.parser import Parser
from pathlib import Path
import os
import runpy
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info
from setuptools.command.sdist import sdist


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_rocm_version() -> str:
    value = os.environ.get("TENSILELITE_ROCM_VERSION")
    if not value:
        raise RuntimeError(
            "TENSILELITE_ROCM_VERSION=X.Y.Z is required to build a TensileLite wheel. "
            "Use the CMake or Invoke build frontend, or supply the selected SDK base version explicitly."
        )
    return value


def _distribution_version() -> str:
    metadata_path = _PROJECT_ROOT / "release_metadata.py"
    if metadata_path.is_file():
        metadata = runpy.run_path(str(metadata_path))
        return metadata["distribution_version"](_build_rocm_version())

    package_info = Path(__file__).with_name("PKG-INFO")
    if package_info.is_file():
        value = Parser().parsestr(package_info.read_text(encoding="utf-8"))["Version"]
        if value:
            return value
    raise RuntimeError("compatibility source package has no release metadata")


_version = _distribution_version()


class CleanBuildPy(build_py):
    """Prevent stale package files in build/lib from leaking into wheels."""

    def run(self):
        build_lib = Path(self.build_lib)
        if build_lib.exists():
            shutil.rmtree(build_lib)
        super().run()


class BuildEggInfo(egg_info):
    """Keep generated distribution metadata out of the source root."""

    def finalize_options(self):
        if self.egg_base is None:
            egg_base = Path(__file__).with_name("build") / "egg-info"
            egg_base.mkdir(parents=True, exist_ok=True)
            self.egg_base = str(egg_base)
        super().finalize_options()


class SelfContainedSdist(sdist):
    """Copy shared release metadata into the compatibility source archive."""

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        release_root = Path(base_dir)
        for name in ("VERSION", "release_metadata.py"):
            shutil.copy2(_PROJECT_ROOT / name, release_root / name)


setup(
    version=_version,
    install_requires=[f"tensilelite=={_version}"],
    cmdclass={
        "build_py": CleanBuildPy,
        "egg_info": BuildEggInfo,
        "sdist": SelfContainedSdist,
    },
)
