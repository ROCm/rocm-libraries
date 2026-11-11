# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import runpy
import shutil

from email.parser import Parser

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info
from setuptools.command.sdist import sdist

_metadata = runpy.run_path(str(Path(__file__).with_name("release_metadata.py")))
_BUILD_INFO = os.path.join("tensilelite", "_build_info.py")

# Resolved once per setup.py run. pip runs setup.py several times for one
# install; the resolver gives the same answer each time for the same tree.
_REVISION = _metadata["source_revision"]()


def _distribution_version():
    """Return this build's version.

    Only TheRock's CMake build passes a ROCm identity (TENSILELITE_ROCM_VERSION);
    every other build is a development build labelled with its commit. An
    unpacked sdist keeps the version it was made with, so a wheel built from it
    carries the same identity without the variable being passed again.
    """
    identity = os.environ.get("TENSILELITE_ROCM_VERSION") or None
    version = _metadata["distribution_version"](_REVISION, identity)
    pkg_info = Path(__file__).with_name("PKG-INFO")
    if not pkg_info.is_file():
        return version
    recorded = Parser().parsestr(pkg_info.read_text(encoding="utf-8"))["Version"]
    if identity is not None and recorded != version:
        raise RuntimeError(
            f"TENSILELITE_ROCM_VERSION={identity} gives {version}, but this sdist was made "
            f"as {recorded}"
        )
    return recorded


def _write_build_info(path):
    # The file may be a hard link to the source tree (sdist release trees are
    # linked when possible), so replace it rather than writing through it.
    if os.path.lexists(path):
        os.unlink(path)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(_metadata["render_build_info"](_REVISION))


class CleanBuildPy(build_py):
    """
    Regenerate ``build/lib`` so stale files cannot leak into a wheel.

    Setuptools does not prune files left by earlier package layouts or exclusions.

    Also stamps the source revision into ``build_lib``, never into the source
    tree. Editable installs get no stamp; they compute the revision from git
    when they need it, so it cannot go stale.
    """

    def run(self):
        build_lib = Path(self.build_lib)
        if build_lib.exists():
            shutil.rmtree(build_lib)
        super().run()
        if not self.editable_mode:
            _write_build_info(os.path.join(self.build_lib, _BUILD_INFO))

    def get_outputs(self, include_bytecode=True):
        outputs = super().get_outputs(include_bytecode)
        if self.editable_mode:
            return outputs
        stamp = os.path.join(self.build_lib, _BUILD_INFO)
        return outputs if stamp in outputs else [*outputs, stamp]


class BuildEggInfo(egg_info):
    """Keep generated distribution metadata out of the source root."""

    def finalize_options(self):
        if self.egg_base is None:
            egg_base = Path(__file__).with_name("build") / "egg-info"
            egg_base.mkdir(parents=True, exist_ok=True)
            self.egg_base = str(egg_base)
        super().finalize_options()


class StampedSdist(sdist):
    """Save the source revision in the sdist, where git is no longer available."""

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        _write_build_info(os.path.join(base_dir, _BUILD_INFO))


setup(
    version=_distribution_version(),
    install_requires=[
        "packaging",
        "pyyaml",
        "msgpack",
        "joblib>=1.4.0",
        "filelock",
        "numpy",
        "pandas",
    ],
    cmdclass={"build_py": CleanBuildPy, "egg_info": BuildEggInfo, "sdist": StampedSdist},
)
