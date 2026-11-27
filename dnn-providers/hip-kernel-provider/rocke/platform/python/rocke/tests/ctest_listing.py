# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The CTest entries registered in a directory, as ``ctest --show-only=json-v1`` lists them.

One home for reading a registration without running it: the runners and checks
that need entry names, commands, labels or properties ask CTest rather than
parsing ``CTestTestfile.cmake``. CTest omits an entry's ``command`` when it
cannot find the program, so list a built tree, with the test interpreter on
``PATH`` for script entries: the installed entries run a bare ``python``, so
list an installed tree with :func:`list_installed_ctest_entries`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from .platform_build import configure_platform

__all__ = [
    "GPU_RESOURCE_LOCK",
    "PROVIDER_CTEST_SUBDIR",
    "configure_installed_ctest",
    "list_ctest_entries",
    "list_installed_ctest_entries",
    "python_on_path_environment",
]

# The provider build installs rocKE's CTest entries into bin/<this>
# (ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR).
PROVIDER_CTEST_SUBDIR = "hip_kernel_provider"
# The CTest resource lock of every entry whose tests launch GPU work
# (_ROCKE_INSTALLED_GPU_RESOURCE_LOCK in platform/CMakeLists.txt).
GPU_RESOURCE_LOCK = "rocke_gpu"


def list_ctest_entries(
    test_dir: Path,
    *,
    config: str | None = None,
    name_pattern: str | None = None,
    env: Mapping[str, str] | None = None,
) -> list[dict]:
    """The json-v1 ``tests`` of ``test_dir``, optionally for one build
    ``config`` and only names matching the ``name_pattern`` regex (``-R``).
    ``env`` replaces the environment CTest evaluates the registration in."""
    command = ["ctest", "--test-dir", str(test_dir)]
    if config is not None:
        command += ["-C", config]
    command.append("--show-only=json-v1")
    if name_pattern is not None:
        command += ["-R", name_pattern]
    listed = subprocess.run(
        command, check=True, capture_output=True, text=True, env=env
    )
    return json.loads(listed.stdout)["tests"]


def python_on_path_environment(
    scratch: Path, environment: Mapping[str, str] | None = None
) -> dict[str, str]:
    """A copy of ``environment`` (default ``os.environ``) in which the bare
    ``python`` that installed entries run resolves: unchanged when ``python``
    is on its ``PATH``, else with a directory under ``scratch`` holding a
    ``python`` link to this interpreter put first on ``PATH``. Pass it as
    ``env`` to :func:`list_ctest_entries` to list an installed tree's Python
    entries with their commands where no ``python`` is on ``PATH`` (a venv
    used by its full path). The link is a symlink, so this is for POSIX hosts."""
    resolved = dict(os.environ if environment is None else environment)
    if shutil.which("python", path=resolved.get("PATH", "")) is None:
        interpreter_dir = scratch / "python-bin"
        interpreter_dir.mkdir(parents=True, exist_ok=True)
        link = interpreter_dir / "python"
        link.unlink(missing_ok=True)
        link.symlink_to(sys.executable)
        resolved["PATH"] = os.pathsep.join(
            part for part in (str(interpreter_dir), resolved.get("PATH")) if part
        )
    return resolved


def list_installed_ctest_entries(
    ctest_dir: Path,
    scratch: Path,
    *,
    config: str | None = None,
    name_pattern: str | None = None,
    env: Mapping[str, str] | None = None,
) -> list[dict]:
    """The entries of the installed test tree ``ctest_dir``, as
    :func:`list_ctest_entries` lists them, with the bare ``python`` of its
    Python entries resolved, so each lists its command: CTest evaluates the
    registration in :func:`python_on_path_environment` of ``env`` (default
    ``os.environ``), which may link the interpreter under ``scratch``."""
    return list_ctest_entries(
        ctest_dir,
        config=config,
        name_pattern=name_pattern,
        env=python_on_path_environment(scratch, env),
    )


def configure_installed_ctest(
    scratch: Path, *cmake_arguments: str
) -> tuple[Path, Path]:
    """Configure the platform source tree with BUILD_TESTING into
    ``scratch/build`` and return that build directory and ``scratch/installed-ctest``,
    a directory holding its installed CTestTestfile as an install lays it out.
    Nothing is built. A -D in ``cmake_arguments`` overrides the default
    ROCKE_INSTALL_TEST_GPU_REFERENCES=OFF, because CMake keeps the last value."""
    build = scratch / "build"
    configure_platform(
        build,
        "-DBUILD_TESTING=ON",
        "-DROCKE_INSTALL_TEST_GPU_REFERENCES=OFF",
        *cmake_arguments,
    )
    ctest_dir = scratch / "installed-ctest"
    ctest_dir.mkdir()
    shutil.copy2(
        build / "CTestTestfile.cmake.install", ctest_dir / "CTestTestfile.cmake"
    )
    return build, ctest_dir
