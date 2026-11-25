# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Install contract for rocKE embedded in the hip-kernel-provider.

In provider mode the only rocKE content a release ships is the plugin. Every
other rocKE install rule is either test content (under bin/hip_kernel_provider/)
or developer content (public headers, the ckc CMake package, static archives)
that no TheRock artifact component claims. These tests configure the rocKE
project the way the provider does and read the generated install scripts, so
they need CMake but no compiler build and no GPU.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from rocke.assets import platform_root
from rocke.tests.cmake_embedding import configure_embedded_rocke

pytestmark = pytest.mark.integration

PROVIDER_SUBDIR = "hip_kernel_provider"

# What the provider passes to rocke/ when HIPKERNELPROVIDER_ENABLE_TESTS=OFF.
TESTS_OFF = (
    "-DBUILD_TESTING=OFF",
    "-DROCKE_INSTALL_TESTS=OFF",
    "-DROCKE_INSTALL_PYTHON_TESTS=OFF",
    "-DROCKE_INSTALL_PLATFORM_PYTEST=OFF",
    "-DROCKE_INSTALL_PYTHON_PACKAGE=OFF",
    "-DROCKE_INSTALL_TEST_GPU_REFERENCES=OFF",
    "-DROCKE_INSTALL_DEVELOPMENT=OFF",
)


@pytest.fixture
def rocke() -> Path:
    root = platform_root().parent
    if not (root / "CMakeLists.txt").is_file():
        pytest.skip("rocKE source CMake project is absent from this test artifact")
    if shutil.which("cmake") is None:
        pytest.skip("cmake is required for the source configuration contract")
    return root


@pytest.fixture
def provider_cmake(rocke: Path) -> str:
    path = rocke.parent / "CMakeLists.txt"
    if not path.is_file():
        pytest.skip("hip-kernel-provider CMake project is absent from this checkout")
    return path.read_text()


def _configure(rocke: Path, build: Path, *options: str) -> str:
    """Configure rocke/ inside a wrapper project, as the provider embeds it, and
    return the text of every generated install script."""
    result = configure_embedded_rocke(
        rocke,
        build,
        "-DROCKE_BUILD_PYENV=OFF",
        # GNUInstallDirs picks lib64 on native 64-bit Fedora/RHEL; pin it so the
        # lib/... destinations asserted below do not depend on the host.
        "-DCMAKE_INSTALL_LIBDIR=lib",
        *options,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    scripts = sorted(build.rglob("cmake_install.cmake"))
    assert scripts
    return "\n".join(path.read_text() for path in scripts)


def _installs_under(install: str, directory: str) -> bool:
    """True when an install rule targets <prefix>/<directory> or below it."""
    return (
        re.search(
            rf'DESTINATION "\$\{{CMAKE_INSTALL_PREFIX\}}/{re.escape(directory)}(/|")',
            install,
        )
        is not None
    )


def test_provider_follows_tests_switch(provider_cmake: str) -> None:
    block = provider_cmake.split('message(STATUS "rocKE features enabled")', 1)[1]
    block = block.split("endblock()", 1)[0]
    for gate in (
        "ROCKE_INSTALL_TESTS",
        "ROCKE_INSTALL_PYTHON_TESTS",
        "ROCKE_INSTALL_PLATFORM_PYTEST",
        "ROCKE_INSTALL_PYTHON_PACKAGE",
    ):
        assert re.search(
            rf"set\({gate} \$\{{HIPKERNELPROVIDER_ENABLE_TESTS\}}\)", block
        ), f"{gate} must follow HIPKERNELPROVIDER_ENABLE_TESTS"
    assert "set(BUILD_TESTING ${HIPKERNELPROVIDER_ENABLE_TESTS})" in block
    assert "set(ROCKE_INSTALL_DEVELOPMENT OFF)" in block


def test_development_install_is_on_by_default(rocke: Path, tmp_path: Path) -> None:
    install = _configure(rocke, tmp_path / "build")

    assert "ckcTargets" in install
    assert _installs_under(install, "lib/cmake/ckc")
    assert _installs_under(install, "include")
    assert "TYPE STATIC_LIBRARY" in install


def test_development_install_can_be_disabled(rocke: Path, tmp_path: Path) -> None:
    install = _configure(rocke, tmp_path / "build", "-DROCKE_INSTALL_DEVELOPMENT=OFF")

    # Both halves of the shared ckcTargets export set go together; exporting a
    # target that is not installed would already have failed the configure.
    assert "ckcTargets" not in install
    assert "ckcConfig" not in install
    # The test payload still carries its own copy of librocke_core.a under bin/
    # and the standalone build still installs the Python package; only the
    # developer destinations must be gone.
    assert "TYPE STATIC_LIBRARY" not in install
    assert not _installs_under(install, "lib/cmake")
    assert not _installs_under(install, "include")


def test_provider_tests_off_installs_no_rocke_content(
    rocke: Path, tmp_path: Path
) -> None:
    install = _configure(
        rocke,
        tmp_path / "build",
        f"-DROCKE_INSTALL_PROVIDER_CTEST_SUBDIR={PROVIDER_SUBDIR}",
        "-DROCKE_INSTALL_STANDALONE_CTEST=OFF",
        *TESTS_OFF,
    )

    assert PROVIDER_SUBDIR not in install
    assert "ckcTargets" not in install
    assert "wheels" not in install
    assert not _installs_under(install, "bin")
    assert not _installs_under(install, "lib")
    assert not _installs_under(install, "include")
