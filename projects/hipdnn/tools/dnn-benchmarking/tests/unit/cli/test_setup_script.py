# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for runtime setup script provider configure flags."""

from pathlib import Path


_SETUP_SH = Path(__file__).parents[3] / "setup.sh"


def _cmake_configure_block(script: str, marker: str) -> str:
    start = script.index(marker)
    end = script.index("cmake --build", start)
    return script[start:end]


def test_runtime_provider_setup_disables_developer_clang_checks() -> None:
    """Runtime plugin setup must not require clang-format or clang-tidy."""
    script = _SETUP_SH.read_text()

    hipblaslt_block = _cmake_configure_block(
        script, "Building and installing hipBLASLt provider"
    )
    assert "-DENABLE_CLANG_TIDY=OFF" in hipblaslt_block

    hip_kernel_block = _cmake_configure_block(
        script, "Building and installing hip-kernel-provider"
    )
    assert "-DENABLE_CLANG_FORMAT=OFF" in hip_kernel_block
    assert "-DENABLE_CLANG_TIDY=OFF" in hip_kernel_block
