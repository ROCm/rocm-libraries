# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the .amdhsa_user_sgpr_kernarg_preload ROCm version gate."""

from types import SimpleNamespace

import pytest

from Tensile.CustomKernels import supportsUserSgprKernargPreload
from Tensile.KernelWriterAssembly import KernelWriterAssembly as _KWA

pytestmark = pytest.mark.unit

PRELOAD_LENGTH_DIRECTIVE = ".amdhsa_user_sgpr_kernarg_preload_length 2\n"
PRELOAD_OFFSET_DIRECTIVE = ".amdhsa_user_sgpr_kernarg_preload_offset 0\n"
KERNEL_SOURCE = (
    "s_nop 0\n"
    + PRELOAD_LENGTH_DIRECTIVE
    + PRELOAD_OFFSET_DIRECTIVE
    + "s_endpgm\n"
)


def _rocm_version(major, patch, minor=0):
    return SimpleNamespace(major=major, minor=minor, patch=patch)


@pytest.mark.parametrize(
    "major, patch, supported",
    [
        (5, 99999, False),
        (6, 32649, False),
        # A locally built ROCm 6.x reporting a low build remains unsupported;
        # see supportsUserSgprKernargPreload's docstring for why.
        (6, 0, False),
        (6, 32650, True),
        (6, 40000, True),
        # ROCm 7+ always supports the directive, including builds whose patch
        # number is below the ROCm 6 cut-off.
        (7, 0, True),
        (7, 32649, True),
        (8, 1, True),
    ],
)
def test_supports_user_sgpr_kernarg_preload(major, patch, supported):
    assert supportsUserSgprKernargPreload(_rocm_version(major, patch)) is supported


def test_supports_user_sgpr_kernarg_preload_rejects_local_rocm_6_x_zero_build():
    assert supportsUserSgprKernargPreload(_rocm_version(6, 0, minor=4)) is False


def _make_kernel_writer(rocm_version):
    kwa = object.__new__(_KWA)
    kwa.debugConfig = SimpleNamespace(splitGSU=False)
    kwa.assembler = SimpleNamespace(rocm_version=rocm_version)
    kwa.states = SimpleNamespace()
    return kwa


def _write_custom_kernel(directory, name):
    (directory / (name + ".s")).write_text(KERNEL_SOURCE, encoding="utf-8")
    return {"CustomKernelName": name, "ISA": (9, 4, 2)}


@pytest.mark.parametrize("major, patch", [(5, 99999), (6, 32649)])
def test_custom_kernel_source_strips_preload_for_unsupported_rocm(tmp_path, major, patch):
    kernel = _write_custom_kernel(tmp_path, "kernel")
    kwa = _make_kernel_writer(_rocm_version(major, patch))

    code = kwa._getCustomKernelSource(kernel, str(tmp_path))

    assert "amdhsa_user_sgpr_kernarg_preload" not in code
    assert PRELOAD_LENGTH_DIRECTIVE not in code
    assert PRELOAD_OFFSET_DIRECTIVE not in code
    assert code == "s_nop 0\ns_endpgm\n"


@pytest.mark.parametrize("major, patch", [(6, 32650), (7, 0)])
def test_custom_kernel_source_keeps_preload_for_supported_rocm(tmp_path, major, patch):
    kernel = _write_custom_kernel(tmp_path, "kernel")
    kwa = _make_kernel_writer(_rocm_version(major, patch))

    code = kwa._getCustomKernelSource(kernel, str(tmp_path))

    assert PRELOAD_LENGTH_DIRECTIVE in code
    assert PRELOAD_OFFSET_DIRECTIVE in code
    assert code == KERNEL_SOURCE
