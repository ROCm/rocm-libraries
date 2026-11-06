# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MX scale layout of the checked-in rocRoller custom kernels.

Handwritten custom kernels skip Solution.assignDerivedParameters, so
MXScaleFormat keeps the value in custom.config, and Contractions maps a missing
or "Auto" value to NoSwizzle (0). hipBLASLt reads that field to decide which
scale modes a kernel may serve, so it gives such a kernel packed scales.
rocRoller names a kernel that reads pre-swizzled scales with PSTA_/PTA_/PSTB_/PTB_,
and that kernel must declare a swizzled layout.
"""

from pathlib import Path

import pytest

import tensilelite
from tensilelite.CustomKernels import readCustomKernelConfig

pytestmark = pytest.mark.unit

_PRESWIZZLE_TAGS = ("_PSTA_", "_PTA_", "_PSTB_", "_PTB_")
_SWIZZLED = ("HostPreSwizzle", "InMemorySwizzle")
_PACKED = ("Auto", "NoSwizzle")


def _rocroller_kernels():
    root = Path(tensilelite.__file__).parent / "CustomKernels"
    return sorted(path.stem for path in root.rglob("RR_GEMM_*.s"))


def test_rocroller_custom_kernels_are_found():
    assert _rocroller_kernels()


@pytest.mark.parametrize("name", _rocroller_kernels())
def test_rocroller_kernel_layout_matches_its_scales(name):
    layout = readCustomKernelConfig(name).get("MXScaleFormat", "Auto")
    if any(tag in name for tag in _PRESWIZZLE_TAGS):
        assert layout in _SWIZZLED, f"{name} reads pre-swizzled scales but declares {layout}"
    else:
        assert layout in _PACKED, f"{name} reads packed scales but declares {layout}"
