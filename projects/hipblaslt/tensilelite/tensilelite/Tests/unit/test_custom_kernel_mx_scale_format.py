# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MX scale layout of the checked-in custom kernels.

hipBLASLt reads MXScaleFormat to decide which scale modes a kernel may serve:
packed scales take NoSwizzle or InMemorySwizzle, Block_32_UE8M0_32_8_EXT takes
HostPreSwizzle. A handwritten kernel with MX scales must declare the layout it
reads, and rocRoller names a kernel that reads pre-swizzled scales with
PSTA_/PTA_/PSTB_/PTB_.
"""

from pathlib import Path

import pytest

import tensilelite
from tensilelite.CustomKernels import getCustomKernelConfig, readCustomKernelConfig

pytestmark = pytest.mark.unit

_PRESWIZZLE_TAGS = ("_PSTA_", "_PTA_", "_PSTB_", "_PTB_")


def _mx_custom_kernels():
    root = Path(tensilelite.__file__).parent / "CustomKernels"
    names = []
    for path in sorted(root.rglob("*.s")):
        try:
            problemType = readCustomKernelConfig(path.stem).get("ProblemType", {})
        except RuntimeError:
            continue
        if problemType.get("MXBlockA", 0) or problemType.get("MXBlockB", 0):
            names.append(path.stem)
    return names


def test_mx_custom_kernels_are_found():
    names = _mx_custom_kernels()
    assert any(name.startswith("RR_GEMM_") for name in names)
    assert any(not name.startswith("RR_GEMM_") for name in names)


@pytest.mark.parametrize("name", _mx_custom_kernels())
def test_mx_custom_kernel_declares_its_layout(name):
    layout = getCustomKernelConfig(name, {})["MXScaleFormat"]
    if name.startswith("RR_GEMM_"):
        expected = (
            "HostPreSwizzle" if any(tag in name for tag in _PRESWIZZLE_TAGS) else "NoSwizzle"
        )
        assert layout == expected, f"{name} declares {layout}"
