# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.KernelWriter import KernelWriter

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "internal_support, expected",
    [
        ({"KernArgsVersion": 2}, "0x3fff"),
        ({"KernArgsVersion": 3}, "0xfff"),
        (
            {"KernArgsVersion": 2, "SupportDeviceScalarAlpha": True},
            "0x7ff",
        ),
        (
            {"KernArgsVersion": 3, "SupportDeviceScalarAlpha": True},
            "0x7ff",
        ),
    ],
)
def test_device_scalar_alpha_capability_owns_gsu_bit_11(internal_support, expected):
    kernel = {"InternalSupportParams": internal_support}
    assert KernelWriter.gsuMaskHex(None, kernel) == expected
