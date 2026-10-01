# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for StreamK._depthUForTc MX scale format gating.

HostPreSwizzle / InMemorySwizzle pair KernelWriter's <<5 StridesMXS scaling with
a *32 on _DepthUMXS* so StreamKLocalStart offsets stay in data-K units.
NoSwizzle keeps canonical scale strides; applying *32 there over-
advances scale SRDs for any nonzero StreamKLocalStart.
"""

import pytest

# Prime the component registry before StreamK imports (avoids circular import).
from Tensile.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401

from Tensile.Components.StreamK import StreamK

pytestmark = pytest.mark.unit


def _mx_kernel(mx_scale_format, *, use_subtile=True, depth_u=256, depth_u_mxs=8):
    return {
        "DepthU": depth_u,
        "_DepthUMXSA": depth_u_mxs,
        "_DepthUMXSB": depth_u_mxs,
        "UseSubtileImpl": use_subtile,
        "MXScaleFormat": mx_scale_format,
        "ProblemType": {"Sparse": False},
    }


@pytest.mark.parametrize("fmt", ["HostPreSwizzle", "InMemorySwizzle"])
@pytest.mark.parametrize("tc", ["MXSA", "MXSB"])
def test_depthu_for_tc_swizzled_applies_x32(fmt, tc):
    """Swizzled layouts recover a DepthU-sized StreamK K-step (8 * 32 = 256)."""
    kernel = _mx_kernel(fmt)
    assert StreamK._depthUForTc(kernel, tc) == 256


@pytest.mark.parametrize("tc", ["MXSA", "MXSB"])
def test_depthu_for_tc_noswizzle_uses_canonical(tc):
    """NoSwizzle must not apply the HostPreSwizzle *32 under UseSubtileImpl."""
    kernel = _mx_kernel("NoSwizzle")
    assert StreamK._depthUForTc(kernel, tc) == 8


@pytest.mark.parametrize("fmt", ["HostPreSwizzle", "InMemorySwizzle", "NoSwizzle"])
@pytest.mark.parametrize("tc", ["MXSA", "MXSB"])
def test_depthu_for_tc_without_subtile_never_scales(fmt, tc):
    """Without UseSubtileImpl the *32 contract does not apply."""
    kernel = _mx_kernel(fmt, use_subtile=False)
    assert StreamK._depthUForTc(kernel, tc) == 8


def test_depthu_for_tc_data_tensor_uses_full_depthu():
    kernel = _mx_kernel("NoSwizzle")
    assert StreamK._depthUForTc(kernel, "A") == 256
    assert StreamK._depthUForTc(kernel, "B") == 256


def test_depthu_for_tc_missing_mxs_key_falls_back_to_depthu():
    kernel = {
        "DepthU": 256,
        "UseSubtileImpl": True,
        "MXScaleFormat": "HostPreSwizzle",
        "ProblemType": {"Sparse": False},
    }
    assert StreamK._depthUForTc(kernel, "MXSA") == 256
