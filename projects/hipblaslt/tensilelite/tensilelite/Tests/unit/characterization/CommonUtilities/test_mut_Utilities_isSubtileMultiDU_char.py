# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Mutation-targeted tests for ``Tensile.Common.Utilities.isSubtileMultiDU``."""

import pytest

from Tensile.Common.Utilities import isSubtileMultiDU

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "kernel,expected",
    [
        ({"DepthU": 32}, False),
        ({"DepthU": 32, "_DepthUA": 32, "_DepthUB": 32}, False),
        ({"DepthU": 32, "_DepthUA": 16}, True),
        ({"DepthU": 32, "_DepthUB": 16}, True),
        ({"DepthU": 32, "_DepthUA": 64, "_DepthUB": 32}, False),
    ],
    ids=[
        "implicit-per-tensor-depths",
        "explicit-equal-depths",
        "a-depth-split",
        "b-depth-split",
        "larger-a-is-not-split",
    ],
)
def test_is_subtile_multi_du_depends_on_either_tensor_depth(kernel, expected):
    assert isSubtileMultiDU(kernel) is expected


def test_is_subtile_multi_du_requires_loop_depth_key():
    with pytest.raises(KeyError) as excinfo:
        isSubtileMultiDU({"_DepthUA": 16})

    assert excinfo.value.args == ("DepthU",)
