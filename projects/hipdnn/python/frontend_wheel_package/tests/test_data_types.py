# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for non-FLOAT data type coverage on a conv_fprop graph."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .helpers import build_all_plans, execute_zeros

# numpy has no native bfloat16; a same-width uint16 zero buffer is bit-identical
# to a zeroed bfloat16 buffer, so it's a safe stand-in for an execute-only check.
_NUMPY_DTYPES = {
    hipdnn.DataType.HALF: np.float16,
    hipdnn.DataType.BFLOAT16: np.uint16,
}


def build_typed_conv_graph(
    data_type, n=1, c=2, h=8, w=8, k=4, r=3, s=3, stride=1, pad=1, dil=1
):
    """Build a conv_fprop graph whose tensors use the given data type.

    Compute stays FLOAT (the typical accumulation type); IO and intermediate
    tensors use data_type.
    """
    graph = hipdnn.Graph()
    graph.set_io_data_type(data_type)
    graph.set_intermediate_data_type(data_type)
    graph.set_compute_data_type(hipdnn.DataType.FLOAT)
    graph.set_name("typed_conv_fprop_test")

    x = hipdnn.Tensor.create([n, c, h, w], data_type)
    x.set_name("input_x")

    weight = hipdnn.Tensor.create([k, c, r, s], data_type)
    weight.set_name("weight")

    conv_attrs = hipdnn.ConvFpropAttributes()
    conv_attrs.set_name("conv_fprop_node")
    conv_attrs.set_padding([pad, pad])
    conv_attrs.set_stride([stride, stride])
    conv_attrs.set_dilation([dil, dil])

    y = graph.conv_fprop(x, weight, conv_attrs)
    y.set_name("output_y")
    y.set_output(True)

    return graph, x, weight, y


@pytest.mark.parametrize(
    "data_type",
    [hipdnn.DataType.HALF, hipdnn.DataType.BFLOAT16],
)
class TestNonFloatDataTypes:
    """Validation and lowering coverage for HALF and BFLOAT16."""

    def test_validates(self, data_type):
        """A conv graph with a non-FLOAT data type passes validation (no GPU)."""
        graph, _x, _weight, _y = build_typed_conv_graph(data_type)
        assert graph.validate().is_good()

    @pytest.mark.gpu
    def test_execution_succeeds(self, data_type):
        """A conv graph with a non-FLOAT data type executes against the stub engine."""
        graph, x, weight, y = build_typed_conv_graph(data_type)
        dtype = _NUMPY_DTYPES[data_type]

        handle = build_all_plans(graph)
        execute_zeros(graph, [(x, dtype), (weight, dtype), (y, dtype)], handle)


def test_operation_configuration_enums_are_importable():
    """Operation attribute enums expose their public values to Python."""
    assert hipdnn.ReductionMode.ADD.name == "ADD"
    assert hipdnn.ResampleMode.MAXPOOL.name == "MAXPOOL"
    assert hipdnn.PaddingMode.ZERO_PAD.name == "ZERO_PAD"
    assert hipdnn.NormFwdPhase.TRAINING.name == "TRAINING"
    assert hipdnn.DiagonalAlignment.BOTTOM_RIGHT.name == "BOTTOM_RIGHT"
    assert hipdnn.AttentionImplementation.UNIFIED.name == "UNIFIED"
    assert hipdnn.MoeGroupedMatmulMode.SCATTER.name == "SCATTER"
