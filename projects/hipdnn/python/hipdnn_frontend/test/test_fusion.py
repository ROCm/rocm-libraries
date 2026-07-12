# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for operation fusion (conv + pointwise)."""

import pytest

import hipdnn_frontend as hipdnn

from .helpers import build_operation_graph, create_float_graph


def build_conv_relu_fusion_graph(
    n=1, c=2, h=8, w=8, k=4, r=3, s=3, stride=1, pad=1, dil=1
):
    """Build a fused conv_fprop -> ReLU graph returning (graph, x, weight, y).

    The convolution output is a virtual (intermediate) tensor consumed by the
    pointwise ReLU, so the two ops form a single fusable graph.
    """
    graph = create_float_graph()
    graph.set_name("conv_relu_fusion_test")

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("input_x")

    weight = hipdnn.Tensor.create([k, c, r, s], hipdnn.DataType.FLOAT)
    weight.set_name("weight")

    conv_attrs = hipdnn.ConvFpropAttributes()
    conv_attrs.set_name("conv_fprop_node")
    conv_attrs.set_padding([pad, pad])
    conv_attrs.set_stride([stride, stride])
    conv_attrs.set_dilation([dil, dil])

    conv_out = graph.conv_fprop(x, weight, conv_attrs)
    conv_out.set_name("conv_out")
    conv_out.set_is_virtual(True)

    relu_attrs = hipdnn.PointwiseAttributes()
    relu_attrs.set_name("relu_node")
    relu_attrs.set_mode(hipdnn.PointwiseMode.RELU_FWD)

    y = graph.pointwise(conv_out, relu_attrs)
    y.set_name("output_y")
    y.set_output(True)

    return graph, x, weight, y


@pytest.mark.gpu
class TestConvReluFusion:
    """Tests for a fused convolution + ReLU graph."""

    def test_fused_graph_builds_operation_graph(self):
        """A fused conv + ReLU graph lowers to a backend operation graph.

        Execution is intentionally not exercised: fusion coverage only needs to
        confirm the fused graph is buildable.
        """
        graph, x, weight, y = build_conv_relu_fusion_graph()

        build_operation_graph(graph)
