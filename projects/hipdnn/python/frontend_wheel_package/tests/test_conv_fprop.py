# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for convolution forward propagation."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .helpers import (
    call_attribute_methods,
    build_all_plans,
    create_float_graph,
    execute_graph,
)


def build_conv_fprop_graph(
    n=16, c=16, h=16, w=16, k=16, r=3, s=3, stride=1, pad=1, dil=1
):
    """Build a conv_fprop graph returning (graph, x, weight, y)."""
    graph = create_float_graph()
    graph.set_name("conv_fprop_test")

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("input_x")

    weight = hipdnn.Tensor.create([k, c, r, s], hipdnn.DataType.FLOAT)
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


@pytest.mark.gpu
class TestConvFprop:
    """Tests for convolution forward propagation end-to-end pipeline."""

    def test_execution_produces_nonzero_output(self):
        """Full end-to-end conv_fprop: execute and verify non-zero output."""
        graph, x, weight, y = build_conv_fprop_graph()

        handle = build_all_plans(graph)

        x_data = np.random.uniform(0.0, 1.0, x.get_dim()).astype(np.float32)
        w_data = np.random.uniform(0.0, 1.0, weight.get_dim()).astype(np.float32)
        y_data = np.zeros(y.get_dim(), dtype=np.float32)

        tensor_data = {
            x.get_uid(): x_data,
            weight.get_uid(): w_data,
            y.get_uid(): y_data,
        }

        results = execute_graph(graph, tensor_data, handle)
        y_result = results[y.get_uid()]

        assert not np.all(y_result == 0), "Conv fprop output is all zeros"

    def test_execution_matches_hardcoded_values(self):
        """Conv_fprop on a hand-checked 3x3 input matches hardcoded output.

        Cross-correlation of a 3x3 input with an identity-diagonal 2x2 kernel
        (stride=1, pad=0): each output is x[i,j] + x[i+1,j+1].
        """
        graph, x, weight, y = build_conv_fprop_graph(
            n=1, c=1, h=3, w=3, k=1, r=2, s=2, stride=1, pad=0
        )

        handle = build_all_plans(graph)

        x_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32).reshape(
            x.get_dim()
        )
        w_data = np.array([[1, 0], [0, 1]], dtype=np.float32).reshape(weight.get_dim())
        y_data = np.zeros(y.get_dim(), dtype=np.float32)

        tensor_data = {
            x.get_uid(): x_data,
            weight.get_uid(): w_data,
            y.get_uid(): y_data,
        }

        results = execute_graph(graph, tensor_data, handle)
        y_result = results[y.get_uid()]

        expected = np.array([[6, 8], [12, 14]], dtype=np.float32).reshape(y.get_dim())
        np.testing.assert_allclose(y_result, expected, rtol=2e-3, atol=2e-3)


class TestConvFpropAttributeBindings:
    """Every forward-convolution attribute binding round-trips through its getter (no GPU required)."""

    def test_alias_identity(self):
        """ConvFpropAttributes is the same class as ConvolutionFpropAttributes."""
        assert hipdnn.ConvFpropAttributes is hipdnn.ConvolutionFpropAttributes

    def test_methods_are_callable(self):
        x = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        w = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        y = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.ConvFpropAttributes(),
            (
                ("set_name", ("conv_fprop",), "get_name", "conv_fprop"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_x", (x,), "get_x", x),
                ("set_w", (w,), "get_w", w),
                ("set_y", (y,), "get_y", y),
                ("set_padding", ([9],), "get_pre_padding", [9]),
                ("set_padding", ([9],), "get_post_padding", [9]),
                ("set_pre_padding", ([1],), "get_pre_padding", [1]),
                ("set_post_padding", ([2],), "get_post_padding", [2]),
                ("set_stride", ([3],), "get_stride", [3]),
                ("set_dilation", ([4],), "get_dilation", [4]),
                (
                    "set_convolution_mode",
                    (hipdnn.ConvolutionMode.CONVOLUTION,),
                    "get_convolution_mode",
                    hipdnn.ConvolutionMode.CONVOLUTION,
                ),
            ),
        )
