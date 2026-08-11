# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for convolution backward weight gradient: plan-building and stubbed execution."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .helpers import (
    call_attribute_methods,
    build_all_plans,
    create_float_graph,
    execute_zeros,
)


def build_conv_wgrad_graph(
    n=16, c=16, h=16, w=16, k=16, r=3, s=3, stride=1, pad=1, dil=1
):
    """Build a conv_wgrad graph returning (graph, dy, x, dw)."""
    out_h = (h + 2 * pad - dil * (r - 1) - 1) // stride + 1
    out_w = (w + 2 * pad - dil * (s - 1) - 1) // stride + 1

    graph = create_float_graph()
    graph.set_name("conv_wgrad_test")

    dy = hipdnn.Tensor.create([n, k, out_h, out_w], hipdnn.DataType.FLOAT)
    dy.set_name("output_gradient_dy")

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("input_x")

    conv_attrs = hipdnn.ConvWgradAttributes()
    conv_attrs.set_name("conv_wgrad_node")
    conv_attrs.set_pre_padding([pad, pad])
    conv_attrs.set_post_padding([pad, pad])
    conv_attrs.set_stride([stride, stride])
    conv_attrs.set_dilation([dil, dil])

    dw = graph.conv_wgrad(dy, x, conv_attrs)
    dw.set_dim([k, c, r, s])
    dw.set_name("weight_gradient_dw")
    dw.set_output(True)

    return graph, dy, x, dw


@pytest.mark.gpu
class TestConvWgrad:
    """Tests for convolution backward weight gradient plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        """Conv_wgrad builds plans and executes against the stub engine without erroring."""
        graph, dy, x, dw = build_conv_wgrad_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(dy, np.float32), (x, np.float32), (dw, np.float32)],
            handle,
        )


class TestConvWgradAttributeBindings:
    """Every weight-gradient convolution attribute binding round-trips through its getter (no GPU required)."""

    def test_alias_identity(self):
        """ConvWgradAttributes is the same class as ConvolutionWgradAttributes."""
        assert hipdnn.ConvWgradAttributes is hipdnn.ConvolutionWgradAttributes

    def test_methods_are_callable(self):
        x = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        dy = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        dw = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.ConvWgradAttributes(),
            (
                ("set_name", ("conv_wgrad",), "get_name", "conv_wgrad"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_x", (x,), "get_x", x),
                ("set_dy", (dy,), "get_dy", dy),
                ("set_dw", (dw,), "get_dw", dw),
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
