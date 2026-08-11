# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for convolution backward data gradient: plan-building and stubbed execution."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .helpers import (
    call_attribute_methods,
    build_all_plans,
    create_float_graph,
    execute_zeros,
)


def build_conv_dgrad_graph(
    n=16, c=16, h=16, w=16, k=16, r=3, s=3, stride=1, pad=1, dil=1
):
    """Build a conv_dgrad graph returning (graph, dy, weight, dx)."""
    out_h = (h + 2 * pad - dil * (r - 1) - 1) // stride + 1
    out_w = (w + 2 * pad - dil * (s - 1) - 1) // stride + 1

    graph = create_float_graph()
    graph.set_name("conv_dgrad_test")

    dy = hipdnn.Tensor.create([n, k, out_h, out_w], hipdnn.DataType.FLOAT)
    dy.set_name("output_gradient_dy")

    weight = hipdnn.Tensor.create([k, c, r, s], hipdnn.DataType.FLOAT)
    weight.set_name("weight")

    conv_attrs = hipdnn.ConvDgradAttributes()
    conv_attrs.set_name("conv_dgrad_node")
    conv_attrs.set_pre_padding([pad, pad])
    conv_attrs.set_post_padding([pad, pad])
    conv_attrs.set_stride([stride, stride])
    conv_attrs.set_dilation([dil, dil])

    dx = graph.conv_dgrad(dy, weight, conv_attrs)
    dx.set_dim([n, c, h, w])
    dx.set_name("input_gradient_dx")
    dx.set_output(True)

    return graph, dy, weight, dx


@pytest.mark.gpu
class TestConvDgrad:
    """Tests for convolution backward data gradient plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        """Conv_dgrad builds plans and executes against the stub engine without erroring."""
        graph, dy, weight, dx = build_conv_dgrad_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(dy, np.float32), (weight, np.float32), (dx, np.float32)],
            handle,
        )


class TestConvDgradAttributeBindings:
    """Every data-gradient convolution attribute binding round-trips through its getter (no GPU required)."""

    def test_alias_identity(self):
        """ConvDgradAttributes is the same class as ConvolutionDgradAttributes."""
        assert hipdnn.ConvDgradAttributes is hipdnn.ConvolutionDgradAttributes

    def test_methods_are_callable(self):
        dy = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        w = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        dx = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.ConvDgradAttributes(),
            (
                ("set_name", ("conv_dgrad",), "get_name", "conv_dgrad"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_dy", (dy,), "get_dy", dy),
                ("set_w", (w,), "get_w", w),
                ("set_dx", (dx,), "get_dx", dx),
                ("set_pre_padding", ([1],), "get_pre_padding", [1]),
                ("set_post_padding", ([2],), "get_post_padding", [2]),
                ("set_stride", ([3],), "get_stride", [3]),
                ("set_dilation", ([4],), "get_dilation", [4]),
                ("set_padding", ([5],), "get_pre_padding", [5]),
                ("set_padding", ([5],), "get_post_padding", [5]),
                (
                    "set_convolution_mode",
                    (hipdnn.ConvolutionMode.CROSS_CORRELATION,),
                    "get_convolution_mode",
                    hipdnn.ConvolutionMode.CROSS_CORRELATION,
                ),
            ),
        )
