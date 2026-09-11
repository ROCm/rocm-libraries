# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for convolution forward propagation: plan-building and stubbed execution."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .graph_builders import build_conv_fprop_graph
from .helpers import build_all_plans, call_attribute_methods, execute_zeros


@pytest.mark.gpu
class TestConvFprop:
    """Tests for convolution forward propagation plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        """Conv_fprop builds plans and executes against the stub engine without erroring."""
        graph, x, weight, y = build_conv_fprop_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(x, np.float32), (weight, np.float32), (y, np.float32)],
            handle,
        )


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
