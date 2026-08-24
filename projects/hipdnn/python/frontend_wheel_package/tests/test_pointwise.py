# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for elementwise pointwise operations: plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .graph_builders import build_pointwise_add_graph
from .helpers import build_all_plans, call_attribute_methods, execute_zeros


@pytest.mark.gpu
class TestPointwiseAdd:
    """Tests for pointwise add plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        """Pointwise add builds plans and executes against the stub engine without erroring."""
        graph, a, b, out = build_pointwise_add_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(a, np.float32), (b, np.float32), (out, np.float32)],
            handle,
        )


class TestPointwiseAttributeBindings:
    """Every pointwise attribute binding round-trips through its getter (no GPU required)."""

    def test_methods_are_callable(self):
        in0 = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        in1 = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        in2 = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        out0 = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.PointwiseAttributes(),
            (
                ("set_name", ("pointwise",), "get_name", "pointwise"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                (
                    "set_mode",
                    (hipdnn.PointwiseMode.ADD,),
                    "get_mode",
                    hipdnn.PointwiseMode.ADD,
                ),
                ("set_relu_lower_clip", (0.0,), "get_relu_lower_clip", 0.0),
                ("set_relu_upper_clip", (1.0,), "get_relu_upper_clip", 1.0),
                (
                    "set_relu_lower_clip_slope",
                    (0.25,),
                    "get_relu_lower_clip_slope",
                    0.25,
                ),
                ("set_swish_beta", (0.5,), "get_swish_beta", 0.5),
                ("set_elu_alpha", (2.0,), "get_elu_alpha", 2.0),
                ("set_softplus_beta", (4.0,), "get_softplus_beta", 4.0),
                ("set_axis", (0,), "get_axis", 0),
                ("set_input_0", (in0,), "get_input_0", in0),
                ("set_input_1", (in1,), "get_input_1", in1),
                ("set_input_2", (in2,), "get_input_2", in2),
                ("set_output_0", (out0,), "get_output_0", out0),
            ),
        )
