# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for custom operations: plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .graph_builders import build_custom_op_graph
from .helpers import build_all_plans, call_attribute_methods, execute_zeros


@pytest.mark.gpu
class TestCustomOp:
    """Tests for custom op plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        graph, a, b, y0, y1 = build_custom_op_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [
                (a, np.float32),
                (b, np.float32),
                (y0, np.float32),
                (y1, np.float32),
            ],
            handle,
        )


class TestCustomOpAttributeBindings:
    """Every custom operation attribute binding round-trips through its getter (no GPU required)."""

    def test_methods_are_callable(self):
        input_tensor = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        output_tensor = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.CustomOpAttributes(),
            (
                ("set_name", ("custom_op",), "get_name", "custom_op"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                (
                    "set_custom_op_id",
                    ("example.identity",),
                    "get_custom_op_id",
                    "example.identity",
                ),
                ("set_inputs", ([input_tensor],), "get_inputs", [input_tensor]),
                ("set_outputs", ([output_tensor],), "get_outputs", [output_tensor]),
                ("set_data", ([0, 1, 2],), "get_data", [0, 1, 2]),
            ),
        )
