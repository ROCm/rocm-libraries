# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for reduction operations: plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .graph_builders import build_reduction_graph
from .helpers import build_all_plans, call_attribute_methods, execute_zeros


@pytest.mark.gpu
class TestReduction:
    """Tests for reduction plan-building and stubbed execution."""

    def test_execution_succeeds_without_output(self):
        graph, x, output, _node_output = build_reduction_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(x, np.float32), (output, np.float32)],
            handle,
        )

    def test_execution_succeeds_with_output(self):
        graph, x, output, node_output = build_reduction_graph(with_output_tensor=True)
        assert node_output is output

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(x, np.float32), (output, np.float32)],
            handle,
        )


class TestReductionAttributeBindings:
    """Every reduction attribute binding round-trips through its getter (no GPU required)."""

    def test_methods_are_callable(self):
        x = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        y = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.ReductionAttributes(),
            (
                ("set_name", ("reduction",), "get_name", "reduction"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                (
                    "set_mode",
                    (hipdnn.ReductionMode.ADD,),
                    "get_mode",
                    hipdnn.ReductionMode.ADD,
                ),
                ("set_is_deterministic", (True,), "get_is_deterministic", True),
                ("set_x", (x,), "get_x", x),
                ("set_y", (y,), "get_y", y),
            ),
        )
