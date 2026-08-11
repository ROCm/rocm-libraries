# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for reduction operations: plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .helpers import (
    call_attribute_methods,
    build_all_plans,
    create_float_graph,
    execute_zeros,
)


@pytest.mark.gpu
class TestReduction:
    """Tests for reduction operation-graph construction."""

    def test_builds_operation_graph_without_output(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([4, 8], hipdnn.DataType.FLOAT)
        output = graph.reduction(
            x,
            hipdnn.ReductionAttributes().set_mode(hipdnn.ReductionMode.ADD),
        )
        output.set_dim([1, 8]).set_stride([8, 1]).set_output(True)

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(x, np.float32), (output, np.float32)],
            handle,
        )

    def test_builds_operation_graph_with_output(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([4, 8], hipdnn.DataType.FLOAT)
        output = hipdnn.Tensor.create([1, 8], hipdnn.DataType.FLOAT)
        output.set_output(True)
        assert (
            graph.reduction(
                x,
                output,
                hipdnn.ReductionAttributes().set_mode(hipdnn.ReductionMode.ADD),
            )
            is output
        )

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
