# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for custom operations: plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .helpers import (
    call_attribute_methods,
    build_all_plans_or_skip,
    create_float_graph,
    execute_zeros,
)


@pytest.mark.gpu
class TestCustomOp:
    """Tests for custom operation-graph construction."""

    def test_builds_operation_graph(self):
        graph = create_float_graph()
        a = hipdnn.Tensor.create([4, 8], hipdnn.DataType.FLOAT)
        b = hipdnn.Tensor.create([4, 8], hipdnn.DataType.FLOAT)
        outputs = graph.custom_op(
            [a, b],
            2,
            hipdnn.CustomOpAttributes().set_custom_op_id("example.identity"),
        )
        for output in outputs:
            output.set_dim([4, 8]).set_stride([8, 1]).set_output(True)
            output.set_data_type(hipdnn.DataType.FLOAT)

        handle = build_all_plans_or_skip(graph)
        execute_zeros(
            graph,
            [(a, np.float32), (b, np.float32)]
            + [(output, np.float32) for output in outputs],
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
