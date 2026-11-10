# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for gather: shape inference, plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .graph_builders import build_gather_graph
from .helpers import (
    build_all_plans,
    call_attribute_methods,
    create_float_graph,
    execute_zeros,
)


@pytest.mark.gpu
class TestGather:
    """Tests for gather plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        graph, table, ids, y = build_gather_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(table, np.float32), (ids, np.int32), (y, np.float32)],
            handle,
        )


class TestGatherShapeInference:
    """Output shape follows x.dims[:axis] ++ indices.dims ++ x.dims[axis + 1:] (no GPU required)."""

    def test_middle_axis(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([2, 5, 4], hipdnn.DataType.FLOAT)
        indices = hipdnn.Tensor.create([3], hipdnn.DataType.INT64)

        y = graph.gather(x, indices, hipdnn.GatherAttributes().set_axis(1))
        y.set_output(True)
        assert graph.validate().is_good()

        assert y.get_dim() == [2, 3, 4]
        assert y.get_stride() == [12, 4, 1]


class TestGatherAttributeBindings:
    """Every gather attribute binding round-trips through its getter (no GPU required)."""

    def test_methods_are_callable(self):
        x = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        indices = hipdnn.Tensor.create([1], hipdnn.DataType.INT32)
        y = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.GatherAttributes(),
            (
                ("set_name", ("gather",), "get_name", "gather"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_x", (x,), "get_x", x),
                ("set_indices", (indices,), "get_indices", indices),
                ("set_y", (y,), "get_y", y),
                ("set_axis", (1,), "get_axis", 1),
            ),
        )
