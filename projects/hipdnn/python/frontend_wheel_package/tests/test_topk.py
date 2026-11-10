# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for topk: shape and data-type inference, plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .graph_builders import build_topk_graph
from .helpers import (
    build_all_plans,
    call_attribute_methods,
    create_float_graph,
    execute_zeros,
)


@pytest.mark.gpu
class TestTopk:
    """Tests for topk plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        graph, x, values, indices = build_topk_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(x, np.float32), (values, np.float32), (indices, np.int64)],
            handle,
        )


class TestTopkShapeInference:
    """Outputs take x's dims with the last dim k (no GPU required)."""

    def test_defaults(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([2, 5, 8], hipdnn.DataType.FLOAT)

        values, indices = graph.topk(x, hipdnn.TopkAttributes().set_k(3))
        values.set_output(True)
        indices.set_output(True)
        assert graph.validate().is_good()

        assert values.get_dim() == [2, 5, 3]
        assert values.get_stride() == [15, 3, 1]
        assert values.get_data_type() == hipdnn.DataType.FLOAT
        assert indices.get_dim() == [2, 5, 3]
        assert indices.get_data_type() == hipdnn.DataType.INT64

    def test_explicit_int32_indices(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([4, 6], hipdnn.DataType.FLOAT)

        values, indices = graph.topk(
            x, hipdnn.TopkAttributes().set_k(1).set_largest(False)
        )
        values.set_output(True)
        indices.set_output(True)
        indices.set_data_type(hipdnn.DataType.INT32)
        assert graph.validate().is_good()

        assert indices.get_dim() == [4, 1]
        assert indices.get_data_type() == hipdnn.DataType.INT32

    def test_k_out_of_range_fails_validation(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([4, 6], hipdnn.DataType.FLOAT)

        values, indices = graph.topk(x, hipdnn.TopkAttributes().set_k(7))
        values.set_output(True)
        indices.set_output(True)
        assert not graph.validate().is_good()


class TestTopkAttributeBindings:
    """Every topk attribute binding round-trips through its getter (no GPU required)."""

    def test_methods_are_callable(self):
        x = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        values = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        indices = hipdnn.Tensor.create([1], hipdnn.DataType.INT64)
        call_attribute_methods(
            hipdnn.TopkAttributes(),
            (
                ("set_name", ("topk",), "get_name", "topk"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_x", (x,), "get_x", x),
                ("set_values", (values,), "get_values", values),
                ("set_indices", (indices,), "get_indices", indices),
                ("set_k", (2,), "get_k", 2),
                ("set_largest", (False,), "get_largest", False),
            ),
        )
