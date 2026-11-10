# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for softmax: shape inference, validation, plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .graph_builders import build_softmax_graph
from .helpers import (
    build_all_plans,
    call_attribute_methods,
    create_float_graph,
    execute_zeros,
)


@pytest.mark.gpu
class TestSoftmax:
    """Tests for softmax plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        graph, p, s, stats = build_softmax_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(p, np.float32), (s, np.float32), (stats, np.float32)],
            handle,
        )


class TestSoftmaxShapeInference:
    """s takes the dims of p; stats takes p's dims with the last dim 1 (no GPU required)."""

    def test_outputs_inferred(self):
        graph, p, s, stats = build_softmax_graph()
        assert graph.validate().is_good()

        assert s.get_dim() == p.get_dim()
        assert s.get_stride() == p.get_stride()
        assert stats.get_dim() == [2, 4, 8, 1]

    def test_stats_absent_by_default(self):
        graph = create_float_graph()
        p = hipdnn.Tensor.create([3, 5], hipdnn.DataType.FLOAT)

        s, stats = graph.softmax(p, hipdnn.SoftmaxAttributes())
        s.set_output(True)
        assert stats is None
        assert graph.validate().is_good()

    def test_rejects_mismatched_s_dims(self):
        graph = create_float_graph()
        p = hipdnn.Tensor.create([3, 5], hipdnn.DataType.FLOAT)

        s, _ = graph.softmax(p, hipdnn.SoftmaxAttributes())
        s.set_dim([3, 4])
        s.set_output(True)
        assert not graph.validate().is_good()


class TestSoftmaxAttributeBindings:
    """Every softmax attribute binding round-trips through its getter (no GPU required)."""

    def test_methods_are_callable(self):
        p = hipdnn.Tensor.create([1, 2], hipdnn.DataType.FLOAT)
        s = hipdnn.Tensor.create([1, 2], hipdnn.DataType.FLOAT)
        stats = hipdnn.Tensor.create([1, 1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.SoftmaxAttributes(),
            (
                ("set_name", ("softmax",), "get_name", "softmax"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_p", (p,), "get_p", p),
                ("set_s", (s,), "get_s", s),
                ("set_stats", (stats,), "get_stats", stats),
                ("set_stats_use_log2", (True,), "get_stats_use_log2", True),
                ("set_generate_stats", (True,), "get_generate_stats", True),
            ),
        )
