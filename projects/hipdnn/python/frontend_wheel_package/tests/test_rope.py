# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for rope: shape inference, validation, plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .graph_builders import build_rope_graph
from .helpers import (
    build_all_plans,
    call_attribute_methods,
    create_float_graph,
    execute_zeros,
)


@pytest.mark.gpu
class TestRope:
    """Tests for rope plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        graph, x, freqs, y = build_rope_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(x, np.float32), (freqs, np.float32), (y, np.float32)],
            handle,
        )


class TestRopeShapeInference:
    """y takes the dims of x (no GPU required)."""

    def test_output_matches_input(self):
        graph, x, _, y = build_rope_graph()
        assert graph.validate().is_good()

        assert y.get_dim() == x.get_dim()
        assert y.get_stride() == x.get_stride()

    def test_rejects_freqs_last_dim_not_rope_dim(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([1, 2, 3, 8], hipdnn.DataType.FLOAT)
        freqs = hipdnn.Tensor.create([1, 1, 3, 4], hipdnn.DataType.FLOAT)

        y = graph.rope(x, freqs, hipdnn.RopeAttributes())
        y.set_output(True)
        assert not graph.validate().is_good()


class TestRopeAttributeBindings:
    """Every rope attribute binding round-trips through its getter (no GPU required)."""

    def test_methods_are_callable(self):
        x = hipdnn.Tensor.create([1, 2], hipdnn.DataType.FLOAT)
        freqs = hipdnn.Tensor.create([1, 2], hipdnn.DataType.FLOAT)
        y = hipdnn.Tensor.create([1, 2], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.RopeAttributes(),
            (
                ("set_name", ("rope",), "get_name", "rope"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_x", (x,), "get_x", x),
                ("set_freqs", (freqs,), "get_freqs", freqs),
                ("set_y", (y,), "get_y", y),
                ("set_output_scale", (0.5,), "get_output_scale", 0.5),
                ("set_rope_dim", (2,), "get_rope_dim", 2),
                ("set_rope_offset", (0,), "get_rope_offset", 0),
            ),
        )

    def test_rope_offset_defaults_to_none(self):
        assert hipdnn.RopeAttributes().get_rope_offset() is None
