# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for elementwise pointwise operations."""

import pytest

import hipdnn_frontend as hipdnn

from .helpers import build_operation_graph, create_float_graph


def build_pointwise_add_graph(n=16, c=16, h=16, w=16):
    """Build an elementwise-add pointwise graph returning (graph, a, b, out)."""
    graph = create_float_graph()
    graph.set_name("pointwise_add_test")

    a = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    a.set_name("a")

    b = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    b.set_name("b")

    attrs = hipdnn.PointwiseAttributes()
    attrs.set_name("pointwise_add_node")
    attrs.set_mode(hipdnn.PointwiseMode.ADD)

    out = graph.pointwise(a, b, attrs)
    out.set_name("out")
    out.set_output(True)

    return graph, a, b, out


@pytest.mark.gpu
class TestPointwiseAdd:
    """Tests for pointwise add end-to-end pipeline."""

    def test_builds_operation_graph(self):
        """Pointwise add graph validates and lowers to a backend operation graph.

        Execution is not exercised here: no provider in the python wheel test
        environment supplies an engine for a standalone pointwise op (the
        miopen provider only exposes fused pointwise support).
        """
        graph, a, b, out = build_pointwise_add_graph()

        build_operation_graph(graph)


class TestPointwiseAttributes:
    """Round-trip tests for PointwiseAttributes accessors (no GPU required)."""

    def test_name_round_trip(self):
        """set_name()/get_name() round-trip."""
        attrs = hipdnn.PointwiseAttributes()
        attrs.set_name("relu")
        assert attrs.get_name() == "relu"

    def test_mode_round_trip(self):
        """set_mode()/get_mode() round-trip."""
        attrs = hipdnn.PointwiseAttributes()
        attrs.set_mode(hipdnn.PointwiseMode.RELU_FWD)
        assert attrs.get_mode() == hipdnn.PointwiseMode.RELU_FWD
