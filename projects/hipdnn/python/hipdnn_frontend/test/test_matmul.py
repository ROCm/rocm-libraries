# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for matrix multiplication."""

import pytest

import hipdnn_frontend as hipdnn

# Dimensions: A [M, K], B [K, N] -> C [M, N]
M, K, N = 4, 3, 5


def _build_matmul_graph(graph):
    """Build a matmul graph returning (graph, a, b, c)."""
    graph.set_name("matmul_test")

    a = hipdnn.Tensor.create([M, K], hipdnn.DataType.FLOAT)
    a.set_name("A")

    b = hipdnn.Tensor.create([K, N], hipdnn.DataType.FLOAT)
    b.set_name("B")

    attrs = hipdnn.MatmulAttributes()
    attrs.set_name("matmul_node")

    c = graph.matmul(a, b, attrs)
    c.set_name("C")
    c.set_output(True)

    return graph, a, b, c


@pytest.mark.gpu
class TestMatmul:
    """Tests for matrix multiplication graph building."""

    def test_graph_validates(self, graph):
        """Create a matmul graph and verify validation passes."""
        graph, a, b, c = _build_matmul_graph(graph)

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

    def test_operation_graph_builds(self, graph, handle):
        """Validate and build matmul operation graph."""
        graph, a, b, c = _build_matmul_graph(graph)

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

        result = graph.build_operation_graph(handle)
        assert result.is_good(), f"Build operation graph failed: {result.get_message()}"
