# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for matrix multiplication."""

import pytest

from .helpers import build_matmul_graph


@pytest.mark.gpu
class TestMatmul:
    """Tests for matrix multiplication graph building."""

    def test_graph_validates(self, graph):
        """Create a matmul graph and verify validation passes."""
        graph, a, b, c = build_matmul_graph(graph)

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

    def test_operation_graph_builds(self, graph, handle):
        """Validate and build matmul operation graph."""
        graph, a, b, c = build_matmul_graph(graph)

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

        result = graph.build_operation_graph(handle)
        assert result.is_good(), f"Build operation graph failed: {result.get_message()}"
