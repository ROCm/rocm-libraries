# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for matrix multiplication."""

import pytest

import hipdnn_frontend as hipdnn

from .helpers import build_operation_graph, create_float_graph


def build_matmul_graph(m=4, k=3, n=5):
    """Build a matmul graph (A [M, K] x B [K, N] -> C [M, N])."""
    graph = create_float_graph()
    graph.set_name("matmul_test")

    a = hipdnn.Tensor.create([m, k], hipdnn.DataType.FLOAT)
    a.set_name("A")

    b = hipdnn.Tensor.create([k, n], hipdnn.DataType.FLOAT)
    b.set_name("B")

    attrs = hipdnn.MatmulAttributes()
    attrs.set_name("matmul_node")

    c = graph.matmul(a, b, attrs)
    c.set_name("C")
    c.set_output(True)

    return graph, a, b, c


@pytest.mark.gpu
class TestMatmul:
    """Tests for matrix multiplication end-to-end pipeline."""

    def test_builds_operation_graph(self):
        """Matmul graph validates and lowers to a backend operation graph.

        Execution is not exercised here: matmul requires the hipblaslt
        provider, which is not loaded in the python wheel test environment
        (only the miopen provider is available).
        """
        graph, a, b, c = build_matmul_graph()

        build_operation_graph(graph)


class TestMatmulAttributes:
    """Round-trip tests for MatmulAttributes accessors (no GPU required)."""

    def test_name_round_trip(self):
        """set_name()/get_name() round-trip."""
        attrs = hipdnn.MatmulAttributes()
        attrs.set_name("matmul")
        assert attrs.get_name() == "matmul"

    def test_compute_data_type_round_trip(self):
        """set_compute_data_type()/get_compute_data_type() round-trip."""
        attrs = hipdnn.MatmulAttributes()
        attrs.set_compute_data_type(hipdnn.DataType.FLOAT)
        assert attrs.get_compute_data_type() == hipdnn.DataType.FLOAT
