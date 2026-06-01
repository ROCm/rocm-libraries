# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for matrix multiplication."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .helpers import build_all_plans, create_float_graph, execute_graph


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
@pytest.mark.integration
class TestMatmul:
    """Tests for matrix multiplication end-to-end pipeline."""

    def test_graph_validates(self):
        """Create a matmul graph and verify validation passes."""
        graph, a, b, c = build_matmul_graph()

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

    def test_execution_produces_nonzero_output(self):
        """Full end-to-end matmul: execute and verify non-zero output."""
        graph, a, b, c = build_matmul_graph()

        handle = build_all_plans(graph)

        a_data = np.random.uniform(0.0, 1.0, a.get_dim()).astype(np.float32)
        b_data = np.random.uniform(0.0, 1.0, b.get_dim()).astype(np.float32)
        c_data = np.zeros(c.get_dim(), dtype=np.float32)

        tensor_data = {
            a.get_uid(): a_data,
            b.get_uid(): b_data,
            c.get_uid(): c_data,
        }

        results = execute_graph(graph, tensor_data, handle)
        c_result = results[c.get_uid()]

        assert not np.all(c_result == 0), "Matmul output is all zeros"
