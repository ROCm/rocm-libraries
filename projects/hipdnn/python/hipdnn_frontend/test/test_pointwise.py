# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for elementwise pointwise operations."""

import numpy as np
import pytest

from .helpers import build_all_plans, build_pointwise_add_graph, execute_graph


@pytest.mark.gpu
@pytest.mark.integration
class TestPointwiseAdd:
    """Tests for pointwise add end-to-end pipeline."""

    def test_graph_validates_successfully(self):
        """Build a pointwise add graph and verify validation passes."""
        graph, a, b, out = build_pointwise_add_graph()

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

    def test_execution_produces_expected_sum(self):
        """Full end-to-end pointwise add: execute and verify a + b."""
        graph, a, b, out = build_pointwise_add_graph()

        handle = build_all_plans(graph)

        a_data = np.random.uniform(0.0, 1.0, a.get_dim()).astype(np.float32)
        b_data = np.random.uniform(0.0, 1.0, b.get_dim()).astype(np.float32)
        out_data = np.zeros(out.get_dim(), dtype=np.float32)

        tensor_data = {
            a.get_uid(): a_data,
            b.get_uid(): b_data,
            out.get_uid(): out_data,
        }

        results = execute_graph(graph, tensor_data, handle)
        out_result = results[out.get_uid()]

        np.testing.assert_allclose(out_result, a_data + b_data, rtol=1e-5, atol=1e-5)
