# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for convolution forward propagation."""

import numpy as np
import pytest

from .helpers import build_all_plans, build_conv_fprop_graph, execute_graph


@pytest.mark.gpu
@pytest.mark.integration
class TestConvFprop:
    """Tests for convolution forward propagation end-to-end pipeline."""

    def test_graph_validates_successfully(self):
        """Build a conv_fprop graph and verify validation passes."""
        graph, x, weight, y = build_conv_fprop_graph()

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

    def test_execution_produces_nonzero_output(self):
        """Full end-to-end conv_fprop: execute and verify non-zero output."""
        graph, x, weight, y = build_conv_fprop_graph()

        handle = build_all_plans(graph)

        x_data = np.random.uniform(0.0, 1.0, x.get_dim()).astype(np.float32)
        w_data = np.random.uniform(0.0, 1.0, weight.get_dim()).astype(np.float32)
        y_data = np.zeros(y.get_dim(), dtype=np.float32)

        tensor_data = {
            x.get_uid(): x_data,
            weight.get_uid(): w_data,
            y.get_uid(): y_data,
        }

        results = execute_graph(graph, tensor_data, handle)
        y_result = results[y.get_uid()]

        assert not np.all(y_result == 0), "Conv fprop output is all zeros"
