# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for convolution forward propagation."""

import numpy as np
import pytest

from . import helpers
from .helpers import build_all_plans, build_conv_fprop_graph, execute_graph


@pytest.mark.gpu
@pytest.mark.integration
class TestConvFprop:
    """Tests for convolution forward propagation end-to-end pipeline."""

    def test_graph_validates_successfully(self, graph):
        """Build a conv_fprop graph and verify validation passes."""
        graph, x, weight, y = build_conv_fprop_graph(graph)

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

    def test_operation_graph_builds(self, graph, handle):
        """Build a conv_fprop operation graph with backend handle."""
        graph, x, weight, y = build_conv_fprop_graph(graph)

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

        result = graph.build_operation_graph(handle)
        assert result.is_good(), f"Build operation graph failed: {result.get_message()}"

    def test_execution_plans_created(self, graph, handle):
        """Build execution plans for conv_fprop."""
        graph, x, weight, y = build_conv_fprop_graph(graph)

        build_all_plans(graph, handle)

    def test_execution_produces_nonzero_output(self, graph, handle):
        """Full end-to-end conv_fprop: execute and verify non-zero output."""
        graph, x, weight, y = build_conv_fprop_graph(graph)

        build_all_plans(graph, handle)

        x_data = np.random.uniform(
            0.0, 1.0, [helpers.N, helpers.C, helpers.H, helpers.W]
        ).astype(np.float32)
        w_data = np.random.uniform(
            0.0, 1.0, [helpers.K, helpers.C, helpers.R, helpers.S]
        ).astype(np.float32)
        y_data = np.zeros(
            [helpers.N, helpers.K, helpers.OUT_H, helpers.OUT_W], dtype=np.float32
        )

        tensor_data = {
            x.get_uid(): x_data,
            weight.get_uid(): w_data,
            y.get_uid(): y_data,
        }

        results = execute_graph(graph, handle, tensor_data)
        y_result = results[y.get_uid()]

        assert not np.all(y_result == 0), "Conv fprop output is all zeros"
