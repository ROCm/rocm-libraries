# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for convolution backward data gradient."""

import numpy as np
import pytest

from . import helpers
from .helpers import build_all_plans, build_conv_dgrad_graph, execute_graph


@pytest.mark.gpu
@pytest.mark.integration
class TestConvDgrad:
    """Tests for convolution backward data gradient end-to-end pipeline."""

    def test_graph_validates_successfully(self, graph):
        """Build a conv_dgrad graph and verify validation passes."""
        graph, dy, weight, dx = build_conv_dgrad_graph(graph)

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

    def test_operation_graph_builds(self, graph, handle):
        """Build a conv_dgrad operation graph with backend handle."""
        graph, dy, weight, dx = build_conv_dgrad_graph(graph)

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

        result = graph.build_operation_graph(handle)
        assert result.is_good(), f"Build operation graph failed: {result.get_message()}"

    def test_execution_plans_created(self, graph, handle):
        """Build execution plans for conv_dgrad."""
        graph, dy, weight, dx = build_conv_dgrad_graph(graph)

        build_all_plans(graph, handle)

    def test_execution_produces_nonzero_output(self, graph, handle):
        """Full end-to-end conv_dgrad: execute and verify non-zero output."""
        graph, dy, weight, dx = build_conv_dgrad_graph(graph)

        build_all_plans(graph, handle)

        dy_data = np.random.uniform(
            0.0, 1.0, [helpers.N, helpers.K, helpers.OUT_H, helpers.OUT_W]
        ).astype(np.float32)
        w_data = np.random.uniform(
            0.0, 1.0, [helpers.K, helpers.C, helpers.R, helpers.S]
        ).astype(np.float32)
        dx_data = np.zeros(
            [helpers.N, helpers.C, helpers.H, helpers.W], dtype=np.float32
        )

        tensor_data = {
            dy.get_uid(): dy_data,
            weight.get_uid(): w_data,
            dx.get_uid(): dx_data,
        }

        results = execute_graph(graph, handle, tensor_data)
        dx_result = results[dx.get_uid()]

        assert not np.all(dx_result == 0), "Conv dgrad output is all zeros"
