# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for convolution backward weight gradient."""

import numpy as np
import pytest

from . import helpers
from .helpers import build_all_plans, build_conv_wgrad_graph, execute_graph


@pytest.mark.gpu
@pytest.mark.integration
class TestConvWgrad:
    """Tests for convolution backward weight gradient end-to-end pipeline."""

    def test_graph_validates_successfully(self, graph):
        """Build a conv_wgrad graph and verify validation passes."""
        graph, dy, x, dw = build_conv_wgrad_graph(graph)

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

    def test_operation_graph_builds(self, graph, handle):
        """Build a conv_wgrad operation graph with backend handle."""
        graph, dy, x, dw = build_conv_wgrad_graph(graph)

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

        result = graph.build_operation_graph(handle)
        assert result.is_good(), f"Build operation graph failed: {result.get_message()}"

    def test_execution_plans_created(self, graph, handle):
        """Build execution plans for conv_wgrad."""
        graph, dy, x, dw = build_conv_wgrad_graph(graph)

        build_all_plans(graph, handle)

    def test_execution_produces_nonzero_output(self, graph, handle):
        """Full end-to-end conv_wgrad: execute and verify non-zero output."""
        graph, dy, x, dw = build_conv_wgrad_graph(graph)

        build_all_plans(graph, handle)

        dy_data = np.random.uniform(
            0.0, 1.0, [helpers.N, helpers.K, helpers.OUT_H, helpers.OUT_W]
        ).astype(np.float32)
        x_data = np.random.uniform(
            0.0, 1.0, [helpers.N, helpers.C, helpers.H, helpers.W]
        ).astype(np.float32)
        dw_data = np.zeros(
            [helpers.K, helpers.C, helpers.R, helpers.S], dtype=np.float32
        )

        tensor_data = {
            dy.get_uid(): dy_data,
            x.get_uid(): x_data,
            dw.get_uid(): dw_data,
        }

        results = execute_graph(graph, handle, tensor_data)
        dw_result = results[dw.get_uid()]

        assert not np.all(dw_result == 0), "Conv wgrad output is all zeros"
