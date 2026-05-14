# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for convolution forward propagation."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from conftest import build_conv_fprop_graph, execute_graph

# Dimensions used across tests
N, C, H, W = 16, 16, 16, 16
K, R, S = 16, 3, 3
STRIDE, PAD, DIL = 1, 1, 1
OUT_H = (H + 2 * PAD - DIL * (R - 1) - 1) // STRIDE + 1
OUT_W = (W + 2 * PAD - DIL * (S - 1) - 1) // STRIDE + 1


@pytest.mark.gpu
@pytest.mark.integration
class TestConvFprop:
    """Tests for convolution forward propagation end-to-end pipeline."""

    def test_graph_validates_successfully(self, graph):
        """Build a conv_fprop graph and verify validation passes."""
        graph, x, weight, y, out_h, out_w = build_conv_fprop_graph(
            graph,
            n=N,
            c=C,
            h=H,
            w=W,
            k=K,
            r=R,
            s=S,
            stride=STRIDE,
            pad=PAD,
            dilation=DIL,
        )

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

    def test_operation_graph_builds(self, graph, handle):
        """Build a conv_fprop operation graph with backend handle."""
        graph, x, weight, y, out_h, out_w = build_conv_fprop_graph(
            graph,
            n=N,
            c=C,
            h=H,
            w=W,
            k=K,
            r=R,
            s=S,
            stride=STRIDE,
            pad=PAD,
            dilation=DIL,
        )

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

        result = graph.build_operation_graph(handle)
        assert result.is_good(), f"Build operation graph failed: {result.get_message()}"

    def test_execution_plans_created(self, graph, handle):
        """Build execution plans for conv_fprop."""
        graph, x, weight, y, out_h, out_w = build_conv_fprop_graph(
            graph,
            n=N,
            c=C,
            h=H,
            w=W,
            k=K,
            r=R,
            s=S,
            stride=STRIDE,
            pad=PAD,
            dilation=DIL,
        )

        assert graph.validate().is_good()
        assert graph.build_operation_graph(handle).is_good()
        assert graph.create_execution_plans().is_good()
        assert graph.check_support().is_good()
        assert graph.build_plans().is_good()

    def test_execution_produces_nonzero_output(self, graph, handle):
        """Full end-to-end conv_fprop: execute and verify non-zero output."""
        graph, x, weight, y, out_h, out_w = build_conv_fprop_graph(
            graph,
            n=N,
            c=C,
            h=H,
            w=W,
            k=K,
            r=R,
            s=S,
            stride=STRIDE,
            pad=PAD,
            dilation=DIL,
        )

        assert graph.validate().is_good()
        assert graph.build_operation_graph(handle).is_good()
        assert graph.create_execution_plans().is_good()
        assert graph.check_support().is_good()
        assert graph.build_plans().is_good()

        x_data = np.random.uniform(0.0, 1.0, [N, C, H, W]).astype(np.float32)
        w_data = np.random.uniform(0.0, 1.0, [K, C, R, S]).astype(np.float32)
        y_data = np.zeros([N, K, out_h, out_w], dtype=np.float32)

        tensor_data = {
            x.get_uid(): x_data,
            weight.get_uid(): w_data,
            y.get_uid(): y_data,
        }

        results = execute_graph(graph, handle, tensor_data)
        y_result = results[y.get_uid()]

        assert not np.all(y_result == 0), "Conv fprop output is all zeros"
