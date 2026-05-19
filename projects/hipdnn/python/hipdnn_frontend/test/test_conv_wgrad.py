# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for convolution backward weight gradient."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .helpers import execute_graph

# Dimensions used across tests
N, C, H, W = 16, 16, 16, 16
K, R, S = 16, 3, 3
STRIDE, PAD, DIL = 1, 1, 1
OUT_H = (H + 2 * PAD - DIL * (R - 1) - 1) // STRIDE + 1
OUT_W = (W + 2 * PAD - DIL * (S - 1) - 1) // STRIDE + 1


def _build_conv_wgrad_graph(graph):
    """Build a conv_wgrad graph returning (graph, dy, x, dw)."""
    graph.set_name("conv_wgrad_test")

    dy = hipdnn.Tensor.create([N, K, OUT_H, OUT_W], hipdnn.DataType.FLOAT)
    dy.set_name("output_gradient_dy")

    x = hipdnn.Tensor.create([N, C, H, W], hipdnn.DataType.FLOAT)
    x.set_name("input_x")

    conv_attrs = hipdnn.ConvWgradAttributes()
    conv_attrs.set_name("conv_wgrad_node")
    conv_attrs.set_pre_padding([PAD, PAD])
    conv_attrs.set_post_padding([PAD, PAD])
    conv_attrs.set_stride([STRIDE, STRIDE])
    conv_attrs.set_dilation([DIL, DIL])

    dw = graph.conv_wgrad(dy, x, conv_attrs)
    dw.set_name("weight_gradient_dw")
    dw.set_output(True)

    return graph, dy, x, dw


@pytest.mark.gpu
@pytest.mark.integration
class TestConvWgrad:
    """Tests for convolution backward weight gradient end-to-end pipeline."""

    def test_graph_validates_successfully(self, graph):
        """Build a conv_wgrad graph and verify validation passes."""
        graph, dy, x, dw = _build_conv_wgrad_graph(graph)

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

    def test_operation_graph_builds(self, graph, handle):
        """Build a conv_wgrad operation graph with backend handle."""
        graph, dy, x, dw = _build_conv_wgrad_graph(graph)

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

        result = graph.build_operation_graph(handle)
        assert result.is_good(), f"Build operation graph failed: {result.get_message()}"

    def test_execution_plans_created(self, graph, handle):
        """Build execution plans for conv_wgrad."""
        graph, dy, x, dw = _build_conv_wgrad_graph(graph)

        assert graph.validate().is_good()
        assert graph.build_operation_graph(handle).is_good()
        assert graph.create_execution_plans().is_good()
        assert graph.check_support().is_good()
        assert graph.build_plans().is_good()

    def test_execution_produces_nonzero_output(self, graph, handle):
        """Full end-to-end conv_wgrad: execute and verify non-zero output."""
        graph, dy, x, dw = _build_conv_wgrad_graph(graph)

        assert graph.validate().is_good()
        assert graph.build_operation_graph(handle).is_good()
        assert graph.create_execution_plans().is_good()
        assert graph.check_support().is_good()
        assert graph.build_plans().is_good()

        dy_data = np.random.uniform(
            0.0,
            1.0,
            [N, K, OUT_H, OUT_W],
        ).astype(np.float32)
        x_data = np.random.uniform(0.0, 1.0, [N, C, H, W]).astype(np.float32)
        dw_data = np.zeros([K, C, R, S], dtype=np.float32)

        tensor_data = {
            dy.get_uid(): dy_data,
            x.get_uid(): x_data,
            dw.get_uid(): dw_data,
        }

        results = execute_graph(graph, handle, tensor_data)
        dw_result = results[dw.get_uid()]

        assert not np.all(dw_result == 0), "Conv wgrad output is all zeros"
