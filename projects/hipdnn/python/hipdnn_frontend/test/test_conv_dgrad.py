# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for convolution backward data gradient."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from conftest import execute_graph

# Dimensions used across tests
N, C, H, W = 16, 16, 16, 16
K, R, S = 16, 3, 3
STRIDE, PAD, DIL = 1, 1, 1
OUT_H = (H + 2 * PAD - DIL * (R - 1) - 1) // STRIDE + 1
OUT_W = (W + 2 * PAD - DIL * (S - 1) - 1) // STRIDE + 1


def _build_conv_dgrad_graph(graph):
    """Build a conv_dgrad graph returning (graph, dy, weight, dx)."""
    graph.set_name("conv_dgrad_test")

    dy = hipdnn.Tensor.create([N, K, OUT_H, OUT_W], hipdnn.DataType.FLOAT)
    dy.set_name("output_gradient_dy")

    weight = hipdnn.Tensor.create([K, C, R, S], hipdnn.DataType.FLOAT)
    weight.set_name("weight")

    conv_attrs = hipdnn.ConvDgradAttributes()
    conv_attrs.set_name("conv_dgrad_node")
    conv_attrs.set_pre_padding([PAD, PAD])
    conv_attrs.set_post_padding([PAD, PAD])
    conv_attrs.set_stride([STRIDE, STRIDE])
    conv_attrs.set_dilation([DIL, DIL])

    dx = graph.conv_dgrad(dy, weight, conv_attrs)
    dx.set_name("input_gradient_dx")
    dx.set_output(True)

    return graph, dy, weight, dx


@pytest.mark.gpu
@pytest.mark.integration
class TestConvDgrad:
    """Tests for convolution backward data gradient end-to-end pipeline."""

    def test_graph_validates_successfully(self, graph):
        """Build a conv_dgrad graph and verify validation passes."""
        graph, dy, weight, dx = _build_conv_dgrad_graph(graph)

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

    def test_operation_graph_builds(self, graph, handle):
        """Build a conv_dgrad operation graph with backend handle."""
        graph, dy, weight, dx = _build_conv_dgrad_graph(graph)

        result = graph.validate()
        assert result.is_good(), f"Validation failed: {result.get_message()}"

        result = graph.build_operation_graph(handle)
        assert result.is_good(), f"Build operation graph failed: {result.get_message()}"

    def test_execution_plans_created(self, graph, handle):
        """Build execution plans for conv_dgrad."""
        graph, dy, weight, dx = _build_conv_dgrad_graph(graph)

        assert graph.validate().is_good()
        assert graph.build_operation_graph(handle).is_good()
        assert graph.create_execution_plans().is_good()
        assert graph.check_support().is_good()
        assert graph.build_plans().is_good()

    def test_execution_produces_nonzero_output(self, graph, handle):
        """Full end-to-end conv_dgrad: execute and verify non-zero output."""
        graph, dy, weight, dx = _build_conv_dgrad_graph(graph)

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
        w_data = np.random.uniform(0.0, 1.0, [K, C, R, S]).astype(np.float32)
        dx_data = np.zeros([N, C, H, W], dtype=np.float32)

        tensor_data = {
            dy.get_uid(): dy_data,
            weight.get_uid(): w_data,
            dx.get_uid(): dx_data,
        }

        results = execute_graph(graph, handle, tensor_data)
        dx_result = results[dx.get_uid()]

        assert not np.all(dx_result == 0), "Conv dgrad output is all zeros"
