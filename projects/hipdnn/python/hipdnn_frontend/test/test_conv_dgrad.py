# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for convolution backward data gradient."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .helpers import build_all_plans, create_float_graph, execute_graph


def build_conv_dgrad_graph(
    n=16, c=16, h=16, w=16, k=16, r=3, s=3, stride=1, pad=1, dil=1
):
    """Build a conv_dgrad graph returning (graph, dy, weight, dx)."""
    out_h = (h + 2 * pad - dil * (r - 1) - 1) // stride + 1
    out_w = (w + 2 * pad - dil * (s - 1) - 1) // stride + 1

    graph = create_float_graph()
    graph.set_name("conv_dgrad_test")

    dy = hipdnn.Tensor.create([n, k, out_h, out_w], hipdnn.DataType.FLOAT)
    dy.set_name("output_gradient_dy")

    weight = hipdnn.Tensor.create([k, c, r, s], hipdnn.DataType.FLOAT)
    weight.set_name("weight")

    conv_attrs = hipdnn.ConvDgradAttributes()
    conv_attrs.set_name("conv_dgrad_node")
    conv_attrs.set_pre_padding([pad, pad])
    conv_attrs.set_post_padding([pad, pad])
    conv_attrs.set_stride([stride, stride])
    conv_attrs.set_dilation([dil, dil])

    dx = graph.conv_dgrad(dy, weight, conv_attrs)
    dx.set_name("input_gradient_dx")
    dx.set_output(True)

    return graph, dy, weight, dx


@pytest.mark.gpu
class TestConvDgrad:
    """Tests for convolution backward data gradient end-to-end pipeline."""

    def test_execution_produces_nonzero_output(self):
        """Full end-to-end conv_dgrad: execute and verify non-zero output."""
        graph, dy, weight, dx = build_conv_dgrad_graph()

        handle = build_all_plans(graph)

        dy_data = np.random.uniform(0.0, 1.0, dy.get_dim()).astype(np.float32)
        w_data = np.random.uniform(0.0, 1.0, weight.get_dim()).astype(np.float32)
        dx_data = np.zeros(dx.get_dim(), dtype=np.float32)

        tensor_data = {
            dy.get_uid(): dy_data,
            weight.get_uid(): w_data,
            dx.get_uid(): dx_data,
        }

        results = execute_graph(graph, tensor_data, handle)
        dx_result = results[dx.get_uid()]

        assert not np.all(dx_result == 0), "Conv dgrad output is all zeros"

    def test_execution_matches_hardcoded_values(self):
        """Conv_dgrad on a hand-checked case matches hardcoded dx.

        dx is the scatter-add of each dy element times the 2x2 identity-diagonal
        kernel into a 3x3 grid (stride=1, pad=0).
        """
        graph, dy, weight, dx = build_conv_dgrad_graph(
            n=1, c=1, h=3, w=3, k=1, r=2, s=2, stride=1, pad=0
        )

        handle = build_all_plans(graph)

        dy_data = np.array([[1, 2], [3, 4]], dtype=np.float32).reshape(dy.get_dim())
        w_data = np.array([[1, 0], [0, 1]], dtype=np.float32).reshape(weight.get_dim())
        dx_data = np.zeros(dx.get_dim(), dtype=np.float32)

        tensor_data = {
            dy.get_uid(): dy_data,
            weight.get_uid(): w_data,
            dx.get_uid(): dx_data,
        }

        results = execute_graph(graph, tensor_data, handle)
        dx_result = results[dx.get_uid()]

        expected = np.array(
            [[1, 2, 0], [3, 5, 2], [0, 3, 4]], dtype=np.float32
        ).reshape(dx.get_dim())
        np.testing.assert_allclose(dx_result, expected, rtol=2e-3, atol=2e-3)


class TestConvDgradAttributes:
    """Attribute accessors and aliases for ConvDgrad (no GPU required)."""

    def test_alias_identity(self):
        """ConvDgradAttributes is the same class as ConvolutionDgradAttributes."""
        assert hipdnn.ConvDgradAttributes is hipdnn.ConvolutionDgradAttributes

    def test_pre_post_padding_chain(self):
        """ConvDgrad pre/post padding setters chain and store the name."""
        attrs = hipdnn.ConvDgradAttributes()
        result = (
            attrs.set_name("dgrad")
            .set_pre_padding([1, 1])
            .set_post_padding([1, 1])
            .set_stride([1, 1])
            .set_dilation([1, 1])
        )
        assert result is attrs
        assert attrs.get_name() == "dgrad"
