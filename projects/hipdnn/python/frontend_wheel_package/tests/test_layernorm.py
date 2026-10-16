# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for layer normalization: plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .graph_builders import build_layernorm_backward_graph, build_layernorm_graph
from .helpers import build_all_plans, call_attribute_methods, execute_zeros


@pytest.mark.gpu
class TestLayernorm:
    """Tests for layer normalization plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        graph, x, scale, bias, y, mean, inv_variance = build_layernorm_graph()
        # INFERENCE phase produces no statistics.
        assert (mean, inv_variance) == (None, None)

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(x, np.float32), (scale, np.float32), (bias, np.float32), (y, np.float32)],
            handle,
        )


@pytest.mark.gpu
class TestLayernormBackward:
    """Tests for layer normalization backward plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        graph, dy, x, scale, dx, dscale, dbias = build_layernorm_backward_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [
                (dy, np.float32),
                (x, np.float32),
                (scale, np.float32),
                (dx, np.float32),
                (dscale, np.float32),
                (dbias, np.float32),
            ],
            handle,
        )


class TestLayernormAttributeBindings:
    """Every LayerNorm attribute binding round-trips through its getter (no GPU required)."""

    def test_forward_methods_are_callable(self):
        x = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        scale = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        bias = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        epsilon = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        y = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        mean = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        inv_variance = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.LayernormAttributes(),
            (
                ("set_name", ("layernorm",), "get_name", "layernorm"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_x", (x,), "get_x", x),
                ("set_scale", (scale,), "get_scale", scale),
                ("set_bias", (bias,), "get_bias", bias),
                ("set_epsilon", (epsilon,), "get_epsilon", epsilon),
                ("set_y", (y,), "get_y", y),
                ("set_mean", (mean,), "get_mean", mean),
                ("set_inv_variance", (inv_variance,), "get_inv_variance", inv_variance),
                (
                    "set_forward_phase",
                    (hipdnn.NormFwdPhase.INFERENCE,),
                    "get_forward_phase",
                    hipdnn.NormFwdPhase.INFERENCE,
                ),
            ),
        )

    def test_backward_methods_are_callable(self):
        dy = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        x = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        scale = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        mean = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        inv_variance = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        epsilon = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        dx = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        dscale = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        dbias = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        saved_mean = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        saved_inv_variance = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.LayernormBackwardAttributes(),
            (
                ("set_name", ("layernorm_backward",), "get_name", "layernorm_backward"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_dy", (dy,), "get_dy", dy),
                ("set_x", (x,), "get_x", x),
                ("set_scale", (scale,), "get_scale", scale),
                ("set_mean", (mean,), "get_mean", mean),
                ("set_inv_variance", (inv_variance,), "get_inv_variance", inv_variance),
                ("set_epsilon", (epsilon,), "get_epsilon", epsilon),
                ("set_dx", (dx,), "get_dx", dx),
                ("set_dscale", (dscale,), "get_dscale", dscale),
                ("set_dbias", (dbias,), "get_dbias", dbias),
                ("set_normalized_dim_count", (1,), "get_normalized_dim_count", 1),
                (
                    "set_saved_mean_and_inv_variance",
                    (saved_mean, saved_inv_variance),
                    "get_mean",
                    saved_mean,
                ),
                (
                    "set_saved_mean_and_inv_variance",
                    (saved_mean, saved_inv_variance),
                    "get_inv_variance",
                    saved_inv_variance,
                ),
            ),
        )
