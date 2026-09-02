# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for batch normalization training and backward passes: plan-building
and stubbed execution.

Batchnorm inference is covered separately in test_normalization.py.
"""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .graph_builders import (
    build_batchnorm_backward_graph,
    build_batchnorm_training_graph,
)
from .helpers import build_all_plans, call_attribute_methods, execute_zeros


@pytest.mark.gpu
class TestBatchnormTraining:
    """Tests for batchnorm training plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        """Batchnorm training builds plans and executes against the stub engine without erroring."""
        graph, x, scale, bias, y, mean, inv_variance = build_batchnorm_training_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [
                (x, np.float32),
                (scale, np.float32),
                (bias, np.float32),
                (y, np.float32),
                (mean, np.float32),
                (inv_variance, np.float32),
            ],
            handle,
        )


@pytest.mark.gpu
class TestBatchnormBackward:
    """Tests for batchnorm backward plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        """Batchnorm backward builds plans and executes against the stub engine without erroring."""
        graph, dy, x, scale, dx, dscale, dbias = build_batchnorm_backward_graph()

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


class TestBatchnormAttributeBindings:
    """Every Batchnorm attribute binding round-trips through its getter (no GPU required)."""

    def test_batchnorm_methods_are_callable(self):
        x = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        scale = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        bias = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        epsilon = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        peer_stat = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        prev_running_mean = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        prev_running_variance = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        momentum = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        y = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        mean = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        inv_variance = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        next_running_mean = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        next_running_variance = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        combined_mean = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        combined_variance = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        combined_momentum = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.BatchnormAttributes(),
            (
                ("set_name", ("batchnorm",), "get_name", "batchnorm"),
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
                ("set_peer_stats", ([peer_stat],), "get_peer_stats", [peer_stat]),
                (
                    "set_prev_running_mean",
                    (prev_running_mean,),
                    "get_prev_running_mean",
                    prev_running_mean,
                ),
                (
                    "set_prev_running_variance",
                    (prev_running_variance,),
                    "get_prev_running_variance",
                    prev_running_variance,
                ),
                ("set_momentum", (momentum,), "get_momentum", momentum),
                ("set_y", (y,), "get_y", y),
                ("set_mean", (mean,), "get_mean", mean),
                ("set_inv_variance", (inv_variance,), "get_inv_variance", inv_variance),
                (
                    "set_next_running_mean",
                    (next_running_mean,),
                    "get_next_running_mean",
                    next_running_mean,
                ),
                (
                    "set_next_running_variance",
                    (next_running_variance,),
                    "get_next_running_variance",
                    next_running_variance,
                ),
                (
                    "set_previous_running_stats",
                    (combined_mean, combined_variance, combined_momentum),
                    "get_prev_running_mean",
                    combined_mean,
                ),
                (
                    "set_previous_running_stats",
                    (combined_mean, combined_variance, combined_momentum),
                    "get_prev_running_variance",
                    combined_variance,
                ),
                (
                    "set_previous_running_stats",
                    (combined_mean, combined_variance, combined_momentum),
                    "get_momentum",
                    combined_momentum,
                ),
            ),
        )

    def test_backward_methods_are_callable(self):
        dy = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        x = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        scale = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        mean = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        inv_variance = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        dx = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        dscale = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        dbias = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        peer_stat = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        saved_mean = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        saved_inv_variance = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.BatchnormBackwardAttributes(),
            (
                ("set_name", ("batchnorm_backward",), "get_name", "batchnorm_backward"),
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
                ("set_dx", (dx,), "get_dx", dx),
                ("set_dscale", (dscale,), "get_dscale", dscale),
                ("set_dbias", (dbias,), "get_dbias", dbias),
                ("set_peer_stats", ([peer_stat],), "get_peer_stats", [peer_stat]),
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
