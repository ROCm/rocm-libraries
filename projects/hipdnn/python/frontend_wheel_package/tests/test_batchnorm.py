# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for batch normalization training and backward passes: plan-building
and stubbed execution.

Batchnorm inference is covered separately in test_normalization.py.
"""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .helpers import (
    call_attribute_methods,
    build_all_plans,
    create_float_graph,
    execute_zeros,
)


def build_batchnorm_training_graph(n=4, c=8, h=8, w=8):
    """Build a batchnorm (training) graph.

    Returns (graph, x, scale, bias, y, mean, inv_var). Per-channel scale/bias
    use [1, C, 1, 1] shapes; mean and inv_variance are produced as per-channel
    outputs.
    """
    graph = create_float_graph()
    graph.set_name("batchnorm_training_test")

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("x")

    scale = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    scale.set_name("scale")

    bias = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    bias.set_name("bias")

    epsilon = hipdnn.Tensor()
    epsilon.set_name("epsilon")
    epsilon.set_value(1e-5)

    attrs = hipdnn.BatchnormAttributes()
    attrs.set_name("batchnorm_node")
    attrs.set_epsilon(epsilon)

    y, mean, inv_variance, _next_mean, _next_var = graph.batchnorm(
        x, scale, bias, attrs
    )
    y.set_name("y")
    y.set_output(True)
    mean.set_name("mean")
    mean.set_output(True)
    inv_variance.set_name("inv_variance")
    inv_variance.set_output(True)

    return graph, x, scale, bias, y, mean, inv_variance


def build_batchnorm_backward_graph(n=4, c=8, h=8, w=8):
    """Build a batchnorm_backward graph.

    Returns (graph, dy, x, scale, dx, dscale, dbias).
    """
    graph = create_float_graph()
    graph.set_name("batchnorm_backward_test")

    dy = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    dy.set_name("dy")

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("x")

    scale = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    scale.set_name("scale")

    attrs = hipdnn.BatchnormBackwardAttributes()
    attrs.set_name("batchnorm_backward_node")

    dx, dscale, dbias = graph.batchnorm_backward(dy, x, scale, attrs)
    dx.set_name("dx")
    dx.set_output(True)
    dscale.set_name("dscale")
    dscale.set_output(True)
    dbias.set_name("dbias")
    dbias.set_output(True)

    return graph, dy, x, scale, dx, dscale, dbias


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
