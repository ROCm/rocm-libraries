# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for batch normalization training and backward passes.

Batchnorm inference is covered separately in test_normalization.py.
"""

import pytest

import hipdnn_frontend as hipdnn

from .helpers import build_operation_graph, create_float_graph


def build_batchnorm_training_graph(n=4, c=8, h=8, w=8):
    """Build a batchnorm (training) graph returning (graph, x, y, mean, inv_var).

    Per-channel scale/bias use [1, C, 1, 1] shapes; mean and inv_variance are
    produced as per-channel outputs.
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

    return graph, x, y, mean, inv_variance


def build_batchnorm_backward_graph(n=4, c=8, h=8, w=8):
    """Build a batchnorm_backward graph returning (graph, dx, dscale, dbias)."""
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

    return graph, dx, dscale, dbias


@pytest.mark.gpu
class TestBatchnormTraining:
    """Tests for batchnorm training end-to-end pipeline."""

    def test_builds_operation_graph(self):
        """A batchnorm training graph lowers to a backend operation graph."""
        graph, x, y, mean, inv_variance = build_batchnorm_training_graph()

        build_operation_graph(graph)


@pytest.mark.gpu
class TestBatchnormBackward:
    """Tests for batchnorm backward end-to-end pipeline."""

    def test_builds_operation_graph(self):
        """A batchnorm_backward graph lowers to a backend operation graph."""
        graph, dx, dscale, dbias = build_batchnorm_backward_graph()

        build_operation_graph(graph)
