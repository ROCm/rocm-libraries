# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for batch normalization training and backward passes.

Batchnorm inference is covered separately in test_normalization.py.
"""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .helpers import build_all_plans, create_float_graph, execute_graph


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
    """Tests for batchnorm training end-to-end pipeline."""

    def test_execution_produces_nonzero_output(self):
        """Full end-to-end batchnorm training: execute and verify outputs."""
        graph, x, scale, bias, y, mean, inv_variance = build_batchnorm_training_graph()

        handle = build_all_plans(graph)

        x_data = np.random.uniform(-2.0, 2.0, x.get_dim()).astype(np.float32)
        scale_data = np.random.uniform(0.5, 1.5, scale.get_dim()).astype(np.float32)
        bias_data = np.random.uniform(0.0, 1.0, bias.get_dim()).astype(np.float32)
        y_data = np.zeros(y.get_dim(), dtype=np.float32)
        mean_data = np.zeros(mean.get_dim(), dtype=np.float32)
        inv_var_data = np.zeros(inv_variance.get_dim(), dtype=np.float32)

        tensor_data = {
            x.get_uid(): x_data,
            scale.get_uid(): scale_data,
            bias.get_uid(): bias_data,
            y.get_uid(): y_data,
            mean.get_uid(): mean_data,
            inv_variance.get_uid(): inv_var_data,
        }

        results = execute_graph(graph, tensor_data, handle)

        assert not np.all(results[y.get_uid()] == 0), "Training y is all zeros"
        # Mean over the normalized batch should be near zero per channel.
        np.testing.assert_allclose(
            results[mean.get_uid()],
            x_data.mean(axis=(0, 2, 3)).reshape(mean.get_dim()),
            rtol=2e-3,
            atol=2e-3,
        )


@pytest.mark.gpu
class TestBatchnormBackward:
    """Tests for batchnorm backward end-to-end pipeline."""

    def test_execution_produces_nonzero_output(self):
        """Full end-to-end batchnorm_backward: execute and verify gradients."""
        graph, dy, x, scale, dx, dscale, dbias = build_batchnorm_backward_graph()

        handle = build_all_plans(graph)

        dy_data = np.random.uniform(-1.0, 1.0, dy.get_dim()).astype(np.float32)
        x_data = np.random.uniform(-2.0, 2.0, x.get_dim()).astype(np.float32)
        scale_data = np.random.uniform(0.5, 1.5, scale.get_dim()).astype(np.float32)
        dx_data = np.zeros(dx.get_dim(), dtype=np.float32)
        dscale_data = np.zeros(dscale.get_dim(), dtype=np.float32)
        dbias_data = np.zeros(dbias.get_dim(), dtype=np.float32)

        tensor_data = {
            dy.get_uid(): dy_data,
            x.get_uid(): x_data,
            scale.get_uid(): scale_data,
            dx.get_uid(): dx_data,
            dscale.get_uid(): dscale_data,
            dbias.get_uid(): dbias_data,
        }

        results = execute_graph(graph, tensor_data, handle)

        assert not np.all(results[dx.get_uid()] == 0), "Backward dx is all zeros"
        # dbias is the sum of dy over the batch/spatial dims per channel.
        np.testing.assert_allclose(
            results[dbias.get_uid()],
            dy_data.sum(axis=(0, 2, 3)).reshape(dbias.get_dim()),
            rtol=2e-3,
            atol=2e-3,
        )


class TestBatchnormAttributes:
    """Attribute accessors for batchnorm training/backward (no GPU required)."""

    def test_training_setters_chain(self):
        """Training setters return self for chaining and store the values."""
        attrs = hipdnn.BatchnormAttributes()
        result = attrs.set_name("bn").set_compute_data_type(hipdnn.DataType.FLOAT)
        assert result is attrs
        assert attrs.get_name() == "bn"
        assert attrs.get_compute_data_type() == hipdnn.DataType.FLOAT

    def test_set_epsilon_returns_self(self):
        """set_epsilon returns self and stores the epsilon tensor."""
        epsilon = hipdnn.Tensor()
        epsilon.set_name("epsilon")
        epsilon.set_value(1e-5)

        attrs = hipdnn.BatchnormAttributes()
        result = attrs.set_epsilon(epsilon)
        assert result is attrs
        assert attrs.get_epsilon().get_name() == "epsilon"

    def test_backward_setters_chain(self):
        """Backward setters return self for chaining and store the values."""
        attrs = hipdnn.BatchnormBackwardAttributes()
        result = attrs.set_name("bn_bwd").set_compute_data_type(hipdnn.DataType.FLOAT)
        assert result is attrs
        assert attrs.get_name() == "bn_bwd"
        assert attrs.get_compute_data_type() == hipdnn.DataType.FLOAT
