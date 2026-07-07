# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for batch normalization (inference)."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .helpers import build_all_plans, create_float_graph, execute_graph


def build_batchnorm_inference_graph(n=4, c=8, h=8, w=8):
    """Build a batchnorm inference graph.

    Returns (graph, x, mean, inv_variance, scale, bias, y). Per-channel
    mean/inv_variance/scale/bias use [1, C, 1, 1] shapes.
    """
    graph = create_float_graph()
    graph.set_name("batchnorm_inference_test")

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    x.set_name("x")

    mean = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    mean.set_name("mean")

    inv_variance = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    inv_variance.set_name("inv_variance")

    scale = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    scale.set_name("scale")

    bias = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    bias.set_name("bias")

    attrs = hipdnn.BatchnormInferenceAttributes()
    attrs.set_name("batchnorm_inference_node")

    y = graph.batchnorm_inference(x, mean, inv_variance, scale, bias, attrs)
    y.set_name("y")
    y.set_output(True)

    return graph, x, mean, inv_variance, scale, bias, y


@pytest.mark.gpu
class TestBatchnormInference:
    """Tests for batchnorm inference end-to-end pipeline."""

    def test_execution_produces_nonzero_output(self):
        """Full end-to-end batchnorm inference: execute and verify output."""
        graph, x, mean, inv_variance, scale, bias, y = build_batchnorm_inference_graph()

        handle = build_all_plans(graph)

        x_data = np.random.uniform(0.0, 1.0, x.get_dim()).astype(np.float32)
        mean_data = np.random.uniform(0.0, 1.0, mean.get_dim()).astype(np.float32)
        inv_var_data = np.random.uniform(0.5, 1.5, inv_variance.get_dim()).astype(
            np.float32
        )
        scale_data = np.random.uniform(0.5, 1.5, scale.get_dim()).astype(np.float32)
        bias_data = np.random.uniform(0.0, 1.0, bias.get_dim()).astype(np.float32)
        y_data = np.zeros(y.get_dim(), dtype=np.float32)

        tensor_data = {
            x.get_uid(): x_data,
            mean.get_uid(): mean_data,
            inv_variance.get_uid(): inv_var_data,
            scale.get_uid(): scale_data,
            bias.get_uid(): bias_data,
            y.get_uid(): y_data,
        }

        results = execute_graph(graph, tensor_data, handle)
        y_result = results[y.get_uid()]

        assert not np.all(y_result == 0), "Batchnorm inference output is all zeros"

    def test_execution_matches_reference(self):
        """Batchnorm inference matches the y = scale*(x-mean)*inv_var + bias formula."""
        graph, x, mean, inv_variance, scale, bias, y = build_batchnorm_inference_graph(
            n=2, c=4, h=4, w=4
        )

        handle = build_all_plans(graph)

        x_data = np.random.uniform(-2.0, 2.0, x.get_dim()).astype(np.float32)
        mean_data = np.random.uniform(-1.0, 1.0, mean.get_dim()).astype(np.float32)
        inv_var_data = np.random.uniform(0.5, 1.5, inv_variance.get_dim()).astype(
            np.float32
        )
        scale_data = np.random.uniform(0.5, 1.5, scale.get_dim()).astype(np.float32)
        bias_data = np.random.uniform(-1.0, 1.0, bias.get_dim()).astype(np.float32)
        y_data = np.zeros(y.get_dim(), dtype=np.float32)

        tensor_data = {
            x.get_uid(): x_data,
            mean.get_uid(): mean_data,
            inv_variance.get_uid(): inv_var_data,
            scale.get_uid(): scale_data,
            bias.get_uid(): bias_data,
            y.get_uid(): y_data,
        }

        results = execute_graph(graph, tensor_data, handle)
        y_result = results[y.get_uid()]

        expected = scale_data * (x_data - mean_data) * inv_var_data + bias_data
        np.testing.assert_allclose(y_result, expected, rtol=2e-3, atol=2e-3)


class TestBatchnormInferenceAttributes:
    """Attribute accessors for BatchnormInference (no GPU required)."""

    def test_setters_chain(self):
        """Setters return self for chaining and store the values."""
        attrs = hipdnn.BatchnormInferenceAttributes()
        result = attrs.set_name("bn_inf").set_compute_data_type(hipdnn.DataType.FLOAT)
        assert result is attrs
        assert attrs.get_name() == "bn_inf"
        assert attrs.get_compute_data_type() == hipdnn.DataType.FLOAT
