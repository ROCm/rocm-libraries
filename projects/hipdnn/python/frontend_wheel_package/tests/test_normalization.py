# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for batch normalization (inference)."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .helpers import (
    call_attribute_methods,
    build_all_plans,
    build_operation_graph,
    create_float_graph,
    execute_graph,
)


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


def build_batchnorm_inference_variance_graph(n=4, c=8, h=8, w=8):
    """Build batchnorm inference from variance and a compile-time epsilon."""
    graph = create_float_graph()
    graph.set_name("batchnorm_inference_variance_test")

    x = hipdnn.Tensor.create([n, c, h, w], hipdnn.DataType.FLOAT)
    mean = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    variance = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    scale = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    bias = hipdnn.Tensor.create([1, c, 1, 1], hipdnn.DataType.FLOAT)
    epsilon = hipdnn.Tensor()
    epsilon.set_value(1e-5)

    y = graph.batchnorm_inference_variance_ext(
        x,
        mean,
        variance,
        scale,
        bias,
        epsilon,
        hipdnn.BatchnormInferenceAttributesVarianceExt(),
    )
    y.set_output(True)
    return graph, x, mean, variance, scale, bias, y


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


@pytest.mark.gpu
class TestBatchnormInferenceVariance:
    """GPU integration tests for batchnorm inference with variance."""

    def test_builds_operation_graph(self):
        graph, *_ = build_batchnorm_inference_variance_graph()
        build_operation_graph(graph)

    def test_execution_matches_reference(self):
        graph, x, mean, variance, scale, bias, y = (
            build_batchnorm_inference_variance_graph(n=2, c=4, h=4, w=4)
        )
        handle = hipdnn.create_handle()
        assert graph.validate().is_good()
        assert graph.build_operation_graph(handle).is_good()
        plans = graph.create_execution_plans()
        if plans.is_bad():
            pytest.skip(f"No compatible engine: {plans.get_message()}")
        support = graph.check_support()
        if support.is_bad():
            pytest.skip(f"No supported execution plan: {support.get_message()}")
        assert graph.build_plans().is_good()

        x_data = np.random.uniform(-2.0, 2.0, x.get_dim()).astype(np.float32)
        mean_data = np.random.uniform(-1.0, 1.0, mean.get_dim()).astype(np.float32)
        variance_data = np.random.uniform(0.5, 1.5, variance.get_dim()).astype(
            np.float32
        )
        scale_data = np.random.uniform(0.5, 1.5, scale.get_dim()).astype(np.float32)
        bias_data = np.random.uniform(-1.0, 1.0, bias.get_dim()).astype(np.float32)
        y_data = np.zeros(y.get_dim(), dtype=np.float32)

        results = execute_graph(
            graph,
            {
                x.get_uid(): x_data,
                mean.get_uid(): mean_data,
                variance.get_uid(): variance_data,
                scale.get_uid(): scale_data,
                bias.get_uid(): bias_data,
                y.get_uid(): y_data,
            },
            handle,
        )
        expected = (
            scale_data * (x_data - mean_data) / np.sqrt(variance_data + 1e-5)
            + bias_data
        )
        np.testing.assert_allclose(results[y.get_uid()], expected, rtol=2e-3, atol=2e-3)


class TestBatchnormInferenceAttributeBindings:
    """Every batchnorm inference attribute binding round-trips through its getter (no GPU required)."""

    def test_inference_methods_are_callable(self):
        x = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        mean = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        inv_variance = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        scale = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        bias = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        y = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.BatchnormInferenceAttributes(),
            (
                (
                    "set_name",
                    ("batchnorm_inference",),
                    "get_name",
                    "batchnorm_inference",
                ),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_x", (x,), "get_x", x),
                ("set_mean", (mean,), "get_mean", mean),
                ("set_inv_variance", (inv_variance,), "get_inv_variance", inv_variance),
                ("set_scale", (scale,), "get_scale", scale),
                ("set_bias", (bias,), "get_bias", bias),
                ("set_y", (y,), "get_y", y),
            ),
        )

    def test_variance_extension_methods_are_callable(self):
        x = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        mean = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        variance = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        scale = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        bias = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        epsilon = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        y = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.BatchnormInferenceAttributesVarianceExt(),
            (
                (
                    "set_name",
                    ("batchnorm_inference_variance",),
                    "get_name",
                    "batchnorm_inference_variance",
                ),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_x", (x,), "get_x", x),
                ("set_mean", (mean,), "get_mean", mean),
                ("set_variance", (variance,), "get_variance", variance),
                ("set_scale", (scale,), "get_scale", scale),
                ("set_bias", (bias,), "get_bias", bias),
                ("set_epsilon", (epsilon,), "get_epsilon", epsilon),
                ("set_y", (y,), "get_y", y),
            ),
        )
