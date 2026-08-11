# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for batch normalization (inference): plan-building and stubbed execution."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .helpers import (
    call_attribute_methods,
    build_all_plans,
    create_float_graph,
    execute_zeros,
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
    """Tests for batchnorm inference plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        """Batchnorm inference builds plans and executes against the stub engine without erroring."""
        graph, x, mean, inv_variance, scale, bias, y = build_batchnorm_inference_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [
                (x, np.float32),
                (mean, np.float32),
                (inv_variance, np.float32),
                (scale, np.float32),
                (bias, np.float32),
                (y, np.float32),
            ],
            handle,
        )


@pytest.mark.gpu
class TestBatchnormInferenceVariance:
    """Tests for batchnorm inference (from variance) plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        """Batchnorm inference-variance builds plans and executes against the stub engine without erroring."""
        graph, x, mean, variance, scale, bias, y = (
            build_batchnorm_inference_variance_graph()
        )

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [
                (x, np.float32),
                (mean, np.float32),
                (variance, np.float32),
                (scale, np.float32),
                (bias, np.float32),
                (y, np.float32),
            ],
            handle,
        )


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
