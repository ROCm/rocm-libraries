# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for batch normalization (inference): plan-building and stubbed execution."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn

from .graph_builders import (
    build_batchnorm_inference_graph,
    build_batchnorm_inference_variance_graph,
)
from .helpers import build_all_plans, call_attribute_methods, execute_zeros


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
