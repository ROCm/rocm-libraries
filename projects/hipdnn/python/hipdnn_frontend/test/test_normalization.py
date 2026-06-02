# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for batch normalization (inference)."""

import pytest

import hipdnn_frontend as hipdnn

from .helpers import build_operation_graph, create_float_graph


def build_batchnorm_inference_graph(n=4, c=8, h=8, w=8):
    """Build a batchnorm inference graph returning (graph, x, y).

    Per-channel mean/inv_variance/scale/bias use [1, C, 1, 1] shapes.
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

    return graph, x, y


@pytest.mark.gpu
class TestBatchnormInference:
    """Tests for batchnorm inference end-to-end pipeline."""

    def test_builds_operation_graph(self):
        """A batchnorm inference graph lowers to a backend operation graph."""
        graph, x, y = build_batchnorm_inference_graph()

        build_operation_graph(graph)
