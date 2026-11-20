# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""CPU-tier checks of the graphs the source-tree samples build."""

import importlib.util
from pathlib import Path

import numpy as np

import hipdnn_frontend as hipdnn


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def _load_sample(name):
    spec = importlib.util.spec_from_file_location(name, SAMPLES_DIR / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cudnn_style_conv_bias_relu_graph_validates():
    sample = _load_sample("cudnn_style_conv_bias_relu")
    x = np.zeros((2, 8, 16, 16), np.float32)
    w = np.zeros((16, 8, 3, 3), np.float32)
    b = np.zeros((1, 16, 1, 1), np.float32)
    padding, stride, dilation = [1, 1], [1, 1], [1, 1]

    graph, _, _, _, y = sample.build_graph(x, w, b, padding, stride, dilation)

    assert isinstance(graph, hipdnn.pygraph)
    graph.validate()
    expected = sample.reference_conv_bias_relu(x, w, b, padding, stride, dilation)
    assert y.get_dim() == list(expected.shape)
    assert y.get_data_type() == hipdnn.data_type.FLOAT
