# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for block-scale quantization operations: plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .graph_builders import (
    build_block_scale_dequantize_graph,
    build_block_scale_quantize_graph,
)
from .helpers import build_all_plans, call_attribute_methods, execute_zeros


@pytest.mark.gpu
class TestBlockScaleDequantize:
    """Tests for block-scale dequantization plan-building."""

    def test_plan_build_succeeds(self):
        """Dequantize output y must stay virtual (fused-only op); no standalone execute."""
        graph = build_block_scale_dequantize_graph()[0]

        build_all_plans(graph)


@pytest.mark.gpu
class TestBlockScaleQuantize:
    """Tests for block-scale quantization plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        graph, x, y, scale = build_block_scale_quantize_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(x, np.float32), (y, np.float32), (scale, np.float32)],
            handle,
        )


class TestBlockScaleAttributeBindings:
    """Every block-scale attribute binding round-trips through its getter (no GPU required)."""

    def test_dequantize_methods_are_callable(self):
        x = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        scale = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        y = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.BlockScaleDequantizeAttributes(),
            (
                (
                    "set_name",
                    ("block_scale_dequantize",),
                    "get_name",
                    "block_scale_dequantize",
                ),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_x", (x,), "get_x", x),
                ("set_scale", (scale,), "get_scale", scale),
                ("set_y", (y,), "get_y", y),
                ("set_block_size", ([4],), "get_block_size", [4]),
                ("set_is_negative_scale", (True,), "get_is_negative_scale", True),
            ),
        )

    def test_quantize_methods_are_callable(self):
        x = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        y = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        scale = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.BlockScaleQuantizeAttributes(),
            (
                (
                    "set_name",
                    ("block_scale_quantize",),
                    "get_name",
                    "block_scale_quantize",
                ),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_x", (x,), "get_x", x),
                ("set_y", (y,), "get_y", y),
                ("set_scale", (scale,), "get_scale", scale),
                ("set_block_size", (5,), "get_block_size", 5),
                ("set_axis", (2,), "get_axis", 2),
                ("set_transpose", (True,), "get_transpose", True),
            ),
        )
