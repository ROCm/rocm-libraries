# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for block-scale quantization operations: plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .helpers import (
    call_attribute_methods,
    build_all_plans_or_skip,
    create_float_graph,
    execute_zeros,
)


@pytest.mark.gpu
class TestBlockScaleDequantize:
    """Tests for block-scale dequantization operation-graph construction."""

    def test_builds_operation_graph(self):
        """Dequantize output y must stay virtual (fused-only op); no standalone execute."""
        graph = create_float_graph()
        x = hipdnn.Tensor.create([2, 64, 32, 32], hipdnn.DataType.FLOAT)
        scale = hipdnn.Tensor.create([2, 2, 32, 32], hipdnn.DataType.FLOAT)
        graph.block_scale_dequantize(
            x,
            scale,
            hipdnn.BlockScaleDequantizeAttributes().set_block_size([32]),
        )

        build_all_plans_or_skip(graph)


@pytest.mark.gpu
class TestBlockScaleQuantize:
    """Tests for block-scale quantization operation-graph construction."""

    def test_builds_operation_graph(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([2, 64, 32, 32], hipdnn.DataType.FLOAT)
        outputs = graph.block_scale_quantize(
            x,
            hipdnn.BlockScaleQuantizeAttributes().set_block_size(32),
        )
        assert isinstance(outputs, tuple)
        assert len(outputs) == 2
        for output in outputs:
            output.set_output(True)
            output.set_data_type(hipdnn.DataType.FLOAT)

        handle = build_all_plans_or_skip(graph)
        execute_zeros(
            graph,
            [(x, np.float32)] + [(output, np.float32) for output in outputs],
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
