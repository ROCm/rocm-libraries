# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for resampling operations: plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .helpers import (
    call_attribute_methods,
    build_all_plans,
    create_float_graph,
    execute_zeros,
)


def _resample_fwd_attributes():
    return (
        hipdnn.ResampleFwdAttributes()
        .set_resample_mode(hipdnn.ResampleMode.MAXPOOL)
        .set_padding_mode(hipdnn.PaddingMode.ZERO_PAD)
        .set_pre_padding([0, 0])
        .set_post_padding([0, 0])
        .set_stride([2, 2])
        .set_window([2, 2])
    )


def _resample_bwd_attributes():
    return (
        hipdnn.ResampleBwdAttributes()
        .set_resample_mode(hipdnn.ResampleMode.AVGPOOL_EXCLUDE_PADDING)
        .set_padding_mode(hipdnn.PaddingMode.ZERO_PAD)
        .set_pre_padding([1, 1])
        .set_post_padding([1, 1])
        .set_stride([2, 2])
        .set_window([3, 3])
    )


@pytest.mark.gpu
class TestResample:
    """Tests for resample operation-graph construction."""

    def test_execution_succeeds(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([1, 3, 4, 4], hipdnn.DataType.FLOAT)
        outputs = graph.resample(x, _resample_fwd_attributes())
        assert isinstance(outputs, tuple)
        assert len(outputs) == 2
        y, index = outputs
        assert index is None  # requires set_generate_index(True), not requested here
        y.set_output(True)

        handle = build_all_plans(graph)
        execute_zeros(graph, [(x, np.float32), (y, np.float32)], handle)


@pytest.mark.gpu
class TestResampleFwd:
    """Tests for resample forward operation-graph construction."""

    def test_execution_succeeds(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([1, 3, 4, 4], hipdnn.DataType.FLOAT)
        y = graph.resample_fwd(x, _resample_fwd_attributes())
        y.set_output(True)
        y.set_data_type(hipdnn.DataType.FLOAT)

        handle = build_all_plans(graph)
        execute_zeros(graph, [(x, np.float32), (y, np.float32)], handle)


@pytest.mark.gpu
class TestResampleBwd:
    """Tests for resample backward operation-graph construction."""

    def test_execution_succeeds(self):
        graph = create_float_graph()
        dy = hipdnn.Tensor.create([1, 3, 16, 16], hipdnn.DataType.FLOAT)
        dx = graph.resample_bwd(dy, _resample_bwd_attributes())
        dx.set_output(True)
        dx.set_data_type(hipdnn.DataType.FLOAT)

        handle = build_all_plans(graph)
        execute_zeros(graph, [(dy, np.float32), (dx, np.float32)], handle)

    def test_builds_operation_graph_with_maxpool_index(self):
        """resample_bwd's optional ``index`` argument accepts a real index tensor."""
        graph = create_float_graph()
        attrs = (
            hipdnn.ResampleBwdAttributes()
            .set_resample_mode(hipdnn.ResampleMode.MAXPOOL)
            .set_padding_mode(hipdnn.PaddingMode.ZERO_PAD)
            .set_pre_padding([0, 0])
            .set_post_padding([0, 0])
            .set_stride([2, 2])
            .set_window([2, 2])
        )
        dy = hipdnn.Tensor.create([1, 3, 16, 16], hipdnn.DataType.FLOAT)
        index = hipdnn.Tensor.create([1, 3, 16, 16], hipdnn.DataType.INT32)
        dx = graph.resample_bwd(dy, attrs, index=index)
        dx.set_output(True)
        dx.set_data_type(hipdnn.DataType.FLOAT)

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(dy, np.float32), (index, np.int32), (dx, np.float32)],
            handle,
        )


class TestResampleAttributeBindings:
    """Every resample attribute binding round-trips through its getter (no GPU required)."""

    def test_forward_methods_are_callable(self):
        x = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        y = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        index = hipdnn.Tensor.create([1], hipdnn.DataType.INT32)
        call_attribute_methods(
            hipdnn.ResampleFwdAttributes(),
            (
                ("set_name", ("resample_fwd",), "get_name", "resample_fwd"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_x", (x,), "get_x", x),
                ("set_y", (y,), "get_y", y),
                ("set_index", (index,), "get_index", index),
                ("set_pre_padding", ([1],), "get_pre_padding", [1]),
                ("set_post_padding", ([2],), "get_post_padding", [2]),
                ("set_stride", ([3],), "get_stride", [3]),
                ("set_window", ([4],), "get_window", [4]),
                (
                    "set_resample_mode",
                    (hipdnn.ResampleMode.MAXPOOL,),
                    "get_resample_mode",
                    hipdnn.ResampleMode.MAXPOOL,
                ),
                (
                    "set_padding_mode",
                    (hipdnn.PaddingMode.ZERO_PAD,),
                    "get_padding_mode",
                    hipdnn.PaddingMode.ZERO_PAD,
                ),
                ("set_generate_index", (True,), "get_generate_index", True),
            ),
        )

    def test_backward_methods_are_callable(self):
        dy = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        index = hipdnn.Tensor.create([1], hipdnn.DataType.INT32)
        dx = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.ResampleBwdAttributes(),
            (
                ("set_name", ("resample_bwd",), "get_name", "resample_bwd"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_dy", (dy,), "get_dy", dy),
                ("set_index", (index,), "get_index", index),
                ("set_dx", (dx,), "get_dx", dx),
                ("set_pre_padding", ([1],), "get_pre_padding", [1]),
                ("set_post_padding", ([2],), "get_post_padding", [2]),
                ("set_stride", ([3],), "get_stride", [3]),
                ("set_window", ([4],), "get_window", [4]),
                (
                    "set_resample_mode",
                    (hipdnn.ResampleMode.MAXPOOL,),
                    "get_resample_mode",
                    hipdnn.ResampleMode.MAXPOOL,
                ),
                (
                    "set_padding_mode",
                    (hipdnn.PaddingMode.ZERO_PAD,),
                    "get_padding_mode",
                    hipdnn.PaddingMode.ZERO_PAD,
                ),
            ),
        )
