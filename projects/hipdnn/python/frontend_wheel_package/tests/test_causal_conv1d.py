# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for causal_conv1d: graph building, validation, plan-building and execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .graph_builders import build_causal_conv1d_graph
from .helpers import (
    build_all_plans,
    call_attribute_methods,
    create_float_graph,
    execute_zeros,
)


@pytest.mark.gpu
class TestCausalConv1d:
    """Tests for causal_conv1d plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        graph, x, weight, cu_seqlens, bias, initial_state, y, final_state = (
            build_causal_conv1d_graph()
        )

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [
                (x, np.float32),
                (weight, np.float32),
                (cu_seqlens, np.int32),
                (bias, np.float32),
                (initial_state, np.float32),
                (y, np.float32),
                (final_state, np.float32),
            ],
            handle,
        )


class TestCausalConv1dGraph:
    """Output shape inference and validation (no GPU)."""

    def test_infers_output_shapes(self):
        graph, _, _, _, _, _, y, final_state = build_causal_conv1d_graph()
        assert graph.validate().is_good()
        assert y.get_dim() == [6, 4]
        assert final_state.get_dim() == [2, 4, 2]
        assert final_state.get_stride() == [8, 2, 1]

    def test_final_state_is_none_unless_requested(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([3, 2], hipdnn.DataType.FLOAT)
        weight = hipdnn.Tensor.create([2, 4], hipdnn.DataType.FLOAT)
        cu_seqlens = hipdnn.Tensor.create([2], hipdnn.DataType.INT64)

        y, final_state = graph.causal_conv1d(
            x, weight, cu_seqlens, hipdnn.CausalConv1dAttributes()
        )
        assert final_state is None
        y.set_output(True)
        assert graph.validate().is_good()

    def test_state_pool_infers_pool_rows(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([2, 4], hipdnn.DataType.FLOAT)
        weight = hipdnn.Tensor.create([4, 3], hipdnn.DataType.FLOAT)
        cu_seqlens = hipdnn.Tensor.create([3], hipdnn.DataType.INT32)
        pool = hipdnn.Tensor.create([5, 4, 2], hipdnn.DataType.FLOAT)
        state_indices = hipdnn.Tensor.create([2], hipdnn.DataType.INT32)

        attributes = (
            hipdnn.CausalConv1dAttributes()
            .set_initial_state(pool)
            .set_state_indices(state_indices)
            .set_output_final_state(True)
        )
        y, final_state = graph.causal_conv1d(x, weight, cu_seqlens, attributes)
        y.set_output(True)
        final_state.set_output(True)
        assert graph.validate().is_good()
        assert final_state.get_dim() == [5, 4, 2]

    def test_state_indices_without_initial_state_is_rejected(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([2, 4], hipdnn.DataType.FLOAT)
        weight = hipdnn.Tensor.create([4, 3], hipdnn.DataType.FLOAT)
        cu_seqlens = hipdnn.Tensor.create([3], hipdnn.DataType.INT32)
        state_indices = hipdnn.Tensor.create([2], hipdnn.DataType.INT32)

        y, _ = graph.causal_conv1d(
            x,
            weight,
            cu_seqlens,
            hipdnn.CausalConv1dAttributes().set_state_indices(state_indices),
        )
        y.set_output(True)
        assert not graph.validate().is_good()

    def test_kernel_width_below_two_is_rejected(self):
        graph = create_float_graph()
        x = hipdnn.Tensor.create([2, 4], hipdnn.DataType.FLOAT)
        weight = hipdnn.Tensor.create([4, 1], hipdnn.DataType.FLOAT)
        cu_seqlens = hipdnn.Tensor.create([2], hipdnn.DataType.INT32)

        y, _ = graph.causal_conv1d(
            x, weight, cu_seqlens, hipdnn.CausalConv1dAttributes()
        )
        y.set_output(True)
        assert not graph.validate().is_good()


class TestCausalConv1dAttributeBindings:
    """Every causal_conv1d attribute binding round-trips through its getter (no GPU)."""

    def test_methods_are_callable(self):
        bias = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        initial_state = hipdnn.Tensor.create([1, 1, 1], hipdnn.DataType.FLOAT)
        state_indices = hipdnn.Tensor.create([1], hipdnn.DataType.INT32)
        call_attribute_methods(
            hipdnn.CausalConv1dAttributes(),
            (
                ("set_name", ("conv",), "get_name", "conv"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_bias", (bias,), "get_bias", bias),
                (
                    "set_initial_state",
                    (initial_state,),
                    "get_initial_state",
                    initial_state,
                ),
                (
                    "set_state_indices",
                    (state_indices,),
                    "get_state_indices",
                    state_indices,
                ),
                (
                    "set_activation",
                    (hipdnn.CausalConv1dActivation.SILU,),
                    "get_activation",
                    hipdnn.CausalConv1dActivation.SILU,
                ),
                ("set_output_final_state", (True,), "get_output_final_state", True),
            ),
        )
