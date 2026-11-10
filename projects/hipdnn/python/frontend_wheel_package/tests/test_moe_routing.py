# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for MoE routing: shape inference, validation, plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .graph_builders import build_moe_routing_graph
from .helpers import (
    build_all_plans,
    call_attribute_methods,
    create_float_graph,
    execute_zeros,
)


@pytest.mark.gpu
class TestMoeRouting:
    """Tests for MoE routing plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        graph, topk_ids, first_token_offset, token_index, token_ks = (
            build_moe_routing_graph()
        )

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [
                (topk_ids, np.int32),
                (first_token_offset, np.int32),
                (token_index, np.int32),
                (token_ks, np.int32),
            ],
            handle,
        )


class TestMoeRoutingShapeInference:
    """Outputs are [E + 1, 1, 1] and [1, S * top_k, 1], INT32 (no GPU required)."""

    def test_infers_outputs(self):
        graph, _, first_token_offset, token_index, token_ks = build_moe_routing_graph()
        assert graph.validate().is_good()

        assert first_token_offset.get_dim() == [5, 1, 1]
        assert token_index.get_dim() == [1, 8, 1]
        assert token_ks.get_dim() == [1, 8, 1]
        for tensor in (first_token_offset, token_index, token_ks):
            assert tensor.get_data_type() == hipdnn.DataType.INT32

    def test_single_expert_is_rejected(self):
        graph = create_float_graph()
        topk_ids = hipdnn.Tensor.create([4, 2], hipdnn.DataType.INT32)
        graph.moe_routing(topk_ids, hipdnn.MoeRoutingAttributes().set_num_experts(1))
        assert not graph.validate().is_good()

    def test_non_int32_ids_are_rejected(self):
        graph = create_float_graph()
        topk_ids = hipdnn.Tensor.create([4, 2], hipdnn.DataType.FLOAT)
        graph.moe_routing(topk_ids, hipdnn.MoeRoutingAttributes().set_num_experts(4))
        assert not graph.validate().is_good()


class TestMoeRoutingAttributeBindings:
    """Every MoE routing attribute binding round-trips through its getter (no GPU required)."""

    def test_methods_are_callable(self):
        topk_ids = hipdnn.Tensor.create([1, 1], hipdnn.DataType.INT32)
        first_token_offset = hipdnn.Tensor.create([3, 1, 1], hipdnn.DataType.INT32)
        token_index = hipdnn.Tensor.create([1, 1, 1], hipdnn.DataType.INT32)
        token_ks = hipdnn.Tensor.create([1, 1, 1], hipdnn.DataType.INT32)
        call_attribute_methods(
            hipdnn.MoeRoutingAttributes(),
            (
                ("set_name", ("moe_routing",), "get_name", "moe_routing"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_topk_ids", (topk_ids,), "get_topk_ids", topk_ids),
                (
                    "set_first_token_offset",
                    (first_token_offset,),
                    "get_first_token_offset",
                    first_token_offset,
                ),
                ("set_token_index", (token_index,), "get_token_index", token_index),
                ("set_token_ks", (token_ks,), "get_token_ks", token_ks),
                ("set_num_experts", (2,), "get_num_experts", 2),
            ),
        )
