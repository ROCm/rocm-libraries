# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for MoE grouped matrix multiplication: plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .graph_builders import build_moe_grouped_matmul_graph
from .helpers import build_all_plans, call_attribute_methods, execute_zeros


@pytest.mark.gpu
class TestMoeGroupedMatmul:
    """Tests for MoE grouped matmul plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        graph, token, weight, first_token_offset, token_index, token_ks, output = (
            build_moe_grouped_matmul_graph()
        )

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [
                (token, np.float32),
                (weight, np.float32),
                (first_token_offset, np.int32),
                (token_index, np.int32),
                (token_ks, np.int32),
                (output, np.float32),
            ],
            handle,
        )


class TestMoeGroupedMatmulAttributeBindings:
    """Every MoE grouped matmul attribute binding round-trips through its getter (no GPU required)."""

    def test_methods_are_callable(self):
        token = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        weight = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        first_token_offset = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        token_index = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        token_ks = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        output = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.MoeGroupedMatmulAttributes(),
            (
                ("set_name", ("moe_grouped_matmul",), "get_name", "moe_grouped_matmul"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_token", (token,), "get_token", token),
                ("set_weight", (weight,), "get_weight", weight),
                (
                    "set_first_token_offset",
                    (first_token_offset,),
                    "get_first_token_offset",
                    first_token_offset,
                ),
                ("set_token_index", (token_index,), "get_token_index", token_index),
                ("set_token_ks", (token_ks,), "get_token_ks", token_ks),
                ("set_output", (output,), "get_output", output),
                (
                    "set_mode",
                    (hipdnn.MoeGroupedMatmulMode.SCATTER,),
                    "get_mode",
                    hipdnn.MoeGroupedMatmulMode.SCATTER,
                ),
                ("set_top_k", (1,), "get_top_k", 1),
            ),
        )
