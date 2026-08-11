# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for matrix multiplication: plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .helpers import (
    call_attribute_methods,
    build_all_plans,
    create_float_graph,
    execute_zeros,
)


def build_matmul_graph(m=4, k=3, n=5):
    """Build a matmul graph (A [M, K] x B [K, N] -> C [M, N])."""
    graph = create_float_graph()
    graph.set_name("matmul_test")

    a = hipdnn.Tensor.create([m, k], hipdnn.DataType.FLOAT)
    a.set_name("A")

    b = hipdnn.Tensor.create([k, n], hipdnn.DataType.FLOAT)
    b.set_name("B")

    attrs = hipdnn.MatmulAttributes()
    attrs.set_name("matmul_node")

    c = graph.matmul(a, b, attrs)
    c.set_name("C")
    c.set_output(True)

    return graph, a, b, c


@pytest.mark.gpu
class TestMatmul:
    """Tests for matmul plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        """Matmul builds plans and executes against the stub engine without erroring."""
        graph, a, b, c = build_matmul_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(a, np.float32), (b, np.float32), (c, np.float32)],
            handle,
        )


class TestMatmulAttributeBindings:
    """Every matmul attribute binding round-trips through its getter (no GPU required)."""

    def test_methods_are_callable(self):
        a = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        b = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        c = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        call_attribute_methods(
            hipdnn.MatmulAttributes(),
            (
                ("set_name", ("matmul",), "get_name", "matmul"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_a", (a,), "get_a", a),
                ("set_b", (b,), "get_b", b),
                ("set_c", (c,), "get_c", c),
            ),
        )
