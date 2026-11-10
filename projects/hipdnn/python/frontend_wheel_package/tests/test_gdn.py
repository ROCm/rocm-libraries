# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for Gated DeltaNet (GDN): shape inference, validation, plan-building and stubbed
execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .graph_builders import build_gdn_graph
from .helpers import (
    build_all_plans,
    call_attribute_methods,
    create_float_graph,
    execute_zeros,
)


def _gdn_inputs(
    graph_tokens=4, heads_q=2, heads_v=2, key_dim=4, value_dim=2, sequences=2
):
    q = hipdnn.Tensor.create([graph_tokens, heads_q, key_dim], hipdnn.DataType.FLOAT)
    k = hipdnn.Tensor.create([graph_tokens, heads_q, key_dim], hipdnn.DataType.FLOAT)
    v = hipdnn.Tensor.create([graph_tokens, heads_v, value_dim], hipdnn.DataType.FLOAT)
    heads_out = max(heads_q, heads_v)
    g = hipdnn.Tensor.create([graph_tokens, heads_out], hipdnn.DataType.FLOAT)
    beta = hipdnn.Tensor.create([graph_tokens, heads_out], hipdnn.DataType.FLOAT)
    cu_seqlens = hipdnn.Tensor.create([sequences + 1], hipdnn.DataType.INT32)
    return q, k, v, g, beta, cu_seqlens


@pytest.mark.gpu
class TestGdn:
    """Tests for GDN plan-building and stubbed execution."""

    def test_execution_succeeds(self):
        (
            graph,
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            initial_state,
            o,
            final_state,
            state_checkpoints,
        ) = build_gdn_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [
                (q, np.float32),
                (k, np.float32),
                (v, np.float32),
                (g, np.float32),
                (beta, np.float32),
                (cu_seqlens, np.int32),
                (initial_state, np.float32),
                (o, np.float32),
                (final_state, np.float32),
                (state_checkpoints, np.float32),
            ],
            handle,
        )


class TestGdnShapeInference:
    """Output shapes follow the GDN layout (no GPU required)."""

    def test_outputs_inferred(self):
        graph, *_, o, final_state, state_checkpoints = build_gdn_graph()
        assert graph.validate().is_good()

        # o [T, HO, V]; states [N, HO, V, K]; checkpoints [max(T // n + N, 1), HO, V, K].
        assert o.get_dim() == [6, 2, 2]
        assert final_state.get_dim() == [2, 2, 2, 4]
        assert state_checkpoints.get_dim() == [5, 2, 2, 4]

    def test_optional_outputs_absent_by_default(self):
        graph = create_float_graph()
        o, final_state, state_checkpoints = graph.gdn(
            *_gdn_inputs(), hipdnn.GdnAttributes()
        )
        o.set_output(True)

        assert final_state is None
        assert state_checkpoints is None
        assert graph.validate().is_good()
        assert o.get_dim() == [4, 2, 2]

    def test_cross_field_rule_rejected(self):
        graph = create_float_graph()
        o, _, _ = graph.gdn(
            *_gdn_inputs(),
            hipdnn.GdnAttributes()
            .set_safe_gate(True)
            .set_gate_domain(hipdnn.GateDomain.LINEAR),
        )
        o.set_output(True)

        assert not graph.validate().is_good()


class TestGdnAttributeBindings:
    """Every GDN attribute binding round-trips through its getter (no GPU required)."""

    def test_methods_are_callable(self):
        initial_state = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        a_log = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        dt_bias = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
        state_indices = hipdnn.Tensor.create([1], hipdnn.DataType.INT32)
        call_attribute_methods(
            hipdnn.GdnAttributes(),
            (
                ("set_name", ("gdn",), "get_name", "gdn"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                (
                    "set_initial_state",
                    (initial_state,),
                    "get_initial_state",
                    initial_state,
                ),
                ("set_a_log", (a_log,), "get_a_log", a_log),
                ("set_dt_bias", (dt_bias,), "get_dt_bias", dt_bias),
                (
                    "set_state_indices",
                    (state_indices,),
                    "get_state_indices",
                    state_indices,
                ),
                ("set_scale", (0.5,), "get_scale", 0.5),
                ("set_output_final_state", (True,), "get_output_final_state", True),
                ("set_use_qk_l2norm", (True,), "get_use_qk_l2norm", True),
                (
                    "set_checkpoint_every_n_tokens",
                    (64,),
                    "get_checkpoint_every_n_tokens",
                    64,
                ),
                ("set_use_beta_sigmoid", (True,), "get_use_beta_sigmoid", True),
                ("set_allow_neg_eigval", (True,), "get_allow_neg_eigval", True),
                ("set_safe_gate", (True,), "get_safe_gate", True),
                (
                    "set_gate_domain",
                    (hipdnn.GateDomain.LINEAR,),
                    "get_gate_domain",
                    hipdnn.GateDomain.LINEAR,
                ),
                ("set_batch_invariant", (True,), "get_batch_invariant", True),
                (
                    "set_overwrite_initial_state",
                    (True,),
                    "get_overwrite_initial_state",
                    True,
                ),
            ),
        )
