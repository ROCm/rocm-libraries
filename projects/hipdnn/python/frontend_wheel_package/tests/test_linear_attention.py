# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the GDN and KDA linear-attention graph ops (no GPU required)."""

import pytest

import hipdnn_frontend as hipdnn

from .helpers import call_attribute_methods, create_float_graph

# Token-packed: N=2 sequences of 4 tokens (T=8 total), H=2 heads, K=V=8.
N, T, H, K, V = 2, 8, 2, 8, 8

_TENSOR_FIELDS = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "cu_seqlens",
    "initial_state",
    "a_log",
    "dt_bias",
    "state_indices",
    "o",
    "final_state",
    "state_checkpoints",
)


def _attribute_calls(gate_lower_bound):
    tensor = hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
    calls = [
        ("set_name", ("linear_attention",), "get_name", "linear_attention"),
        (
            "set_compute_data_type",
            (hipdnn.DataType.FLOAT,),
            "get_compute_data_type",
            hipdnn.DataType.FLOAT,
        ),
        ("set_scale", (0.5,), "get_scale", 0.5),
        ("set_output_final_state", (True,), "get_output_final_state", True),
        ("set_use_qk_l2norm", (True,), "get_use_qk_l2norm", True),
        ("set_checkpoint_every_n_tokens", (4,), "get_checkpoint_every_n_tokens", 4),
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
        ("set_overwrite_initial_state", (True,), "get_overwrite_initial_state", True),
    ]
    calls += [
        (f"set_{name}", (tensor,), f"get_{name}", tensor) for name in _TENSOR_FIELDS
    ]
    if gate_lower_bound:
        calls.append(("set_gate_lower_bound", (-4.0,), "get_gate_lower_bound", -4.0))
    return tuple(calls)


def _inputs(gate_dims):
    create = hipdnn.Tensor.create
    return (
        create([T, H, K], hipdnn.DataType.FLOAT),
        create([T, H, K], hipdnn.DataType.FLOAT),
        create([T, H, V], hipdnn.DataType.FLOAT),
        create(gate_dims, hipdnn.DataType.FLOAT),
        create([T, H], hipdnn.DataType.FLOAT),
        create([N + 1], hipdnn.DataType.INT32),  # cu_seqlens
    )


class TestLinearAttentionAttributeBindings:
    """Every GDN/KDA attribute binding round-trips through its getter."""

    def test_gdn_methods_are_callable(self):
        call_attribute_methods(
            hipdnn.GdnAttributes(), _attribute_calls(gate_lower_bound=False)
        )

    def test_kda_methods_are_callable(self):
        call_attribute_methods(
            hipdnn.KdaAttributes(), _attribute_calls(gate_lower_bound=True)
        )


@pytest.mark.parametrize(
    ("method", "attributes", "gate_dims"),
    [
        ("gdn_ext", hipdnn.GdnAttributes, [T, H]),
        ("kda_ext", hipdnn.KdaAttributes, [T, H, K]),
    ],
)
class TestLinearAttentionGraph:
    """The graph methods create only the requested outputs and infer their shapes."""

    def test_default_creates_only_o(self, method, attributes, gate_dims):
        graph = create_float_graph()
        o, final_state, state_checkpoints = getattr(graph, method)(
            *_inputs(gate_dims), attributes()
        )

        assert final_state is None
        assert state_checkpoints is None
        assert graph.validate().is_good()
        assert o.get_dim() == [T, H, V]

    def test_requested_outputs_are_created_and_shaped(
        self, method, attributes, gate_dims
    ):
        graph = create_float_graph()
        attrs = attributes()
        attrs.set_output_final_state(True)
        attrs.set_checkpoint_every_n_tokens(2)
        o, final_state, state_checkpoints = getattr(graph, method)(
            *_inputs(gate_dims), attrs
        )

        assert graph.validate().is_good()
        assert o.get_dim() == [T, H, V]
        assert final_state.get_dim() == [N, H, V, K]
        # One checkpoint per 2 tokens over T tokens, plus one partial-chunk row per sequence.
        assert state_checkpoints.get_dim() == [T // 2 + N, H, V, K]

    def test_wrong_gate_shape_fails_validation(self, method, attributes, gate_dims):
        graph = create_float_graph()
        wrong_gate = gate_dims[:-1] + [gate_dims[-1] + 1]
        getattr(graph, method)(*_inputs(wrong_gate), attributes())

        assert not graph.validate().is_good()
