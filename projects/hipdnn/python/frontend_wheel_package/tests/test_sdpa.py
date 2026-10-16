# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for scaled dot-product attention: plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .graph_builders import build_sdpa_backward_graph, build_sdpa_graph
from .helpers import build_all_plans, call_attribute_methods, execute_zeros


@pytest.mark.gpu
class TestSdpa:
    """Tests for SDPA plan-building and stubbed execution."""

    def test_execution_succeeds_when_enabled(self):
        if not hasattr(hipdnn.Graph, "sdpa"):
            pytest.skip("SDPA disabled")

        graph, q, k, v, o, _stats = build_sdpa_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [(q, np.float32), (k, np.float32), (v, np.float32), (o, np.float32)],
            handle,
        )


@pytest.mark.gpu
class TestSdpaBackward:
    """Tests for SDPA backward plan-building and stubbed execution."""

    def test_execution_succeeds_when_enabled(self):
        if not hasattr(hipdnn.Graph, "sdpa_backward"):
            pytest.skip("SDPA disabled")

        graph, q, k, v, o, d_o, stats, dq, dk, dv = build_sdpa_backward_graph()

        handle = build_all_plans(graph)
        execute_zeros(
            graph,
            [
                (q, np.float32),
                (k, np.float32),
                (v, np.float32),
                (o, np.float32),
                (d_o, np.float32),
                (stats, np.float32),
                (dq, np.float32),
                (dk, np.float32),
                (dv, np.float32),
            ],
            handle,
        )


class TestSdpaAttributeBindings:
    """Every SDPA attribute binding round-trips through its getter or property."""

    def test_forward_methods_and_properties_are_accessible(self):
        attributes = hipdnn.SdpaAttributes()
        # (setter suffix, getter suffix) -- most match; a few getters use a shorter
        # or differently-worded name than their setter (page_table_k/v, max, sum_exp).
        tensor_fields = (
            ("q", "q"),
            ("k", "k"),
            ("v", "v"),
            ("bias", "bias"),
            ("attn_scale", "attn_scale"),
            ("seq_len_q", "seq_len_q"),
            ("seq_len_kv", "seq_len_kv"),
            ("seed", "seed"),
            ("offset", "offset"),
            ("dropout_mask", "dropout_mask"),
            ("dropout_scale", "dropout_scale"),
            ("paged_attention_k_table", "page_table_k"),
            ("paged_attention_v_table", "page_table_v"),
            ("block_mask", "block_mask"),
            ("sink_token", "sink_token"),
            ("descale_q", "descale_q"),
            ("descale_k", "descale_k"),
            ("descale_v", "descale_v"),
            ("descale_s", "descale_s"),
            ("scale_s", "scale_s"),
            ("scale_o", "scale_o"),
            ("o", "o"),
            ("stats", "stats"),
            ("logit_max", "max"),
            ("score_sum_exp", "sum_exp"),
            ("rng_dump", "rng_dump"),
            ("amax_s", "amax_s"),
            ("amax_o", "amax_o"),
        )
        tensors = {
            setter_suffix: hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
            for setter_suffix, _ in tensor_fields
        }
        call_attribute_methods(
            attributes,
            (
                (
                    f"set_{setter_suffix}",
                    (tensors[setter_suffix],),
                    f"get_{getter_suffix}",
                    tensors[setter_suffix],
                )
                for setter_suffix, getter_suffix in tensor_fields
            ),
        )
        call_attribute_methods(
            attributes,
            (
                ("set_name", ("sdpa",), "get_name", "sdpa"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                # These fluent setters have no getter method; the paired def_rw
                # property is the only way to read the value back. Each value is
                # read immediately after it is set, while the fields set later are
                # still at their defaults, so a setter wired to the wrong member
                # fails here.
                ("set_generate_stats", (True,), "generate_stats", True),
                ("set_alibi_mask", (True,), "alibi_mask", True),
                ("set_padding_mask", (True,), "padding_mask", True),
                ("set_causal_mask", (True,), "causal_mask", True),
                (
                    "set_causal_mask_bottom_right",
                    (True,),
                    "causal_mask_bottom_right",
                    True,
                ),
                # set_dropout(prob, mask, scale) is a convenience wrapper over the
                # standalone dropout setters; mask and scale round-trip there.
                (
                    "set_dropout",
                    (0.125, tensors["dropout_mask"], tensors["dropout_scale"]),
                    "dropout_probability",
                    0.125,
                ),
                ("set_dropout_probability", (0.25,), "dropout_probability", 0.25),
                ("set_attn_scale", (0.5,), "attn_scale_value", 0.5),
                ("set_diagonal_band_left_bound", (1,), "left_bound", 1),
                ("set_diagonal_band_right_bound", (2,), "right_bound", 2),
                ("set_paged_attention_max_seq_len_kv", (3,), "max_seq_len_kv", 3),
                (
                    "set_diagonal_alignment",
                    (hipdnn.DiagonalAlignment.TOP_LEFT,),
                    "diagonal_alignment",
                    hipdnn.DiagonalAlignment.TOP_LEFT,
                ),
                (
                    "set_mma_core_mode",
                    (hipdnn.DataType.FLOAT,),
                    "mma_core_mode",
                    hipdnn.DataType.FLOAT,
                ),
                (
                    "set_implementation",
                    (hipdnn.AttentionImplementation.UNIFIED,),
                    "implementation",
                    hipdnn.AttentionImplementation.UNIFIED,
                ),
                ("set_unfuse_fma", (True,), "unfuse_fma_hint", True),
            ),
        )

    def test_backward_methods_and_properties_are_accessible(self):
        attributes = hipdnn.SdpaBackwardAttributes()
        tensor_fields = (
            "q",
            "k",
            "v",
            "o",
            "do",
            "stats",
            "attn_scale",
            "bias",
            "seq_len_q",
            "seq_len_kv",
            "seed",
            "offset",
            "dropout_mask",
            "dropout_scale",
            "dropout_scale_inv",
            "dq",
            "dk",
            "dv",
            "dbias",
        )
        tensors = {
            suffix: hipdnn.Tensor.create([1], hipdnn.DataType.FLOAT)
            for suffix in tensor_fields
        }
        call_attribute_methods(
            attributes,
            (
                (f"set_{suffix}", (tensors[suffix],), f"get_{suffix}", tensors[suffix])
                for suffix in tensor_fields
            ),
        )
        call_attribute_methods(
            attributes,
            (
                ("set_name", ("sdpa_backward",), "get_name", "sdpa_backward"),
                (
                    "set_compute_data_type",
                    (hipdnn.DataType.FLOAT,),
                    "get_compute_data_type",
                    hipdnn.DataType.FLOAT,
                ),
                ("set_alibi_mask", (True,), "alibi_mask", True),
                ("set_padding_mask", (True,), "padding_mask", True),
                ("set_causal_mask", (True,), "causal_mask", True),
                (
                    "set_causal_mask_bottom_right",
                    (True,),
                    "causal_mask_bottom_right",
                    True,
                ),
                (
                    "set_dropout",
                    (0.25, tensors["dropout_mask"], tensors["dropout_scale"]),
                    "dropout_probability",
                    0.25,
                ),
                ("set_attn_scale", (0.5,), "attn_scale_value", 0.5),
                ("set_diagonal_band_left_bound", (1,), "left_bound", 1),
                ("set_diagonal_band_right_bound", (2,), "right_bound", 2),
                (
                    "set_diagonal_alignment",
                    (hipdnn.DiagonalAlignment.TOP_LEFT,),
                    "diagonal_alignment",
                    hipdnn.DiagonalAlignment.TOP_LEFT,
                ),
            ),
        )
