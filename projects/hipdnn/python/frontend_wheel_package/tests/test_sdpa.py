# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for scaled dot-product attention: plan-building and stubbed execution."""

import pytest

import hipdnn_frontend as hipdnn

import numpy as np

from .helpers import (
    access_attribute_properties,
    call_attribute_methods,
    build_all_plans_or_skip,
    create_float_graph,
    execute_zeros,
)


def test_methods_follow_the_build_feature_gate():
    assert hasattr(hipdnn.Graph, "sdpa") == hasattr(hipdnn.Graph, "sdpa_backward")


@pytest.mark.gpu
class TestSdpa:
    """Tests for SDPA operation-graph construction."""

    def test_builds_operation_graph_when_enabled(self):
        if not hasattr(hipdnn.Graph, "sdpa"):
            pytest.skip("SDPA disabled")

        graph = create_float_graph()
        q = hipdnn.Tensor.create([2, 8, 16, 64], hipdnn.DataType.FLOAT)
        k = hipdnn.Tensor.create([2, 8, 32, 64], hipdnn.DataType.FLOAT)
        v = hipdnn.Tensor.create([2, 8, 32, 64], hipdnn.DataType.FLOAT)
        outputs = graph.sdpa(q, k, v, hipdnn.SdpaAttributes().set_generate_stats(True))
        assert isinstance(outputs, tuple)
        assert len(outputs) == 2
        o = outputs[0]
        o.set_output(True)

        handle = build_all_plans_or_skip(graph)
        execute_zeros(
            graph,
            [(q, np.float32), (k, np.float32), (v, np.float32), (o, np.float32)],
            handle,
        )


@pytest.mark.gpu
class TestSdpaBackward:
    """Tests for SDPA backward operation-graph construction."""

    def test_builds_operation_graph_when_enabled(self):
        if not hasattr(hipdnn.Graph, "sdpa_backward"):
            pytest.skip("SDPA disabled")

        graph = create_float_graph()
        q = hipdnn.Tensor.create([2, 8, 16, 64], hipdnn.DataType.FLOAT)
        k = hipdnn.Tensor.create([2, 8, 32, 64], hipdnn.DataType.FLOAT)
        v = hipdnn.Tensor.create([2, 8, 32, 64], hipdnn.DataType.FLOAT)
        o = hipdnn.Tensor.create([2, 8, 16, 64], hipdnn.DataType.FLOAT)
        d_o = hipdnn.Tensor.create([2, 8, 16, 64], hipdnn.DataType.FLOAT)
        stats = hipdnn.Tensor.create([2, 8, 16, 1], hipdnn.DataType.FLOAT)
        outputs = graph.sdpa_backward(
            q, k, v, o, d_o, stats, hipdnn.SdpaBackwardAttributes()
        )
        assert isinstance(outputs, tuple)
        assert len(outputs) == 3
        for output in outputs:
            output.set_output(True)

        handle = build_all_plans_or_skip(graph)
        execute_zeros(
            graph,
            [
                (q, np.float32),
                (k, np.float32),
                (v, np.float32),
                (o, np.float32),
                (d_o, np.float32),
                (stats, np.float32),
            ]
            + [(output, np.float32) for output in outputs],
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
                # These fluent setters have no matching getter method; the paired
                # def_rw property (verified below via access_attribute_properties)
                # is the only way to read the value back.
                ("set_generate_stats", (True,), None, None),
                ("set_alibi_mask", (True,), None, None),
                ("set_padding_mask", (True,), None, None),
                ("set_causal_mask", (True,), None, None),
                ("set_causal_mask_bottom_right", (True,), None, None),
                # set_dropout(prob, mask, scale) is a convenience wrapper over the
                # standalone dropout setters; its components round-trip there.
                (
                    "set_dropout",
                    (0.25, tensors["dropout_mask"], tensors["dropout_scale"]),
                    None,
                    None,
                ),
                ("set_dropout_probability", (0.25,), None, None),
                ("set_attn_scale", (0.5,), None, None),
                ("set_diagonal_band_left_bound", (1,), None, None),
                ("set_diagonal_band_right_bound", (1,), None, None),
                ("set_paged_attention_max_seq_len_kv", (1,), None, None),
                (
                    "set_diagonal_alignment",
                    (hipdnn.DiagonalAlignment.TOP_LEFT,),
                    None,
                    None,
                ),
                ("set_mma_core_mode", (hipdnn.DataType.FLOAT,), None, None),
                (
                    "set_implementation",
                    (hipdnn.AttentionImplementation.UNIFIED,),
                    None,
                    None,
                ),
                ("set_unfuse_fma", (True,), None, None),
            ),
        )
        access_attribute_properties(
            attributes,
            (
                ("generate_stats", True),
                ("alibi_mask", True),
                ("padding_mask", True),
                ("causal_mask", True),
                ("causal_mask_bottom_right", True),
                ("dropout_probability", 0.25),
                ("attn_scale_value", 0.5),
                ("left_bound", 1),
                ("right_bound", 1),
                ("max_seq_len_kv", 1),
                ("diagonal_alignment", hipdnn.DiagonalAlignment.TOP_LEFT),
                ("mma_core_mode", hipdnn.DataType.FLOAT),
                ("implementation", hipdnn.AttentionImplementation.UNIFIED),
                ("unfuse_fma_hint", True),
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
                ("set_alibi_mask", (True,), None, None),
                ("set_padding_mask", (True,), None, None),
                ("set_causal_mask", (True,), None, None),
                ("set_causal_mask_bottom_right", (True,), None, None),
                (
                    "set_dropout",
                    (0.25, tensors["dropout_mask"], tensors["dropout_scale"]),
                    None,
                    None,
                ),
                ("set_attn_scale", (0.5,), None, None),
                ("set_diagonal_band_left_bound", (1,), None, None),
                ("set_diagonal_band_right_bound", (1,), None, None),
                (
                    "set_diagonal_alignment",
                    (hipdnn.DiagonalAlignment.TOP_LEFT,),
                    None,
                    None,
                ),
            ),
        )
        access_attribute_properties(
            attributes,
            (
                ("alibi_mask", True),
                ("padding_mask", True),
                ("causal_mask", True),
                ("causal_mask_bottom_right", True),
                ("dropout_probability", 0.25),
                ("attn_scale_value", 0.5),
                ("left_bound", 1),
                ("right_bound", 1),
                ("diagonal_alignment", hipdnn.DiagonalAlignment.TOP_LEFT),
            ),
        )
