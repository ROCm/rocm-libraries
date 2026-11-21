# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""``hipdnn-vllm probe``: layer signatures from a model config (no GPU)."""

import types

import pytest

pytest.importorskip("vllm")

from hipdnn_vllm.attention.routing import prefilter  # noqa: E402
from hipdnn_vllm.probe import model_layers  # noqa: E402


def _model_config(attn_type="decoder", heads=16, kv_heads=4, head=64, window=None, **text):
    cfg = types.SimpleNamespace(architectures=text.pop("architectures", ["LlamaForCausalLM"]), **text)
    return types.SimpleNamespace(
        model="m",
        attn_type=attn_type,
        dtype="bfloat16",
        max_model_len=2048,
        hf_config=cfg,
        hf_text_config=cfg,
        get_head_size=lambda: head,
        get_num_attention_heads=lambda parallel: heads,
        get_num_kv_heads=lambda parallel: kv_heads,
        get_sliding_window=lambda: window,
        get_hidden_size=lambda: heads * head,
    )


def test_decoder_single_layer_kind():
    info = model_layers(_model_config(rms_norm_eps=1e-6))
    (layer,) = info.layers.values()
    assert (layer.num_heads, layer.num_kv_heads, layer.head_size) == (16, 4, 64)
    assert layer.scale == pytest.approx(64**-0.5)
    assert layer.sliding_window == (-1, -1)
    assert info.rms_norm_eps == 1e-6 and not info.requires


def test_encoder_window_is_symmetric():
    info = model_layers(_model_config(attn_type="encoder_only", kv_heads=16, window=128))
    assert info.layers["attention"].sliding_window == (127, 127)
    assert info.layers["attention"].attn_type == "encoder_only"


def test_gemma_style_layer_types_scale_and_softcap():
    info = model_layers(
        _model_config(
            window=512,
            layer_types=["sliding_attention", "full_attention", "sliding_attention"],
            query_pre_attn_scalar=256,
            attn_logit_softcapping=50.0,
        )
    )
    assert set(info.layers) == {"sliding_attention", "full_attention"}
    assert info.layers["sliding_attention"].sliding_window == (511, 0)
    assert info.layers["full_attention"].sliding_window == (-1, -1)
    assert info.layers["full_attention"].scale == pytest.approx(256**-0.5)
    assert prefilter(info.layers["full_attention"]).reason == "UNSUPPORTED_FEATURE:softcap"


def test_chunk_lookback_model_needs_patch_and_falls_back():
    info = model_layers(
        _model_config(window=1024, layer_types=["chunked_attention", "full_attention"])
    )
    assert "patch 0001" in info.requires
    assert prefilter(info.layers["chunked_attention"]).reason == "UNSUPPORTED_FEATURE:chunk_lookback"
    assert prefilter(info.layers["full_attention"]) is None


def test_sinks_detected_for_gpt_oss_and_not_prefiltered():
    info = model_layers(_model_config(architectures=["GptOssForCausalLM"]))
    layer = info.layers["attention"]
    assert layer.has_sinks
    # The engine decides: the probe builds the graph with a sink_token.
    assert prefilter(layer) is None


def test_cli_dispatch_help(capsys):
    from hipdnn_vllm.__main__ import main

    assert main(["--help"]) == 0
    assert "probe --model" in capsys.readouterr().out
