# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""chunk_lookback layers under HIPDNN (needs local vLLM patch 0001)."""

import pytest

vllm = pytest.importorskip("vllm")

from vllm.v1.attention.backend import AttentionBackend  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(AttentionBackend, "supports_chunk_lookback"),
    reason="vLLM lacks patches/vllm/0001 (run scripts/apply_vllm_patches.sh)",
)


def test_hipdnn_inherits_chunk_lookback_capability():
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    from hipdnn_vllm.attention.backend import HipdnnAttentionBackend

    assert TritonAttentionBackend.supports_chunk_lookback()
    assert HipdnnAttentionBackend.supports_chunk_lookback()
    assert not AttentionBackend.supports_chunk_lookback()


def test_attention_layer_with_chunk_lookback_under_hipdnn():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention

    from hipdnn_vllm import register
    from hipdnn_vllm.attention.backend import HipdnnAttentionBackend, HipdnnAttentionImpl
    from hipdnn_vllm.attention.metadata import BatchClass
    from hipdnn_vllm.attention.routing import select

    register()  # the entry point is not installed in the test venv

    with set_current_vllm_config(VllmConfig()):
        layer = Attention(
            num_heads=8,
            head_size=64,
            scale=0.125,
            num_kv_heads=2,
            per_layer_sliding_window=64,
            attn_backend=HipdnnAttentionBackend,
            chunk_lookback=1,
            prefix="model.layers.0.self_attn.attn",
        )
    impl = layer.impl
    assert isinstance(impl, HipdnnAttentionImpl)
    assert impl._layer_static.chunk_lookback == 1
    for bc in BatchClass:
        r = select(impl._layer_static, bc, True)
        assert r.path is None and r.reason == "UNSUPPORTED_FEATURE:chunk_lookback"
