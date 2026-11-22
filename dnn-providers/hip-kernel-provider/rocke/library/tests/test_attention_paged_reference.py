# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host pieces of the torch-free paged attention numeric tests.

The host encoding accepts only fp16 and bf16, so a test written for another
dtype fails loudly instead of having its values read as bf16. The ALiBi and
QQ-bias helpers turn the kernels' bias inputs into additive score biases, and
the paged reference applies them on the gathered keys.

No GPU required.
"""

from __future__ import annotations

import numpy as np
import pytest

from ._attention_paged_reference import (
    alibi_score_bias,
    decode_half,
    encode_half,
    paged_attention_reference,
    qq_score_bias,
    round_to_half,
)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_half_dtypes_round_trip_exact_values(dtype):
    values = np.array([0.0, 1.0, -2.5, 0.125], dtype=np.float32)
    np.testing.assert_array_equal(round_to_half(values, dtype), values)


@pytest.mark.parametrize("helper", [encode_half, decode_half, round_to_half])
@pytest.mark.parametrize("dtype", ["fp8e4m3", "fp32", "float16", ""])
def test_non_half_dtype_is_refused(helper, dtype):
    values = np.zeros(4, dtype=np.float32)
    with pytest.raises(
        ValueError, match=rf"dtype={dtype!r}, expected one of \('fp16', 'bf16'\)"
    ):
        helper(values, dtype)


def test_alibi_bias_counts_keys_from_the_bottom_right_offset():
    # q_len 2, kv_len 5: context_len 3, so key k gets slope * (k - 3).
    bias = alibi_score_bias([0.5, -1.0], q_len=2, kv_len=5)
    assert bias.shape == (2, 2, 5)
    np.testing.assert_array_equal(bias[0, 1], [-1.5, -1.0, -0.5, 0.0, 0.5])
    np.testing.assert_array_equal(bias[1, 0], [3.0, 2.0, 1.0, 0.0, -1.0])
    np.testing.assert_array_equal(bias[:, 0], bias[:, 1])


def test_qq_bias_reads_the_query_window_and_is_zero_outside():
    # q_len 2, kv_len 4: context_len 2; keys 2 and 3 read columns 0 and 1, and
    # a row stride of 1 leaves key 3 out of range.
    qq = np.array([[10.0], [20.0]])
    np.testing.assert_array_equal(
        qq_score_bias(qq, q_len=2, kv_len=4), [[0, 0, 10, 0], [0, 0, 20, 0]]
    )


def test_paged_reference_applies_softcap_bias_and_sinks_on_the_gathered_keys():
    """Paged through a shuffled block table, the result equals a direct
    computation over the contiguous keys."""
    rng = np.random.default_rng(0)
    q_len, kv_len, block_size, heads, head_size = 3, 6, 4, 2, 8
    q = rng.standard_normal((q_len, heads, head_size))
    keys = rng.standard_normal((kv_len, heads, head_size))
    values = rng.standard_normal((kv_len, heads, head_size))
    block_ids = np.array([2, 0])
    k_cache = np.zeros((3, block_size, heads, head_size))
    v_cache = np.zeros_like(k_cache)
    for page, block in enumerate(block_ids):
        tokens = slice(page * block_size, min((page + 1) * block_size, kv_len))
        k_cache[block, : tokens.stop - tokens.start] = keys[tokens]
        v_cache[block, : tokens.stop - tokens.start] = values[tokens]
    bias = alibi_score_bias([0.25, 0.5], q_len, kv_len)
    sinks = np.array([0.5, -1.0])
    softcap, scale = 2.0, 0.3

    got = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        kv_len,
        block_ids,
        scale,
        score_bias=bias,
        softcap=softcap,
        sinks=sinks,
    )

    want = np.zeros_like(got)
    for head in range(heads):
        scores = softcap * np.tanh(scale * q[:, head] @ keys[:, head].T / softcap)
        scores = scores + bias[head]
        visible = np.arange(kv_len)[None, :] <= np.arange(q_len)[:, None] + 3
        scores = np.where(visible, scores, -np.inf)
        weights = np.exp(scores)
        weights /= weights.sum(-1, keepdims=True) + np.exp(sinks[head])
        want[:, head] = weights @ values[:, head]
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
