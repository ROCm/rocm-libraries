# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The logit softcap and the attention sinks of ``rocke.numeric.sdpa_reference``.

Each case is checked against a scalar loop that writes the definitions out one
row at a time: cap the scaled score, add the bias, keep the keys the causal
window admits, and put the head's sink logit in the softmax denominator.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from rocke.numeric.sdpa_reference import sdpa_reference

NUM_QUERY_HEADS, NUM_KV_HEADS, HEAD_SIZE = 4, 2, 8
Q_LENS, KV_LENS = (3, 5), (7, 4)  # the second sequence has q_len > kv_len
WINDOW = 3  # keys per query row, the diagonal included
SCALE = 0.5


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((sum(Q_LENS), NUM_QUERY_HEADS, HEAD_SIZE))
    k = rng.standard_normal((sum(KV_LENS), NUM_KV_HEADS, HEAD_SIZE))
    v = rng.standard_normal((sum(KV_LENS), NUM_KV_HEADS, HEAD_SIZE))
    bias = rng.standard_normal(
        (len(Q_LENS), NUM_QUERY_HEADS, max(Q_LENS), max(KV_LENS))
    )
    sinks = rng.standard_normal(NUM_QUERY_HEADS) * 2.0
    return q, k, v, bias, sinks


def _offsets(lengths):
    return [0, *np.cumsum(lengths).tolist()]


def _scalar_reference(q, k, v, bias, *, softcap, sinks):
    """Bottom-right causal attention with a WINDOW-key window, row by row."""
    group = NUM_QUERY_HEADS // NUM_KV_HEADS
    out = np.zeros(q.shape)
    lse = np.full(q.shape[:2], -math.inf)
    q_starts, kv_starts = _offsets(Q_LENS), _offsets(KV_LENS)
    for seq, (q_len, kv_len) in enumerate(zip(Q_LENS, KV_LENS)):
        for head in range(NUM_QUERY_HEADS):
            sink = -math.inf if sinks is None else float(sinks[head])
            for row in range(q_len):
                diagonal = row + kv_len - q_len
                logits = {}
                for key in range(kv_len):
                    if not diagonal - WINDOW < key <= diagonal:
                        continue
                    query = q[q_starts[seq] + row, head]
                    stored = k[kv_starts[seq] + key, head // group]
                    score = SCALE * float(query @ stored)
                    if softcap:
                        score = softcap * math.tanh(score / softcap)
                    logits[key] = score + float(bias[seq, head, row, key])
                peak = max([sink, *logits.values()])
                if peak == -math.inf:
                    continue
                denominator = math.exp(sink - peak) + sum(
                    math.exp(logit - peak) for logit in logits.values()
                )
                for key, logit in logits.items():
                    weight = math.exp(logit - peak) / denominator
                    out[q_starts[seq] + row, head] += (
                        weight * v[kv_starts[seq] + key, head // group]
                    )
                lse[q_starts[seq] + row, head] = peak + math.log(denominator)
    return out, lse


def _oracle(q, k, v, bias, **modifiers):
    return sdpa_reference(
        q,
        k,
        v,
        scale=SCALE,
        bias=bias,
        left_bound=WINDOW - 1,
        right_bound=0,
        top_left=False,
        q_offsets=_offsets(Q_LENS),
        kv_offsets=_offsets(KV_LENS),
        **modifiers,
    )


@pytest.mark.parametrize("softcap", [0.0, 1.5])
@pytest.mark.parametrize("with_sinks", [False, True])
def test_softcap_and_sinks_match_the_definition(softcap, with_sinks):
    q, k, v, bias, sinks = _inputs()
    sinks = sinks if with_sinks else None
    out, lse = _oracle(q, k, v, bias, softcap=softcap, sinks=sinks)
    want_out, want_lse = _scalar_reference(q, k, v, bias, softcap=softcap, sinks=sinks)
    np.testing.assert_allclose(out, want_out, rtol=0, atol=1e-12)
    np.testing.assert_allclose(lse, want_lse, rtol=0, atol=1e-12)


def test_a_row_with_no_key_keeps_only_its_sink():
    """The second sequence's first query rows see no key: their output is 0
    and their LSE is the sink logit alone."""
    q, k, v, bias, sinks = _inputs()
    out, lse = _oracle(q, k, v, bias, sinks=sinks)
    empty_rows = slice(Q_LENS[0], Q_LENS[0] + Q_LENS[1] - KV_LENS[1])
    assert empty_rows.stop > empty_rows.start
    assert np.all(out[empty_rows] == 0.0)
    assert np.array_equal(
        lse[empty_rows], np.broadcast_to(sinks, lse[empty_rows].shape)
    )


def test_the_modifiers_change_the_result():
    """Guards against a modifier being accepted and ignored."""
    q, k, v, bias, sinks = _inputs()
    plain, _ = _oracle(q, k, v, bias)
    for modifiers in ({"softcap": 1.5}, {"sinks": sinks}):
        changed, _ = _oracle(q, k, v, bias, **modifiers)
        assert np.abs(changed - plain).max() > 1e-3, modifiers


@pytest.mark.parametrize(
    "modifiers, match",
    [
        ({"softcap": -1.0}, "softcap must be finite and >= 0"),
        ({"softcap": math.nan}, "softcap must be finite and >= 0"),
        ({"sinks": np.zeros(NUM_QUERY_HEADS + 1)}, r"sinks must have shape \(4,\)"),
    ],
)
def test_bad_modifiers_are_refused(modifiers, match):
    q, k, v, bias, _ = _inputs()
    with pytest.raises(ValueError, match=match):
        _oracle(q, k, v, bias, **modifiers)
