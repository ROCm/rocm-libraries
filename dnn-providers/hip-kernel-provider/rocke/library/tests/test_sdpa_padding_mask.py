# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only tests for the SDPA padding-mask spec.

``kernels.common._sdpa_padding_mask`` is the host-side specification of the
padding half of the SDPA mask -- the ``keep_key`` / ``row_is_valid`` predicates
the shell's mask hook and epilogue will emit. These tests lock the boundaries
(``<`` not ``<=``) and prove the predicates compose to the same result as
``dense_attention_reference`` with padding, so the kernel emitter has an exact
host reference to match. No GPU.
"""

from __future__ import annotations

import math

import numpy as np

from kernels.common._sdpa_padding_mask import keep_key, row_is_valid
from rocke.numeric.references import dense_attention_reference


def _qkv(Sq=5, Sk=6, H=3, D=8, seed=0xBEEF):
    rng = np.random.default_rng(seed)
    Q = (rng.standard_normal((Sq, H, D)) * 0.3).astype(np.float16)
    K = (rng.standard_normal((Sk, H, D)) * 0.3).astype(np.float16)
    V = (rng.standard_normal((Sk, H, D)) * 0.3).astype(np.float16)
    return Q, K, V


def _emulate_kernel_padding(Q, K, V, *, len_q, len_kv):
    """Apply the padding mask the way the shell will -- per-column ``keep_key``
    into the score, per-row ``row_is_valid`` at the epilogue -- and run softmax,
    so the predicates can be checked against the vectorized oracle."""
    Sq, _, D = Q.shape
    Sk = K.shape[0]
    scores = np.einsum("ihd,jhd->ihj", Q.astype(np.float32), K.astype(np.float32))
    scores /= math.sqrt(D)
    keep = keep_key(np.arange(Sk), len_kv)  # (Sk,) bool
    scores = np.where(keep[None, None, :], scores, -1e30)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out = np.einsum("ihj,jhd->ihd", probs, V.astype(np.float32))
    if not keep.any():  # fully masked (len_kv == 0): O = 0
        out[:] = 0.0
    valid = row_is_valid(np.arange(Sq), len_q)  # padded rows -> 0
    return np.where(valid[:, None, None], out, 0.0)


def test_keep_key_boundary():
    # keep iff key < len_kv (strict), so len_kv itself is masked.
    np.testing.assert_array_equal(
        keep_key(np.arange(5), 3), [True, True, True, False, False]
    )
    assert keep_key(0, 1) and not keep_key(0, 0)


def test_row_is_valid_boundary():
    np.testing.assert_array_equal(
        row_is_valid(np.arange(4), 2), [True, True, False, False]
    )
    assert row_is_valid(0, 1) and not row_is_valid(1, 1)


def test_predicates_compose_to_oracle():
    for len_q, len_kv in ((5, 4), (3, 6), (2, 2), (5, 6)):
        Q, K, V = _qkv()
        emulated = _emulate_kernel_padding(Q, K, V, len_q=len_q, len_kv=len_kv)
        oracle = dense_attention_reference(
            Q, K, V, causal=False, seqlen_q=len_q, seqlen_kv=len_kv
        )
        np.testing.assert_allclose(emulated, oracle, atol=1e-6, rtol=0)


def test_fully_masked_and_padded_rows_are_zero():
    Q, K, V = _qkv()
    # len_kv = 0 -> every row fully masked -> all zero.
    np.testing.assert_array_equal(
        _emulate_kernel_padding(Q, K, V, len_q=5, len_kv=0),
        np.zeros_like(Q, dtype=np.float32),
    )
    # len_q = 2 -> rows 2.. are padded -> zero; earlier rows nonzero.
    out = _emulate_kernel_padding(Q, K, V, len_q=2, len_kv=6)
    assert np.all(out[2:] == 0.0) and not np.all(out[:2] == 0.0)
