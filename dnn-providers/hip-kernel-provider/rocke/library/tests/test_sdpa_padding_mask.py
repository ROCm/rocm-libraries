# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only tests for the SDPA padding-mask spec.

``kernels.common._sdpa_padding_mask`` is the host-side specification of the
padding half of the SDPA mask -- the ``keep_key`` / ``row_is_valid`` /
``row_has_mass`` predicates the shell's mask hook and epilogue will emit. These
tests lock the boundaries (``<`` not ``<=``) and prove the predicates compose to
the same result as ``dense_attention_reference`` with padding, so the kernel
emitter has an exact host reference to match. No GPU.
"""

from __future__ import annotations

import math

import numpy as np

from kernels.common._sdpa_padding_mask import keep_key, row_has_mass, row_is_valid
from rocke.numeric.references import dense_attention_reference


def _qkv(Sq=5, Sk=6, H=3, D=8, seed=0xBEEF):
    rng = np.random.default_rng(seed)
    Q = (rng.standard_normal((Sq, H, D)) * 0.3).astype(np.float16)
    K = (rng.standard_normal((Sk, H, D)) * 0.3).astype(np.float16)
    V = (rng.standard_normal((Sk, H, D)) * 0.3).astype(np.float16)
    return Q, K, V


def _emulate_kernel_padding(Q, K, V, *, len_q, len_kv, causal=False):
    """Model the shell's masking + epilogue by running the REAL online-softmax
    math, not a special case.

    Per-column ``keep_key`` (and the causal predicate) drive masked scores to the
    finite sentinel ``neg_inf = -1e30``, exactly as the kernel's mask hook does.
    The running max ``m`` is then taken over the masked scores; a fully-masked row
    keeps ``m == neg_inf`` and would softmax to ``mean(V)`` under a bare ``1/l``.
    The epilogue selects on ``row_has_mass(m, neg_inf)`` to write ``O = 0`` for
    such a row -- the real guard, so the test fails if the spec drops it rather
    than passing by a hand-zeroed shortcut. ``row_is_valid`` zeroes padded rows.
    """
    Sq, _, D = Q.shape
    Sk = K.shape[0]
    neg_inf = -1e30
    scores = np.einsum("ihd,jhd->ihj", Q.astype(np.float32), K.astype(np.float32))
    scores /= math.sqrt(D)
    keep = keep_key(np.arange(Sk), len_kv)[None, None, :]  # (1, 1, Sk)
    if causal:
        q_pos = np.arange(Sq)[:, None, None]
        k_pos = np.arange(Sk)[None, None, :]
        keep = keep & (k_pos <= q_pos)
    keep = np.broadcast_to(keep, scores.shape)
    scores = np.where(keep, scores, neg_inf)
    m = scores.max(axis=-1, keepdims=True)  # sentinel if the row kept no key
    probs = np.exp(scores - m)
    out = np.einsum("ihj,jhd->ihd", probs, V.astype(np.float32))
    out /= probs.sum(axis=-1, keepdims=True)
    out = np.where(row_has_mass(m, neg_inf), out, 0.0)  # fully-masked -> O = 0
    valid = row_is_valid(np.arange(Sq), len_q)[:, None, None]  # padded rows -> 0
    return np.where(valid, out, 0.0)


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


def test_row_has_mass_guards_the_sentinel():
    # A row whose running max never rose above the finite sentinel saw no key.
    assert not row_has_mass(-1e30, -1e30)
    assert row_has_mass(-1e29, -1e30)
    np.testing.assert_array_equal(
        row_has_mass(np.array([-1e30, 0.0, 5.0]), -1e30), [False, True, True]
    )


def test_predicates_compose_to_oracle():
    # Includes fully-masked (len_kv == 0) and causal rows so the online-softmax
    # recurrence, not a shortcut, has to produce the oracle's zeros.
    cases = (
        (5, 4, False),
        (3, 6, False),
        (2, 2, False),
        (5, 6, False),
        (5, 0, False),  # every row fully masked
        (5, 4, True),  # causal + KV padding
        (3, 2, True),  # causal + Q and KV padding
        (5, 0, True),  # causal + fully masked
    )
    for len_q, len_kv, causal in cases:
        Q, K, V = _qkv()
        emulated = _emulate_kernel_padding(
            Q, K, V, len_q=len_q, len_kv=len_kv, causal=causal
        )
        oracle = dense_attention_reference(
            Q, K, V, causal=causal, seqlen_q=len_q, seqlen_kv=len_kv
        )
        np.testing.assert_allclose(
            emulated,
            oracle,
            atol=1e-6,
            rtol=0,
            err_msg=f"len_q={len_q} len_kv={len_kv} causal={causal}",
        )


def test_fully_masked_and_padded_rows_are_zero():
    Q, K, V = _qkv()
    # len_kv = 0 -> every row fully masked -> all zero (via the row_has_mass guard).
    np.testing.assert_array_equal(
        _emulate_kernel_padding(Q, K, V, len_q=5, len_kv=0),
        np.zeros_like(Q, dtype=np.float32),
    )
    # len_q = 2 -> rows 2.. are padded -> zero; earlier rows nonzero.
    out = _emulate_kernel_padding(Q, K, V, len_q=2, len_kv=6)
    assert np.all(out[2:] == 0.0) and not np.all(out[:2] == 0.0)
