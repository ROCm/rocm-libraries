# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Explicit-mask dropout of ``rocke.numeric.sdpa_reference`` (SdpaSemanticsRegister SEM-015)."""

from __future__ import annotations

import numpy as np
import pytest
from rocke.numeric.philox_dropout import dropout_scale, philox_keep_mask
from rocke.numeric.sdpa_reference import sdpa_reference

B, HQ, HKV, D = 2, 4, 2, 8
Q_LENS, KV_LENS = (5, 3), (6, 7)


def _dense(seed=0, sq=5, skv=6):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, HQ, sq, D))
    k = rng.standard_normal((B, HKV, skv, D))
    v = rng.standard_normal((B, HKV, skv, D))
    return q, k, v


def test_dense_matches_the_definition():
    q, k, v = _dense()
    p = 0.3
    mask = philox_keep_mask(B, HQ, 5, 6, p, seed=7, offset=11)
    o, lse = sdpa_reference(
        q, k, v, causal=True, dropout_mask=mask, dropout_scale=dropout_scale(p)
    )
    o_ref, lse_ref = sdpa_reference(q, k, v, causal=True)
    np.testing.assert_array_equal(lse, lse_ref)  # LSE stays pre-dropout
    scale = 1.0 / np.sqrt(D)
    for b in range(B):
        for h in range(HQ):
            s = q[b, h] @ k[b, h // 2].T * scale
            s = np.where(np.tril(np.ones(s.shape, bool)), s, -np.inf)
            prob = np.exp(s - lse_ref[b, h][:, None])
            want = (prob * mask[b, h] / (1.0 - p)) @ v[b, h // 2]
            np.testing.assert_allclose(o[b, h], want, rtol=1e-12, atol=1e-12)
    assert not np.allclose(o, o_ref)


def test_all_kept_with_scale_one_is_no_dropout():
    q, k, v = _dense(1)
    o, lse = sdpa_reference(q, k, v, dropout_mask=np.ones(1), dropout_scale=1.0)
    o_ref, lse_ref = sdpa_reference(q, k, v)
    np.testing.assert_array_equal(o, o_ref)
    np.testing.assert_array_equal(lse, lse_ref)


def test_ragged_uses_in_batch_positions():
    # SEM-015: a packed graph's mask is the dense [B, H, max_sq, max_skv] one,
    # read at each sequence's own positions.
    rng = np.random.default_rng(2)
    q_off = [0, *np.cumsum(Q_LENS)]
    kv_off = [0, *np.cumsum(KV_LENS)]
    q = rng.standard_normal((q_off[-1], HQ, D))
    k = rng.standard_normal((kv_off[-1], HKV, D))
    v = rng.standard_normal((kv_off[-1], HKV, D))
    mask = philox_keep_mask(B, HQ, max(Q_LENS), max(KV_LENS), 0.5, seed=3, offset=0)
    o, _ = sdpa_reference(
        q,
        k,
        v,
        q_offsets=q_off,
        kv_offsets=kv_off,
        dropout_mask=mask,
        dropout_scale=2.0,
    )
    for b in range(B):
        qs, ks = slice(q_off[b], q_off[b + 1]), slice(kv_off[b], kv_off[b + 1])
        want, _ = sdpa_reference(
            q[qs].transpose(1, 0, 2)[None],
            k[ks].transpose(1, 0, 2)[None],
            v[ks].transpose(1, 0, 2)[None],
            dropout_mask=mask[b : b + 1, :, : Q_LENS[b], : KV_LENS[b]],
            dropout_scale=2.0,
        )
        np.testing.assert_allclose(
            o[qs], want[0].transpose(1, 0, 2), rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(dropout_mask=np.ones(1)), "^dropout_mask and dropout_scale go together$"),
        (dict(dropout_scale=2.0), "^dropout_mask and dropout_scale go together$"),
        (
            dict(dropout_mask=np.ones((3, 6)), dropout_scale=1.0),
            r"^dropout_mask shape \(1, 1, 3, 6\) not broadcastable to \(2, 4, 5, 6\)$",
        ),
    ],
)
def test_bad_dropout_is_refused(kwargs, match):
    q, k, v = _dense()
    with pytest.raises(ValueError, match=match):
        sdpa_reference(q, k, v, **kwargs)
