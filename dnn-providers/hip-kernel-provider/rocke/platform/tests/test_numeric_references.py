# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only tests for the shared numpy numeric references.

``rocke.numeric.references.dense_attention_reference`` is the single canonical
reference every attention verify harness measures kernel output against. Those
harnesses are Tier-2 (need a matching GPU), so on the host-only gate the
reference is otherwise never exercised -- a corrupted reference would silently
pass a broken kernel. These tests lock the contract with no GPU and no torch:
an independently-written oracle, a causal-mask property, and the dtype contract.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from rocke.numeric.references import dense_attention_reference


def _independent_attention(Q, K, V, *, causal):
    """Second, independent softmax-attention implementation (fp64, explicit
    loops + np.dot) -- deliberately NOT the einsum, so a transposed index or a
    broken scale/mask in the reference is caught rather than mirrored."""
    Sq, H, D = Q.shape
    Sk = K.shape[0]
    Qf, Kf, Vf = Q.astype(np.float64), K.astype(np.float64), V.astype(np.float64)
    out = np.zeros((Sq, H, D), dtype=np.float64)
    scale = 1.0 / math.sqrt(D)
    for h in range(H):
        for i in range(Sq):
            scores = np.array(
                [float(np.dot(Qf[i, h], Kf[j, h])) * scale for j in range(Sk)]
            )
            if causal:
                scores[np.arange(Sk) > i] = -np.inf
            scores -= scores.max()
            w = np.exp(scores)
            w /= w.sum()
            out[i, h] = sum(w[j] * Vf[j, h] for j in range(Sk))
    return out


class TestDenseAttentionReference(unittest.TestCase):
    def _qkv(self, Sq=5, Sk=5, H=3, D=8, seed=0xA11E):
        rng = np.random.default_rng(seed)
        Q = (rng.standard_normal((Sq, H, D)) * 0.3).astype(np.float16)
        K = (rng.standard_normal((Sk, H, D)) * 0.3).astype(np.float16)
        V = (rng.standard_normal((Sk, H, D)) * 0.3).astype(np.float16)
        return Q, K, V

    def test_matches_independent_softmax_reference(self):
        for causal in (False, True):
            Q, K, V = self._qkv()
            got = dense_attention_reference(Q, K, V, causal=causal)  # fp32
            want = _independent_attention(Q, K, V, causal=causal)  # fp64
            self.assertEqual(got.dtype, np.float32)
            # fp32-vs-fp64 math on the same fp16 inputs: differences are ~1e-6.
            np.testing.assert_allclose(got, want, atol=1e-4, rtol=0)

    def test_causal_mask_excludes_future_keys(self):
        Q, K, V = self._qkv()
        causal = dense_attention_reference(Q, K, V, causal=True)
        full = dense_attention_reference(Q, K, V, causal=False)
        # Masking must actually change the result.
        self.assertFalse(np.allclose(causal, full))
        # For query row i, causal attention over all keys equals non-causal
        # attention over just the prefix keys[0..i] -- i.e. no future key leaks.
        for i in range(Q.shape[0]):
            prefix = dense_attention_reference(
                Q[i : i + 1], K[: i + 1], V[: i + 1], causal=False
            )
            np.testing.assert_allclose(causal[i], prefix[0], atol=1e-6, rtol=0)

    def test_out_dtype_none_is_fp32_else_casts(self):
        Q, K, V = self._qkv()
        ref32 = dense_attention_reference(Q, K, V, causal=False)
        ref16 = dense_attention_reference(Q, K, V, causal=False, out_dtype=np.float16)
        self.assertEqual(ref32.dtype, np.float32)
        self.assertEqual(ref16.dtype, np.float16)
        # The fp16 output is exactly the fp32 result cast down.
        self.assertTrue(np.array_equal(ref16, ref32.astype(np.float16)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
