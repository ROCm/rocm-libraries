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

from rocke.numeric.references import _keep_mask, dense_attention_reference, keep


def _independent_attention(Q, K, V, *, causal, scale=None):
    """Second, independent softmax-attention implementation (fp64, explicit
    loops + np.dot) -- deliberately NOT the einsum, so a transposed index or a
    broken scale/mask in the reference is caught rather than mirrored."""
    Sq, H, D = Q.shape
    Sk = K.shape[0]
    Qf, Kf, Vf = Q.astype(np.float64), K.astype(np.float64), V.astype(np.float64)
    out = np.zeros((Sq, H, D), dtype=np.float64)
    scale = 1.0 / math.sqrt(D) if scale is None else scale
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

    def test_arbitrary_scale_matches_independent(self):
        # hipDNN supplies its own scale rather than assuming 1/sqrt(d); the
        # reference must apply whatever scale it is given, checked against the
        # independent fp64 oracle.
        for scale in (0.123, 1.0, 2.5):
            for causal in (False, True):
                Q, K, V = self._qkv()
                got = dense_attention_reference(Q, K, V, causal=causal, scale=scale)
                want = _independent_attention(Q, K, V, causal=causal, scale=scale)
                np.testing.assert_allclose(got, want, atol=1e-4, rtol=0)

    def test_scale_is_not_ignored(self):
        # A non-default scale must change the result -- guards against the arg
        # being accepted but dropped.
        Q, K, V = self._qkv()
        default = dense_attention_reference(Q, K, V, causal=False)
        scaled = dense_attention_reference(Q, K, V, causal=False, scale=3.0)
        self.assertFalse(np.allclose(default, scaled))

    def test_scale_none_preserves_default(self):
        # scale=None must stay equivalent to an explicit 1/sqrt(d), so existing
        # callers that rely on the implicit default are unaffected.
        Q, K, V = self._qkv()
        d = Q.shape[-1]
        implicit = dense_attention_reference(Q, K, V, causal=True)
        explicit = dense_attention_reference(
            Q, K, V, causal=True, scale=1.0 / math.sqrt(d)
        )
        np.testing.assert_allclose(implicit, explicit, atol=1e-6, rtol=0)

    def test_out_dtype_none_is_fp32_else_casts(self):
        Q, K, V = self._qkv()
        ref32 = dense_attention_reference(Q, K, V, causal=False)
        ref16 = dense_attention_reference(Q, K, V, causal=False, out_dtype=np.float16)
        self.assertEqual(ref32.dtype, np.float32)
        self.assertEqual(ref16.dtype, np.float16)
        # The fp16 output is exactly the fp32 result cast down.
        self.assertTrue(np.array_equal(ref16, ref32.astype(np.float16)))


def _hipdnn_is_masked(sq, skv, seq_q, seq_kv, left, right, top_left):
    """Transliteration of hipDNN's ``CpuFpReferenceSdpa::isMasked`` (C++)."""
    offset = 0 if top_left else seq_kv - seq_q
    if right >= 0 and skv >= sq + 1 + offset + right:
        return True
    if left >= 0 and skv < sq + offset - left:
        return True
    return False


class TestKeep(unittest.TestCase):
    BOUNDS = (-1, 0, 1, 3)

    def _grid(self):
        for s_q in range(1, 9):
            for s_kv in range(1, 9):
                for diagonal in ("top_left", "bottom_right"):
                    for left in self.BOUNDS:
                        for right in self.BOUNDS:
                            yield s_q, s_kv, diagonal, left, right

    def test_matches_hipdnn_is_masked_exhaustively(self):
        for s_q, s_kv, diagonal, left, right in self._grid():
            for q in range(s_q):
                for k in range(s_kv):
                    want = not _hipdnn_is_masked(
                        q, k, s_q, s_kv, left, right, diagonal == "top_left"
                    )
                    got = keep(q, k, s_q, s_kv, diagonal, left, right)
                    self.assertEqual(
                        got, want, (q, k, s_q, s_kv, diagonal, left, right)
                    )

    def test_mask_grid_agrees_with_keep(self):
        for s_q, s_kv, diagonal, left, right in self._grid():
            grid = _keep_mask(s_q, s_kv, diagonal, left, right)
            want = np.array(
                [
                    [keep(q, k, s_q, s_kv, diagonal, left, right) for k in range(s_kv)]
                    for q in range(s_q)
                ]
            )
            np.testing.assert_array_equal(grid, want)

    def test_known_shapes(self):
        # Causal top-left keeps the lower triangle; bottom-right on a 2x4
        # problem shifts the diagonal right by s_kv - s_q = 2.
        tl = _keep_mask(3, 3, "top_left", -1, 0)
        np.testing.assert_array_equal(tl, np.tril(np.ones((3, 3), dtype=bool)))
        br = _keep_mask(2, 4, "bottom_right", -1, 0)
        np.testing.assert_array_equal(br, [[1, 1, 1, 0], [1, 1, 1, 1]])
        # left_bound = W keeps W + 1 keys: one more than rocKE's dist < W window.
        window = _keep_mask(6, 6, "top_left", 2, 0)
        self.assertEqual(window.sum(axis=1).max(), 3)

    def test_rejects_unknown_diagonal(self):
        with self.assertRaises(ValueError):
            keep(0, 0, 1, 1, "center", -1, -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
