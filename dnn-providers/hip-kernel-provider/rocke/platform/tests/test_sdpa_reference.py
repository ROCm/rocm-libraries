# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only tests for the float64 SDPA oracle (``rocke.numeric.sdpa_reference``).

Covers the diagonal-band predicate, fully-masked-row semantics, GQA/MQA head
mapping, bias broadcasting, padded and ragged layouts, and cross-checks the
vectorised oracle against a scalar pure-python implementation.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from rocke.numeric.references import sdpa_reference as exported_sdpa_reference
from rocke.numeric.sdpa_reference import keep, resolve_diagonal_band, sdpa_reference

INF = math.inf


def _brute(
    q,
    k,
    v,
    *,
    scale=None,
    bias=None,
    left=-1,
    right=-1,
    top_left=True,
    lens_q=None,
    lens_kv=None,
):
    """Scalar python-loop SDPA on dense [B,H,S,D] inputs (nested loops, math only)."""
    B, Hq, Sq, D = q.shape
    Hk, Skv, Hv, Dv = k.shape[1], k.shape[2], v.shape[1], v.shape[3]
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    o = np.zeros((B, Hq, Sq, Dv))
    lse = np.full((B, Hq, Sq), -INF)
    for b in range(B):
        nq = Sq if lens_q is None else lens_q[b]
        nk = Skv if lens_kv is None else lens_kv[b]
        for h in range(Hq):
            hk = h * Hk // Hq
            hv = h * Hv // Hq
            for i in range(nq):
                scores = {}
                for j in range(nk):
                    off = 0 if top_left else nk - nq
                    if right >= 0 and j - i - off > right:
                        continue
                    if left >= 0 and i + off - j > left:
                        continue
                    dot = 0.0
                    for d in range(D):
                        dot += float(q[b, h, i, d]) * float(k[b, hk, j, d])
                    s = dot * scale
                    if bias is not None:
                        bb = np.asarray(bias)
                        idx = (b, h, i, j)[4 - bb.ndim :]
                        idx = tuple(
                            0 if bb.shape[n] == 1 else idx[n] for n in range(bb.ndim)
                        )
                        s += float(bb[idx])
                    scores[j] = s
                if not scores:
                    continue
                m = max(scores.values())
                if m == -INF:
                    continue
                den = sum(math.exp(s - m) for s in scores.values())
                lse[b, h, i] = m + math.log(den)
                for dd in range(Dv):
                    acc = 0.0
                    for j, s in scores.items():
                        acc += math.exp(s - m) / den * float(v[b, hv, j, dd])
                    o[b, h, i, dd] = acc
    return o, lse


def _rand(shape, rng):
    return rng.standard_normal(shape)


def _assert_same(tc, a, b):
    o1, l1 = a
    o2, l2 = b
    np.testing.assert_allclose(o1, o2, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.isneginf(l1), np.isneginf(l2))
    fin = np.isfinite(l1)
    np.testing.assert_allclose(l1[fin], l2[fin], rtol=1e-12, atol=1e-12)


class TestKeep(unittest.TestCase):
    def test_unbounded_keeps_everything(self):
        for q in range(4):
            for k in range(6):
                self.assertTrue(keep(q, k, 4, 6, "top_left", -1, -1))

    def test_causal_top_left(self):
        self.assertTrue(keep(2, 2, 4, 4, "top_left", -1, 0))
        self.assertFalse(keep(2, 3, 4, 4, "top_left", -1, 0))

    def test_left_bound_keeps_w_plus_one_keys(self):
        for w in (0, 1, 3, 5):
            kept = [k for k in range(16) if keep(8, k, 16, 16, "top_left", w, 0)]
            self.assertEqual(len(kept), w + 1)
            self.assertEqual(kept, list(range(8 - w, 9)))

    def test_left_and_right_band(self):
        kept = [k for k in range(16) if keep(8, k, 16, 16, "top_left", 2, 3)]
        self.assertEqual(kept, list(range(6, 12)))

    def test_bottom_right_offset(self):
        # s_q=2, s_kv=5 -> offset 3: row 0 sees keys 0..3
        kept = [k for k in range(5) if keep(0, k, 2, 5, "bottom_right", -1, 0)]
        self.assertEqual(kept, [0, 1, 2, 3])
        kept = [k for k in range(5) if keep(1, k, 2, 5, "bottom_right", -1, 0)]
        self.assertEqual(kept, [0, 1, 2, 3, 4])

    def test_bool_diagonal(self):
        self.assertEqual(keep(0, 1, 2, 5, True, -1, 0), False)
        self.assertEqual(keep(0, 1, 2, 5, False, -1, 0), True)

    def test_invalid_bounds(self):
        with self.assertRaises(ValueError):
            keep(0, 0, 1, 1, "top_left", -2, -1)
        with self.assertRaises(ValueError):
            keep(0, 0, 1, 1, "top_left", -1, -2)
        with self.assertRaises(ValueError):
            keep(0, 0, 1, 1, "diag", -1, -1)

    def test_matches_reference_formula_exhaustive(self):
        for sq in range(1, 6):
            for skv in range(1, 7):
                for left in (-1, 0, 1, 3):
                    for right in (-1, 0, 2):
                        for tl in (True, False):
                            off = 0 if tl else skv - sq
                            for q in range(sq):
                                for k in range(skv):
                                    masked = (
                                        right >= 0 and k >= q + 1 + off + right
                                    ) or (left >= 0 and k < q + off - left)
                                    self.assertEqual(
                                        keep(q, k, sq, skv, tl, left, right), not masked
                                    )


class TestResolve(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(resolve_diagonal_band(), (-1, -1, True))

    def test_causal_overrides_bounds_and_alignment(self):
        self.assertEqual(
            resolve_diagonal_band(
                left_bound=4, right_bound=2, top_left=False, causal=True
            ),
            (-1, 0, True),
        )
        self.assertEqual(
            resolve_diagonal_band(
                left_bound=4, right_bound=2, causal_bottom_right=True
            ),
            (-1, 0, False),
        )

    def test_both_causal_rejected(self):
        with self.assertRaises(ValueError):
            resolve_diagonal_band(causal=True, causal_bottom_right=True)

    def test_below_minus_one_rejected(self):
        with self.assertRaises(ValueError):
            resolve_diagonal_band(left_bound=-2)
        with self.assertRaises(ValueError):
            resolve_diagonal_band(right_bound=-5)


class TestSdpaReference(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1234)

    def _qkv(self, B, Hq, Hk, Sq, Skv, D, Hv=None, Dv=None):
        Hv = Hk if Hv is None else Hv
        Dv = D if Dv is None else Dv
        r = self.rng
        return (
            _rand((B, Hq, Sq, D), r),
            _rand((B, Hk, Skv, D), r),
            _rand((B, Hv, Skv, Dv), r),
        )

    def test_reexport(self):
        self.assertIs(exported_sdpa_reference, sdpa_reference)

    def test_plain_softmax_and_lse_natural_log(self):
        q, k, v = self._qkv(1, 1, 1, 3, 4, 8)
        o, lse = sdpa_reference(q, k, v)
        s = q[0, 0] @ k[0, 0].T / math.sqrt(8)
        p = np.exp(s) / np.exp(s).sum(1, keepdims=True)
        np.testing.assert_allclose(o[0, 0], p @ v[0, 0], atol=1e-12)
        np.testing.assert_allclose(lse[0, 0], np.log(np.exp(s).sum(1)), atol=1e-12)
        self.assertEqual(lse.shape, (1, 1, 3))

    def test_explicit_scale(self):
        q, k, v = self._qkv(1, 2, 2, 4, 4, 8)
        _assert_same(
            self, sdpa_reference(q, k, v, scale=0.3), _brute(q, k, v, scale=0.3)
        )

    def test_window_off_by_one_numeric(self):
        # With V = one-hot over keys, O row reveals exactly which keys were kept.
        S = 12
        q = self.rng.standard_normal((1, 1, S, 4))
        k = self.rng.standard_normal((1, 1, S, 4))
        v = np.eye(S)[None, None]
        for w in (0, 1, 3):
            o, _ = sdpa_reference(q, k, v, left_bound=w, right_bound=0)
            for i in range(S):
                nz = np.flatnonzero(o[0, 0, i] > 0)
                self.assertEqual(list(nz), list(range(max(0, i - w), i + 1)))
                self.assertEqual(len(nz), min(w, i) + 1)

    def test_bottom_right_sq_gt_skv_fully_masked_rows(self):
        q, k, v = self._qkv(1, 2, 2, 5, 2, 8)
        o, lse = sdpa_reference(q, k, v, causal_bottom_right=True)
        # offset = -3: rows 0..2 see no key, row 3 sees key 0, row 4 sees keys 0..1
        for i in range(3):
            self.assertTrue(np.all(o[0, :, i] == 0.0))
            self.assertTrue(np.all(np.isneginf(lse[0, :, i])))
        self.assertTrue(np.all(np.isfinite(lse[0, :, 3:])))
        self.assertFalse(np.any(np.isnan(o)))
        np.testing.assert_allclose(o[0, :, 3], v[0, :, 0], atol=1e-12)
        _assert_same(self, (o, lse), _brute(q, k, v, left=-1, right=0, top_left=False))

    def test_fully_masked_via_bias_row(self):
        q, k, v = self._qkv(1, 1, 1, 3, 4, 4)
        bias = np.zeros((3, 4))
        bias[1, :] = -INF
        o, lse = sdpa_reference(q, k, v, bias=bias)
        self.assertTrue(np.all(o[0, 0, 1] == 0.0))
        self.assertTrue(np.isneginf(lse[0, 0, 1]))
        self.assertTrue(np.all(np.isfinite(lse[0, 0, [0, 2]])))

    def test_lse_fp32_convertible(self):
        q, k, v = self._qkv(1, 1, 1, 5, 2, 4)
        _, lse = sdpa_reference(q, k, v, causal_bottom_right=True)
        l32 = lse.astype(np.float32)
        np.testing.assert_array_equal(np.isneginf(l32), np.isneginf(lse))
        self.assertFalse(np.any(np.isnan(l32)))

    def test_gqa_mqa_head_mapping(self):
        for Hq, Hk in ((8, 8), (8, 4), (8, 2), (8, 1), (6, 3)):
            q, k, v = self._qkv(2, Hq, Hk, 5, 7, 8)
            got = sdpa_reference(q, k, v, left_bound=2, right_bound=1)
            _assert_same(self, got, _brute(q, k, v, left=2, right=1))
            # contiguous grouping: head h behaves like MHA with KV head h // (Hq//Hk)
            g = Hq // Hk
            ke = np.repeat(k, g, axis=1)
            ve = np.repeat(v, g, axis=1)
            _assert_same(
                self, got, sdpa_reference(q, ke, ve, left_bound=2, right_bound=1)
            )

    def test_k_and_v_head_counts_independent_and_dv(self):
        q, k, v = self._qkv(1, 4, 2, 3, 5, 8, Hv=4, Dv=16)
        got = sdpa_reference(q, k, v)
        self.assertEqual(got[0].shape, (1, 4, 3, 16))
        _assert_same(self, got, _brute(q, k, v))

    def test_bad_group_ratio(self):
        q, k, v = self._qkv(1, 6, 4, 2, 2, 4, Hv=3)
        with self.assertRaises(ValueError):
            sdpa_reference(q, k, v)

    def test_bias_added_after_scaling(self):
        q, k, v = self._qkv(1, 1, 1, 1, 3, 4)
        bias = np.array([0.0, 5.0, 0.0])
        _, lse = sdpa_reference(q, k, v, scale=2.0, bias=bias)
        s = (q[0, 0] @ k[0, 0].T) * 2.0 + bias
        np.testing.assert_allclose(lse[0, 0], np.log(np.exp(s).sum(1)), atol=1e-12)

    def test_bias_broadcast_variants(self):
        B, H, Sq, Skv = 2, 4, 3, 5
        q, k, v = self._qkv(B, H, 2, Sq, Skv, 8)
        shapes = [
            (Skv,),
            (1,),
            (Sq, Skv),
            (Sq, 1),
            (1, Skv),
            (1, Sq, Skv),
            (H, Sq, Skv),
            (1, 1, Sq, Skv),
            (B, 1, 1, Skv),
            (B, H, Sq, Skv),
            (B, 1, Sq, 1),
            (1, H, 1, 1),
        ]
        for shp in shapes:
            bias = self.rng.standard_normal(shp)
            got = sdpa_reference(q, k, v, bias=bias, right_bound=1)
            _assert_same(self, got, _brute(q, k, v, bias=bias, right=1))
            full = np.broadcast_to(
                bias.reshape((1,) * (4 - bias.ndim) + bias.shape), (B, H, Sq, Skv)
            )
            _assert_same(self, got, sdpa_reference(q, k, v, bias=full, right_bound=1))

    def test_bias_bad_shape(self):
        q, k, v = self._qkv(1, 2, 2, 3, 4, 4)
        with self.assertRaises(ValueError):
            sdpa_reference(q, k, v, bias=np.zeros((3, 5)))
        with self.assertRaises(ValueError):
            sdpa_reference(q, k, v, bias=np.zeros((1, 1, 1, 1, 4)))

    def test_padded_seq_lens(self):
        B, H, Sq, Skv = 3, 2, 6, 8
        q, k, v = self._qkv(B, H, 1, Sq, Skv, 8)
        lq, lk = [6, 3, 0], [8, 5, 4]
        for kw in (
            dict(),
            dict(causal=True),
            dict(causal_bottom_right=True),
            dict(left_bound=1, right_bound=1),
        ):
            o, lse = sdpa_reference(q, k, v, seq_len_q=lq, seq_len_kv=lk, **kw)
            left, right, tl = resolve_diagonal_band(**kw)
            ref = _brute(
                q, k, v, left=left, right=right, top_left=tl, lens_q=lq, lens_kv=lk
            )
            _assert_same(self, (o, lse), ref)
            # padded rows documented as zero / -inf
            self.assertTrue(
                np.all(o[1, :, 3:] == 0) and np.all(np.isneginf(lse[1, :, 3:]))
            )
            self.assertTrue(np.all(o[2] == 0) and np.all(np.isneginf(lse[2])))

    def test_padding_keys_never_attended(self):
        q, k, v = self._qkv(1, 1, 1, 2, 6, 4)
        k2, v2 = k.copy(), v.copy()
        k2[..., 3:, :] = 1e6
        v2[..., 3:, :] = 1e6
        a = sdpa_reference(q, k, v, seq_len_kv=[3])
        b = sdpa_reference(q, k2, v2, seq_len_kv=[3])
        _assert_same(self, a, b)

    def test_bottom_right_uses_valid_lengths(self):
        q, k, v = self._qkv(1, 1, 1, 6, 8, 4)
        a = sdpa_reference(
            q, k, v, seq_len_q=[3], seq_len_kv=[5], causal_bottom_right=True
        )
        b = sdpa_reference(
            q[:, :, :3], k[:, :, :5], v[:, :, :5], causal_bottom_right=True
        )
        np.testing.assert_allclose(a[0][:, :, :3], b[0], atol=1e-12)
        np.testing.assert_allclose(a[1][:, :, :3], b[1], atol=1e-12)

    def test_ragged_matches_per_sequence_dense(self):
        Hq, Hk, D = 4, 2, 8
        lq, lk = [3, 5, 1], [4, 2, 6]
        qo = np.concatenate([[0], np.cumsum(lq)])
        ko = np.concatenate([[0], np.cumsum(lk)])
        q = self.rng.standard_normal((qo[-1], Hq, D))
        k = self.rng.standard_normal((ko[-1], Hk, D))
        v = self.rng.standard_normal((ko[-1], Hk, D))
        for kw in (
            dict(),
            dict(causal_bottom_right=True),
            dict(left_bound=1, right_bound=0),
        ):
            o, lse = sdpa_reference(q, k, v, q_offsets=qo, kv_offsets=ko, **kw)
            self.assertEqual(o.shape, q.shape)
            self.assertEqual(lse.shape, (qo[-1], Hq))
            for b in range(3):
                dq = q[qo[b] : qo[b + 1]].transpose(1, 0, 2)[None]
                dk = k[ko[b] : ko[b + 1]].transpose(1, 0, 2)[None]
                dv = v[ko[b] : ko[b + 1]].transpose(1, 0, 2)[None]
                od, ld = sdpa_reference(dq, dk, dv, **kw)
                np.testing.assert_allclose(
                    o[qo[b] : qo[b + 1]], od[0].transpose(1, 0, 2), atol=1e-12
                )
                ls = lse[qo[b] : qo[b + 1]].T
                np.testing.assert_array_equal(np.isneginf(ls), np.isneginf(ld[0]))
                fin = np.isfinite(ld[0])
                np.testing.assert_allclose(ls[fin], ld[0][fin], atol=1e-12)

    def test_ragged_with_bias_and_trailing_rows(self):
        qo, ko = [0, 2, 5], [0, 3, 4]
        q = self.rng.standard_normal((7, 2, 4))  # two rows beyond the last segment
        k = self.rng.standard_normal((4, 2, 4))
        v = self.rng.standard_normal((4, 2, 4))
        bias = self.rng.standard_normal((1, 1, 3, 3))
        o, lse = sdpa_reference(q, k, v, bias=bias, q_offsets=qo, kv_offsets=ko)
        self.assertTrue(np.all(o[5:] == 0) and np.all(np.isneginf(lse[5:])))
        # batch 1 (kv len 1): bias column 0 of the in-batch grid, constant per row => no effect on softmax
        no_bias, _ = sdpa_reference(q, k, v, q_offsets=qo, kv_offsets=ko)
        np.testing.assert_allclose(o[2:5], no_bias[2:5], atol=1e-12)

    def test_ragged_argument_errors(self):
        q = np.zeros((2, 1, 4))
        with self.assertRaises(ValueError):
            sdpa_reference(q, q, q, q_offsets=[0, 2])
        with self.assertRaises(ValueError):
            sdpa_reference(q, q, q, q_offsets=[0, 2], kv_offsets=[0, 2], seq_len_q=[2])
        with self.assertRaises(ValueError):
            sdpa_reference(q, q, q, q_offsets=[0, 3], kv_offsets=[0, 2])

    def test_random_sweep_against_brute_force(self):
        rng = np.random.default_rng(7)
        for _ in range(40):
            B = int(rng.integers(1, 3))
            Hk = int(rng.choice([1, 2, 3]))
            Hq = Hk * int(rng.choice([1, 2, 4]))
            Sq = int(rng.integers(1, 9))
            Skv = int(rng.integers(1, 9))
            D = int(rng.choice([2, 4, 8]))
            left = int(rng.choice([-1, 0, 1, 4]))
            right = int(rng.choice([-1, 0, 2]))
            tl = bool(rng.integers(0, 2))
            lq = [int(rng.integers(0, Sq + 1)) for _ in range(B)]
            lk = [int(rng.integers(0, Skv + 1)) for _ in range(B)]
            q, k, v = (
                _rand((B, Hq, Sq, D), rng),
                _rand((B, Hk, Skv, D), rng),
                _rand((B, Hk, Skv, D), rng),
            )
            bias = rng.standard_normal((Sq, Skv)) if rng.integers(0, 2) else None
            got = sdpa_reference(
                q,
                k,
                v,
                bias=bias,
                left_bound=left,
                right_bound=right,
                top_left=tl,
                seq_len_q=lq,
                seq_len_kv=lk,
            )
            ref = _brute(
                q,
                k,
                v,
                bias=bias,
                left=left,
                right=right,
                top_left=tl,
                lens_q=lq,
                lens_kv=lk,
            )
            _assert_same(self, got, ref)


if __name__ == "__main__":
    unittest.main()
