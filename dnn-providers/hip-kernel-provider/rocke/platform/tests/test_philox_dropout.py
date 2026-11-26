# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SEM-015 Philox dropout: the numpy oracle and the IR helpers of ``rocke.helpers.rng``
and ``rocke.helpers.attention_dropout``.

The oracle is pinned to Random123's known answers, to masks drawn by hipDNN's
conformance reference and to the f32 keep thresholds of SEM-015. The IR helpers
run through a small evaluator of the ops they emit, which also fails on any
``nsw`` / ``nuw`` op whose exact result leaves its range (LLVM poison), so the
bits the GPU computes are the ones checked here.
"""

from __future__ import annotations

import unittest

import numpy as np

from rocke.core.ir import I32, IRBuilder
from rocke.helpers import attention_dropout
from rocke.helpers import rng
from rocke.numeric import philox_dropout as oracle
from rocke.tests.int_eval import IntPoison, evaluate_int_ops

M32 = 0xFFFFFFFF

# Random123 known-answer vectors for philox4x32-10: (counter, key, output).
KAT = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((M32,) * 4, (M32, M32), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
]

# Masks drawn by hipdnn_sdpa._philox.philox_mask, rows of (B, Hq, max_sq) in order:
# (B, Hq, max_sq, max_sk, p, seed, offset) -> rows. That oracle's double-precision
# threshold differs from the f32 one by 51 at p = 0.3 (0 at 0.5); no word of
# these masks falls between the two.
PINNED_MASKS = [
    (
        (1, 2, 3, 7, 0.3, 0x123456789ABCDEF0, (1 << 32) - 5),
        ["0001011", "1010011", "0010010", "0001111", "1110011", "1100000"],
    ),
    (
        (1, 1, 2, 9, 0.5, (1 << 64) - 1, (1 << 64) - 2),
        ["011011011", "110011000"],
    ),
]


def _scalar_keep(z, h, q, kv, *, Hq, max_sq, max_sk, p, seed, offset):
    """One element of the mask from the SEM-015 formula, on Python ints."""
    ctr = (
        offset
        + oracle.philox_counter_row_base(z, h, q, Hq=Hq, max_sq=max_sq, max_sk=max_sk)
        + kv // 4
    ) & ((1 << 64) - 1)
    words = oracle.philox4x32_10(ctr & M32, ctr >> 32, 0, 0, seed & M32, seed >> 32)
    w = int(words[kv % 4])
    return (w - (1 << 32) if w >= (1 << 31) else w) > oracle.dropout_threshold(p)


class OracleTest(unittest.TestCase):
    def test_known_answers(self):
        for ctr, key, want in KAT:
            got = oracle.philox4x32_10(*ctr, *key)
            self.assertEqual(tuple(int(w) for w in got), want)

    def test_pinned_hipdnn_masks(self):
        for args, rows in PINNED_MASKS:
            B, Hq, sq, sk = args[:4]
            mask = oracle.philox_keep_mask(*args)
            got = ["".join("1" if x else "0" for x in r) for r in mask.reshape(-1, sk)]
            self.assertEqual(got, rows, args)

    def test_mask_matches_scalar_formula(self):
        B, Hq, sq, sk, p, seed, off = (
            2,
            3,
            5,
            11,
            0.4,
            0xDEADBEEF01234567,
            (1 << 32) - 7,
        )
        mask = oracle.philox_keep_mask(B, Hq, sq, sk, p, seed, off)
        for z, h, q, kv in np.ndindex(mask.shape):
            want = _scalar_keep(
                z, h, q, kv, Hq=Hq, max_sq=sq, max_sk=sk, p=p, seed=seed, offset=off
            )
            self.assertEqual(bool(mask[z, h, q, kv]), want, (z, h, q, kv))

    def test_chunking_is_invisible(self):
        args = (2, 3, 17, 13, 0.4, 99, (1 << 32) - 3)
        whole = oracle.philox_keep_mask(*args)
        saved = oracle._CHUNK_COUNTERS
        try:
            oracle._CHUNK_COUNTERS = 7
            np.testing.assert_array_equal(oracle.philox_keep_mask(*args), whole)
        finally:
            oracle._CHUNK_COUNTERS = saved

    def test_threshold_is_the_f32_formula(self):
        """``int32(f32(p - 0.5) * 2**32)`` in f32, not the same expression in double."""
        want = {
            0.1: -1717986944,
            0.2: -1288490240,
            0.3: -858993408,
            0.7: 858993408,
            0.9: 1717986816,
        }
        for p, t in want.items():
            self.assertEqual(oracle.dropout_threshold(p), t, p)

    def test_threshold_and_scale_edges(self):
        self.assertEqual(oracle.dropout_threshold(0.0), -(2**31))
        self.assertEqual(oracle.dropout_threshold(0.5), 0)
        self.assertEqual(oracle.dropout_threshold(1.0), 2**31 - 1)
        for p in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                oracle.dropout_threshold(p)
        self.assertFalse(oracle.philox_keep_mask(1, 2, 8, 33, 1.0, 5, 0).any())
        self.assertEqual(oracle.dropout_scale(0.2), 1.25)
        self.assertEqual(oracle.dropout_scale(1.0), 0.0)

    def test_keep_rate(self):
        for p in (0.1, 0.5, 0.9):
            mask = oracle.philox_keep_mask(2, 4, 64, 256, p, 2024, 0)
            n = mask.size
            sigma = (n * p * (1 - p)) ** 0.5
            self.assertLess(abs(int(mask.sum()) - n * (1 - p)), 6 * sigma)


class IRHelperTest(unittest.TestCase):
    def test_philox_core_matches_known_answers(self):
        b = IRBuilder("philox")
        ps = [b.param(f"x{i}", I32) for i in range(6)]
        words = rng.philox4x32_10(b, *ps)
        for ctr, key, want in KAT:
            inputs = {p.name: v for p, v in zip(ps, ctr + key)}
            self.assertEqual(tuple(evaluate_int_ops(b, inputs, words)), want)

    def test_add_and_mul_wrap_without_poison(self):
        b = IRBuilder("wrap")
        x = b.param("x", I32)
        y = b.param("y", I32)
        s = rng.add_u32_wrap(b, x, y)
        m = rng.mul_lo_u32_const(b, x, rng.PHILOX_M1)
        for xv, yv in (
            (M32, M32),
            (0x80000000, 0x80000000),
            (0x7FFFFFFF, 1),
            (12345, 678),
        ):
            got = evaluate_int_ops(b, {"x": xv, "y": yv}, [s, m])
            self.assertEqual(got, [(xv + yv) & M32, (xv * rng.PHILOX_M1) & M32])

    def test_plain_i32_multiply_would_be_poison(self):
        """Why the helpers avoid ``b.mul`` on u32: it lowers to ``mul nsw``."""
        b = IRBuilder("nsw")
        x = b.param("x", I32)
        m = b.mul(x, b.const_i32(rng.PHILOX_M0 - (1 << 32)))
        with self.assertRaises(IntPoison):
            evaluate_int_ops(b, {"x": 3}, [m])

    def test_evaluator_div_mod_truncate(self):
        """The evaluator's div / mod follow the lowering's ``sdiv`` / ``srem``."""
        b = IRBuilder("divmod")
        x, y = b.param("x", I32), b.param("y", I32)
        outs = [b.div(x, y), b.mod(x, y)]
        for xv, yv in ((7, 2), (-7, 2), (7, -2), (63, 32)):
            got = evaluate_int_ops(b, {"x": xv & M32, "y": yv & M32}, outs)
            want = [int(xv / yv), xv - int(xv / yv) * yv]
            self.assertEqual(got, [w & M32 for w in want], (xv, yv))
        with self.assertRaises(IntPoison):
            evaluate_int_ops(b, {"x": 1, "y": 0}, outs)

    def test_row_keep_matches_oracle(self):
        """The per-element path a kernel runs: row counter, ``+ kv // 4``, words, compare,
        in each form a kernel may select (``xor3`` = gfx950 ``v_bitop3_b32``, ``mad64``).
        """
        cases = [
            (2, 3, 5, 11, 0.4, 0xDEADBEEF01234567, (1 << 32) - 7),
            (1, 1, 2, 9, 0.5, (1 << 64) - 1, (1 << 64) - 2),
            (1, 2, 3, 130, 0.1, 7, (1 << 63) - 40),
        ]
        for xor3 in (False, True):
            for mad64 in (False, True):
                with self.subTest(xor3=xor3, mad64=mad64):
                    self._check_row_keep(cases, xor3=xor3, mad64=mad64)

    def _check_row_keep(self, cases, *, xor3, mad64):
        b = IRBuilder("row")
        names = ("seed_lo", "seed_hi", "off_lo", "off_hi", "thr", "k", "row", "sk")
        seed_lo, seed_hi, off_lo, off_hi, thr, k, row, max_sk = (
            b.param(n, I32) for n in names
        )
        r_lo, r_hi = attention_dropout.dropout_row_counter(
            b, offset_lo=off_lo, offset_hi=off_hi, row=row, max_sk=max_sk
        )
        c_lo, c_hi = rng.philox_counter_add(b, r_lo, r_hi, k)
        words = attention_dropout.dropout_words(
            b,
            seed_lo=seed_lo,
            seed_hi=seed_hi,
            ctr_lo=c_lo,
            ctr_hi=c_hi,
            xor3=xor3,
            mad64=mad64,
        )
        keeps = [attention_dropout.dropout_keep(b, w, thr) for w in words]
        for B, Hq, sq, sk, p, seed, off in cases:
            mask = oracle.philox_keep_mask(B, Hq, sq, sk, p, seed, off)
            for z, h, q in np.ndindex(B, Hq, sq):
                for k4 in range((sk + 3) // 4):
                    inputs = {
                        "seed_lo": seed & M32,
                        "seed_hi": seed >> 32,
                        "off_lo": off & M32,
                        "off_hi": off >> 32,
                        "thr": oracle.dropout_threshold(p) & M32,
                        "k": k4,
                        "row": (z * Hq + h) * sq + q,
                        "sk": sk,
                    }
                    got = evaluate_int_ops(b, inputs, keeps)
                    want = mask[z, h, q, 4 * k4 : 4 * k4 + 4]
                    self.assertEqual([bool(g) for g in got[: len(want)]], list(want))


if __name__ == "__main__":
    unittest.main()
