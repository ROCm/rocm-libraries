# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import unittest

from kernels.gfx1151.wmma_fmha_fwd import (
    WmmaFmhaFwdSpec,
    _wmma_op_id_for_arch,
    is_valid_spec,
)
from rocke.core.ir import BF16, F16


class TestWmmaFmhaFwdSpec(unittest.TestCase):
    def test_mha_ok(self):
        spec = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4)
        self.assertEqual(spec.kv_heads, 4)

    def test_divisible_gqa_ok(self):
        spec = WmmaFmhaFwdSpec(head_size=64, num_query_heads=8, num_kv_heads=2)
        self.assertEqual(spec.kv_heads, 2)

    def test_non_divisible_gqa_rejected(self):
        with self.assertRaisesRegex(ValueError, "multiple of num_kv_heads"):
            WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, num_kv_heads=3)


class TestWmmaFmhaFwdSpecDtype(unittest.TestCase):
    """The dtype knob: fp16 and bf16 I/O, fp32 accumulate on both."""

    def test_fp16_default_and_alias(self):
        for dt in ("fp16", "f16"):
            spec = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, dtype=dt)
            self.assertIs(spec.dtype_ir, F16)
            # The name tag stays literally "fp16" for both spellings so the
            # pre-bf16 goldens are inert.
            self.assertEqual(spec.dtype_tag, "fp16")
            self.assertIn("_fp16_", spec.kernel_name())

    def test_bf16_accepted(self):
        spec = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, dtype="bf16")
        self.assertIs(spec.dtype_ir, BF16)
        self.assertEqual(spec.dtype_tag, "bf16")
        self.assertIn("_bf16_", spec.kernel_name())

    def test_unsupported_dtype_rejected(self):
        with self.assertRaisesRegex(ValueError, "dtype must be one of"):
            WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, dtype="fp8e4m3")

    def test_op_id_tracks_arch_and_dtype(self):
        self.assertEqual(
            _wmma_op_id_for_arch("gfx1151", "bf16"), "wmma_f32_16x16x16_bf16"
        )
        self.assertEqual(
            _wmma_op_id_for_arch("gfx1201", "bf16"), "wmma_gfx12_f32_16x16x16_bf16"
        )
        self.assertEqual(
            _wmma_op_id_for_arch("gfx1151", "fp16"), "wmma_f32_16x16x16_f16"
        )
        self.assertEqual(
            _wmma_op_id_for_arch("gfx1201", "fp16"), "wmma_gfx12_f32_16x16x16_f16"
        )

    def test_is_valid_spec_accepts_bf16_on_rdna(self):
        for arch in ("gfx1151", "gfx1201"):
            for dt in ("fp16", "bf16"):
                spec = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, dtype=dt)
                ok, why = is_valid_spec(spec, arch=arch)
                self.assertTrue(ok, f"{arch}/{dt}: {why}")


class TestBf16HostCodec(unittest.TestCase):
    """The hand-rolled bf16 host encoding used by the numeric harness.

    numpy has no bf16 and numpy is the only hard dependency, so the verify
    harness encodes by hand. A silently-wrong encoding would show up as kernel
    "error", so it is pinned here (CPU-only, no device needed).
    """

    def setUp(self):
        self.np = __import__("numpy")
        mod = __import__(
            "builders.gfx1151.attention.wmma_fmha_fwd_verify",
            fromlist=["_f32_to_bf16"],
        )
        self.enc, self.dec = mod._f32_to_bf16, mod._bf16_to_f32

    def test_representable_values_round_trip_exactly(self):
        np = self.np
        v = np.array(
            [0.0, -0.0, 1.0, -2.5, 0.5, 3.0, 256.0, -0.015625], dtype=np.float32
        )
        np.testing.assert_array_equal(self.dec(self.enc(v)), v)

    def test_ties_round_to_even(self):
        np = self.np
        # Exact halfway cases on the retained-mantissa boundary: one rounds
        # down to even, one rounds up to even.
        u = np.array([0x3F800000 | 0x8000, 0x3F800000 | 0x18000], dtype=np.uint32).view(
            np.float32
        )
        got = self.enc(u)
        self.assertEqual((int(got[0]), int(got[1])), (0x3F80, 0x3F82))

    def test_error_meets_round_to_nearest_bound(self):
        np = self.np
        r = np.random.default_rng(0).standard_normal(200_000).astype(np.float32)
        rel = float((np.abs(self.dec(self.enc(r)) - r) / np.abs(r)).max())
        # bf16 carries 8 significand bits: RNE is bounded by a half ULP, 2**-8.
        # Plain truncation would reach 2**-7, so this also proves it is RNE.
        self.assertLessEqual(rel, 2**-8 + 1e-9)


if __name__ == "__main__":
    unittest.main()
