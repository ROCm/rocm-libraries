# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tuned config selection for the fp8 D64 dense prefill cohort (gfx950).

Step-0 lever sweep result: fp8 dense prefill is VALU-bound + occupancy-starved,
so the cohort resolves to waves_per_eu=4; sliding-window additionally takes the
smaller bm128/bn32 tile. bf16, non-D64 fp8, and explicit pins are unaffected.
CPU-only (spec resolution, no GPU / build).
"""
from __future__ import annotations

import unittest

from dispatch.attention.common import AttentionRequest
from dispatch.attention.gfx950 import dense_spec_for_request


def _req(**over) -> AttentionRequest:
    base = dict(
        op="attention",
        arch="gfx950",
        batch=1,
        nhead_q=64,
        nhead_k=8,
        seqlen_q=2048,
        seqlen_k=2048,
        hdim_q=64,
        hdim_v=64,
        dtype="bf16",
        mask_type=1,
        dense_tile="auto",
    )
    base.update(over)
    return AttentionRequest(**base)


class Fp8DenseTuneTest(unittest.TestCase):
    def test_fp8_d64_flash_gets_wpe4_default_tile(self):
        s = dense_spec_for_request(_req(use_fp8=True))
        self.assertEqual(s.kv_storage_dtype, "fp8e4m3")
        self.assertFalse(s.persistent)  # fp8 is grid-only
        self.assertEqual(s.waves_per_eu, 4)
        self.assertEqual((s.block_m, s.block_n), (256, 64))

    def test_fp8_d64_swa_gets_wpe4_small_tile(self):
        s = dense_spec_for_request(_req(use_fp8=True, sliding_window=128))
        self.assertEqual(s.waves_per_eu, 4)
        self.assertEqual((s.block_m, s.block_n), (128, 32))

    def test_bf16_unchanged(self):
        s = dense_spec_for_request(_req())
        self.assertIsNone(s.kv_storage_dtype)
        self.assertEqual(s.waves_per_eu, 2)

    def test_fp8_d128_not_tuned(self):
        s = dense_spec_for_request(
            _req(use_fp8=True, hdim_q=128, hdim_v=128, nhead_q=128)
        )
        self.assertEqual(s.waves_per_eu, 2)
        self.assertEqual((s.block_m, s.block_n), (256, 64))

    def test_explicit_wpe_pin_wins(self):
        s = dense_spec_for_request(_req(use_fp8=True, dense_waves_per_eu=2))
        self.assertEqual(s.waves_per_eu, 2)

    def test_explicit_tile_pin_wins_for_swa(self):
        # A pinned tile keeps its geometry; only the wpe default is applied.
        s = dense_spec_for_request(
            _req(use_fp8=True, sliding_window=128, dense_tile="default")
        )
        self.assertEqual((s.block_m, s.block_n), (256, 64))
        self.assertEqual(s.waves_per_eu, 4)


if __name__ == "__main__":
    unittest.main()
