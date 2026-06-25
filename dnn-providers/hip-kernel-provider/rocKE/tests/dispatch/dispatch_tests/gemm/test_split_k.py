# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Split-K selection heuristic + GEMM dispatch wiring tests."""

from __future__ import annotations

import unittest

from rocke.dispatch import GemmRequest, dispatch_gemm_bf16
from rocke.helpers.split_k import (
    GFX950_TARGET_CTAS,
    TARGET_SLICE_K,
    select_split_k,
)


def _bf16(M, N, K, arch="gfx950"):
    return GemmRequest(M=M, N=N, K=K, arch=arch, dtype="bf16")


class TestSelectSplitK(unittest.TestCase):
    def test_decode_shape_engages_split_k(self):
        # M=2 N=4096 K=4096, decode tile 16x64x32 -> grid 64 << 256.
        d = select_split_k(M=2, N=4096, K=4096, tile_m=16, tile_n=64, tile_k=32, env={})
        self.assertGreater(d.split_k, 1)
        # chosen must evenly slice K and tile cleanly
        self.assertEqual(4096 % d.split_k, 0)
        self.assertEqual((4096 // d.split_k) % 32, 0)

    def test_k_depth_targets_slice_512(self):
        # The degree tracks per-slice K-depth (~K / TARGET_SLICE_K), confirmed by
        # the gfx950 decode sweep: K=4096 -> 8 (slice 512), K=2048 -> 4 (slice
        # 512). The earlier fill-only heuristic under-split K=4096 (picked 4).
        d4096 = select_split_k(
            M=2, N=4096, K=4096, tile_m=16, tile_n=64, tile_k=32, env={}
        )
        self.assertEqual(d4096.split_k, 8)
        self.assertEqual(4096 // d4096.split_k, TARGET_SLICE_K)
        d2048 = select_split_k(
            M=2, N=2048, K=2048, tile_m=16, tile_n=64, tile_k=32, env={}
        )
        self.assertEqual(d2048.split_k, 4)
        self.assertEqual(2048 // d2048.split_k, TARGET_SLICE_K)
        # Shallow K (K=768) -> 768/512 ~= 2 (the measured moe_down optimum).
        d768 = select_split_k(
            M=2, N=2048, K=768, tile_m=16, tile_n=64, tile_k=32, env={}
        )
        self.assertEqual(d768.split_k, 2)

    def test_square_shape_stays_one(self):
        # M=N=K=4096 with cshuffle tile fills the device -> no split-K.
        d = select_split_k(
            M=4096, N=4096, K=4096, tile_m=128, tile_n=128, tile_k=32, env={}
        )
        self.assertEqual(d.split_k, 1)

    def test_validity_fallback(self):
        # K not divisible by any plausible factor below the raw target -> 1.
        d = select_split_k(M=2, N=64, K=97, tile_m=16, tile_n=64, tile_k=97, env={})
        self.assertEqual(d.split_k, 1)

    def test_env_off_forces_one(self):
        d = select_split_k(
            M=2,
            N=4096,
            K=4096,
            tile_m=16,
            tile_n=64,
            tile_k=32,
            env={"ROCKE_GEMM_SPLIT_K": "off"},
        )
        self.assertEqual(d.split_k, 1)

    def test_env_force_snaps_to_valid(self):
        # Force 7 (not a factor of 4096) -> snapped down to a valid factor.
        d = select_split_k(
            M=2,
            N=4096,
            K=4096,
            tile_m=16,
            tile_n=64,
            tile_k=32,
            env={"ROCKE_GEMM_SPLIT_K": "7"},
        )
        self.assertGreater(d.split_k, 1)
        self.assertLessEqual(d.split_k, 7)
        self.assertEqual(4096 % d.split_k, 0)

    def test_env_force_overrides_square(self):
        d = select_split_k(
            M=4096,
            N=4096,
            K=4096,
            tile_m=128,
            tile_n=128,
            tile_k=32,
            env={"ROCKE_GEMM_SPLIT_K": "4"},
        )
        self.assertEqual(d.split_k, 4)

    def test_target_constant(self):
        self.assertEqual(GFX950_TARGET_CTAS, 256)


class TestDispatchWiring(unittest.TestCase):
    def test_decode_dispatch_sets_split_k_and_z_grid(self):
        r = dispatch_gemm_bf16(_bf16(2, 4096, 4096))
        self.assertEqual(r.candidate.spec_id, "cdna_decode_16x64")
        self.assertGreater(r.spec.trait.split_k, 1)
        # grid Z dim == split_k
        self.assertEqual(r.grid[2], r.spec.trait.split_k)

    def test_square_dispatch_unchanged(self):
        r = dispatch_gemm_bf16(_bf16(4096, 4096, 4096))
        self.assertEqual(r.candidate.spec_id, "cdna_cshuffle_default")
        self.assertEqual(r.spec.trait.split_k, 1)
        self.assertEqual(r.grid[2], 1)

    def test_decode_candidate_off_for_large_m(self):
        # Large M must not pick the decode candidate.
        r = dispatch_gemm_bf16(_bf16(256, 256, 256))
        self.assertNotEqual(r.candidate.spec_id, "cdna_decode_16x64")


if __name__ == "__main__":
    unittest.main()
