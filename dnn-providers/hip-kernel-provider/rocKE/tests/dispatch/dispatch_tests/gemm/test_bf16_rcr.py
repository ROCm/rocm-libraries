# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection + support tests for the BF16 RCR GEMM dispatcher case."""

from __future__ import annotations

import unittest

from rocke.dispatch import GemmRequest, dispatch_gemm_bf16
from rocke.dispatch.gemm import gemm_bf16_candidates
from rocke.dispatch.gemm.bf16_rcr import build_kernel


def _bf16(M, N, K, arch):
    return GemmRequest(M=M, N=N, K=K, arch=arch, dtype="bf16")


class TestBf16RcrDispatch(unittest.TestCase):
    def test_dtype_gate_rejects_non_bf16(self):
        with self.assertRaises(ValueError):
            dispatch_gemm_bf16(GemmRequest(M=128, N=128, K=32, arch="gfx950"))  # fp16

    def test_cdna_cshuffle_selected_when_tile_divides(self):
        r = dispatch_gemm_bf16(_bf16(256, 256, 256, "gfx950"))
        self.assertEqual(r.candidate.spec_id, "cdna_cshuffle_default")
        self.assertEqual((r.spec.tile.tile_m, r.spec.tile.tile_n), (128, 128))
        # bf16 CDNA has no 32x32 atom -> 16x16x16 warp tile, 4x4 warp grid.
        self.assertEqual((r.spec.tile.warp_tile_m, r.spec.tile.warp_tile_n), (16, 16))
        self.assertEqual((r.spec.tile.warp_m, r.spec.tile.warp_n), (4, 4))

    def test_cdna_mem_selected_when_cshuffle_does_not_divide(self):
        # Same divergence shape the arch-family gate fix covers, for bf16.
        r = dispatch_gemm_bf16(_bf16(64, 128, 32, "gfx950"))
        self.assertEqual(r.candidate.spec_id, "cdna_mem_64x128")
        self.assertNotIn("rdna", r.candidate.name)

    def test_rdna_arch_selects_wmma_candidate(self):
        r = dispatch_gemm_bf16(_bf16(64, 32, 16, "gfx1151"))
        self.assertEqual(r.candidate.spec_id, "rdna_wmma_default")

    def test_rdna_candidates_unsupported_on_cdna(self):
        req = _bf16(64, 32, 16, "gfx950")
        for c in gemm_bf16_candidates():
            if "rdna" in c.name:
                ok, why = c.supports(req)
                self.assertFalse(ok)
                self.assertIn("family", why)

    def test_unique_candidate_names(self):
        names = [c.name for c in gemm_bf16_candidates()]
        self.assertEqual(len(names), len(set(names)))

    def test_build_kernel_lowers(self):
        r = dispatch_gemm_bf16(_bf16(256, 256, 256, "gfx950"))
        mod = build_kernel(r)
        self.assertIsNotNone(mod)


if __name__ == "__main__":
    unittest.main()
