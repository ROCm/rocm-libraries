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


_EXPECTED_BF16_PTRS = {
    "A": "ptr<bf16, global>",
    "B": "ptr<bf16, global>",
    "C": "ptr<bf16, global>",
}


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
                ok, why = c.admits(req)
                self.assertFalse(ok)
                self.assertIn("arch 'gfx950' not in", why)

    def test_unique_candidate_names(self):
        names = [c.name for c in gemm_bf16_candidates()]
        self.assertEqual(len(names), len(set(names)))

    def test_build_kernel_lowers(self):
        r = dispatch_gemm_bf16(_bf16(256, 256, 256, "gfx950"))
        mod = build_kernel(r)
        self.assertIsNotNone(mod)


class TestBf16RcrSignature(unittest.TestCase):
    """Every candidate must advertise bf16 ptr types, not the helper's fp16 default.

    ``helpers.manifest.gemm_args_signature`` defaults to ``dtype="fp16"`` and
    ``bf16_rcr._make_candidate`` overrides it explicitly. No bf16 sweep feeds
    these signatures into a manifest yet -- ``benchmark/gemm/fp16_rcr_sweep.py``
    is the only ``result.signature -> make_gemm_manifest`` wiring today -- so
    this guards dispatch metadata rather than a live verify path. It matters
    because the bf16 sweep will be written by copying the fp16 one, and at that
    point a dropped override silently advertises ``ptr<f16, global>``, which the
    manifest runner reads back to pick its reference dtype.

    Assert against literal strings rather than a second call to the helper, so
    the assertion cannot agree with the code by construction.
    """

    @staticmethod
    def _ptr_types(signature):
        return {a["name"]: a["type"] for a in signature if a["name"] in ("A", "B", "C")}

    def test_every_candidate_signature_is_bf16(self):
        candidates = gemm_bf16_candidates()
        self.assertTrue(candidates, "no bf16 RCR candidates registered")
        for c in candidates:
            with self.subTest(candidate=c.name):
                ptrs = self._ptr_types(c.signature(None))
                self.assertEqual(ptrs, _EXPECTED_BF16_PTRS)


if __name__ == "__main__":
    unittest.main()
