# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Regression test for the GEMM candidate arch-family gate.

Before the fix, the RDNA/WMMA candidates were NOT gated to the RDNA arch family.
On a CDNA arch (e.g. gfx950) ``_make_spec`` rebuilds an RDNA candidate with the
target's native wave64 and a 16x16x16 MFMA atom that *also* exists on CDNA, so
``gemm_config_supported`` accepted it. The prio-10 ``rdna_wmma`` candidate then
wrongly out-ranked the intended prio-20 ``cdna_mem`` candidate whenever the
128x128 cshuffle tile did not divide the problem.

These tests fail on the old (ungated) code and pass after the arch-family gate.
"""

from __future__ import annotations

import unittest

from rocke.dispatch import GemmRequest, dispatch_gemm_fp16
from rocke.dispatch.gemm import gemm_fp16_candidates


def _by_spec_id(spec_id: str):
    for c in gemm_fp16_candidates():
        if c.spec_id == spec_id:
            return c
    raise AssertionError(f"no candidate with spec_id {spec_id!r}")


class TestArchFamilyGate(unittest.TestCase):
    def test_rdna_candidates_unsupported_on_cdna_arch(self):
        # On gfx950 (CDNA), neither RDNA/WMMA candidate may report support, even
        # for shapes whose tiles happen to divide the problem.
        req = GemmRequest(M=64, N=32, K=16, arch="gfx950")
        for spec_id in ("rdna_wmma_default", "rdna_wmma_32x32"):
            ok, why = _by_spec_id(spec_id).supports(req)
            self.assertFalse(ok, f"{spec_id} wrongly supported on gfx950")
            self.assertIn("family", why)

    def test_cdna_candidates_unsupported_on_rdna_arch(self):
        # Symmetric: CDNA candidates must not report support on an RDNA arch.
        req = GemmRequest(M=128, N=128, K=32, arch="gfx1151")
        for spec_id in ("cdna_cshuffle_default", "cdna_mem_64x128"):
            ok, why = _by_spec_id(spec_id).supports(req)
            self.assertFalse(ok, f"{spec_id} wrongly supported on gfx1151")
            self.assertIn("family", why)

    def test_cdna_selection_prefers_mem_when_cshuffle_tile_does_not_divide(self):
        # M=64 breaks the 128x128 cshuffle tile but the 64x128 mem tile divides;
        # the intended pick is cdna_mem. Pre-fix this returned rdna_wmma.
        result = dispatch_gemm_fp16(GemmRequest(M=64, N=128, K=32, arch="gfx950"))
        self.assertEqual(result.candidate.spec_id, "cdna_mem_64x128")
        self.assertNotIn("rdna", result.candidate.name)

    def test_rdna_arch_still_selects_rdna_candidate(self):
        # The gate must not break the RDNA path it protects.
        result = dispatch_gemm_fp16(GemmRequest(M=64, N=32, K=16, arch="gfx1151"))
        self.assertEqual(result.candidate.spec_id, "rdna_wmma_default")

    def test_cdna_cshuffle_still_selected_when_it_divides(self):
        result = dispatch_gemm_fp16(GemmRequest(M=128, N=128, K=32, arch="gfx950"))
        self.assertEqual(result.candidate.spec_id, "cdna_cshuffle_default")


if __name__ == "__main__":
    unittest.main()
