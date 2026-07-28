# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection + support + arch-gating tests for the conv dispatcher family."""

from __future__ import annotations

import unittest

from rocke.dispatch.families.conv import (
    ConvRequest,
    conv_candidates,
    dispatch_conv,
)


def _conv(arch, **kw):
    base = dict(N=8, C=64, K=64, Hi=56, Wi=56, Y=3, X=3, pad_h=1, pad_w=1, arch=arch)
    base.update(kw)
    return ConvRequest(**base)


class TestConvDispatch(unittest.TestCase):
    def test_rejects_unsupported_dtype(self):
        with self.assertRaises(ValueError):
            dispatch_conv(_conv("gfx950", dtype="fp32"))

    def test_bf16_supported(self):
        # AICK-1752: bf16 is a required dtype (performance-gated).
        r = dispatch_conv(_conv("gfx950", dtype="bf16"))
        self.assertEqual(r.spec.data.dtype_a, "bf16")
        self.assertEqual(r.spec.data.dtype_d, "bf16")

    def test_rejects_indivisible_groups(self):
        # C/K not divisible by G is rejected before codegen.
        with self.assertRaises(ValueError):
            dispatch_conv(_conv("gfx950", G=3))

    def test_grouped_selected_and_grid_has_group_z(self):
        # AICK-1752: regular grouped conv (g4, cpg=kpg=16) dispatches; the grid
        # gains z = groups (CK-style grid-per-group).
        r = dispatch_conv(_conv("gfx950", G=4))
        self.assertEqual(r.spec.problem.groups, 4)
        self.assertEqual(r.grid[2], 4)

    def test_cardinality_grouped_g32_cpg8(self):
        # Hero case g32/cpg8: kpg=8 < tile_n so N_gemm doesn't tile-align, but the
        # kernel is tail-safe, so the request is supported and grid.z == 32.
        r = dispatch_conv(
            _conv("gfx950", N=2, C=256, K=256, Hi=8, Wi=8, G=32, dtype="bf16")
        )
        self.assertEqual(r.spec.problem.groups, 32)
        self.assertEqual(r.spec.problem.cpg, 8)
        self.assertEqual(r.grid[2], 32)

    def test_rejects_unknown_arch(self):
        with self.assertRaises(ValueError):
            dispatch_conv(_conv("gfx000"))

    def test_rejects_degenerate_output(self):
        # 5x5 filter on 3x3 input, no pad -> Ho<=0.
        with self.assertRaises(ValueError):
            dispatch_conv(_conv("gfx950", Hi=3, Wi=3, Y=5, X=5, pad_h=0, pad_w=0))

    def test_cshuffle_selected_when_kgemm_div64(self):
        # N8 C64 K64 R3S3: M=25088, Ng=64, K_gemm=576 (=9*64) -> cshuffle.
        r = dispatch_conv(_conv("gfx950"))
        self.assertEqual(r.candidate.spec_id, "cdna_cshuffle_64x64")
        self.assertEqual((r.spec.tile_m, r.spec.tile_n, r.spec.tile_k), (64, 64, 64))
        self.assertEqual(r.spec.epilogue, "cshuffle")

    def test_mem_selected_when_kgemm_only_div32(self):
        # C=32 R3S3: K_gemm=288 -> %64!=0, %32==0 -> mem tile_k=32.
        r = dispatch_conv(_conv("gfx950", C=32, Hi=16, Wi=16, N=4))
        self.assertEqual(r.candidate.spec_id, "cdna_mem_64x64")
        self.assertEqual(r.spec.tile_k, 32)

    def test_gfx942_uses_mem_no_32x32x16(self):
        # gfx942 lacks the 32x32x16 f16 atom, so the cshuffle candidate is
        # unsupported there; mem (16x16x16) is the only CDNA pick.
        r = dispatch_conv(_conv("gfx942", C=32, Hi=16, Wi=16, N=4))
        self.assertEqual(r.candidate.spec_id, "cdna_mem_64x64")

    def test_cshuffle_unsupported_on_gfx942(self):
        for c in conv_candidates():
            if c.spec_id == "cdna_cshuffle_64x64":
                ok, why = c.supports(_conv("gfx942"))
                self.assertFalse(ok)

    def test_gfx1250_selects_wmma_16x16x32(self):
        # gfx1250: wave32 WMMA with the 16x16x32 atom (K=32).
        r = dispatch_conv(_conv("gfx1250", C=64, K=64, Hi=16, Wi=16, N=2, dtype="bf16"))
        self.assertEqual(r.candidate.spec_id, "gfx1250_wmma_64x64")
        self.assertEqual(
            (r.spec.warp_tile_m, r.spec.warp_tile_n, r.spec.warp_tile_k), (16, 16, 32)
        )
        self.assertEqual(r.spec.wave_size, 32)

    def test_gfx1250_grouped_selects_wmma(self):
        r = dispatch_conv(
            _conv("gfx1250", N=2, C=256, K=256, Hi=8, Wi=8, G=32, dtype="bf16")
        )
        self.assertEqual(r.candidate.spec_id, "gfx1250_wmma_64x64")
        self.assertEqual(r.spec.problem.groups, 32)
        self.assertEqual(r.grid[2], 32)

    def test_wave64_cdna_candidates_unsupported_on_gfx1250(self):
        # The wave64 cshuffle/mem specs must not be selectable on wave32 gfx1250.
        req = _conv("gfx1250", C=64, K=64, Hi=16, Wi=16, N=2, dtype="bf16")
        for c in conv_candidates():
            if c.spec_id in ("cdna_cshuffle_64x64", "cdna_mem_64x64"):
                ok, _why = c.supports(req)
                self.assertFalse(ok)

    def test_rdna_arch_selects_wmma(self):
        r = dispatch_conv(
            ConvRequest(N=1, C=32, K=32, Hi=16, Wi=16, Y=1, X=1, arch="gfx1151")
        )
        self.assertEqual(r.candidate.spec_id, "rdna_wmma_32x32")

    def test_rdna_candidate_unsupported_on_cdna(self):
        req = _conv("gfx950")
        for c in conv_candidates():
            if "rdna" in c.name:
                ok, why = c.supports(req)
                self.assertFalse(ok)
                self.assertIn("family", why)

    def test_cdna_candidates_unsupported_on_rdna(self):
        req = ConvRequest(N=1, C=32, K=32, Hi=16, Wi=16, Y=1, X=1, arch="gfx1151")
        for c in conv_candidates():
            if "cdna" in c.name:
                ok, why = c.supports(req)
                self.assertFalse(ok)
                self.assertIn("family", why)

    def test_grid_ceildiv_nm(self):
        r = dispatch_conv(_conv("gfx950"))
        p = r.spec.problem
        gm = (p.M + r.spec.tile_m - 1) // r.spec.tile_m
        gn = (p.N_gemm + r.spec.tile_n - 1) // r.spec.tile_n
        self.assertEqual(r.grid, (gn, gm, 1))

    def test_unique_candidate_names(self):
        names = [c.name for c in conv_candidates()]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
