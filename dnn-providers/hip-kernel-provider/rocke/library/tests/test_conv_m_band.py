# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU: the output-plane band order (m_band) is a bijection of the M tiles,
whole-tile planes and planes that are not a whole number of tiles alike, and
the spec rules / name tags of the fwd and dgrad knobs."""

from __future__ import annotations

import unittest

from kernels.common.conv_m_band import python_m_band_order


class TestMBandOrder(unittest.TestCase):
    def test_bijection(self):
        cases = [
            (16384, 128 * 128, 256, 16),  # fwd s1/s2: 64 tiles per plane
            (16384, 128 * 128, 256, 32),
            (4096, 64 * 64, 256, 4),
            (258 * 16900 // 256 + 1, 16900, 256, 16),  # dgrad s1: 66.02 tiles per plane
            (258 * 16900 // 512 + 1, 16900, 512, 11),
            (50, 30, 8, 3),
            (7, 100, 256, 4),  # fewer tiles than a row: identity
            (99, 17, 4, 5),
        ]
        for T, plane, tm, mb in cases:
            with self.subTest(T=T, plane=plane, tile_m=tm, mb=mb):
                got = [python_m_band_order(i, num_m_tiles=T, plane=plane, tile_m=tm, m_band=mb)
                       for i in range(T)]
                self.assertEqual(sorted(got), list(range(T)))

    def test_whole_planes(self):
        # 64 tiles per plane, 256 planes: L = (band * 256 + z) * mb + i.
        for mb in (16, 32):
            for L in range(0, 16384, 37):
                band, r = divmod(L, 256 * mb)
                want = (r // mb) * 64 + band * mb + r % mb
                self.assertEqual(
                    python_m_band_order(L, num_m_tiles=16384, plane=16384, tile_m=256, m_band=mb),
                    want,
                )


class TestMBandSpec(unittest.TestCase):
    def test_fwd_rules(self):
        from kernels.common.conv_implicit_gemm import ImplicitGemmConvSpec, _fwd_kloop_conflict
        from tests.test_conv_fwd_3d_kloop_knobs import _problem

        base = dict(
            problem=_problem(C=128), tile_m=64, tile_n=64, tile_k=32, warp_m=2,
            warp_n=2, warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            pipeline="compv4", uniform_k_decode=True, reg_prefetch=1, lds_xor=True,
        )
        ok = ImplicitGemmConvSpec(**base, xcd_remap=6, m_band=16)
        self.assertEqual(_fwd_kloop_conflict(ok), "")
        self.assertIn("mb16", ok.kernel_name())
        self.assertNotIn("mb", ImplicitGemmConvSpec(**base, xcd_remap=6).kernel_name().split("xcd6")[1])
        self.assertNotEqual(_fwd_kloop_conflict(ImplicitGemmConvSpec(**base, m_band=16)), "")

    def test_dgrad_rules(self):
        from kernels.common.conv_implicit_gemm_dgrad import (
            build_implicit_gemm_conv_dgrad,
            is_valid_dgrad_spec,
        )
        from tests.test_conv_dgrad_pingpong import _spec

        s = _spec(xcd_remap=6, m_band=8)
        ok, why = is_valid_dgrad_spec(s, "gfx942")
        self.assertTrue(ok, why)
        self.assertIn("mb8", s.kernel_name())
        build_implicit_gemm_conv_dgrad(s, arch="gfx942")
        self.assertNotIn("_mb", _spec(xcd_remap=6).kernel_name())
        ok, _ = is_valid_dgrad_spec(_spec(m_band=8), "gfx942")  # needs xcd_remap
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
