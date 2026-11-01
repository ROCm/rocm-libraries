# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gemm_skinny native layouts (NN / TN / TT = RRR / CRR / CCR) in the GEMM
dispatcher: the native form of a plan, the native table, admission, knobs and
the no-refusal guarantee for every layout."""

from __future__ import annotations

import math
import random
import unittest
from unittest import mock

from rocke.dispatch import GemmRequest, dispatch_gemm_bf16, dispatch_gemm_fp16
from rocke.dispatch.gemm import skinny as SK
from rocke.dispatch.gemm.skinny_native_table import (
    SKINNY_NATIVE_TABLE,
    skinny_native_entry,
)
from rocke.instances.common.gemm_skinny import (
    SkinnyGemmSpec,
    is_valid_spec,
    skinny_launch_error,
)

_DISPATCH = {"fp16": dispatch_gemm_fp16, "bf16": dispatch_gemm_bf16}
_NATIVE = ("RRR", "CRR", "CCR")


def _req(M, N, K, dt, lay, arch="gfx950"):
    return GemmRequest(
        M=M,
        N=N,
        K=K,
        arch=arch,
        dtype=dt,
        layout=lay,
        trans_a=lay[0] == "C",
        trans_b=lay[1] == "C",
    )


class TestNativeForm(unittest.TestCase):
    def test_identity_on_rcr(self):
        spec = SkinnyGemmSpec(kalign=2, split=3)
        self.assertIs(SK.skinny_native(spec, _req(8, 8, 8, "bf16", "RCR")), spec)

    def test_alignment_class_and_read_mode(self):
        spec = SkinnyGemmSpec(kalign=2)
        cases = (
            ("RRR", 30, 4096, "gfx950", 8, "tr"),  # B [K, N]: N decides
            ("RRR", 30, 4098, "gfx950", 2, "tr"),
            ("CCR", 4097, 64, "gfx950", 1, "tr"),  # A [K, M]: M decides
            ("CRR", 4096, 4098, "gfx942", 2, "u16"),  # both: the coarsest
        )
        for lay, M, N, arch, mna, lds in cases:
            with self.subTest(lay=lay, M=M, N=N, arch=arch):
                n = SK.skinny_native(spec, _req(M, N, 64, "bf16", lay, arch))
                self.assertEqual(n.layout, lay)
                self.assertEqual(n.mnalign, mna)
                self.assertEqual(n.mn_lds, lds)
                self.assertEqual(n.mn_pad, SK.SKINNY_MN_PAD)
                # no K-contiguous operand left: kalign is unused (canonical 8)
                self.assertEqual(n.kalign, 8 if lay == "CRR" else 2)

    def test_fit_shrinks_the_staging_on_gfx942(self):
        req = _req(3001, 2003, 1024, "fp16", "CRR", "gfx942")
        spec = SK.skinny_plan(req)
        self.assertTrue(is_valid_spec(spec, "gfx942")[0])
        self.assertIsNone(skinny_launch_error(spec, 3001, 2003, 1024, spec.split))
        self.assertGreater(spec.tile_m * spec.tile_n, 16 * 16)  # not the last resort


class TestNativeDispatch(unittest.TestCase):
    def test_decode_shapes_pick_native_skinny(self):
        for arch in ("gfx942", "gfx950"):
            for lay in _NATIVE:
                for dt in ("fp16", "bf16"):
                    for M, N, K in ((1, 4096, 4096), (4096, 7, 4000), (16, 999, 1001)):
                        with self.subTest(arch=arch, lay=lay, dt=dt, shape=(M, N, K)):
                            r = _DISPATCH[dt](_req(M, N, K, dt, lay, arch))
                            self.assertIsInstance(r.spec, SkinnyGemmSpec)
                            self.assertEqual(r.spec.layout, lay)
                            self.assertEqual(r.grid[2], r.spec.split)

    def test_knob_off_restores_round_one(self):
        with mock.patch.object(SK, "SKINNY_NATIVE", False):
            for lay in _NATIVE:
                r = dispatch_gemm_fp16(_req(4, 4096, 4096, "fp16", lay))
                self.assertNotIsInstance(r.spec, SkinnyGemmSpec)
            r = dispatch_gemm_fp16(_req(4, 4096, 4096, "fp16", "RCR"))
            self.assertIsInstance(r.spec, SkinnyGemmSpec)

    def test_table_rule_knob(self):
        req = _req(193, 545, 9544, "fp16", "CCR")
        with mock.patch.object(SK, "SKINNY_NATIVE_TABLE_RULE", False):
            off = SK.skinny_plan(req)
        on = SK.skinny_plan(req)
        e = skinny_native_entry("gfx950", 193, 545, 9544, "CCR", True)
        self.assertIsNotNone(e)
        self.assertEqual((on.wave_m_atoms, on.wave_n_atoms), e[:2])
        self.assertEqual(off.layout, "CCR")

    def test_every_native_shape_gets_a_valid_kernel(self):
        rng = random.Random(4321)
        shapes = [(1, 1, 1), (129, 131, 7), (4097, 3, 33), (9, 65536, 8)]
        for _ in range(60):
            shapes.append(
                tuple(int(round(2 ** rng.uniform(0, 14.5))) for _ in range(3))
            )
        for arch in ("gfx942", "gfx950"):
            for lay in _NATIVE:
                for M, N, K in shapes:
                    dt = rng.choice(("fp16", "bf16"))
                    with self.subTest(arch=arch, lay=lay, shape=(M, N, K)):
                        r = _DISPATCH[dt](_req(M, N, K, dt, lay, arch))
                        if isinstance(r.spec, SkinnyGemmSpec):
                            self.assertEqual(r.spec.layout, lay)
                            self.assertTrue(is_valid_spec(r.spec, arch)[0])
                            self.assertIsNone(
                                skinny_launch_error(r.spec, M, N, K, r.spec.split)
                            )
                        else:
                            self.assertEqual(r.spec.data.layout, lay)


class TestNativeTable(unittest.TestCase):
    def test_entries_plan_valid_launches(self):
        items = [
            (arch, k, e)
            for arch in sorted(SKINNY_NATIVE_TABLE)
            for k, e in sorted(SKINNY_NATIVE_TABLE[arch].items())
        ]
        for arch, (lay, ms, odd, bs, bg, bk), e in items[::7]:
            s = max(1, int(round(2 ** (bs + 0.5)))) if bs else 1
            g = int(round(2 ** (bg + 0.5)))
            K = max(64, int(round(2 ** (bk + 0.5))))
            M, N = (s, g) if ms else (g, s)
            a_mn, b_mn = lay[0] == "C", lay[1] == "R"

            def adj(x, mn):
                return x if not mn else ((x | 1) if odd else max(8, x // 8 * 8))

            M, N = adj(M, a_mn), adj(N, b_mn)
            if SK._ld_odd(_req(M, N, K, "bf16", lay)) != bool(odd):
                continue
            if max(M, N) * K * 2 >= 2**31:
                continue  # an operand past the i32 byte offsets: no gemm_skinny launch
            with self.subTest(arch=arch, key=(lay, ms, odd, bs, bg, bk)):
                self.assertEqual(len(e), 10)
                req = _req(M, N, K, "bf16", lay, arch)
                spec = SK.skinny_native_table_plan(req)
                # an entry that does not fit is repaired or left to the RCR plan
                spec = spec or SK.skinny_plan(req)
                self.assertTrue(is_valid_spec(spec, arch)[0])
                self.assertIsNone(skinny_launch_error(spec, M, N, K, spec.split))

    def test_lookup_bins(self):
        self.assertIsNone(skinny_native_entry("gfx950", 2048, 4096, 64, "RRR", False))
        self.assertIsNone(skinny_native_entry("gfx90a", 200, 300, 5000, "RRR", False))
        for arch in ("gfx950", "gfx942"):
            e = skinny_native_entry(arch, 200, 300, 5000, "RRR", False)
            self.assertEqual(
                e,
                SKINNY_NATIVE_TABLE[arch][
                    ("RRR", 1, 0, 7, int(math.log2(300)), int(math.log2(5000)))
                ],
            )

    def test_tiles_rule(self):
        # the "tiles" rule (gfx950 ships "tiles2":
        # test_gemm_merge_r2.TestSkinnyNativeThresholds)
        p = mock.patch.dict(SK.SKINNY_NATIVE_RULE, {"gfx950": "tiles"})
        p.start()
        self.addCleanup(p.stop)
        self.assertEqual(SK.SKINNY_NATIVE_RULE["gfx950"], "tiles")
        cases = (
            # odd MN leading dimension: always (no DTL ring reads it in place)
            ("RRR", 333, 8001, 4096, True),
            ("CCR", 8001, 333, 4096, True),
            # M / N multiples of 8, not of 64: below 128 128x128 tiles
            ("RRR", 1000, 1000, 4096, True),
            ("RRR", 1608, 1608, 4096, False),
        )
        for lay, M, N, K, want in cases:
            with self.subTest(lay=lay, shape=(M, N, K)):
                ok, _ = SK.skinny_admit_region(_req(M, N, K, "bf16", lay))
                self.assertEqual(ok, want)
        # multiples of 64: below 64 tiles with min(M, N) < 256
        for M, N, want in ((192, 2048, True), (256, 2048, False), (192, 8192, False)):
            with self.subTest(shape=(M, N)):
                ok, _ = SK.skinny_admit_region(_req(M, N, 8192, "bf16", "CRR"))
                self.assertEqual(ok, want)
        with mock.patch.dict(SK.SKINNY_NATIVE_RULE, {"gfx950": "table"}):
            req = _req(256, 2048, 8192, "bf16", "CRR")
            e = skinny_native_entry("gfx950", 256, 2048, 8192, "CRR", False)
            ok, _ = SK.skinny_admit_region(req)
            self.assertEqual(ok, e[-2] * 100 >= e[-1] * SK.SKINNY_NATIVE_MARGIN)

    def test_admission_follows_the_predictions(self):
        for arch in ("gfx942",):
            for lay in _NATIVE:
                for M, N, K in ((200, 300, 5000), (600, 900, 20000), (300, 5001, 2000)):
                    req = _req(M, N, K, "bf16", lay, arch)
                    e = skinny_native_entry(arch, M, N, K, lay, SK._ld_odd(req))
                    ok, _ = SK.skinny_admit_region(req)
                    with self.subTest(arch=arch, lay=lay, shape=(M, N, K)):
                        self.assertEqual(
                            ok, e[-2] * 100 >= e[-1] * SK.SKINNY_NATIVE_MARGIN
                        )


class TestSkinnyOverVgpr(unittest.TestCase):
    """gfx942 RCR: gemm_skinny ahead of the register-staged body for small
    min(M, N) with a moderate max(M, N) (SKINNY_OVER_VGPR). With the
    gfx942 g5_plan model on, the model arbitrates skinny vs the register-staged body instead; the
    region rule applies with the model off (or with
    common.MERGE2_SKINNY_OVER_VGPR_WITH_MODEL)."""

    def setUp(self):
        from rocke.dispatch.gemm import g5_plan

        p = mock.patch.object(g5_plan, "G5P_ON", False)
        p.start()
        self.addCleanup(p.stop)

    def test_region_with_model_knob(self):
        from rocke.dispatch.gemm import common as C
        from rocke.dispatch.gemm import g5_plan

        req = _req(24, 1856, 23104, "bf16", "RCR", "gfx942")
        with mock.patch.object(g5_plan, "G5P_ON", True):
            with mock.patch.object(C, "MERGE2_SKINNY_OVER_VGPR_WITH_MODEL", True):
                self.assertIn("cdna_skinny", dispatch_gemm_bf16(req).candidate.name)
                with mock.patch.dict(SK.SKINNY_OVER_VGPR, clear=True):
                    with mock.patch.object(g5_plan, "G5P_SKINNY_ARB", False):
                        self.assertIn(
                            "cdna_vgpr", dispatch_gemm_bf16(req).candidate.name
                        )

    def test_region(self):
        rule = SK.SKINNY_OVER_VGPR["gfx942"]
        cases = (
            (320, 16, 1152, True),
            (24, 1856, 23104, True),
            (512, 36, 24000, True),
            (rule["max_small"], 320, 1728, False),
            (27712, 25, 26752, False),  # max(M, N) >= max_big
        )
        for M, N, K, want in cases:
            with self.subTest(shape=(M, N, K)):
                r = dispatch_gemm_bf16(_req(M, N, K, "bf16", "RCR", "gfx942"))
                self.assertEqual(
                    "cdna_skinny" in r.candidate.name, want, r.candidate.name
                )
                if not want:
                    self.assertIn("cdna_vgpr", r.candidate.name)

    def test_knobs(self):
        req = _req(320, 16, 1152, "bf16", "RCR", "gfx942")
        with mock.patch.dict(SK.SKINNY_OVER_VGPR, clear=True):
            self.assertIn("cdna_vgpr", dispatch_gemm_bf16(req).candidate.name)
        with mock.patch.object(SK, "SKINNY_DISPATCH", False):
            self.assertFalse(SK.skinny_takes_over_vgpr(req))
        # gfx950 has no such rule; native layouts are never affected
        self.assertFalse(SK.skinny_takes_over_vgpr(_req(320, 16, 1152, "bf16", "RCR")))
        self.assertFalse(
            SK.skinny_takes_over_vgpr(_req(320, 16, 1152, "bf16", "RRR", "gfx942"))
        )


if __name__ == "__main__":
    unittest.main()
