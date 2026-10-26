# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gemm_skinny dispatch: region picks, the no-refusal fallback, the table, knobs, binding."""

from __future__ import annotations

import random
import struct
import unittest
from unittest import mock

from rocke.dispatch import GemmRequest, dispatch_gemm_bf16, dispatch_gemm_fp16
from rocke.dispatch.gemm import common as _GC
from rocke.dispatch.gemm import skinny as SK
from rocke.dispatch.gemm.skinny_table import SKINNY_TABLE, skinny_table_entry
from rocke.instances.common.gemm_skinny import (
    SkinnyGemmSpec,
    is_valid_spec,
    skinny_launch_error,
)

_DISPATCH = {"fp16": dispatch_gemm_fp16, "bf16": dispatch_gemm_bf16}


def _req(M, N, K, dt, arch="gfx950", **kw):
    return GemmRequest(M=M, N=N, K=K, arch=arch, dtype=dt, **kw)


class TestSkinnyRegion(unittest.TestCase):
    def test_decode_shapes_pick_skinny_on_cdna(self):
        for arch in ("gfx942", "gfx950"):
            for dt in ("fp16", "bf16"):
                for M, N, K in (
                    (1, 4096, 4096),
                    (7, 192, 26816),
                    (4096, 3, 333),
                    (32, 1000, 1001),
                ):
                    with self.subTest(arch=arch, dt=dt, shape=(M, N, K)):
                        r = _DISPATCH[dt](_req(M, N, K, dt, arch))
                        self.assertEqual(r.candidate.spec_id, "cdna_skinny_splitk")
                        self.assertIsInstance(r.spec, SkinnyGemmSpec)
                        self.assertEqual(r.grid[2], r.spec.split)
                        self.assertEqual(r.block, (r.spec.block_size, 1, 1))

    def test_rdna_is_untouched(self):
        for arch in ("gfx1151", "gfx1201"):
            r = dispatch_gemm_fp16(_req(64, 32, 16, "fp16", arch))
            self.assertEqual(r.candidate.spec_id, "rdna_wmma_default")

    def test_large_aligned_anchors_keep_their_bodies(self):
        # outside the region the beat bodies are unchanged
        for dt in ("fp16", "bf16"):
            for M, N, K in ((8192, 8192, 8192), (4608, 4608, 288), (2304, 2304, 2304)):
                r = _DISPATCH[dt](_req(M, N, K, dt))
                self.assertNotIn("skinny", r.candidate.name, (dt, M, N, K))

    def test_region_bound_is_table_driven(self):
        # between always_small and max_small the table's predicted ratio decides
        for (dt, ms, bs, bg, bk), e in list(SKINNY_TABLE.items())[::97]:
            if not 6 <= bs <= 8:  # (bs <= 5: always / defer rule; bs 9: mid band)
                continue
            s = 2**bs
            g = 2**bg
            K = 2**bk
            M, N = (s, g) if ms else (g, s)
            ok, why = SK.skinny_admit_region(
                _req(M, N, K, "fp16" if dt == "f16" else "bf16")
            )
            self.assertEqual(ok, e[-1] >= SK.SKINNY_REGION["gfx950"]["min_ratio"], why)


class TestNoRefusals(unittest.TestCase):
    def test_every_shape_gets_a_valid_kernel(self):
        rng = random.Random(1234)
        shapes = [
            (1, 1, 1),
            (130, 128, 32),
            (4104, 4104, 4104),
            (7176, 1440, 4104),
            (9, 65536, 8),
        ]
        for _ in range(150):
            shapes.append(
                tuple(int(round(2 ** rng.uniform(0, 15.2))) for _ in range(3))
            )
        for arch in ("gfx942", "gfx950"):
            for dt in ("fp16", "bf16"):
                for M, N, K in shapes:
                    with self.subTest(arch=arch, dt=dt, shape=(M, N, K)):
                        r = _DISPATCH[dt](_req(M, N, K, dt, arch))
                        if isinstance(r.spec, SkinnyGemmSpec):
                            self.assertTrue(is_valid_spec(r.spec, arch)[0])
                            self.assertIsNone(
                                skinny_launch_error(r.spec, M, N, K, r.spec.split)
                            )

    def test_fallback_knob_restores_refusals(self):
        from rocke.dispatch.gemm import common as _gc

        with mock.patch.object(SK, "SKINNY_FALLBACK", False), mock.patch.object(
            SK, "SKINNY_DISPATCH", False
        ), mock.patch.object(SK, "RING_ANY", False):
            # the padded fallbacks and the universal fallback also serve
            # the shape; with the skinny knobs off it never reaches a
            # gemm_skinny candidate
            r = dispatch_gemm_fp16(_req(130, 128, 32, "fp16"))
            self.assertNotIn("skinny", r.candidate.name)
            self.assertNotIn("ring_small_any", r.candidate.name)
            with mock.patch.object(_gc, "G1_PAD_FALLBACK", False), mock.patch.object(
                _gc, "G1_RING_ANY", False
            ):
                r = dispatch_gemm_fp16(_req(130, 128, 32, "fp16"))
                self.assertIn("universal_fallback", r.candidate.name)
        r = dispatch_gemm_fp16(_req(130, 128, 32, "fp16"))
        self.assertIn("skinny", r.candidate.name)

    def test_dispatch_knob_off_falls_back_to_universal_or_fallback(self):
        with mock.patch.object(SK, "SKINNY_DISPATCH", False):
            r = dispatch_gemm_bf16(_req(2, 4096, 4096, "bf16"))
            self.assertEqual(r.candidate.spec_id, "cdna_decode_16x64")


class TestSkinnyTable(unittest.TestCase):
    def test_every_entry_plans_a_valid_launch(self):
        for (dt, ms, bs, bg, bk), e in SKINNY_TABLE.items():
            s = max(1, 2**bs + (2**bs) // 3)
            g = 2**bg + 7
            K = 2**bk + 24
            M, N = (s, g) if ms else (g, s)
            for arch in ("gfx942", "gfx950"):
                spec = SK.skinny_plan(
                    _req(M, N, K, "fp16" if dt == "f16" else "bf16", arch)
                )
                self.assertTrue(is_valid_spec(spec, arch)[0], (dt, ms, bs, bg, bk))
                self.assertIsNone(skinny_launch_error(spec, M, N, K, spec.split))

    def test_lookup_bins(self):
        self.assertIsNone(skinny_table_entry(1024, 1024, 512, "fp16"))
        self.assertIsNotNone(skinny_table_entry(1023, 4096, 512, "fp16"))
        self.assertIsNotNone(skinny_table_entry(1, 100, 10, "fp16"))  # clamped bins
        self.assertIsNotNone(skinny_table_entry(300, 2**20, 2**20, "bf16"))


class TestSkinnyBinding(unittest.TestCase):
    def test_packed_args_and_exact_check(self):
        from dispatch_tests.core.test_binding import (
            FakeRuntime,
            _write_reference_output,
        )

        for dt in ("fp16", "bf16"):
            with self.subTest(dt=dt):
                req = _req(7, 192, 1024, dt)
                result = _DISPATCH[dt](req)
                self.assertIsInstance(result.spec, SkinnyGemmSpec)
                binding = result.bind(verify=True)
                rt = FakeRuntime()
                args, ptrs = binding.make_args(rt)
                self.assertEqual(len(ptrs), 5)
                a, b, c, w, n, M, N, K, split, kps = struct.unpack("<QQQQQiiiii", args)
                self.assertEqual((a, b, c, w, n), ptrs)
                self.assertEqual((M, N, K, split), (7, 192, 1024, result.spec.split))
                self.assertGreaterEqual(kps * split * result.spec.step_k, 1024)
                self.assertTrue(all(x == 0 for x in rt.mem[n]))  # counters zeroed
                _write_reference_output(rt, binding, ptrs, req, dt)
                _d, bad, total = binding.check(rt, ptrs)
                self.assertEqual((bad, total), (0, 7 * 192))
                rt0 = FakeRuntime()
                _a, p0 = binding.make_args(rt0)
                self.assertGreater(binding.check(rt0, p0)[1], 0)


if __name__ == "__main__":
    unittest.main()


class TestRingSmallAny(unittest.TestCase):
    def test_takes_sub_half_wave_shapes_over_the_generic_bodies(self):
        for dt in ("fp16", "bf16"):
            for M, N, K in (
                (512, 512, 256),
                (1024, 1280, 256),
                (256, 7424, 7168),
                (3584, 512, 5376),
            ):
                # (ring_any is ordered ahead of it where both admit)
                r = _DISPATCH[dt](_req(M, N, K, dt))
                self.assertIn(
                    r.candidate.spec_id,
                    ("cdna_ring_small_any", "cdna_dtl_ring_any_first"),
                    (dt, M, N, K),
                )
                self.assertTrue(r.spec.trait.direct_to_lds)
                with mock.patch.object(_GC, "MERGE_RING_ANY_FIRST", False):
                    r = _DISPATCH[dt](_req(M, N, K, dt))
                self.assertEqual(
                    r.candidate.spec_id, "cdna_ring_small_any", (dt, M, N, K)
                )
                self.assertTrue(r.spec.trait.direct_to_lds)
        with mock.patch.object(SK, "RING_ANY", False):
            r = dispatch_gemm_fp16(_req(512, 512, 256, "fp16"))
            self.assertEqual(r.candidate.spec_id, "cdna_cshuffle_default")

    def test_measured_k_floor_keeps_the_default(self):
        # aligned shapes below the ring's K floor stay
        # with the cshuffle default
        for M, N, K in ((2048, 2048, 64), (2048, 2048, 32), (4608, 4608, 96)):
            r = dispatch_gemm_fp16(_req(M, N, K, "fp16"))
            self.assertEqual(r.candidate.spec_id, "cdna_cshuffle_default", (M, N, K))

    def test_split_rule(self):
        # split degrees: few tiles + long K split, many
        # tiles or short K do not
        f = SK.ring_any_split
        self.assertEqual(f(256, 256, 17152, 128, 128, 64), 4)
        self.assertEqual(f(256, 2304, 10752, 128, 128, 64), 4)
        self.assertEqual(f(1672, 896, 14592, 128, 128, 64), 2)
        self.assertEqual(f(256, 7424, 7168, 128, 128, 64), 1)
        self.assertEqual(f(512, 2048, 768, 128, 128, 64), 1)
        self.assertEqual(f(1792, 192, 1984, 128, 128, 64), 1)
        self.assertEqual(f(256, 256, 1536, 128, 128, 64), 3)
        self.assertEqual(f(256, 256, 1000, 128, 128, 64), 1)  # K % tile_k
        for M, N, K in ((256, 256, 17152), (1672, 896, 14592), (300, 200, 8000)):
            r = dispatch_gemm_bf16(_req(M, N, K, "bf16"))
            if r.candidate.spec_id == "cdna_ring_small_any":
                t = r.spec.tile
                s = SK.ring_any_split(M, N, K, t.tile_m, t.tile_n, t.tile_k)
                self.assertEqual(r.spec.trait.split_k, s)
                self.assertEqual(r.grid[2], s)
                if s > 1:
                    self.assertFalse(r.spec.trait.pad_fast_tile)

    def test_small_side_bound(self):
        r = dispatch_gemm_fp16(_req(40, 4096, 4096, "fp16"))
        self.assertNotEqual(r.candidate.spec_id, "cdna_ring_small_any")


class TestSkinnyRulesDetail(unittest.TestCase):
    def test_ring_split_needs_long_k_unless_very_few_tiles(self):
        self.assertEqual(SK.ring_any_split(512, 512, 2048, 128, 128, 64), 1)
        self.assertEqual(SK.ring_any_split(256, 256, 2048, 128, 128, 64), 4)

    def test_ring_split_spec_drops_pad_fast_tile(self):
        from rocke.dispatch.gemm.bf16_rcr import _spec_cdna_dtl_ring_small

        req = _req(256, 256, 17152, "bf16")
        spec = SK.ring_small_any_spec(req, _spec_cdna_dtl_ring_small(req, "t"))
        self.assertEqual(spec.trait.split_k, 4)
        self.assertFalse(spec.trait.pad_fast_tile)
        ok, why = SK.ring_small_any_admit(req, spec)
        self.assertTrue(ok, why)

    def test_plan_last_resort_is_valid(self):
        bad = SkinnyGemmSpec(wave_m_atoms=8, wave_n_atoms=8, kc=32, prefetch=4)
        with mock.patch.object(SK, "_fallback_plan", lambda *a: bad), mock.patch(
            "rocke.dispatch.gemm.skinny_table.skinny_table_plan", lambda *a: None
        ):
            for arch in ("gfx942", "gfx950"):
                spec = SK.skinny_plan(_req(100, 3000, 5000, "fp16", arch))
                self.assertTrue(is_valid_spec(spec, arch)[0])

    def test_binding_zeroes_the_counters(self):
        from dispatch_tests.core.test_binding import FakeRuntime

        class DirtyRuntime(FakeRuntime):
            def alloc(self, n):
                p = super().alloc(n)
                self.mem[p][:] = b"\xff" * n
                return p

        result = dispatch_gemm_bf16(_req(7, 192, 1024, "bf16"))
        rt = DirtyRuntime()
        _args, ptrs = result.bind().make_args(rt)
        self.assertTrue(all(x == 0 for x in rt.mem[ptrs[4]]))

    def test_always_small_defers_to_the_ring_when_predicted_slower(self):
        # held-out regression of the first table: the ring beat the body there
        e = skinny_table_entry(22208, 8, 22144, "fp16")
        self.assertLess(e[-1], SK.SKINNY_REGION["gfx950"]["defer_ratio"])
        self.assertTrue(SK._ring_small_takes(_req(22208, 8, 22144, "fp16")))
        r = dispatch_gemm_fp16(_req(22208, 8, 22144, "fp16"))
        self.assertEqual(r.candidate.spec_id, "cdna_dtl_ring_small")

    def test_mid_band_needs_a_confident_prediction(self):
        reg = SK.SKINNY_REGION["gfx950"]
        found = 0
        for (dt, ms, bs, bg, bk), e in SKINNY_TABLE.items():
            if (
                bs != 9
                or bg < 10
                or not reg["min_ratio"] <= e[-1] < reg["mid_min_ratio"]
            ):
                continue
            s, g, K = 2**bs + 100, 2**bg + 5, 2**bk + 8
            M, N = (s, g) if ms else (g, s)
            ok, _why = SK.skinny_admit_region(
                _req(M, N, K, "fp16" if dt == "f16" else "bf16")
            )
            self.assertFalse(ok)
            found += 1
        self.assertGreater(found, 0)
