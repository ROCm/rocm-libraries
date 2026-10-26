# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the ``vgpr_prefetch`` lever (gfx942).

``vgpr_prefetch`` turns the compv4 VGPR-staged double-buffer body into a
register-staged prefetch loop (the global reads of K tile i + 2 are issued one
whole tile before their LDS write). The checks: the validity scope, the kernel
name flag, the Python-engine-only refusal of the C++ converters, the emitted
loop structure (buffer loads, no vmcnt(0) barrier drain, the split swizzled LDS
writes) and that the default stays byte-identical.
"""

from __future__ import annotations

import re
import unittest
from dataclasses import replace

from rocke.core import backend
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    launch_shape_error,
)


def _spec(geo=(256, 256, 32, 2, 2, 1, 32, 32, 8), dt="fp16", **trait):
    tile = TileSpec(*geo)
    base = dict(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        ab_double_buffer=True,
        chiplet_swizzle=True,
    )
    base.update(trait)
    return UniversalGemmSpec(
        name="g5t",
        tile=tile,
        trait=TraitSpec(**base),
        data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt),
    )


def _ir(spec, arch="gfx942"):
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    return lower_kernel_to_llvm(build_universal_gemm(spec, arch=arch), arch=arch)


class TestVgprPrefetchValidity(unittest.TestCase):
    def test_admitted_bodies(self):
        for mode in (1, 2, 3):
            for geo in (
                (256, 256, 32, 2, 2, 1, 32, 32, 8),
                (256, 256, 32, 4, 4, 1, 16, 16, 16),
                (128, 128, 64, 2, 2, 1, 32, 32, 8),
            ):
                for swz in (False, True):
                    extra = dict(lds_swizzle=True, lds_swizzle_r=1) if swz else {}
                    s = _spec(geo, vgpr_prefetch=mode, **extra)
                    with self.subTest(mode=mode, geo=geo, swz=swz):
                        self.assertEqual(is_valid_spec(s, arch="gfx942"), (True, "ok"))
                        self.assertIn(f"_vp{mode}", s.kernel_name())

    def test_refused(self):
        cases = {
            "mode": dict(vgpr_prefetch=5),
            "mode 4 with the double buffer": dict(vgpr_prefetch=4),
            "mode 3 single buffer": dict(vgpr_prefetch=3, ab_double_buffer=False),
            "depth": dict(vgpr_prefetch=1, vgpr_prefetch_depth=4),
            "depth without the lever": dict(vgpr_prefetch_depth=2),
            "pad_k with split-K": dict(vgpr_prefetch=1, pad_k=True, split_k=2),
            "dtl": dict(
                vgpr_prefetch=1,
                direct_to_lds=True,
                dtl_prefetch=True,
                ab_double_buffer=None,
            ),
            "single buffer": dict(vgpr_prefetch=1, ab_double_buffer=False),
            "preshuffle_b": dict(vgpr_prefetch=1, preshuffle_b=True),
            "mem pipeline": dict(
                vgpr_prefetch=1, pipeline="mem", ab_double_buffer=None
            ),
        }
        for what, tr in cases.items():
            with self.subTest(what=what):
                ok, why = is_valid_spec(_spec(**tr), arch="gfx942")
                self.assertFalse(ok, what)
        # mode 3 needs >= 2 K atoms per tile
        s = _spec((256, 256, 16, 4, 4, 1, 16, 16, 16), vgpr_prefetch=3)
        self.assertFalse(is_valid_spec(s, arch="gfx942")[0])
        self.assertTrue(
            is_valid_spec(
                replace(s, trait=replace(s.trait, vgpr_prefetch=1)), arch="gfx942"
            )[0]
        )

    def test_swizzle_split_writes_admitted_only_with_vgpr_prefetch(self):
        sw = dict(lds_swizzle=True, lds_swizzle_r=1)
        self.assertFalse(is_valid_spec(_spec(**sw), arch="gfx942")[0])
        self.assertTrue(is_valid_spec(_spec(vgpr_prefetch=1, **sw), arch="gfx942")[0])

    def test_mode4_single_buffer(self):
        for geo in (
            (256, 256, 64, 2, 2, 1, 16, 16, 16),
            (128, 64, 128, 2, 2, 1, 16, 16, 16),
        ):
            s = _spec(
                geo,
                vgpr_prefetch=4,
                ab_double_buffer=False,
                lds_swizzle=True,
                lds_swizzle_r=0,
            )
            with self.subTest(geo=geo):
                self.assertEqual(is_valid_spec(s, arch="gfx942"), (True, "ok"))
                self.assertIn("_absb", s.kernel_name())
        # cshuffle aliases C onto the single AB buffer: refused
        s = _spec(vgpr_prefetch=4, ab_double_buffer=False, epilogue="cshuffle")
        self.assertFalse(is_valid_spec(s, arch="gfx942")[0])

    def test_pad_k_launch(self):
        s = _spec(
            (256, 256, 64, 2, 2, 1, 16, 16, 16),
            vgpr_prefetch=4,
            ab_double_buffer=False,
            pad_k=True,
        )
        self.assertTrue(is_valid_spec(s, arch="gfx942")[0])
        self.assertIsNone(launch_shape_error(s, 4096, 4096, 4104))
        # a 16-byte vector would straddle K
        self.assertIsNotNone(launch_shape_error(s, 4096, 4096, 4100))

    def test_launch_range(self):
        s = _spec(vgpr_prefetch=1)
        self.assertIsNone(launch_shape_error(s, 8192, 8192, 8192))
        self.assertIsNotNone(launch_shape_error(s, 65536, 256, 32768))
        sp = _spec(vgpr_prefetch=1, pad_m=True, pad_n=True)
        self.assertIsNone(launch_shape_error(sp, 1000, 1000, 1024))
        self.assertIsNotNone(launch_shape_error(sp, 40000, 256, 32768))

    def test_python_only(self):
        s = _spec(vgpr_prefetch=1)
        with self.assertRaises(NotImplementedError):
            backend.universal_gemm_spec_to_dict(s)
        backend.universal_gemm_spec_to_dict(_spec())  # default converts


class TestG5Dispatch(unittest.TestCase):
    def _pick(self, dt, M, N, K, arch="gfx942"):
        from rocke.dispatch import dispatch_gemm_bf16, dispatch_gemm_fp16
        from rocke.dispatch.gemm.common import GemmRequest

        fn = dispatch_gemm_fp16 if dt == "fp16" else dispatch_gemm_bf16
        return fn(GemmRequest(M=M, N=N, K=K, arch=arch, dtype=dt))

    def test_gfx942_takes_the_register_staged_body(self):
        for dt in ("fp16", "bf16"):
            for M, N, K in (
                (8192, 8192, 8192),
                (4096, 4096, 512),
                (2048, 2048, 2048),
                (11840, 6272, 5184),
                (4096, 4096, 4104),
                (64, 4096, 4096),
            ):
                with self.subTest(dt=dt, shape=(M, N, K)):
                    r = self._pick(dt, M, N, K)
                    self.assertEqual(r.kernel_id.spec_id, "cdna_vgpr")
                    tr = r.spec.trait
                    self.assertIn(tr.vgpr_prefetch, (3, 4))
                    self.assertEqual(is_valid_spec(r.spec, arch="gfx942"), (True, "ok"))
                    self.assertIsNone(launch_shape_error(r.spec, M, N, K))
                    t = r.spec.tile
                    self.assertEqual(tr.pad_m, M % t.tile_m != 0)
                    self.assertEqual(tr.pad_n, N % t.tile_n != 0)
                    # a K tail, or (gfx942 plan model) split-K
                    # slices that do not divide K: whole-tile slices + the
                    # masked tail (split_k_aligned)
                    ragged = tr.split_k > 1 and K % (tr.split_k * t.tile_k) != 0
                    self.assertEqual(tr.pad_k, K % t.tile_k != 0 or ragged)
                    if ragged:
                        self.assertTrue(tr.split_k_aligned)

    def test_big_square_geometry(self):
        from unittest import mock

        from rocke.dispatch.gemm import common as C
        from rocke.dispatch.gemm import g5_plan

        # the fixed menu (G5_VGPR_GEOS) and its cost model; the plan
        # model has its own tests (test_gemm_r2_nt_tune.py)
        with mock.patch.object(g5_plan, "G5P_ON", False):
            r = self._pick("fp16", 8192, 8192, 8192)
        t = r.spec.tile
        # a 4-wave tile_k 64 single-buffer body, the cheapest plan of the model
        self.assertEqual((t.tile_k, t.warp_m, t.warp_n), (64, 2, 2))
        self.assertGreaterEqual(min(t.tile_m, t.tile_n), 192)
        self.assertEqual(r.spec.trait.vgpr_prefetch, 4)
        self.assertIs(r.spec.trait.ab_double_buffer, False)
        geo, s = C.g5_vgpr_plan(8192, 8192, 8192, "gfx942")
        self.assertEqual(
            (C.G5_VGPR_GEOS[geo][0], C.G5_VGPR_GEOS[geo][1]), (t.tile_m, t.tile_n)
        )
        self.assertEqual(s, 1)

    def test_gfx950_unchanged(self):
        for M, N, K in ((8192, 8192, 8192), (2048, 2048, 2048), (3072, 1920, 576)):
            r = self._pick("fp16", M, N, K, arch="gfx950")
            self.assertNotEqual(r.kernel_id.spec_id, "cdna_vgpr")
            self.assertEqual(r.spec.trait.vgpr_prefetch, 0)

    def test_knob_off_restores_the_base_picks(self):
        from unittest import mock

        from rocke.dispatch.gemm import common as C

        with mock.patch.object(C, "G5_VGPR", False):
            r = self._pick("fp16", 8192, 8192, 8192)
        self.assertEqual(r.kernel_id.spec_id, "cdna_dtl_pingpong_256x256")

    def test_k_tail_and_split(self):
        from unittest import mock

        from rocke.dispatch.gemm import common as C
        from rocke.dispatch.gemm import skinny as SK

        # gemm_skinny takes 16 x 4096 ahead of this body on gfx942
        # (SKINNY_OVER_VGPR); the body's own split rule is tested without it
        p = mock.patch.dict(SK.SKINNY_OVER_VGPR, clear=True)
        p.start()
        self.addCleanup(p.stop)
        from rocke.dispatch.gemm import g5_plan

        with mock.patch.object(g5_plan, "G5P_ON", False):
            self._k_tail_and_split(C)

    def _k_tail_and_split(self, C):
        r = self._pick("fp16", 4096, 4096, 4104)
        self.assertTrue(r.spec.trait.pad_k)
        self.assertEqual(r.spec.trait.split_k, 1)
        for M, N, K in ((128, 1024, 16384), (16, 4096, 4096)):
            r = self._pick("fp16", M, N, K)
            sk = r.spec.trait.split_k
            t = r.spec.tile
            self.assertEqual(K % (sk * t.tile_k), 0)
            self.assertFalse(r.spec.trait.pad_fast_tile)
            geo, s = C.g5_vgpr_plan(M, N, K, "gfx942")
            self.assertEqual(s, sk)


class TestVgprPrefetchEmission(unittest.TestCase):
    def test_default_unchanged(self):
        s = _spec()
        self.assertEqual(
            _ir(s), _ir(replace(s, trait=replace(s.trait, vgpr_prefetch=0)))
        )

    def test_loop_shape(self):
        for mode in (1, 2, 3, 4):
            s = _spec(
                vgpr_prefetch=mode,
                lds_swizzle=True,
                lds_swizzle_r=1,
                ab_double_buffer=mode != 4,
            )
            ir = _ir(s)
            with self.subTest(mode=mode):
                # register-staged global reads through buffer resources
                self.assertIn("llvm.amdgcn.raw.ptr.buffer.load", ir)
                # one 16-byte global read is written as two 8-byte LDS stores
                # (the 4-half swizzle group of the 32x32x8 atom)
                self.assertNotIn("store <8 x half>", ir.split("define")[1])
                self.assertIn("store <4 x half>", ir)

    def test_barriers_with_tiny_atoms(self):
        # mode 4 needs exactly two barriers per K tile (+ one in the prologue)
        # and mode 3 one, also when an atom has fewer MFMAs than the
        # interleave plan has items (16-row tiles: 2 MFMAs per K atom)
        geos = [
            ((16, 128, 128, 1, 4, 1, 16, 16, 16), 4),
            ((128, 16, 128, 4, 1, 1, 16, 16, 16), 4),
            ((16, 64, 128, 1, 4, 1, 16, 16, 16), 4),  # 1 MFMA per K atom
            ((256, 256, 64, 2, 2, 1, 16, 16, 16), 4),
            ((128, 128, 64, 2, 2, 1, 32, 32, 8), 3),
            ((16, 64, 128, 1, 4, 1, 16, 16, 16), 3),
            ((256, 256, 32, 2, 2, 1, 32, 32, 8), 1),
            ((256, 256, 32, 2, 2, 1, 32, 32, 8), 2),
        ]
        for geo, mode in geos:
            s = _spec(
                geo,
                vgpr_prefetch=mode,
                ab_double_buffer=mode != 4,
                pad_m=True,
                pad_n=True,
            )
            with self.subTest(geo=geo, mode=mode):
                self.assertTrue(is_valid_spec(s, arch="gfx942")[0])
                ir = _ir(s)
                self.assertEqual(
                    ir.count("call void @llvm.amdgcn.s.barrier()"),
                    3 if mode == 4 else 2,
                )

    def test_k_tail_mask_and_clamps(self):
        # pad_k: a vector past K gets the out-of-range voffset (0x7FFFFFF0)
        geo = (256, 256, 64, 2, 2, 1, 16, 16, 16)
        base = dict(vgpr_prefetch=4, ab_double_buffer=False)
        ir_k = _ir(_spec(geo, pad_k=True, **base))
        ir_0 = _ir(_spec(geo, **base))
        self.assertIn("2147483632", ir_k)
        self.assertNotIn("2147483632", ir_0)
        # the tile-1 prefetch (prologue) and the tile-i+2 prefetch (loop) are
        # both clamped to the last tile
        for mode in (1, 3, 4):
            s = _spec(
                geo if mode == 4 else (256, 256, 32, 2, 2, 1, 32, 32, 8),
                vgpr_prefetch=mode,
                ab_double_buffer=mode != 4,
            )
            with self.subTest(mode=mode):
                self.assertEqual(_ir(s).count("call i32 @llvm.smin.i32("), 2)

    def test_padded_rows_use_the_tight_range(self):
        geo = (256, 256, 64, 2, 2, 1, 16, 16, 16)
        base = dict(vgpr_prefetch=4, ab_double_buffer=False)
        n0 = _ir(_spec(geo, **base)).count("call i32 @llvm.smax.i32(")
        n_m = _ir(_spec(geo, pad_m=True, **base)).count("call i32 @llvm.smax.i32(")
        n_mn = _ir(_spec(geo, pad_m=True, pad_n=True, **base)).count(
            "call i32 @llvm.smax.i32("
        )
        self.assertEqual((n_m - n0, n_mn - n0), (1, 2))

    def test_swizzle_period_rule(self):
        # 16-row tile, 4-half groups over tile_k 128 (W = 5), R = 1: period 64
        s = _spec(
            (16, 128, 128, 1, 4, 1, 16, 16, 16),
            vgpr_prefetch=4,
            ab_double_buffer=False,
            lds_swizzle=True,
            lds_swizzle_r=1,
        )
        ok, why = is_valid_spec(s, arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("period", why)

    def test_no_swizzle_keeps_wide_writes(self):
        ir = _ir(_spec(vgpr_prefetch=1))
        self.assertIn("store <8 x half>", ir)


if __name__ == "__main__":
    unittest.main()
