# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the gfx942 NT levers on the register-staged
``vgpr_prefetch`` body.

* ``split_k_serial``: the in-launch split-K reduction (validity scope, kernel
  arguments, workspace sizes, the two synchronisation modes in the IR).
* split-K with a K tail (``pad_k`` + ``split_k_aligned``) on the register-staged body.
* ``mfma_swap_ab`` on the interleaved register-staged bodies.
* the mode-3 interleave plan of a 1-MFMA K atom: the LDS writes of the next
  tile must be issued before the barrier that precedes the reads of it.
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
    split_k_serial_workspace,
)


def _spec(geo=(64, 64, 128, 2, 2, 1, 16, 16, 16), dt="fp16", **trait):
    base = dict(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        ab_double_buffer=False,
        chiplet_swizzle=True,
        lds_swizzle=True,
        lds_swizzle_r=0,
        vgpr_prefetch=4,
    )
    base.update(trait)
    return UniversalGemmSpec(
        name="r2t",
        tile=TileSpec(*geo),
        trait=TraitSpec(**base),
        data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt),
    )


def _ir(spec, arch="gfx942"):
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    return lower_kernel_to_llvm(build_universal_gemm(spec, arch=arch), arch=arch)


def _params(spec, arch="gfx942"):
    k = build_universal_gemm(spec, arch=arch)
    return [p.name for p in k.params]


class TestSplitKSerial(unittest.TestCase):
    def test_validity_scope(self):
        ok = _spec(split_k=4, split_k_serial=2)
        self.assertEqual(is_valid_spec(ok, arch="gfx942"), (True, "ok"))
        self.assertIn("_sks2", ok.kernel_name())
        self.assertIn("_sks1", _spec(split_k=4, split_k_serial=1).kernel_name())
        refused = {
            "no split": dict(split_k_serial=2),
            "mode": dict(split_k=4, split_k_serial=4),  # (3: atomic + ticket mode)
            "no vgpr_prefetch": dict(
                split_k=4,
                split_k_serial=2,
                vgpr_prefetch=0,
                ab_double_buffer=True,
                lds_swizzle=False,
            ),
        }
        for what, tr in refused.items():
            with self.subTest(what=what):
                self.assertFalse(is_valid_spec(_spec(**tr), arch="gfx942")[0])

    def test_arguments_and_workspace(self):
        ser = _spec(split_k=4, split_k_serial=2)
        self.assertEqual(
            _params(ser), ["A", "B", "C", "W", "Cnt", "M", "N", "K"]
        )
        self.assertEqual(
            _params(_spec(split_k=4)), ["A", "B", "Cf32", "M", "N", "K"]
        )
        # 4 slices x 4 x 4 output tiles of 64 x 64; 4 waves per tile
        self.assertEqual(
            split_k_serial_workspace(ser, 256, 256), (4 * 16 * 64 * 64, 16 * 4)
        )
        # ragged M / N: whole tiles
        self.assertEqual(
            split_k_serial_workspace(
                _spec(split_k=2, split_k_serial=2, pad_m=True, pad_n=True), 65, 129
            ),
            (2 * 2 * 3 * 64 * 64, 2 * 3 * 4),
        )
        self.assertEqual(split_k_serial_workspace(_spec(split_k=4), 256, 256), (0, 0))
        # slot byte offsets must fit i32
        big = _spec(split_k=32, split_k_serial=2)
        self.assertIsNotNone(launch_shape_error(big, 8192, 8192, 8192))
        self.assertIsNone(launch_shape_error(ser, 256, 256, 8192))

    def test_sync_modes_in_ir(self):
        # mode 1 is refused by is_valid_spec (measured wrong results); its emission is
        # kept behind SPLIT_K_SERIAL_FENCED_VALID
        from unittest import mock

        from rocke.instances.common import gemm_universal as _gu

        with mock.patch.object(_gu, "SPLIT_K_SERIAL_FENCED_VALID", True):
            ir1 = _ir(_spec(split_k=4, split_k_serial=1))
        ir2 = _ir(_spec(split_k=4, split_k_serial=2))
        # mode 1: agent-scope release before the ticket, acquire after it
        self.assertIn('fence syncscope("agent") release', ir1)
        self.assertIn('fence syncscope("agent") acquire', ir1)
        self.assertIn("atomicrmw add", ir1)
        # mode 2: write-through stores / loads, no L2 fences, relaxed ticket
        self.assertIn("global_store_dwordx4 $0, $1, off sc1", ir2)
        self.assertIn("global_load_dwordx4 $0,", ir2)
        self.assertNotIn('fence syncscope("agent")', ir2)
        self.assertRegex(ir2, r"atomicrmw add .* monotonic")
        # the counter is reset and C is stored (no f32 atomics into C)
        for ir in (ir1, ir2):
            self.assertNotIn("atomicrmw fadd", ir)

    def test_converted(self):
        # the in-launch reduction is mirrored by the C++ engine (gemm_build_sks.cpp)
        d = backend.universal_gemm_spec_to_dict(_spec(split_k=4, split_k_serial=2))
        self.assertEqual((d["split_k"], d["split_k_serial"]), (4, 2))


class TestSplitKTail(unittest.TestCase):
    def test_pad_k_split_needs_aligned_slices(self):
        s = _spec(split_k=4, pad_k=True)
        self.assertFalse(is_valid_spec(s, arch="gfx942")[0])
        a = replace(s, trait=replace(s.trait, split_k_aligned=True))
        self.assertEqual(is_valid_spec(a, arch="gfx942"), (True, "ok"))
        # any K that is a multiple of the 8-element load vector launches
        self.assertIsNone(launch_shape_error(a, 256, 256, 20744))
        ser = replace(a, trait=replace(a.trait, split_k_serial=2))
        self.assertEqual(is_valid_spec(ser, arch="gfx942"), (True, "ok"))


class TestSwapAB(unittest.TestCase):
    def test_admitted_on_interleaved_bodies(self):
        s = _spec((256, 288, 32, 4, 2, 1, 16, 16, 16), mfma_swap_ab=True, acc_agpr=True)
        self.assertEqual(is_valid_spec(s, arch="gfx942"), (True, "ok"))
        ir = _ir(s)
        self.assertIn("store <4 x half>", ir.split("define")[1])
        # still refused with pad_n (the 8-byte runs need whole rows)
        p = replace(s, trait=replace(s.trait, pad_n=True))
        self.assertFalse(is_valid_spec(p, arch="gfx942")[0])


class TestOneMfmaAtomPlan(unittest.TestCase):
    """Mode 3 with one MFMA per K atom has no free interleave slot before the
    last atom. The LDS writes of the next tile go into the other LDS half, and
    the next tile's first-atom fragments are read from that half right after
    the loop's barrier: every write must be issued before that barrier."""

    def _last(self, ir, pat):
        idx = [m.start() for m in re.finditer(pat, ir)]
        return idx[-1] if idx else -1

    def test_mode3_writes_before_the_barrier(self):
        for geo in (
            (32, 32, 32, 2, 2, 1, 16, 16, 16),
            (16, 64, 32, 1, 4, 1, 16, 16, 16),
            (64, 32, 32, 4, 2, 1, 16, 16, 16),
        ):
            s = _spec(geo, vgpr_prefetch=3, ab_double_buffer=True)
            with self.subTest(geo=geo):
                self.assertEqual(is_valid_spec(s, arch="gfx942"), (True, "ok"))
                ir = _ir(s).split("define")[1]
                last_bar = self._last(ir, r"call void @llvm\.amdgcn\.s\.barrier\(\)")
                last_lds = self._last(ir, r"store <[0-9]+ x half> [^\n]*addrspace\(3\)")
                self.assertGreater(last_bar, 0)
                self.assertGreater(last_lds, 0)
                self.assertLess(last_lds, last_bar)


if __name__ == "__main__":
    unittest.main()


class TestG5PlanDispatch(unittest.TestCase):
    """The gfx942 register-staged plan model (``dispatch.gemm.g5_plan``)."""

    def _pick(self, dt, M, N, K, arch="gfx942"):
        from rocke.dispatch import dispatch_gemm_bf16, dispatch_gemm_fp16
        from rocke.dispatch.gemm.common import GemmRequest

        fn = dispatch_gemm_fp16 if dt == "fp16" else dispatch_gemm_bf16
        return fn(GemmRequest(M=M, N=N, K=K, arch=arch, dtype=dt))

    def test_shipped_model_matches_the_features(self):
        from rocke.dispatch.gemm import g5_plan as gp

        self.assertTrue(gp.load_menu("gfx942"))
        self.assertIsNotNone(gp.load_model("gfx942"))
        self.assertEqual(gp.load_menu("gfx950"), {})
        for geo in gp.load_menu("gfx942"):
            tm, tn, tk, wm, wn, mode, swr = gp.parse_geo(geo)
            self.assertIn(mode, (3, 4))
            tile, trait = gp.plan_tile_trait(gp.G5Plan(geo), 4096, 4096, 4096)
            s = UniversalGemmSpec(
                name="r2t", tile=tile, trait=trait, data=DataSpec("fp16", "fp16", "fp16")
            )
            with self.subTest(geo=geo):
                self.assertEqual(is_valid_spec(s, arch="gfx942"), (True, "ok"))

    def test_picks_are_launchable(self):
        from rocke.instances.common.gemm_skinny import SkinnyGemmSpec

        shapes = [
            (8192, 8192, 8192), (1440, 4104, 7176), (4608, 4608, 288),
            (3072, 1920, 576), (256, 256, 20736), (512, 512, 8192),
            (1000, 3000, 1000), (4099, 4101, 1000), (17, 832, 3200),
            (2752, 34, 2432), (20096, 1408, 192), (640, 448, 9600),
        ]  # fmt: skip
        for dt in ("fp16", "bf16"):
            for M, N, K in shapes:
                r = self._pick(dt, M, N, K)
                with self.subTest(dt=dt, shape=(M, N, K)):
                    if isinstance(r.spec, SkinnyGemmSpec):
                        continue
                    self.assertEqual(is_valid_spec(r.spec, arch="gfx942"), (True, "ok"))
                    self.assertIsNone(launch_shape_error(r.spec, M, N, K))
                    names = [a["name"] for a in r.signature]
                    if r.spec.trait.split_k > 1 and r.spec.trait.split_k_serial:
                        self.assertEqual(names[2:5], ["C", "W", "Cnt"])

    def test_menu_geometries_reach_the_dispatcher(self):
        # a menu plan the fixed G5_VGPR_GEOS geometries do not have (one wave of
        # 256 x 288 tiles at 4608^2 x 288)
        r = self._pick("fp16", 4608, 4608, 288)
        t = r.spec.tile
        self.assertEqual(r.kernel_id.spec_id, "cdna_vgpr")
        self.assertNotIn((t.tile_m, t.tile_n, t.tile_k), ((256, 64, 64),))

    def test_knob_off_restores_round1(self):
        from unittest import mock

        from rocke.dispatch.gemm import common as C
        from rocke.dispatch.gemm import g5_plan

        with mock.patch.object(g5_plan, "G5P_ON", False):
            for M, N, K in ((8192, 8192, 8192), (1440, 4104, 7176)):
                r = self._pick("fp16", M, N, K)
                geo, s = C.g5_vgpr_plan(M, N, K, "gfx942")
                t = r.spec.tile
                self.assertEqual(
                    (t.tile_m, t.tile_n, t.tile_k), tuple(C.G5_VGPR_GEOS[geo][:3])
                )
                self.assertEqual(r.spec.trait.split_k_serial, 0)

    def test_analytic_order_without_the_model(self):
        from unittest import mock

        from rocke.dispatch.gemm import g5_plan as gp

        a = gp.candidate_plans(1440, 4104, 7176, "gfx942")
        with mock.patch.object(gp, "G5P_MODEL", False):
            r = gp.rank_plans(1440, 4104, 7176, "fp16", "gfx942")
        self.assertEqual([p for _t, p in r], [p for _t, p in a[: gp.G5P_PRESELECT]])
        times = [t for t, _p in a]
        self.assertEqual(times, sorted(times))

    def test_skinny_arbitration(self):
        from unittest import mock

        from rocke.dispatch.gemm import common as C
        from rocke.dispatch.gemm import g5_plan as gp
        from rocke.instances.common.gemm_skinny import SkinnyGemmSpec

        # find a request the model hands to gemm_skinny, then switch the
        # arbitration off: the register-staged body takes it back
        handed = None
        for M, N, K in ((17, 832, 3200), (320, 128, 576), (56, 960, 2752)):
            if isinstance(self._pick("fp16", M, N, K).spec, SkinnyGemmSpec):
                handed = (M, N, K)
                break
        self.assertIsNotNone(handed)
        C._G5P_CACHE.clear()
        with mock.patch.object(gp, "G5P_SKINNY_ARB", False):
            r = self._pick("fp16", *handed)
        C._G5P_CACHE.clear()
        self.assertEqual(r.kernel_id.spec_id, "cdna_vgpr")

    def test_feature_row(self):
        from rocke.dispatch.gemm import g5_plan as gp

        menu = gp.load_menu("gfx942")
        geo = sorted(menu)[0]
        x = gp.plan_features(1000, 2000, 3000, False, gp.G5Plan(geo, 2, 2), menu, 304)
        self.assertEqual(len(x), len(gp.FEATURES))
        f = dict(zip(gp.FEATURES, x))
        self.assertEqual((f["split"], f["serial"], f["kind"]), (2.0, 2.0, 0.0))
        self.assertNotEqual(f["ln_pred"], gp.G5P_NA)
        sk = gp.plan_features(16, 2000, 3000, True, gp.G5Plan("skinny"), menu, 304)
        f = dict(zip(gp.FEATURES, sk))
        self.assertEqual((f["kind"], f["bf16"], f["ln_pred"]), (1.0, 1.0, gp.G5P_NA))

    def test_vp_unroll_rule(self):
        from unittest import mock

        from rocke.dispatch.gemm import common as C
        from rocke.dispatch.gemm import g5_plan as gp

        menu = gp.load_menu("gfx942")
        small = [g for g in menu if gp.plan_unroll(gp.G5Plan(g))]
        big = [g for g in menu if g not in small]
        self.assertTrue(small and big)
        for geo, want in ((small[0], 1), (big[0], 0)):
            _t, tr = gp.plan_tile_trait(gp.G5Plan(geo, 2, 2), 2048, 2048, 4096)
            self.assertEqual(tr.vp_unroll, want)
            s = UniversalGemmSpec(name="r2t", tile=_t, trait=tr, data=DataSpec("fp16", "fp16", "fp16"))
            self.assertEqual(is_valid_spec(s, arch="gfx942"), (True, "ok"))
        with mock.patch.object(gp, "G5P_VP_UNROLL_MAX_ACC", 0):
            self.assertEqual(gp.plan_tile_trait(gp.G5Plan(small[0]), 2048, 2048, 4096)[1].vp_unroll, 0)
        # the ranking scales unrolled plans by G5P_VP_UNROLL_SCALE (and the dispatcher cache keys on it)
        C._G5P_CACHE.clear()
        base = dict((p, t) for t, p in gp.rank_plans(1000, 3000, 1000, "fp16", "gfx942"))
        with mock.patch.object(gp, "G5P_VP_UNROLL_SCALE", 1.0):
            off = dict((p, t) for t, p in gp.rank_plans(1000, 3000, 1000, "fp16", "gfx942"))
        for p, t in base.items():
            self.assertAlmostEqual(t / off[p], 0.97 if gp.plan_unroll(p) else 1.0, places=9)
