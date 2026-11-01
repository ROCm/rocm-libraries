# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU checks of the rules where the GEMM levers interact (coverage levers,
gemm_skinny, native layouts, the learned selector, the gfx942 register-staged
prefetch).

* An MN-contiguous operand's leading dimension must be a whole number of loads
  (``mn_launch_error``): ``mn_vec`` on the VGPR-staged loader, 8 on the DTL
  ring. The VMM-guarded GPU fuzz found faults at 128x110x768 (RRR) and
  347x240x16 (CRR) with the default ``mn_vec`` 8.
* The DTL K tail and the register-staged prefetch are K-contiguous-only.
* The ragged-row launch rule accepts every bounded global-load path
  (``pad_row_clamp``, ``kc_row_guard`` / any non-RCR layout,
  ``vgpr_prefetch``) and nothing else.
* The per-request adaptation (G1_ADAPT) leaves register-staged bodies alone;
  the bf16 decode body takes the guarded loader only when the adaptation is off.
* Registry order: the gfx942 register-staged body ahead of gemm_skinny where
  both admit.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

from rocke.dispatch.gemm import (
    GemmRequest,
    dispatch_gemm_bf16,
    dispatch_gemm_fp16,
    dispatch_gemm_fp16_all,
)
from rocke.dispatch.gemm import common as _gc
from rocke.dispatch.gemm import skinny as _sk
from rocke.instances.common.gemm_mn_operand import mn_launch_error
from rocke.instances.common.gemm_skinny import SkinnyGemmSpec
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    is_valid_spec,
    launch_shape_error,
)

_PAD = TraitSpec(
    pipeline="compv4", epilogue="default", pad_m=True, pad_n=True, pad_k=True
)


def _spec(tile, trait, layout="RCR", dt="bf16"):
    return UniversalGemmSpec(
        name="t",
        tile=tile,
        trait=trait,
        data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt, layout=layout),
    )


def _req(M, N, K, dt="bf16", lay="RCR", arch="gfx950"):
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


_T128x256 = TileSpec(128, 256, 32, 2, 2, 1, 32, 32, 16)
_T128 = TileSpec(128, 128, 16, 2, 2, 1, 32, 32, 8)


class TestMnLaunchRule(unittest.TestCase):
    def test_fuzz_faults_refused(self):
        # the two fuzz faults (generic VGPR-staged MN loader, mn_vec 8)
        s1 = _spec(_T128x256, replace(_PAD, kc_row_guard=True), "RRR")
        self.assertTrue(is_valid_spec(s1, arch="gfx950")[0])
        self.assertIn("mn_vec=8", launch_shape_error(s1, 128, 110, 768))
        s2 = _spec(_T128, replace(_PAD, pad_k=False, kc_row_guard=True), "CRR")
        self.assertIn("mn_vec=8", launch_shape_error(s2, 347, 240, 16))
        # a dividing load width is admitted
        s1b = replace(s1, trait=replace(s1.trait, mn_vec=2))
        self.assertIsNone(launch_shape_error(s1b, 128, 110, 768))
        s2b = replace(s2, trait=replace(s2.trait, mn_vec=1))
        self.assertIsNone(launch_shape_error(s2b, 347, 240, 16))

    def test_only_mn_operands_count(self):
        # RRR: B is N-contiguous, A is K-contiguous -> only N matters
        s = _spec(_T128x256, replace(_PAD, kc_row_guard=True), "RRR")
        self.assertIsNone(mn_launch_error(s, 130, 112, 768))
        self.assertIsNotNone(mn_launch_error(s, 128, 114, 768))
        # CCR: A is M-contiguous, B K-contiguous -> only M matters
        c = replace(s, data=replace(s.data, layout="CCR"))
        self.assertIsNone(mn_launch_error(c, 128, 114, 768))
        self.assertIsNotNone(mn_launch_error(c, 130, 112, 768))
        # RCR: no MN operand
        r = replace(s, data=replace(s.data, layout="RCR"))
        self.assertIsNone(mn_launch_error(r, 131, 117, 768))

    def test_dtl_ring_needs_16_byte_rows(self):
        r = dispatch_gemm_bf16(_req(4096, 4096, 4096, lay="RRR"))
        self.assertTrue(r.spec.trait.direct_to_lds)
        self.assertIsNone(mn_launch_error(r.spec, 4096, 4096, 4096))
        self.assertIn("8 (16-byte", mn_launch_error(r.spec, 4096, 4092, 4096))

    def test_dispatcher_picks_satisfy_the_rule(self):
        for lay in ("RRR", "CRR", "CCR"):
            for M, N, K in ((347, 240, 16), (128, 110, 768), (1000, 1002, 520)):
                rs = dispatch_gemm_fp16_all(_req(M, N, K, "fp16", lay))
                self.assertTrue(rs)
                for r in rs:
                    if isinstance(r.spec, SkinnyGemmSpec):
                        continue
                    with self.subTest(lay=lay, shape=(M, N, K), c=r.candidate.name):
                        self.assertIsNone(launch_shape_error(r.spec, M, N, K))


class TestLeverScopes(unittest.TestCase):
    def test_mn_k_tail_needs_exact_bounds(self):
        # the native rings take K % tile_k != 0 with the DTL K tail: the
        # K-contiguous operand's lanes are masked, an MN-contiguous operand's
        # K-tail rows read zeros through its exact-bounds resource
        for lay in ("RRR", "CRR", "CCR"):
            for M, N, K in ((4096, 4096, 4104), (2304, 1216, 1000)):
                r = dispatch_gemm_bf16(_req(M, N, K, lay=lay))
                s = r.spec
                with self.subTest(lay=lay, shape=(M, N, K), c=r.candidate.name):
                    self.assertIsNone(launch_shape_error(s, M, N, K))
                    if not s.trait.direct_to_lds:
                        continue
                    self.assertTrue(s.trait.dtl_k_tail)
                    self.assertTrue(s.trait.dtl_tight_rsrc)
                    loose = replace(s, trait=replace(s.trait, dtl_tight_rsrc=False))
                    err = launch_shape_error(loose, M, N, K) or ""
                    self.assertTrue(err)
                    if M % s.tile.tile_m == 0 and N % s.tile.tile_n == 0:
                        self.assertIn("MN-contiguous", err)
        # a native ring for each layout x K-tail mode builds
        from rocke.instances.common.gemm_universal import build_universal_gemm

        for lay in ("RRR", "CRR", "CCR"):
            s = dispatch_gemm_bf16(_req(2304, 1216, 1000, lay=lay)).spec
            if not s.trait.direct_to_lds:
                continue
            for mode in (1, 2):
                if mode == 2 and s.trait.pingpong:
                    continue
                k = replace(s, trait=replace(s.trait, dtl_k_tail=mode))
                with self.subTest(lay=lay, mode=mode):
                    self.assertTrue(is_valid_spec(k, arch="gfx950")[0])
                    build_universal_gemm(k, arch="gfx950")

    def test_vgpr_prefetch_is_k_contiguous_only(self):
        vp = _spec(
            TileSpec(256, 256, 64, 2, 4, 1, 16, 16, 16),
            TraitSpec(
                pipeline="compv4",
                epilogue="default",
                ab_double_buffer=False,
                vgpr_prefetch=4,
                lds_swizzle=True,
                lds_swizzle_r=0,
            ),
        )
        self.assertTrue(is_valid_spec(vp, arch="gfx942")[0])
        for lay in ("RRR", "CRR", "CCR"):
            ok, why = is_valid_spec(
                replace(vp, data=replace(vp.data, layout=lay)), arch="gfx942"
            )
            self.assertFalse(ok, lay)
            self.assertIn("vgpr_prefetch", why)

    def test_ragged_rows_need_a_bounded_path(self):
        base = _spec(TileSpec(128, 128, 32, 2, 2, 1, 32, 32, 16), _PAD)
        self.assertIn("pad_row_clamp", launch_shape_error(base, 130, 128, 64))
        for kw in (dict(pad_row_clamp=True), dict(kc_row_guard=True)):
            s = replace(base, trait=replace(base.trait, **kw))
            self.assertIsNone(launch_shape_error(s, 130, 128, 64), kw)
        # a non-RCR layout guards its rows on the VGPR-staged loader
        n = replace(base, data=replace(base.data, layout="RRR"))
        self.assertIsNone(launch_shape_error(n, 130, 128, 64))

    def test_g1_adapt_leaves_vgpr_prefetch_bodies(self):
        tt = _gc.g5_vgpr_tile_trait(
            GemmRequest(M=3000, N=5000, K=4096, arch="gfx942", dtype="bf16")
        )
        self.assertIsNotNone(tt)
        s = _spec(*tt)
        req = GemmRequest(M=3000, N=5000, K=4104, arch="gfx942", dtype="bf16")
        a = _gc.g1_adapt_spec(req, replace(s, trait=replace(s.trait, pad_k=True)))
        self.assertFalse(a.trait.pad_row_clamp)
        self.assertFalse(a.trait.pad_k_fast)

    def test_decode_guard_follows_g1_adapt(self):
        dec = GemmRequest(M=5, N=4736, K=2624, arch="gfx950", dtype="bf16")
        with mock.patch.object(_sk, "SKINNY_DISPATCH", False), mock.patch.object(
            _sk, "SKINNY_FALLBACK", False
        ), mock.patch.object(_sk, "RING_ANY", False):
            r = dispatch_gemm_bf16(dec)
            if r.candidate.name == "universal_gemm_bf16_cdna_decode":
                self.assertTrue(r.spec.trait.pad_row_clamp)
                self.assertFalse(r.spec.trait.kc_row_guard)
            with mock.patch.object(_gc, "G1_ADAPT", False):
                r = dispatch_gemm_bf16(dec)
                self.assertIsNone(launch_shape_error(r.spec, 5, 4736, 2624))
                if r.candidate.name == "universal_gemm_bf16_cdna_decode":
                    self.assertTrue(r.spec.trait.kc_row_guard)


class TestKTailCompileCliff(unittest.TestCase):
    def test_k_tail_drops_iglp(self):
        req = GemmRequest(M=4864, N=5120, K=392, arch="gfx950", dtype="fp16")
        for r in dispatch_gemm_fp16_all(req):
            if isinstance(r.spec, SkinnyGemmSpec):
                continue
            with self.subTest(c=r.candidate.name):
                tr = r.spec.trait
                self.assertFalse(tr.dtl_k_tail and tr.iglp_opt is not None)
        ring = dispatch_gemm_fp16(
            GemmRequest(M=4864, N=5120, K=384, arch="gfx950", dtype="fp16")
        ).spec
        ig = replace(ring, trait=replace(ring.trait, iglp_opt=1))
        a = _gc.g1_adapt_spec(req, ig)
        self.assertIsNone(a.trait.iglp_opt)
        self.assertTrue(a.trait.dtl_k_tail)
        with mock.patch.object(_gc, "G1_K_TAIL_DROP_IGLP", False):
            a = _gc.g1_adapt_spec(req, ig)
            self.assertEqual(a.trait.iglp_opt, 1)
            self.assertEqual(a.trait.dtl_k_tail, 1)


class TestNativeRingAny(unittest.TestCase):
    def setUp(self):
        # the native gemm_skinny (priority 3) takes some of these
        # few-tile shapes; the ring knobs are tested without it
        p = mock.patch.object(_sk, "SKINNY_NATIVE", False)
        p.start()
        self.addCleanup(p.stop)

    def test_knob(self):
        # few-tile NN shape: ring_any natively (ahead of
        # ring_small_any); with MERGE_NATIVE_RING_ANY off neither serves a
        # non-RCR request (both are registered RCR-only)
        req = _req(256, 512, 1024, lay="RRR")
        r = dispatch_gemm_bf16(req)
        self.assertEqual(
            r.candidate.name, "universal_gemm_bf16_cdna_dtl_ring_any_first"
        )
        self.assertEqual(r.spec.data.layout, "RRR")
        self.assertTrue(r.spec.trait.direct_to_lds)
        with mock.patch.object(_gc, "MERGE_RING_ANY_FIRST", False):
            r = dispatch_gemm_bf16(req)
            self.assertEqual(
                r.candidate.name, "universal_gemm_bf16_cdna_ring_small_any"
            )
        with mock.patch.object(_gc, "MERGE_NATIVE_RING_ANY", False):
            r0 = dispatch_gemm_bf16(req)
            self.assertNotIn("ring_small_any", r0.candidate.name)
            self.assertNotIn("dtl_ring_any", r0.candidate.name)
            # RCR keeps both
            r1 = dispatch_gemm_bf16(_req(256, 512, 1024))
            self.assertIn(
                r1.candidate.name,
                (
                    "universal_gemm_bf16_cdna_dtl_ring_any_first",
                    "universal_gemm_bf16_cdna_skinny",
                ),
            )

    def test_launchable_form_first(self):
        req = _req(4096, 4096, 4104, lay="RRR")
        good = dispatch_gemm_bf16(req).spec
        bad = replace(good, trait=replace(good.trait, dtl_tight_rsrc=False))
        self.assertIsNotNone(launch_shape_error(bad, 4096, 4096, 4104))
        alts = ([("bad", bad), ("good", good)], "ok")
        with mock.patch.object(_gc, "native_layout_alternatives", return_value=alts):
            s, _ = _gc.native_layout_spec(req, good)
            self.assertIs(s, good)
            with mock.patch.object(_gc, "NATIVE_LAYOUT_LAUNCHABLE_FIRST", False):
                s, _ = _gc.native_layout_spec(req, good)
                self.assertIs(s, bad)

    def test_native_forms_adapted_for_the_k_tail(self):
        rcr = dispatch_gemm_bf16(_req(4096, 4096, 4104)).spec
        for lay in ("RRR", "CRR", "CCR"):
            req = _req(4096, 4096, 4104, lay=lay)
            alts, why = _gc.native_layout_alternatives(req, rcr)
            self.assertTrue(alts, why)
            for nm, s in alts:
                with self.subTest(lay=lay, form=nm):
                    if s.trait.direct_to_lds:
                        self.assertTrue(s.trait.dtl_k_tail)
                        self.assertTrue(s.trait.dtl_tight_rsrc)
                        self.assertFalse(s.trait.dtl_soff0)
                        self.assertFalse(s.trait.dtl_tight_ragged_only)
                    self.assertIsNone(launch_shape_error(s, 4096, 4096, 4104))

    def test_adapt_tightens_mn_k_tail(self):
        rcr = dispatch_gemm_bf16(_req(4096, 4096, 4096)).spec
        req = _req(4096, 4096, 4104, lay="CCR")
        s = replace(rcr, data=replace(rcr.data, layout="CCR"))
        a = _gc.g1_adapt_spec(req, s)
        self.assertTrue(a.trait.dtl_k_tail)
        self.assertTrue(a.trait.dtl_tight_rsrc)
        self.assertFalse(a.trait.dtl_soff0)
        # RCR: the K tail without the exact-bounds resource (unchanged)
        a2 = _gc.g1_adapt_spec(_req(4096, 4096, 4104), rcr)
        self.assertTrue(a2.trait.dtl_k_tail)
        self.assertEqual(a2.trait.dtl_tight_rsrc, rcr.trait.dtl_tight_rsrc)


class TestRegistryOrder(unittest.TestCase):
    def test_gfx942_vgpr_ahead_of_skinny(self):
        r = dispatch_gemm_fp16(
            GemmRequest(M=64, N=4096, K=4096, arch="gfx942", dtype="fp16")
        )
        self.assertEqual(r.candidate.name, "universal_gemm_fp16_cdna_vgpr")
        # below G5_VGPR_MIN_MN the register-direct skinny body takes it
        r = dispatch_gemm_fp16(
            GemmRequest(M=8, N=4096, K=4096, arch="gfx942", dtype="fp16")
        )
        self.assertIn("skinny", r.candidate.name)
        with mock.patch.object(_gc, "G5_VGPR", False):
            r = dispatch_gemm_fp16(
                GemmRequest(M=64, N=4096, K=4096, arch="gfx942", dtype="fp16")
            )
            self.assertNotEqual(r.candidate.name, "universal_gemm_fp16_cdna_vgpr")


if __name__ == "__main__":
    unittest.main()
