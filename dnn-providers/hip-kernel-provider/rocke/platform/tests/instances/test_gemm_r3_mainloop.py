# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 NT long-K main-loop levers. CPU only.

* ``common.R3_KPROXY``: K % 64 != 0 (K % 8 == 0) RCR requests on gfx950 are dispatched as K rounded up to a
  multiple of 64 and run with the DTL K tail (``g1_adapt_spec``) instead of the tile_k 32 ring geometry;
* ``common.R3_QGEO``: a dispatched 128x128x64 2x4-wave ring is replaced by the 256x192 / 192x256 4-wave
  interleave ring when that ring's quantised work is at most the small ring's (K % 64 != 0 by default);
* ``gemm_universal.STREAMK_KLOOP_UNROLL``: kloop_unroll 2 / 4 is valid on stream-K bodies;
* ``gemm_universal.STREAMK_RESIDENCY_WPE``: a ``waves_per_eu`` hint counts as the register budget of the
  stream-K co-residency bound.
"""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest import mock

from rocke.dispatch.gemm import GemmRequest, dispatch_gemm_bf16, dispatch_gemm_fp16
from rocke.dispatch.gemm import common as C
from rocke.dispatch.gemm import fp16_rcr as F16
from rocke.instances.common import gemm_universal as GU
from rocke.instances.common.gemm_universal import is_valid_spec, launch_shape_error


def _req(M, N, K, lay="RCR", dt="fp16", arch="gfx950"):
    return GemmRequest(
        M=M, N=N, K=K, arch=arch, dtype=dt, layout=lay,
        trans_a=lay[0] == "C", trans_b=lay[1] == "C",
    )  # fmt: skip


def _disp(req):
    return (dispatch_gemm_fp16 if req.dtype == "fp16" else dispatch_gemm_bf16)(req)


def _kp_line(res):
    return [e for e in res.explanation if e.startswith("r3 k-proxy")]


class TestKProxy(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {"ROCKE_GEMM_SELECTOR": "0"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def _with(self, on):
        p = mock.patch.object(C, "R3_KPROXY", on)
        p.start()
        self.addCleanup(p.stop)

    def test_k_tail_pick(self):
        for dt in ("fp16", "bf16"):
            for M, N, K in (
                (4104, 4104, 4104),
                (7176, 1440, 4104),
                (15752, 2056, 2776),
            ):
                req = _req(M, N, K, dt=dt)
                res = _disp(req)
                sp = res.spec
                self.assertTrue(_kp_line(res), (dt, M, N, K))
                self.assertEqual(sp.tile.tile_k, 64)
                self.assertTrue(sp.trait.direct_to_lds)
                self.assertEqual(sp.trait.dtl_k_tail, 2)
                self.assertTrue(sp.trait.pad_k)
                self.assertTrue(is_valid_spec(sp, arch="gfx950")[0])
                self.assertIsNone(launch_shape_error(sp, M, N, K))

    def test_proxy_spec_is_rounded_pick_adapted(self):
        req = _req(4104, 4104, 4104)
        proxy = dispatch_gemm_fp16(replace(req, K=4160))
        with mock.patch.object(C, "R3_QGEO", False):
            self.assertEqual(_disp(req).spec, C.g1_adapt_spec(req, proxy.spec))

    def test_knob_off_restores_dispatch(self):
        req = _req(4104, 4104, 4104)
        self._with(False)
        res = _disp(req)
        self.assertFalse(_kp_line(res))
        self.assertEqual(res.spec.tile.tile_k, 32)

    def test_not_applied(self):
        # aligned K, K % 8 != 0, other layouts, other arches: no K-proxy
        for req in (
            _req(4096, 4096, 4096),
            _req(4104, 4104, 4100),
            _req(4104, 4104, 4104, lay="RRR"),
            _req(4104, 4104, 4104, arch="gfx942"),
        ):
            self.assertIsNone(C.r3_kproxy_dispatch(req, dispatch_gemm_fp16), req)
            self.assertFalse(_kp_line(_disp(req)), req)

    def test_aligned_dispatch_unchanged(self):
        req = _req(8192, 8192, 8192)
        a = _disp(req)
        self._with(False)
        b = _disp(req)
        self.assertEqual(a.spec, b.spec)
        self.assertEqual(a.candidate.name, b.candidate.name)

    def test_non_dtl_proxy_pick_falls_back(self):
        # a proxy pick that is not a direct-to-LDS body (here: gemm_skinny) keeps the normal dispatch
        req = _req(64, 64, 200)
        self.assertIsNone(C.r3_kproxy_dispatch(req, dispatch_gemm_fp16))

    def test_proxy_failure_falls_back(self):
        def boom(_req):
            raise RuntimeError("x")

        self.assertIsNone(C.r3_kproxy_dispatch(_req(4104, 4104, 4104), boom))

    def test_short_k_keeps_measured_bodies(self):
        # K < R3_KPROXY_MIN_K keeps the short-K tile_k 32 bodies; at the bound the proxy applies
        self.assertIsNone(
            C.r3_kproxy_dispatch(_req(4608, 4608, 288), dispatch_gemm_fp16)
        )
        self.assertIsNone(
            C.r3_kproxy_dispatch(_req(4608, 4608, 504), dispatch_gemm_fp16)
        )
        self.assertIsNotNone(
            C.r3_kproxy_dispatch(_req(4608, 4608, 520), dispatch_gemm_fp16)
        )
        with mock.patch.object(C, "R3_KPROXY_MIN_K", 0):
            self.assertIsNotNone(
                C.r3_kproxy_dispatch(_req(4608, 4608, 288), dispatch_gemm_fp16)
            )

    def test_explanation_names_rounded_k(self):
        res = _disp(_req(4104, 4104, 4104))
        self.assertIn("K=4160", _kp_line(res)[0])

    def _fake(self, spec):
        from types import SimpleNamespace

        def fn(_r):
            return SimpleNamespace(spec=spec, candidate=None, explanation=())

        return fn

    def test_arch_gate_without_validity(self):
        # the arch gate alone keeps gfx942 out (validity mocked open)
        ring = F16._spec_cdna_dtl_ring_small(_req(4104, 4104, 4160), "t")
        with mock.patch.object(GU, "is_valid_spec", lambda *a, **k: (True, "ok")):
            self.assertIsNone(
                C.r3_kproxy_dispatch(
                    _req(4104, 4104, 4104, arch="gfx942"), self._fake(ring)
                )
            )
            self.assertIsNotNone(
                C.r3_kproxy_dispatch(_req(4104, 4104, 4104), self._fake(ring))
            )

    def test_tail_and_validity_checks(self):
        ring = F16._spec_cdna_dtl_ring_small(_req(4104, 4104, 4160), "t")
        req = _req(4104, 4104, 4104)
        # an adaptation that does not add the K tail: refused
        with mock.patch.object(C, "g1_adapt_spec", lambda r, s: s):
            self.assertIsNone(C.r3_kproxy_dispatch(req, self._fake(ring)))
        # an adapted spec that is not valid at the real K: refused
        with mock.patch.object(GU, "is_valid_spec", lambda *a, **k: (False, "x")):
            self.assertIsNone(C.r3_kproxy_dispatch(req, self._fake(ring)))
        # a non direct-to-LDS universal pick: refused even when the tail is forced
        nodtl = replace(ring, trait=replace(ring.trait, direct_to_lds=False))
        forced = lambda r, s: replace(
            s, trait=replace(s.trait, dtl_k_tail=2, pad_k=True)
        )  # noqa: E731
        with mock.patch.object(C, "g1_adapt_spec", forced), mock.patch.object(
            GU, "is_valid_spec", lambda *a, **k: (True, "ok")
        ), mock.patch.object(GU, "launch_shape_error", lambda *a, **k: None):
            self.assertIsNone(C.r3_kproxy_dispatch(req, self._fake(nodtl)))
            self.assertIsNotNone(C.r3_kproxy_dispatch(req, self._fake(ring)))
            # K % 8 != 0 stays out (the K tail needs 8-element K chunks)
            self.assertIsNone(
                C.r3_kproxy_dispatch(_req(4104, 4104, 4100), self._fake(ring))
            )

    def test_k_tail_min_tiles(self):
        # K = 200: 4 K tiles of 64 with a tail -> adapted; K = 136 (3 tiles) -> the tail rule refuses, no proxy
        self.assertIsNone(
            C.r3_kproxy_dispatch(_req(4096, 4096, 136), dispatch_gemm_fp16)
        )


class TestQGeo(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {"ROCKE_GEMM_SELECTOR": "0"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def _ring(self, req):
        return C.g1_adapt_spec(
            req,
            F16._spec_cdna_dtl_ring_small(replace(req, K=-(-req.K // 64) * 64), "t"),
        )

    def test_switch_k_tail(self):
        req = _req(7176, 1440, 4104)
        res = _disp(req)
        sp = res.spec
        self.assertEqual(
            (sp.tile.tile_m, sp.tile.tile_n, sp.tile.tile_k), (256, 192, 64)
        )
        self.assertEqual((sp.tile.warp_m, sp.tile.warp_n), (2, 2))
        self.assertEqual(sp.block_size, 256)
        self.assertEqual(res.block[0], 256)
        self.assertFalse(sp.trait.acc_agpr)
        self.assertTrue(sp.trait.mfma_interleave and sp.trait.dtl_frag_prefetch)
        self.assertEqual(sp.trait.dtl_stages, 2)
        self.assertEqual(sp.trait.dtl_k_tail, 2)
        self.assertTrue(is_valid_spec(sp, arch="gfx950")[0])
        self.assertIsNone(launch_shape_error(sp, 7176, 1440, 4104))
        self.assertTrue([e for e in res.explanation if e.startswith("r3 qgeo")])
        with mock.patch.object(C, "R3_QGEO", False):
            sp0 = _disp(req).spec
        self.assertEqual((sp0.tile.tile_m, sp0.tile.tile_n), (128, 128))

    def test_orientation_and_ratio(self):
        # the geometry with the smaller quantised work; none when it exceeds the small ring's
        req = _req(1440, 7176, 4104)
        q = C.r3_qgeo_spec(req, self._ring(req))
        self.assertIsNotNone(q)
        self.assertIn((q.tile.tile_m, q.tile.tile_n), ((256, 192), (192, 256)))
        # M = 10 x 192: 192x256 covers 1920 x 6400 in one wave (250 tiles), 256x192 needs two (272)
        req2 = _req(1920, 6400, 4104)
        q2 = C.r3_qgeo_spec(req2, self._ring(req2))
        self.assertIsNotNone(q2)
        self.assertEqual((q2.tile.tile_m, q2.tile.tile_n), (192, 256))
        few = _req(
            1024, 1024, 4104
        )  # 64 small-ring tiles: one partial wave either way -> keep
        self.assertIsNone(C.r3_qgeo_spec(few, self._ring(few)))
        with mock.patch.object(C, "R3_QGEO_MIN_RATIO", 10.0):
            self.assertIsNone(C.r3_qgeo_spec(req, self._ring(req)))

    def test_aligned_knob(self):
        req = _req(7176, 1440, 4096)
        ring = self._ring(req)
        self.assertIsNone(C.r3_qgeo_spec(req, ring))
        with mock.patch.object(C, "R3_QGEO_KTAIL_ONLY", False):
            q = C.r3_qgeo_spec(req, ring)
        self.assertIsNotNone(q)
        self.assertEqual(q.trait.dtl_k_tail, 0)
        self.assertFalse(q.trait.pad_k)

    def test_min_k(self):
        req = _req(7176, 1440, 488)
        ring = C.g1_adapt_spec(req, self._ring(_req(7176, 1440, 4104)))
        self.assertIsNone(C.r3_qgeo_spec(req, ring))
        with mock.patch.object(C, "R3_QGEO_MIN_K", 0):
            self.assertIsNotNone(C.r3_qgeo_spec(req, ring))

    def test_small_ring_two_per_cu(self):
        # 33 x 33 small-ring tiles = 3 waves at two workgroups per CU vs 2 waves of the 192 ring: switch (at one
        # small-ring workgroup per CU it would not)
        req = _req(4224, 4224, 4104)
        self.assertIsNotNone(C.r3_qgeo_spec(req, self._ring(req)))

    def test_ring_warp_layout(self):
        req = _req(7176, 1440, 4104)
        ring = self._ring(req)
        other = replace(ring, tile=replace(ring.tile, warp_n=2), block_size=0)
        self.assertTrue(is_valid_spec(other, arch="gfx950")[0])
        self.assertIsNone(C.r3_qgeo_spec(req, other))

    def test_arch_gate_without_validity(self):
        req = _req(7176, 1440, 4104)
        ring = self._ring(req)
        with mock.patch.object(GU, "is_valid_spec", lambda *a, **k: (True, "ok")):
            self.assertIsNone(
                C.r3_qgeo_spec(_req(7176, 1440, 4104, arch="gfx942"), ring)
            )
            self.assertIsNotNone(C.r3_qgeo_spec(req, ring))

    def test_agpr_knob(self):
        req = _req(7176, 1440, 4104)
        self.assertFalse(C.r3_qgeo_spec(req, self._ring(req)).trait.acc_agpr)
        with mock.patch.object(C, "R3_QGEO_AGPR", True):
            q = C.r3_qgeo_spec(req, self._ring(req))
        self.assertTrue(q.trait.acc_agpr)

    def test_not_applied(self):
        req = _req(7176, 1440, 4104)
        ring = self._ring(req)
        beat = F16._spec_cdna_dtl_pingpong(_req(8192, 8192, 8192), "t")
        self.assertIsNone(C.r3_qgeo_spec(req, beat))
        for kv in (dict(split_k=2), dict(persistent_ctas=256)):
            self.assertIsNone(
                C.r3_qgeo_spec(req, replace(ring, trait=replace(ring.trait, **kv)))
            )
        self.assertIsNone(C.r3_qgeo_spec(_req(7176, 1440, 4104, arch="gfx942"), ring))
        self.assertIsNone(C.r3_qgeo_spec(_req(7176, 1440, 4104, lay="RRR"), ring))
        with mock.patch.object(C, "R3_QGEO", False):
            self.assertIsNone(C.r3_qgeo_spec(req, ring))


class TestStreamKLevers(unittest.TestCase):
    def _beat(self):
        return F16._spec_cdna_dtl_pingpong(_req(8192, 8192, 8192), "t")

    def _ring(self):
        return F16._spec_cdna_dtl_ring_small(_req(4096, 4096, 8192), "t")

    def test_streamk_kloop_unroll(self):
        b = self._beat()
        self.assertEqual(b.trait.kloop_unroll, 4)
        sk = replace(b, trait=replace(b.trait, persistent_ctas=256, streamk=True))
        self.assertTrue(is_valid_spec(sk, arch="gfx950")[0])
        with mock.patch.object(GU, "STREAMK_KLOOP_UNROLL", False):
            ok, why = is_valid_spec(sk, arch="gfx950")
            self.assertFalse(ok)
            self.assertIn("kloop_unroll", why)
            sk1 = replace(sk, trait=replace(sk.trait, kloop_unroll=1))
            self.assertTrue(is_valid_spec(sk1, arch="gfx950")[0])

    def test_residency_waves_per_eu(self):
        r = self._ring()
        self.assertEqual(r.tile.warp_m * r.tile.warp_n, 8)
        sk = replace(r, trait=replace(r.trait, persistent_ctas=512, streamk=True))
        ok, why = is_valid_spec(sk, arch="gfx950")
        self.assertFalse(ok)
        self.assertIn("co-resident", why)
        w4 = replace(sk, trait=replace(sk.trait, waves_per_eu=4))
        self.assertTrue(is_valid_spec(w4, arch="gfx950")[0])
        # waves_per_eu 2 bounds 1 workgroup of 8 waves per CU: 512 still refused
        w2 = replace(sk, trait=replace(sk.trait, waves_per_eu=2))
        self.assertFalse(is_valid_spec(w2, arch="gfx950")[0])
        # the pool at the bound (256 x 2) passes, one more refused
        w4b = replace(w4, trait=replace(w4.trait, persistent_ctas=520))
        self.assertFalse(is_valid_spec(w4b, arch="gfx950")[0])
        with mock.patch.object(GU, "STREAMK_RESIDENCY_WPE", False):
            self.assertFalse(is_valid_spec(w4, arch="gfx950")[0])

    def test_resident_ctas_bound(self):
        r = self._ring()
        sk = replace(r, trait=replace(r.trait, persistent_ctas=256, streamk=True))
        self.assertEqual(GU.streamk_resident_ctas(sk, "gfx950", 65536)[1], 1)
        w4 = replace(sk, trait=replace(sk.trait, waves_per_eu=4))
        self.assertEqual(GU.streamk_resident_ctas(w4, "gfx950", 65536)[1], 2)
        # the LDS bound still applies
        self.assertEqual(GU.streamk_resident_ctas(w4, "gfx950", 100000)[1], 1)
        wt = replace(sk, trait=replace(sk.trait, waves_per_eu=(4, 8)))
        self.assertEqual(GU.streamk_resident_ctas(wt, "gfx950", 65536)[1], 2)
        # the tuple's minimum is the guarantee (small LDS: the register bound decides)
        self.assertEqual(GU.streamk_resident_ctas(wt, "gfx950", 20000)[1], 2)


if __name__ == "__main__":
    unittest.main()
