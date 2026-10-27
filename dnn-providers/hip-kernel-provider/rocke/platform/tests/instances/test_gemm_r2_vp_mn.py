# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the MN-operand levers of the gfx942 register-staged
body (``vgpr_prefetch``):

* ``vp_mn_kv`` (+ ``vp_mn_order`` / ``vp_mn_worder`` / ``vp_mn_pipe`` /
  ``vp_mn_wrap``): MN-contiguous operands (NN / TN / TT) read as micro-blocks
  and transposed in registers into the K-contiguous LDS tile;
* ``vp_late_reads`` / ``vp_unroll``: the K-loop forms without back-edge copies
  of the prefetch registers;
* the dispatcher's native form (``common.r2_vp_mn_form`` / ``R2_VP_MN``) and the
  plan chooser (``plan.gemm_plan``).

The index math the kernel emits is shared with these tests through
``gemm_mn_operand.IntOps`` (``vp_mn_micro``, ``vp_mn_col``).
"""

from __future__ import annotations

import itertools
import random
import re
import unittest
from dataclasses import replace

from rocke.core import backend
from rocke.instances.common import gemm_mn_operand as mno
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    launch_shape_error,
)

ARCH = "gfx942"


def _g5(geo="256x256x64", layout="RCR", dt="fp16", pads="", **trait):
    from rocke.dispatch.gemm.common import _g5_tile_trait

    tile, tr = _g5_tile_trait(geo, "m" in pads, "n" in pads)
    if "k" in pads:
        tr = replace(tr, pad_k=True)
    tr = replace(tr, **trait)
    return UniversalGemmSpec(
        name="r2t",
        tile=tile,
        trait=tr,
        data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt, layout=layout),
    )


def _ir(spec):
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    return lower_kernel_to_llvm(
        build_universal_gemm(spec, arch=ARCH), arch=ARCH, llvm_flavor="llvm22"
    )


def _req(M, N, K, layout="RCR", dt="fp16"):
    from rocke.dispatch.gemm import GemmRequest

    return GemmRequest(
        M=M,
        N=N,
        K=K,
        arch=ARCH,
        dtype=dt,
        layout=layout,
        trans_a=layout[0] == "C",
        trans_b=layout[1] == "C",
    )


class TestValidity(unittest.TestCase):
    def test_admitted(self):
        for lay in ("RRR", "CRR", "CCR"):
            for kv, mv in ((8, 8), (8, 4), (4, 8), (4, 4), (2, 2)):
                for order in (0, 1, 2, 3):
                    s = _g5(layout=lay, vp_mn_kv=kv, mn_vec=mv, vp_mn_order=order)
                    with self.subTest(lay=lay, kv=kv, mv=mv, order=order):
                        self.assertEqual(is_valid_spec(s, arch=ARCH), (True, "ok"))
                        self.assertIn(f"_vmk{kv}", s.kernel_name())
                        self.assertIn(f"_L{lay}", s.kernel_name())

    def test_refused(self):
        bad = {
            "kv 3": dict(layout="RRR", vp_mn_kv=3),
            "kv 16": dict(layout="RRR", vp_mn_kv=16, vp_mn_wrap=1, geo="128x128x128"),
            "mn_vec 1": dict(layout="RRR", vp_mn_kv=8, mn_vec=1),
            "order 4": dict(layout="RRR", vp_mn_kv=8, vp_mn_order=4),
            "worder 4": dict(layout="RRR", vp_mn_kv=8, vp_mn_worder=4),
            "pipe 2": dict(layout="RRR", vp_mn_kv=8, vp_mn_pipe=2),
            "wrap 2": dict(layout="RRR", vp_mn_kv=8, vp_mn_wrap=2),
            "MN without vp_mn_kv": dict(layout="RRR"),
            "vp_mn_kv on RCR": dict(vp_mn_kv=8),
            "order on RCR": dict(vp_mn_order=2),
            "pipe on RCR": dict(vp_mn_pipe=1),
            "wrap on RCR": dict(vp_mn_wrap=1),
            "vp_mn_kv without vgpr_prefetch": dict(
                layout="RRR", vp_mn_kv=8, vgpr_prefetch=0
            ),
            # 224 x 64 / 64 = 224 micro-blocks for 256 threads
            "partial round": dict(geo="256x224x64", layout="RRR", vp_mn_kv=8),
            "late reads mode 3": dict(
                geo="256x256x32", vp_late_reads=1, vgpr_prefetch=3
            ),
            "late reads depth 2": dict(vp_late_reads=1, vgpr_prefetch_depth=2),
            "unroll 2": dict(vp_unroll=2),
        }
        for nm, kw in bad.items():
            geo = kw.pop("geo", "256x256x64")
            with self.subTest(nm):
                self.assertFalse(is_valid_spec(_g5(geo=geo, **kw), arch=ARCH)[0])

    def test_wrap_admits_partial_rounds(self):
        s = _g5("256x224x64", "RRR", vp_mn_kv=8, vp_mn_wrap=1)
        self.assertEqual(is_valid_spec(s, arch=ARCH), (True, "ok"))
        self.assertIn("_vmx1", s.kernel_name())

    def test_order_group_rule(self):
        self.assertEqual(mno.vp_mn_order_group(2), 4)
        self.assertEqual(mno.vp_mn_order_group(3), 2)
        self.assertEqual(mno.vp_mn_order_group(0), 1)
        # B tile 32 x 128: 32 / 8 = 4 M / N blocks (one group of 4), 64
        # micro-blocks for 256 threads -> only with vp_mn_wrap
        s = _g5("128x32x128", "RRR", vp_mn_kv=8, mn_vec=8, vp_mn_order=2)
        self.assertFalse(is_valid_spec(s, arch=ARCH)[0])
        s = replace(s, trait=replace(s.trait, vp_mn_wrap=1))
        self.assertEqual(is_valid_spec(s, arch=ARCH), (True, "ok"))
        for order in (2, 3):
            s = _g5("128x32x128", "RRR", vp_mn_kv=2, mn_vec=2, vp_mn_order=order)
            self.assertEqual(is_valid_spec(s, arch=ARCH), (True, "ok"))
        # a 16-wide N-contiguous B with mn_vec 8: 2 M / N blocks, not a whole
        # group of 4 (order 2), a whole group of 2 (order 3)
        base = _g5(
            "128x32x128",
            "RRR",
            vp_mn_kv=8,
            mn_vec=8,
            vp_mn_wrap=1,
            lds_swizzle=False,
            lds_swizzle_r=-1,
        )
        t16 = replace(base.tile, tile_n=16, warp_n=1, warp_m=4)
        s16 = replace(base, tile=t16, block_size=0)
        self.assertFalse(
            is_valid_spec(
                replace(s16, trait=replace(s16.trait, vp_mn_order=2)), arch=ARCH
            )[0]
        )
        ok3 = is_valid_spec(
            replace(s16, trait=replace(s16.trait, vp_mn_order=3)), arch=ARCH
        )
        self.assertEqual(ok3, (True, "ok"))
        self.assertEqual(
            is_valid_spec(
                replace(s16, trait=replace(s16.trait, vp_mn_order=0)), arch=ARCH
            ),
            (True, "ok"),
        )

    def test_default_names_unchanged(self):
        s = _g5()
        for flag in ("_vmk", "_vmo", "_vmw", "_vmp", "_vmx", "_vpu", "_vpl"):
            self.assertNotIn(flag, s.kernel_name())

    def test_python_only(self):
        # each lever alone (the converter refuses every one of them)
        for kw in (
            dict(vp_mn_kv=8),
            dict(vp_mn_order=2),
            dict(vp_mn_worder=1),
            dict(vp_mn_pipe=1),
            dict(vp_mn_wrap=1),
            dict(vp_late_reads=1),
            dict(vp_unroll=1),
        ):
            with self.subTest(kw):
                with self.assertRaises(NotImplementedError):
                    backend._refuse_python_only_levers(TraitSpec(**kw))
        backend._refuse_python_only_levers(TraitSpec())


class TestLaunch(unittest.TestCase):
    def test_leading_dimension(self):
        s = _g5(layout="CRR", vp_mn_kv=8, mn_vec=8, pads="mnk")
        self.assertIsNone(launch_shape_error(s, 4096, 4096, 4096))
        self.assertIsNotNone(launch_shape_error(s, 4100, 4096, 4096))
        self.assertIsNotNone(launch_shape_error(s, 4096, 4100, 4096))
        s4 = replace(s, trait=replace(s.trait, mn_vec=4))
        self.assertIsNone(launch_shape_error(s4, 4100, 4100, 4096))

    def test_buffer_range_with_pad_k(self):
        # masked rows: soffset (K * ld * 2) must stay below 2^31
        s = _g5(layout="CCR", vp_mn_kv=8, pads="k")
        self.assertIsNone(launch_shape_error(s, 16384, 256, 65528))
        self.assertIsNotNone(launch_shape_error(s, 16384, 256, 65544 + 8))
        r = _g5(pads="k")
        self.assertIsNone(launch_shape_error(r, 16384, 256, 65544 + 8))

    def test_tn_pad_k_any_k(self):
        # both operands MN-contiguous: the K tail needs no K-vector alignment
        s = _g5(layout="CRR", vp_mn_kv=8, pads="k")
        self.assertIsNone(launch_shape_error(s, 4096, 4096, 4099))
        t = _g5(layout="CCR", vp_mn_kv=8, pads="k")
        self.assertIsNotNone(launch_shape_error(t, 4096, 4096, 4099))


class TestIndexMath(unittest.TestCase):
    def test_micro_block_coverage(self):
        ops = mno.IntOps()
        for tile_mn, tile_k, kv, mv, threads in (
            (256, 64, 8, 8, 256),
            (256, 64, 8, 4, 512),
            (224, 64, 8, 8, 256),
            (192, 64, 4, 8, 512),
            (64, 128, 8, 4, 256),
            (32, 128, 2, 2, 256),
        ):
            n = (tile_mn // mv) * (tile_k // kv)
            for order in (0, 1, 2, 3):
                if (tile_mn // mv) % mno.vp_mn_order_group(order):
                    continue
                seen = {}
                rounds = mno.vp_mn_rounds(tile_mn, tile_k, kv, mv, threads)
                for u in range(rounds):
                    for tid in range(threads):
                        k0, mn0 = mno.vp_mn_micro(
                            ops,
                            u,
                            tid,
                            threads=threads,
                            tile_mn=tile_mn,
                            tile_k=tile_k,
                            kv=kv,
                            mv=mv,
                            order=order,
                        )
                        self.assertEqual(k0 % kv, 0)
                        self.assertEqual(mn0 % mv, 0)
                        self.assertLess(k0, tile_k)
                        self.assertLess(mn0, tile_mn)
                        seen[(k0, mn0)] = seen.get((k0, mn0), 0) + 1
                with self.subTest(tile_mn=tile_mn, kv=kv, mv=mv, order=order):
                    self.assertEqual(len(seen), n)  # every micro-block
                    extra = rounds * threads - n
                    self.assertEqual(sum(seen.values()) - n, extra)

    def test_quad_coalescing_order(self):
        # order 2: 4 consecutive lanes read 4 consecutive M / N blocks of one K row
        ops = mno.IntOps()
        for tid in range(0, 64, 4):
            kk = set()
            mns = []
            for t in range(tid, tid + 4):
                k0, mn0 = mno.vp_mn_micro(
                    ops, 0, t, threads=256, tile_mn=256, tile_k=64, kv=8, mv=8, order=2
                )
                kk.add(k0)
                mns.append(mn0)
            self.assertEqual(len(kk), 1)
            self.assertEqual(mns, [mns[0] + 8 * j for j in range(4)])

    def test_swizzled_column_split(self):
        ops = mno.IntOps()
        rng = random.Random(3)
        n_checked = 0
        for kv, mv in itertools.product((2, 4, 8), (2, 4, 8)):
            for r, w, l_ in itertools.product((0, 1, 2), (1, 2, 3, 4), (1, 2)):
                swz = (r, w, l_)
                if not mno.vp_mn_fast_col_ok(kv, mv, swz):
                    continue
                for _ in range(20):
                    k0 = rng.randrange(0, 128 // kv) * kv
                    mn0 = rng.randrange(0, 256 // mv) * mv
                    cache = {}
                    for i in range(mv):
                        for j0 in range(kv):
                            got = mno.vp_mn_col(
                                ops, k0, mn0, i, j0, kv=kv, mv=mv, swz=swz, cache=cache
                            )
                            want = mno.swizzle_col(ops, k0 + j0, mn0 + i, swz)
                            self.assertEqual(got, want, (kv, mv, swz, k0, mn0, i, j0))
                            n_checked += 1
        self.assertGreater(n_checked, 10000)
        # no swizzle: plain column
        self.assertEqual(
            mno.vp_mn_col(ops, 16, 8, 3, 5, kv=8, mv=8, swz=None, cache={}), 21
        )

    def test_fast_col_condition(self):
        self.assertTrue(mno.vp_mn_fast_col_ok(8, 8, (0, 4, 2)))
        self.assertTrue(mno.vp_mn_fast_col_ok(8, 8, (1, 4, 2)))
        # R > log2(mn_vec): the key's i bits are not static
        self.assertFalse(mno.vp_mn_fast_col_ok(8, 2, (2, 4, 2)))
        # L + log2(mn_vec) - R < log2(kv): D overlaps the immediate bits
        self.assertFalse(mno.vp_mn_fast_col_ok(8, 2, (1, 4, 2)))


class TestEmission(unittest.TestCase):
    def _loads(self, ir):
        return len(re.findall(r"llvm\.amdgcn\.raw\.ptr\.buffer\.load", ir))

    def test_native_forms_build(self):
        for lay in ("RRR", "CRR", "CCR"):
            for kw in (
                dict(vp_mn_kv=8),
                dict(vp_mn_kv=8, vp_mn_pipe=1, vp_late_reads=1),
                dict(vp_mn_kv=4, mn_vec=4, vp_mn_order=2, vp_mn_worder=3),
            ):
                s = _g5(layout=lay, pads="mnk", **kw)
                with self.subTest(lay=lay, kw=kw):
                    ir = _ir(s)
                    self.assertIn("buffer.load", ir)

    def test_same_global_read_count(self):
        # a micro-block of kv x mn_vec reads the same bytes as the RCR vectors
        rcr = self._loads(_ir(_g5()))
        for lay in ("RRR", "CRR", "CCR"):
            n = self._loads(_ir(_g5(layout=lay, vp_mn_kv=8)))
            self.assertEqual(n, rcr, lay)

    def test_late_reads_change_emission_only_when_set(self):
        self.assertNotEqual(_ir(_g5()), _ir(_g5(vp_late_reads=1)))
        self.assertNotEqual(_ir(_g5()), _ir(_g5(vp_unroll=1)))


class TestDispatch(unittest.TestCase):
    def setUp(self):
        from rocke.dispatch.gemm import common as C

        self.C = C
        self.saved = (C.R2_VP_MN, C.R2_VP_MN_WRAP)
        # these tests pin the native machinery without the native-form model;
        # the model (g5_native, tested in test_gemm_r3_g5_native.py) takes
        # precedence when it is on
        from unittest import mock as _mock

        from rocke.dispatch.gemm import g5_native as _gn

        _p = _mock.patch.object(_gn, "G5N_ON", False)
        _p.start()
        self.addCleanup(_p.stop)

    def tearDown(self):
        self.C.R2_VP_MN, self.C.R2_VP_MN_WRAP = self.saved

    def _pick(self, M, N, K, lay, dt="fp16"):
        from rocke.dispatch.gemm import dispatch_gemm_bf16, dispatch_gemm_fp16

        f = dispatch_gemm_fp16 if dt == "fp16" else dispatch_gemm_bf16
        return f(_req(M, N, K, lay, dt))

    def test_native_g5_pick(self):
        for lay in ("RRR", "CRR", "CCR"):
            for dt in ("fp16", "bf16"):
                r = self._pick(4096, 4096, 4096, lay, dt)
                with self.subTest(lay=lay, dt=dt):
                    self.assertTrue(r.candidate.name.endswith("cdna_vgpr"))
                    self.assertEqual(r.spec.data.layout, lay)
                    self.assertGreater(r.spec.trait.vp_mn_kv, 0)
                    self.assertTrue(is_valid_spec(r.spec, arch=ARCH)[0])
                    self.assertIsNone(launch_shape_error(r.spec, 4096, 4096, 4096))

    def test_ragged_leading_dimension(self):
        # N = 4100: mn_vec 8 does not divide B's leading dimension (NN): the
        # form takes a narrower vector and launches
        r = self._pick(4096, 4100, 4096, "RRR")
        self.assertTrue(r.candidate.name.endswith("cdna_vgpr"))
        self.assertEqual(4100 % r.spec.trait.mn_vec, 0)
        self.assertIsNone(launch_shape_error(r.spec, 4096, 4100, 4096))

    def test_knob_off_native_alternatives(self):
        from rocke.dispatch.gemm.common import (
            _g5_plan_tile_trait,
            native_layout_alternatives,
        )

        req = _req(4096, 4096, 4096, "CCR")
        tile, tr = _g5_plan_tile_trait(req, "256x256x64", 1)
        s = UniversalGemmSpec(
            name="p", tile=tile, trait=tr, data=DataSpec("fp16", "fp16", "fp16")
        )
        alts, _ = native_layout_alternatives(req, s)
        self.assertEqual([n for n, _ in alts], ["vp_mn"])
        self.C.R2_VP_MN = False
        alts, _ = native_layout_alternatives(req, s)
        self.assertNotIn("vp_mn", [n for n, _ in alts])

    def test_odd_k_only_tn(self):
        # NN / TT keep a K-contiguous operand (K % 8 for its vector loads)
        for lay in ("RRR", "CCR"):
            r = self._pick(4096, 4096, 4099, lay)
            self.assertFalse(r.candidate.name.endswith("cdna_vgpr"), lay)

    def test_knob_off_restores_round1(self):
        self.C.R2_VP_MN = False
        self.C._R2_PLAN_CACHE.clear()
        for lay in ("RRR", "CRR", "CCR"):
            r = self._pick(4096, 4096, 4096, lay)
            self.assertFalse(r.candidate.name.endswith("cdna_vgpr"), lay)
        self.assertTrue(
            self._pick(4096, 4096, 4096, "RCR").candidate.name.endswith("cdna_vgpr")
        )

    def test_odd_leading_dimension_falls_back(self):
        # an MN-contiguous operand with an odd leading dimension has no vector
        # micro-block: another candidate serves it (no refusal)
        r = self._pick(347, 240, 512, "CRR")
        self.assertFalse(r.candidate.name.endswith("cdna_vgpr"))

    def test_wrap_knob(self):
        from rocke.dispatch.gemm.common import _g5_plan_tile_trait, r2_vp_mn_form

        C = self.C
        saved = (C.R2_VP_MN_USE_FIT, C.R2_VP_MN_USE_TABLE)
        try:
            C.R2_VP_MN_USE_FIT, C.R2_VP_MN_USE_TABLE = False, False
            req = _req(4096, 4096, 4096, "RRR")
            tile, tr = _g5_plan_tile_trait(req, "256x224x64", 1)
            s = UniversalGemmSpec(
                name="p", tile=tile, trait=tr, data=DataSpec("fp16", "fp16", "fp16")
            )
            f = r2_vp_mn_form(req, s)
            # generic order: a whole-round micro-block before any wrapped one
            self.assertEqual(f.trait.vp_mn_wrap, 0)
            C.R2_VP_MN_WRAP = False
            req2 = _req(4096, 4096, 4096, "CRR")
            tile, tr = _g5_plan_tile_trait(req2, "256x256x64w8", 1)
            s2 = UniversalGemmSpec(
                name="p", tile=tile, trait=tr, data=DataSpec("fp16", "fp16", "fp16")
            )
            f2 = r2_vp_mn_form(req2, s2)
            self.assertIsNotNone(f2)
            self.assertEqual(f2.trait.vp_mn_wrap, 0)
        finally:
            C.R2_VP_MN_USE_FIT, C.R2_VP_MN_USE_TABLE = saved

    def test_fitted_micro_block(self):
        from rocke.dispatch.gemm.common import (
            R2_NATIVE_LIN,
            _g5_plan_tile_trait,
            r2_vp_mn_form,
        )

        req = _req(4096, 4096, 4096, "CCR")
        tile, tr = _g5_plan_tile_trait(req, "256x256x64", 1)
        s = UniversalGemmSpec(
            name="p", tile=tile, trait=tr, data=DataSpec("fp16", "fp16", "fp16")
        )
        self.C.R2_VP_MN_USE_TUNED = False
        try:
            f = r2_vp_mn_form(req, s)
        finally:
            self.C.R2_VP_MN_USE_TUNED = True
        fits = [
            (ab[0] + ab[1] * 4096, mic)
            for (L_, g_, mic), ab in R2_NATIVE_LIN.items()
            if L_ == "CCR" and g_ == "256x256x64"
        ]
        self.assertEqual((f.trait.vp_mn_kv, f.trait.mn_vec), min(fits)[1])

    def test_tuned_form(self):
        from rocke.dispatch.gemm.common import (
            R2_VP_MN_TUNED,
            _g5_plan_tile_trait,
            r2_vp_mn_form,
        )

        mic, trait, factor = R2_VP_MN_TUNED[("CCR", "64x64x128")]
        self.assertLess(factor, 1.0)
        req = _req(960, 832, 1472, "CCR")
        tile, tr = _g5_plan_tile_trait(req, "64x64x128", 1)
        s = UniversalGemmSpec(
            name="p", tile=tile, trait=tr, data=DataSpec("fp16", "fp16", "fp16")
        )
        f = r2_vp_mn_form(req, s)
        self.assertEqual((f.trait.vp_mn_kv, f.trait.mn_vec), mic)
        for k, v in trait.items():
            self.assertEqual(getattr(f.trait, k), v, k)
        self.assertTrue(is_valid_spec(f, arch=ARCH)[0])
        # a leading dimension the tuned micro-block does not fit: the tuned
        # traits with a narrower vector
        req2 = _req(964, 832, 1472, "CCR")
        tile, tr = _g5_plan_tile_trait(req2, "64x64x128", 1)
        s2 = UniversalGemmSpec(
            name="p", tile=tile, trait=tr, data=DataSpec("fp16", "fp16", "fp16")
        )
        f2 = r2_vp_mn_form(req2, s2)
        self.assertEqual(964 % f2.trait.mn_vec, 0)
        self.assertEqual(f2.trait.vp_mn_order, trait["vp_mn_order"])
        self.assertIsNone(launch_shape_error(f2, 964, 832, 1472))
        # knob off: the untuned form
        self.C.R2_VP_MN_USE_TUNED = False
        try:
            f3 = r2_vp_mn_form(req, s)
        finally:
            self.C.R2_VP_MN_USE_TUNED = True
        self.assertEqual(f3.trait.vgpr_prefetch, 4)

    def test_tn_any_k(self):
        r = self._pick(4096, 4096, 4099, "CRR")
        self.assertTrue(r.candidate.name.endswith("cdna_vgpr"))
        self.assertTrue(r.spec.trait.pad_k)
        self.assertIsNone(launch_shape_error(r.spec, 4096, 4096, 4099))
        self.C.R2_VP_MN_TN_ANY_K = False
        self.C._R2_PLAN_CACHE.clear()
        try:
            r = self._pick(4096, 4096, 4099, "CRR")
            self.assertFalse(r.candidate.name.endswith("cdna_vgpr"))
        finally:
            self.C.R2_VP_MN_TN_ANY_K = True

    def test_sweep_space_native(self):
        from rocke.dispatch.gemm import GEMM_FP16_REGISTRY

        c = GEMM_FP16_REGISTRY.get("universal_gemm_fp16_cdna_vgpr")
        sp = c.sweep_space(_req(4096, 4096, 4096, "CCR"))
        self.assertTrue(sp)
        for s in sp:
            self.assertEqual(s.data.layout, "CCR")
            self.assertTrue(is_valid_spec(s, arch=ARCH)[0])


class TestPlan(unittest.TestCase):
    def setUp(self):
        from rocke.dispatch.gemm import plan as P

        self.P = P
        self.saved = (P.R2_PLAN, P.R2_PLAN_MARGIN)
        # these tests pin the native machinery without the native-form model;
        # the model (g5_native, tested in test_gemm_r3_g5_native.py) takes
        # precedence when it is on
        from unittest import mock as _mock

        from rocke.dispatch.gemm import g5_native as _gn

        _p = _mock.patch.object(_gn, "G5N_ON", False)
        _p.start()
        self.addCleanup(_p.stop)
        # gemm_skinny's native forms (skinny.SKINNY_NATIVE) take
        # several of these shapes ahead of the slow fallback; the chooser's
        # logic is tested on the register-staged / fallback picks
        from unittest import mock

        from rocke.dispatch.gemm import skinny as SK

        p = mock.patch.object(SK, "SKINNY_NATIVE", False)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        self.P.R2_PLAN, self.P.R2_PLAN_MARGIN = self.saved

    def test_rcr_native(self):
        p = self.P.gemm_plan(_req(4096, 4096, 4096))
        self.assertEqual(p.kind, "native")
        self.assertEqual(p.transposes, ())

    def test_native_for_aligned(self):
        p = self.P.gemm_plan(_req(2048, 2048, 2048, "CCR"))
        self.assertEqual(p.kind, "native")

    def test_prepass_for_narrow_n(self):
        # 22912 x 192 x 2880 NN (pre-pass preferred): the
        # transposed B is small and the RCR body beats the native form by more
        # than the margin
        p = self.P.gemm_plan(_req(22912, 192, 2880, "RRR", "bf16"))
        self.assertEqual(p.kind, "prepass")
        self.assertEqual(p.request.layout, "RCR")
        self.assertEqual([t.operand for t in p.transposes], ["B"])
        self.assertEqual((p.transposes[0].rows, p.transposes[0].cols), (2880, 192))
        self.assertEqual(p.transposes[0].spec.tile_m, 64)

    def test_slow_fallback_prepass(self):
        # an odd MN leading dimension: the native pick is the slow fallback
        p = self.P.gemm_plan(_req(511, 361, 472, "CRR"))
        self.assertEqual(p.kind, "prepass")
        self.assertIn("vgpr", p.dispatch.candidate.name)

    def test_prepass_estimate_uses_copy_rate(self):
        # a side that is not a multiple of 64: the estimate adds the copy's
        # (slow) transpose time to the RCR GEMM
        req = _req(4096, 4100, 4096, "RRR")
        p = self.P.gemm_plan(req)
        rreq = replace(req, layout="RCR", trans_a=False, trans_b=True)
        est_r = self.P._rcr_est(rreq, self.P._dispatch(rreq))
        est_p = p.alt_est_us if p.kind == "native" else p.est_us
        self.assertAlmostEqual(
            est_p, est_r + self.P.r2_transpose_cost_us(4096, 4100, fast=False), places=6
        )

    def test_slow_transpose_model(self):
        # a side that is not a multiple of 64: the caller's copy rate
        self.assertGreater(
            self.P.r2_transpose_cost_us(512, 347, fast=False),
            4 * self.P.r2_transpose_cost_us(512, 347),
        )
        p = self.P.gemm_plan(_req(347, 240, 512, "CCR"))
        self.assertIsNotNone(p.alt_est_us)

    def test_tn_partial_prepass(self):
        # TN with an odd M: transposing A alone gives the NN native form
        p = self.P.gemm_plan(_req(1149, 4396, 3304, "CRR"))
        self.assertEqual(p.kind, "prepass_a")
        self.assertEqual(p.request.layout, "RRR")
        self.assertEqual(
            [(t.operand, t.rows, t.cols) for t in p.transposes], [("A", 3304, 1149)]
        )
        self.assertIn("vgpr", p.dispatch.candidate.name)
        self.assertEqual(p.dispatch.spec.data.layout, "RRR")
        # TN with an odd N: transposing B alone gives the TT native form
        p = self.P.gemm_plan(_req(2820, 195, 2872, "CRR"))
        self.assertEqual(p.kind, "prepass_b")
        self.assertEqual(p.request.layout, "CCR")
        self.P.R2_PLAN_PARTIAL = False
        try:
            p = self.P.gemm_plan(_req(1149, 4396, 3304, "CRR"))
            self.assertIn(p.kind, ("native", "prepass"))
        finally:
            self.P.R2_PLAN_PARTIAL = True

    def test_knob_off(self):
        self.P.R2_PLAN = False
        p = self.P.gemm_plan(_req(22912, 192, 2880, "RRR", "bf16"))
        self.assertEqual(p.kind, "native")

    def test_margin(self):
        self.P.R2_PLAN_MARGIN = 100.0
        self.assertEqual(
            self.P.gemm_plan(_req(22912, 192, 2880, "RRR", "bf16")).kind, "native"
        )
        self.P.R2_PLAN_MARGIN = 0.01
        self.assertEqual(
            self.P.gemm_plan(_req(2048, 2048, 2048, "CCR")).kind, "prepass"
        )

    def test_transposes_tn(self):
        self.P.R2_PLAN_MARGIN = 0.01
        self.P.R2_PLAN_PARTIAL = False
        try:
            p = self.P.gemm_plan(_req(1024, 2048, 512, "CRR"))
        finally:
            self.P.R2_PLAN_PARTIAL = True
        self.assertEqual(
            [(t.operand, t.rows, t.cols) for t in p.transposes],
            [("A", 512, 1024), ("B", 512, 2048)],
        )
        self.assertEqual(p.transposes[0].spec.tile_m, 64)


if __name__ == "__main__":
    unittest.main()
