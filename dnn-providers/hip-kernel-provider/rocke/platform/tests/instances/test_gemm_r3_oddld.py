# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Native NN / TN / TT on an MN leading dimension that is
not a multiple of 8 (gfx950). CPU only.

* ``TraitSpec.mn_ld_any``: the DTL ring reads 2-byte-aligned rows of an
  MN-contiguous operand (the 16-byte DMA returns the right data at any 2-byte
  offset); the last output tile along that operand's dimension is shifted back
  to end at the extent, so no load of the operand reaches past its last element
  (the buffer range check drops a whole dword of a load that straddles the
  operand's end). Validity: DTL ring, no split-K / stream-K / batch; launch:
  every MN leading dimension >= its tile.
* dispatch: ``common.NATIVE_LAYOUT_DTL_LD_ANY`` (the native forms keep the DTL
  ring with mn_ld_any), ``common.BEAT_R2_PAD_NATIVE_N_ANY`` (the padded
  body as the RCR base of NN / TN with N % 8 != 0), ``selector.SELECTOR_LD_ANY``
  (generated mn_ld_any forms), ``skinny.SKINNY_NATIVE_T2_ODD_*`` (gemm_skinny
  only below 32 128x128 tiles or K < 96 on such a leading dimension).
"""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest import mock

from rocke.dispatch.gemm import (
    GEMM_FP16_REGISTRY,
    GemmRequest,
    dispatch_gemm_bf16,
    dispatch_gemm_fp16,
)
from rocke.dispatch.gemm import common as C
from rocke.dispatch.gemm import plan as P
from rocke.dispatch.gemm import selector as S
from rocke.dispatch.gemm import skinny as SK
from rocke.instances.common import gemm_mn_operand as MNO
from rocke.instances.common.gemm_universal import (
    build_universal_gemm,
    is_valid_spec,
    launch_shape_error,
)

GEMM_SELECTOR_TESTS = True  # conftest: keep the shipped selector defaults


def _req(M, N, K, lay="CRR", dt="fp16", arch="gfx950"):
    return GemmRequest(
        M=M, N=N, K=K, arch=arch, dtype=dt, layout=lay,
        trans_a=lay[0] == "C", trans_b=lay[1] == "C",
    )  # fmt: skip


class _Clean(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ)
        self._env.start()
        for k in (S.SELECTOR_ENV, S.NATIVE_SELECTOR_ENV, P.GEMM_PLAN_PREPASS_ENV):
            os.environ.pop(k, None)
        S.clear_cache()

    def tearDown(self):
        self._env.stop()
        S.clear_cache()


def _native(req):
    """The non-skinny native pick of ``req``."""
    with mock.patch.object(SK, "SKINNY_NATIVE", False):
        f = dispatch_gemm_fp16 if req.dtype == "fp16" else dispatch_gemm_bf16
        return f(req).spec


def _n_smin(spec):
    k = build_universal_gemm(spec, arch="gfx950")
    n = 0

    def walk(ops):
        nonlocal n
        for op in ops:
            if op.name == "arith.smin":
                n += 1
            for r in op.regions:
                walk(r.ops)

    walk(k.body.ops)
    return n


class TestSpecRules(_Clean):
    def _form(self, M=4100, N=3002, K=4096, lay="CRR"):
        s = _native(_req(M, N, K, lay))
        self.assertTrue(s.trait.mn_ld_any and s.trait.direct_to_lds, s.kernel_name())
        return s

    def test_default_off_and_name(self):
        s = self._form()
        off = replace(s, trait=replace(s.trait, mn_ld_any=False))
        self.assertTrue(s.kernel_name().endswith("_mnla"))
        self.assertNotIn("mnla", off.kernel_name())
        self.assertFalse(type(s.trait)().mn_ld_any)

    def test_validity(self):
        s = self._form()
        self.assertTrue(is_valid_spec(s, arch="gfx950")[0])
        rcr = replace(s, data=replace(s.data, layout="RCR"))
        self.assertIn("MN-contiguous operand", is_valid_spec(rcr, arch="gfx950")[1])
        for kv in (dict(split_k=2), dict(streamk=True), dict(split_k_serial=1)):
            ok, why = is_valid_spec(
                replace(s, trait=replace(s.trait, **kv)), arch="gfx950"
            )
            self.assertFalse(ok, kv)
        why = MNO.mn_layout_error(
            replace(s, trait=replace(s.trait, split_k=2)), "gfx950"
        )
        self.assertIn("mn_ld_any", why)
        why = MNO.mn_layout_error(replace(s, batched=True), "gfx950")
        self.assertIn("mn_ld_any", why)
        # the VGPR-staged loader (no DTL ring) has no shifted tile
        g = replace(
            s,
            trait=replace(
                s.trait, direct_to_lds=False, dtl_prefetch=False, dtl_stages=0
            ),
        )
        self.assertIn("DTL-ring lever", MNO.mn_layout_error(g, "gfx950"))

    def test_launch_rules(self):
        s = self._form()
        t = s.tile
        self.assertIsNone(launch_shape_error(s, 4100, 3002, 4096))
        self.assertIsNone(launch_shape_error(s, 4097, 3001, 4096))
        # MN leading dimension below the tile (the shifted origin would be < 0),
        # also when it is a multiple of 8
        for M, N in ((t.tile_m - 1, 3002), (4100, t.tile_n - 2), (t.tile_m - 8, 3002)):
            self.assertIn(
                "below the tile", launch_shape_error(s, M, N, 4096) or "", (M, N)
            )
        off = replace(s, trait=replace(s.trait, mn_ld_any=False))
        self.assertIn("not a multiple of 8", launch_shape_error(off, 4100, 3002, 4096))

    def test_shifted_tile_emission(self):
        # one smin per MN-contiguous operand's tile origin (non-persistent body)
        for lay, M, N, n in (
            ("CCR", 4100, 3000, 1),
            ("RRR", 4096, 3002, 1),
            ("CRR", 4100, 3002, 2),
        ):
            s = _native(_req(M, N, 4096, lay))
            s = replace(s, trait=replace(s.trait, persistent_ctas=0))
            off = replace(s, trait=replace(s.trait, mn_ld_any=False))
            self.assertEqual(_n_smin(s) - _n_smin(off), n, lay)
        # the persistent tile walk clamps every tile origin it computes too
        s = _native(_req(4100, 3000, 4096, "CCR"))
        s = replace(s, trait=replace(s.trait, persistent_ctas=256, mn_ld_any=True))
        self.assertTrue(is_valid_spec(s, arch="gfx950")[0])
        off = replace(s, trait=replace(s.trait, mn_ld_any=False))
        self.assertGreaterEqual(_n_smin(s) - _n_smin(off), 2)

    def test_cpp_engine_mirrors_the_lever(self):
        # gemm_mn_operand.cpp / gemm_build_staged.cpp mirror the DTL-ring MN
        # operands (mn_pad / mn_chunk / mn_ld_any); the binding receives them
        from rocke.core import backend as _be

        keys = {k for k, _ in _be._PYTHON_ONLY_GEMM_LEVERS}
        for k in ("mn_ld_any", "mn_pad", "mn_chunk"):
            self.assertNotIn(k, keys)
        s = _native(_req(4100, 3000, 4096, "CCR"))
        d = _be.universal_gemm_spec_to_dict(s)
        self.assertEqual(d["layout"], "CCR")
        self.assertEqual(d["mn_ld_any"], s.trait.mn_ld_any)
        self.assertEqual(d["mn_pad"], s.trait.mn_pad)
        self.assertEqual(d["mn_chunk"], s.trait.mn_chunk)


class TestDispatch(_Clean):
    def test_native_forms_keep_dtl(self):
        req = _req(4100, 3002, 4096, "CRR")
        rcr = dispatch_gemm_fp16(S._rcr_view(req)).spec
        alts, _ = C.native_layout_alternatives(req, rcr)
        self.assertTrue(alts)
        self.assertTrue(all(s.trait.mn_ld_any for _, s in alts if MNO.uses_mn_dtl(s)))
        self.assertTrue(any(MNO.uses_mn_dtl(s) for _, s in alts))
        with mock.patch.object(C, "NATIVE_LAYOUT_DTL_LD_ANY", False):
            alts, _ = C.native_layout_alternatives(req, rcr)
            self.assertEqual([n for n, _ in alts], ["generic"])
        with mock.patch.object(C, "NATIVE_LAYOUT_DTL_LD_ANY_ARCHS", ()):
            alts, _ = C.native_layout_alternatives(req, rcr)
            self.assertEqual([n for n, _ in alts], ["generic"])
        # aligned leading dimensions: no mn_ld_any
        req8 = _req(4104, 3000, 4096, "CRR")
        alts, _ = C.native_layout_alternatives(
            req8, dispatch_gemm_fp16(S._rcr_view(req8)).spec
        )
        self.assertFalse([s for _, s in alts if s.trait.mn_ld_any])

    def test_gfx942_unchanged(self):
        self.assertNotIn("gfx942", C.NATIVE_LAYOUT_DTL_LD_ANY_ARCHS)
        s = _native(_req(4100, 3002, 4096, "CRR", arch="gfx942"))
        self.assertFalse(s.trait.mn_ld_any)

    def test_beat_base_for_odd_n(self):
        # NN / TN with N % 8 != 0: the padded body stays the RCR base
        big = dict(M=8229, N=24555, K=8304)
        for lay, want in (("CRR", True), ("RRR", True), ("CCR", False)):
            self.assertEqual(C.rilv_pad_admits(_req(lay=lay, **big)), want, lay)
        with mock.patch.object(C, "BEAT_R2_PAD_NATIVE_N_ANY", False):
            self.assertFalse(C.rilv_pad_admits(_req(lay="CRR", **big)))
        rcr = replace(_req(lay="CRR", **big), layout="RCR", trans_a=False, trans_b=True)
        self.assertFalse(C.rilv_pad_admits(rcr))

    def test_skinny_thresholds(self):
        # the odd-ld thresholds (the odd long-K rule off: test_gemm_merge_r3)
        with mock.patch.object(SK, "SKINNY_NATIVE_T2_ODD_LONGK_TILES", 0):
            self._skinny_thresholds()

    def _skinny_thresholds(self):
        self.assertEqual(
            (SK.SKINNY_NATIVE_T2_ODD_MAX_TILES, SK.SKINNY_NATIVE_T2_ODD_MIN_K), (32, 96)
        )
        for lay, M, N, K, want in (
            ("CRR", 3097, 4046, 24608, False),  # 25 x 32 tiles: the DTL odd forms
            ("CRR", 3097, 4046, 88, True),  # K < 96: no DTL ring form
            ("CRR", 513, 769, 4096, False),  # 5 x 7 = 35 tiles
            ("CRR", 1001, 500, 4096, False),  # 8 x 4 = 32 tiles
            ("CRR", 1001, 380, 4096, True),  # 8 x 3 = 24 tiles
            ("RRR", 333, 1001, 4096, True),  # 3 x 8 = 24 tiles
            ("CCR", 4095, 1000, 1024, False),  # 32 x 8 = 256 tiles
        ):
            ok, _ = SK.skinny_admit_region(_req(M, N, K, lay, "bf16"))
            self.assertEqual(ok, want, (lay, M, N, K))
        odd_big = _req(3097, 4046, 24608, "CRR", "bf16")
        with mock.patch.object(SK, "SKINNY_NATIVE_T2_ODD_MAX_TILES", 0):
            self.assertTrue(SK.skinny_admit_region(odd_big)[0])
        with mock.patch.object(C, "NATIVE_LAYOUT_DTL_LD_ANY", False):
            self.assertTrue(SK.skinny_admit_region(odd_big)[0])
        # a target without the DTL odd forms keeps the always-skinny rule under tiles2
        odd_942 = _req(3097, 4046, 24608, "CRR", "bf16", arch="gfx942")
        with mock.patch.dict(SK.SKINNY_NATIVE_RULE, {"gfx942": "tiles2"}):
            self.assertTrue(SK._native_tiles2_admit(odd_942)[0])
            self.assertFalse(SK._native_tiles2_admit(odd_big)[0])
        # the default dispatch: a DTL-ring native form
        s = dispatch_gemm_bf16(odd_big).spec
        self.assertTrue(getattr(s, "trait", None) is not None and s.trait.mn_ld_any)

    def test_selector_generated_gate(self):
        req = _req(8229, 24555, 8304, "CRR")
        s = _native(req)
        self.assertTrue(S._ld_any_generated_ok(req, s))
        for kv in (dict(mn_ld_any=False), dict(dtl_tight_rsrc=False)):
            self.assertFalse(
                S._ld_any_generated_ok(req, replace(s, trait=replace(s.trait, **kv))),
                kv,
            )
        with mock.patch.object(S, "SELECTOR_LD_ANY", False):
            self.assertFalse(S._ld_any_generated_ok(req, s))
        with mock.patch.object(C, "NATIVE_LAYOUT_DTL_LD_ANY", False):
            self.assertFalse(S._ld_any_generated_ok(req, s))
        self.assertFalse(
            S._ld_any_generated_ok(_req(8229, 24555, 8304, "CRR", arch="gfx942"), s)
        )
        self.assertIsNone(S.generated_spec_error(req, s, s))
        self.assertIn(
            "multiples of 8", S.generated_spec_error(replace(req, K=8300), s, s)
        )
        # the generated neighbours exist on such a shape (and all carry mn_ld_any)
        cand = GEMM_FP16_REGISTRY.select(req)
        cs = S.native_candidates(
            req, GEMM_FP16_REGISTRY, rule_pick=(cand, cand.select_spec(req))
        )
        nb = [c for c in cs if c.origin.startswith("neighbour")]
        self.assertTrue(nb)
        self.assertTrue(all(c.spec.trait.mn_ld_any for c in nb))

    def test_plan_odd_prepass_rule_inert_on_dtl_picks(self):
        big = _req(6131, 9158, 17696, "CRR", "bf16")
        pl = P.plan_gemm_bf16(big)
        self.assertEqual(pl.kind, "native")
        self.assertTrue(pl.gemm.spec.trait.mn_ld_any)


class TestSkinnyUload(_Clean):
    """gemm_skinny ``mn_uload``: unaligned 16-byte MN loads, the last tile
    shifted to end at the extent."""

    def test_rules_and_name(self):
        from rocke.instances.common.gemm_skinny import (
            SkinnyGemmSpec,
            is_valid_spec as skv,
            skinny_launch_error,
        )

        s = SkinnyGemmSpec(layout="CRR", mnalign=1, mn_uload=True, dtype="fp16")
        self.assertTrue(skv(s, "gfx950")[0])
        self.assertIn("mnul", s.kernel_name())
        self.assertNotIn("mnul", replace(s, mn_uload=False).kernel_name())
        for bad in (replace(s, mnalign=8), replace(s, layout="RCR", mnalign=8)):
            self.assertIn("mn_uload", skv(bad, "gfx950")[1])
        self.assertIn("gfx950 only", skv(replace(s, mn_lds="u16"), "gfx942")[1])
        tm, tn = s.tile_m, s.tile_n
        self.assertIsNone(skinny_launch_error(s, tm + 3, tn + 5, 1024, 1))
        self.assertIn("below the tile", skinny_launch_error(s, tm - 1, tn + 5, 1024, 1))
        self.assertIn("below the tile", skinny_launch_error(s, tm + 3, tn - 2, 1024, 1))
        # RRR: only B is MN-contiguous (M may be anything)
        r = replace(s, layout="RRR")
        self.assertIsNone(skinny_launch_error(r, 3, tn + 5, 1024, 1))

    def test_emission(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.gemm_skinny import SkinnyGemmSpec, build_gemm_skinny

        def ll(sp):
            return lower_kernel_to_llvm(
                build_gemm_skinny(sp, arch="gfx950"), arch="gfx950"
            )

        s = SkinnyGemmSpec(layout="CRR", mnalign=1, dtype="fp16")
        u = ll(replace(s, mn_uload=True))
        self.assertIn("alignbyte", ll(s))
        self.assertNotIn("alignbyte", u)
        smin = "call i32 @llvm.smin.i32"
        self.assertEqual(u.count(smin), ll(s).count(smin) + 2)  # both MN tile origins

    def test_dispatch(self):
        # an odd-ld few-tile native request: gemm_skinny with mn_uload
        req = _req(879, 173, 1888, "CRR")
        s = dispatch_gemm_fp16(req).spec
        self.assertEqual(type(s).__name__, "SkinnyGemmSpec")
        self.assertTrue(s.mn_uload and s.mnalign == 1)
        with mock.patch.object(SK, "SKINNY_MN_ULOAD", False):
            self.assertFalse(dispatch_gemm_fp16(req).spec.mn_uload)
        # a leading dimension below the tile keeps the realigned dword loads
        small = SK.skinny_native(
            replace(s, layout="RCR", mnalign=8, mn_uload=False),
            _req(5, 173, 1888, "CRR"),
        )
        self.assertFalse(small.mn_uload)
        # gfx942: not admitted (not hardware-checked there)
        self.assertNotIn("gfx942", SK.SKINNY_MN_ULOAD_ARCHS)


if __name__ == "__main__":
    unittest.main()
