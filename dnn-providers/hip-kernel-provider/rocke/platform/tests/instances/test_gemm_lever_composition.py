# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of how the overlapping universal-GEMM levers compose.

All of the levers are kept; the overlaps are resolved as follows:

* three StaggerU forms (``k_stagger`` / ``k_stagger_tiles`` /
  ``k_stagger_map``, ``stagger_u`` / ``stagger_u_stride`` / ``stagger_u_map``
  and ``k_stagger_rc``) share one physical-K path and are mutually exclusive;
* the economy levers (``hb_tile_lgkm0``, ``das_late``) and the serpentine
  MFMA order (``mfma_serpentine``) combine on the dispatched aligned bodies;
* the pad bodies take the serpentine order but not ``das_late`` (with the
  exact DTL bounds it makes LLVM spill) and not ``hb_tile_lgkm0`` (no gain);
* the asm-loop emission drops the hb_sched-only economy levers.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from rocke.dispatch.gemm import (
    GemmRequest,
    common as _cm,
    dispatch_gemm_bf16,
    dispatch_gemm_fp16,
)
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    mono_data_spec,
)

_T22 = TileSpec(256, 256, 64, 2, 2, 1, 16, 16, 32)
_HB = TraitSpec(
    pipeline="compv4",
    epilogue="default",
    direct_to_lds=True,
    dtl_prefetch=True,
    lds_swizzle=False,
    dtl_cache_a=0,
    dtl_cache_b=0,
    chiplet_swizzle=True,
    chiplet_wgm=16,
    chiplet_chunk_size=8,
    acc_agpr=True,
    dtl_stages=2,
    dtl_frag_prefetch=True,
    mfma_interleave=True,
    dtl_hoist=True,
    lds_row_interleave=True,
    rilv_pad=32,
    hb_sched=True,
    hb_read_every=2,
    rilv_vec_epilogue=True,
    rve_store_nt=True,
)
# the tuned row-interleave body (long-K class)
_D = replace(
    _HB,
    dtl_soff0=True,
    dtl_addr_spread=True,
    hb_dtl_delay=1,
    kloop_unroll=4,
    hb_skew=1,
)

_ANCHORS = ((8192, 8192, 8192), (23040, 24576, 12288), (15360, 15360, 768))
_PADS = ((30720, 20160, 7680), (38400, 30240, 3072), (7680, 6720, 1536))


def _spec(tr, dt="bf16"):
    return UniversalGemmSpec(name="t", tile=_T22, trait=tr, data=mono_data_spec(dt))


def _ok(tr):
    return is_valid_spec(_spec(tr), arch="gfx950")


def _ir(tr, dt="bf16"):
    return lower_kernel_to_llvm(
        build_universal_gemm(_spec(tr, dt), arch="gfx950"), arch="gfx950"
    )


def _name(tr):
    return build_universal_gemm(_spec(tr), arch="gfx950").name


def _pick(M, N, K, dt="bf16"):
    f = dispatch_gemm_bf16 if dt == "bf16" else dispatch_gemm_fp16
    return f(
        GemmRequest(
            M=M, N=N, K=K, arch="gfx950", dtype="bf16" if dt == "bf16" else "fp16"
        )
    ).spec


class DefaultsTest(unittest.TestCase):
    def test_every_round3_knob_spelled_out_off_is_identical(self):
        spelled = replace(
            _D,
            hb_tile_lgkm0=0,
            dtl_base_fold=False,
            hb_skew_dtl=0,
            das_anchor=False,
            das_late=False,
            k_stagger=0,
            k_stagger_tiles=0,
            k_stagger_map=0,
            dtl_clamp_k=False,
            stagger_u=0,
            stagger_u_stride=1,
            stagger_u_map=0,
            mfma_serpentine=False,
            k_stagger_rc=0,
            dtl_tight_ragged_only=False,
        )
        self.assertEqual(_ir(_D), _ir(spelled))

    def test_stagger_name_flags_distinct(self):
        names = {
            _name(replace(_D, k_stagger=8)),
            _name(replace(_D, stagger_u=8)),
            _name(replace(_D, k_stagger_rc=8)),
        }
        self.assertEqual(len(names), 3)
        self.assertIn("_ksr8", _name(replace(_D, k_stagger_rc=8)))
        self.assertIn("_kst8", _name(replace(_D, k_stagger=8)))
        self.assertIn("_su8s1m0", _name(replace(_D, stagger_u=8)))


class StaggerFormsTest(unittest.TestCase):
    def test_each_form_alone_valid(self):
        for tr in (
            replace(_D, k_stagger=8),
            replace(_D, stagger_u=8),
            replace(_D, k_stagger_rc=8),
        ):
            ok, why = _ok(tr)
            self.assertTrue(ok, why)

    def test_forms_mutually_exclusive(self):
        for tr in (
            replace(_D, k_stagger=8, stagger_u=8),
            replace(_D, k_stagger=8, k_stagger_rc=8),
            replace(_D, stagger_u=8, k_stagger_rc=8),
        ):
            ok, why = _ok(tr)
            self.assertFalse(ok, tr)
            self.assertIn("StaggerU", why)
        ok, why = _ok(replace(_D, stagger_u=8, dtl_clamp_k=True))
        self.assertFalse(ok)
        self.assertIn("dtl_clamp_k", why)

    def test_each_form_moves_the_dma_k(self):
        base = _ir(_D)
        irs = [
            _ir(replace(_D, k_stagger=8)),
            _ir(replace(_D, stagger_u=8)),
            _ir(replace(_D, k_stagger_rc=8)),
        ]
        for ir in irs:
            self.assertNotEqual(ir, base)
        self.assertEqual(len(set(irs)), 3)

    def test_stagger_u_reaches_the_folded_base(self):
        # dtl_base_fold builds the resource from the folded base + K: the
        # stagger_u offset must still move that K (one shared physical-K path)
        dbf = replace(_D, dtl_base_fold=True, dtl_addr_spread=False, hb_dtl_delay=0)
        self.assertTrue(_ok(dbf)[0])
        su = replace(dbf, stagger_u=8)
        self.assertTrue(_ok(su)[0], _ok(su)[1])
        self.assertNotEqual(_ir(dbf), _ir(su))

    def test_stagger_u_reaches_the_spread_steps(self):
        # dtl_addr_spread + das_late: the deferred steps use the same path
        d = replace(_D, hb_tile_lgkm0=1, das_late=True)
        self.assertNotEqual(_ir(d), _ir(replace(d, stagger_u=8)))
        self.assertNotEqual(_ir(d), _ir(replace(d, k_stagger_rc=8)))

    def test_k_stagger_rc_wraps_every_dma_k(self):
        # the k_stagger_rc wrap (compare + select per DTL group) must reach
        # both the deferred (dtl_addr_spread) and the immediate DTL paths
        def sel(ir):
            return ir.count("= select ")

        for tr in (
            replace(_D, hb_tile_lgkm0=1, das_late=True),
            replace(_D, dtl_addr_spread=False, hb_dtl_delay=0),
        ):
            delta = sel(_ir(replace(tr, k_stagger_rc=8))) - sel(_ir(tr))
            self.assertGreaterEqual(delta, 20, tr)


class DispatchTest(unittest.TestCase):
    def test_anchors_carry_both_lever_groups(self):
        for dt in ("bf16", "f16"):
            for M, N, K in _ANCHORS:
                sp = _pick(M, N, K, dt)
                t = sp.trait
                self.assertEqual(t.hb_tile_lgkm0, _cm.RILV_HB_TILE_LGKM0)
                self.assertEqual(t.das_late, _cm.RILV_DAS_LATE)
                self.assertEqual(t.mfma_serpentine, _cm.RILV_SERPENTINE)
                self.assertTrue(t.hb_tile_lgkm0 and t.das_late and t.mfma_serpentine)
                self.assertTrue(is_valid_spec(sp, arch="gfx950")[0])

    def test_pad_bodies_serpentine_without_das_late(self):
        for dt in ("bf16", "f16"):
            for M, N, K in _PADS:
                sp = _pick(M, N, K, dt)
                t = sp.trait
                self.assertTrue(t.dtl_tight_rsrc and t.hb_sched, (M, N, K))
                self.assertTrue(t.mfma_serpentine, (M, N, K))
                self.assertFalse(t.das_late, (M, N, K))
                self.assertEqual(t.hb_tile_lgkm0, 0, (M, N, K))
                self.assertEqual(t.hb_read_every, 2, (M, N, K))
                self.assertTrue(is_valid_spec(sp, arch="gfx950")[0])

    def test_pad_knobs_apply(self):
        old = (_cm.RILV_PAD_HB_TILE_LGKM0, _cm.RILV_PAD_DAS_LATE)
        try:
            _cm.RILV_PAD_HB_TILE_LGKM0 = True
            _cm.RILV_PAD_DAS_LATE = True
            for M, N, K in _PADS:
                sp = _pick(M, N, K)
                self.assertEqual(sp.trait.hb_tile_lgkm0, 1)
                self.assertTrue(sp.trait.das_late)
                self.assertTrue(is_valid_spec(sp, arch="gfx950")[0])
        finally:
            _cm.RILV_PAD_HB_TILE_LGKM0, _cm.RILV_PAD_DAS_LATE = old

    def test_shortk_s2_fill_gate(self):
        # poorly filled 256x256 waves with exact 128x192 tiles -> the small-tile
        # body; well-filled or non-dividing shapes keep the 256x256 body
        for (M, N, K), s2 in (
            ((4608, 4608, 384), True),
            ((4608, 4608, 640), True),
            ((4608, 6144, 384), False),  # fill 0.84
            ((4096, 4096, 512), False),  # 4096 % 192 != 0
        ):
            for dt in ("bf16", "f16"):
                sp = _pick(M, N, K, dt)
                got = (sp.tile.tile_m, sp.tile.tile_n) == (128, 192)
                self.assertEqual(got, s2, (M, N, K, dt))
                self.assertTrue(is_valid_spec(sp, arch="gfx950")[0])
        old = _cm.RILV_SHORTK_SMALL_TILE
        try:
            _cm.RILV_SHORTK_SMALL_TILE = False
            sp = _pick(4608, 4608, 384)
            self.assertEqual((sp.tile.tile_m, sp.tile.tile_n), (256, 256))
        finally:
            _cm.RILV_SHORTK_SMALL_TILE = old

    def test_longk_grid_knob(self):
        # default off: the wgm 16 / chunk 8 grid on every long-K shape
        self.assertEqual(_cm.RILV_LONGK_GRID, ())
        for M, N, K in ((16384, 16384, 8192), (12288, 12288, 12288)):
            t = _pick(M, N, K).trait
            self.assertEqual((t.chiplet_wgm, t.chiplet_chunk_size), (16, 8))
        old = _cm.RILV_LONGK_GRID
        try:
            _cm.RILV_LONGK_GRID = (32, 8)
            t = _pick(16384, 16384, 8192).trait  # 4096 tiles, K 8192
            self.assertEqual((t.chiplet_wgm, t.chiplet_chunk_size), (32, 8))
            t = _pick(8192, 8192, 8192).trait  # 1024 tiles: out of range
            self.assertEqual((t.chiplet_wgm, t.chiplet_chunk_size), (16, 8))
            t = _pick(16384, 16384, 4096).trait  # K below the floor
            self.assertEqual(t.chiplet_wgm, 16)
        finally:
            _cm.RILV_LONGK_GRID = old

    def test_asm_loop_body_drops_hb_only_levers(self):
        old = _cm.RILV_HB_LOOP
        try:
            _cm.RILV_HB_LOOP = "asm_loop"
            for M, N, K in _ANCHORS:
                sp = _pick(M, N, K)
                self.assertEqual(sp.trait.asm_loop, "tensile")
                self.assertEqual(sp.trait.hb_tile_lgkm0, 0)
                self.assertFalse(sp.trait.das_late)
                ok, why = is_valid_spec(sp, arch="gfx950")
                self.assertTrue(ok, why)
        finally:
            _cm.RILV_HB_LOOP = old


if __name__ == "__main__":
    unittest.main()
