# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the universal-GEMM row-interleave / slot-schedule levers
(``lds_row_interleave``, ``hb_sched`` and its slot knobs, ``rilv_vec_epilogue``,
``rve_store_nt``).

They cover validity (each lever is rejected outside the body that implements
it), kernel-name flags, the emitted LDS footprint (padded 1040-byte rows), the
schedule structure in the LLVM IR (release / acquire barriers per K tile, no
XOR swizzle), the 16-byte vector epilogue, and that every lever is default-off
(the default trait keeps the historical emission).
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    hb_last_read_slot,
    hb_slots,
    is_valid_spec,
    mono_data_spec,
)

_W22X16 = TileSpec(256, 256, 64, 2, 2, 1, 16, 16, 32)
_W22X32 = TileSpec(256, 256, 64, 2, 2, 1, 32, 32, 16)
_BASE = TraitSpec(
    pipeline="compv4",
    epilogue="default",
    direct_to_lds=True,
    dtl_prefetch=True,
    lds_swizzle=False,
    dtl_cache_a=0,
    dtl_cache_b=0,
    chiplet_swizzle=True,
    acc_agpr=True,
    dtl_stages=2,
    dtl_frag_prefetch=True,
    mfma_interleave=True,
    dtl_hoist=True,
)
_RILV = replace(_BASE, lds_row_interleave=True)
_HB = replace(_RILV, hb_sched=True)


def _spec(trait, tile=_W22X16, dtype="fp16"):
    return UniversalGemmSpec(
        name="t", tile=tile, trait=trait, data=mono_data_spec(dtype), wave_size=64
    )


def _ir(spec):
    return lower_kernel_to_llvm(
        build_universal_gemm(spec, arch="gfx950"), arch="gfx950"
    )


class TestValidity(unittest.TestCase):
    def test_bodies_valid(self):
        for tr in (
            _RILV,
            _HB,
            replace(_HB, hb_read_every=2, rilv_vec_epilogue=True),
            replace(_HB, hb_read_every=2, rilv_vec_epilogue=True, rve_store_nt=True),
            replace(_HB, hb_split_rel=True, hb_rel_gap=6),
            replace(_HB, hb_acq_b_slot=100),
            replace(_HB, persistent_ctas=256, rilv_vec_epilogue=True),
        ):
            for dt in ("fp16", "bf16"):
                ok, why = is_valid_spec(_spec(tr, dtype=dt), "gfx950")
                self.assertTrue(ok, why)

    def test_rejections(self):
        bad = [
            (_RILV, _W22X32, "gfx950"),  # 32x32 atom
            (replace(_RILV, lds_swizzle=True), _W22X16, "gfx950"),
            (_RILV, _W22X16, "gfx942"),  # 1-dword DTL
            (
                replace(
                    _RILV, dtl_stages=0, dtl_frag_prefetch=False, mfma_interleave=False
                ),
                _W22X16,
                "gfx950",
            ),
            (replace(_BASE, hb_sched=True, mfma_interleave=False), _W22X16, "gfx950"),
            (replace(_HB, ilv_merge=True), _W22X16, "gfx950"),
            (replace(_HB, hb_rel_slot=5), _W22X16, "gfx950"),  # before the last read
            (replace(_HB, hb_rel_slot=128), _W22X16, "gfx950"),
            (replace(_BASE, hb_read_every=2), _W22X16, "gfx950"),  # needs hb_sched
            (replace(_HB, hb_split_rel=True, hb_rel_slot=40), _W22X16, "gfx950"),
            (replace(_HB, hb_acq_b_slot=10), _W22X16, "gfx950"),  # before the A acquire
            (replace(_BASE, rilv_vec_epilogue=True), _W22X16, "gfx950"),
            (replace(_HB, rilv_vec_epilogue=True, pad_n=True), _W22X16, "gfx950"),
            (
                replace(_HB, rilv_vec_epilogue=True, epilogue="cshuffle"),
                _W22X16,
                "gfx950",
            ),
            (replace(_HB, rve_store_nt=True), _W22X16, "gfx950"),
        ]
        for tr, tile, arch in bad:
            ok, _ = is_valid_spec(_spec(tr, tile=tile), arch)
            self.assertFalse(ok, tr)

    def test_slots(self):
        s = _spec(_HB)
        self.assertEqual(hb_last_read_slot(s), 15)
        self.assertEqual(hb_slots(s), (19, 67))
        s2 = _spec(replace(_HB, hb_read_every=2))
        self.assertEqual(hb_last_read_slot(s2), 30)
        self.assertEqual(hb_slots(s2), (34, 67))


class TestEmission(unittest.TestCase):
    def test_default_off(self):
        # the default trait spells none of the new flags and builds the same IR
        # as an explicit all-default copy of the new fields
        tr = replace(
            _BASE,
            lds_swizzle=True,
            lds_row_interleave=False,
            hb_sched=False,
            rilv_vec_epilogue=False,
        )
        name = _spec(tr).kernel_name()
        for flag in ("rilv", "hbs", "rve", "rvnt", "hsr"):
            self.assertNotIn(f"_{flag}", name)
        self.assertEqual(_ir(_spec(tr)), _ir(_spec(replace(tr))))

    def test_names(self):
        n = _spec(
            replace(
                _HB,
                hb_read_every=2,
                hb_split_rel=True,
                rilv_vec_epilogue=True,
                rve_store_nt=True,
            )
        ).kernel_name()
        for flag in ("rilv", "hbs", "he2", "hsr", "rve", "rvnt"):
            self.assertIn(f"_{flag}", n)

    def test_lds_and_barriers(self):
        ir = _ir(_spec(_HB))
        # two stages x (A + B) of 32 padded rows of 1040 bytes = 133120
        self.assertTrue("[133120 x i8]" in ir)
        # per 2-tile unrolled main loop: release + acquire per tile
        self.assertGreaterEqual(ir.count("@llvm.amdgcn.s.barrier"), 4)

    def test_vec_epilogue(self):
        ir = _ir(_spec(replace(_HB, rilv_vec_epilogue=True)))
        self.assertTrue("store <8 x half>" in ir)
        ir_nt = _ir(_spec(replace(_HB, rilv_vec_epilogue=True, rve_store_nt=True)))
        self.assertTrue("!nontemporal" in ir_nt)


if __name__ == "__main__":
    unittest.main()
