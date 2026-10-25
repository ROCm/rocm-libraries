# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of ``hb_skew`` and how the tuned-body levers compose.

* ``hb_skew``: two main-loop copies per SIMD (1) or wave-index (2)
  parity on the computed hb_sched schedule; the odd copy moves every fragment
  read, DTL issue and DTL address step one MFMA later.
* ``asm_loop`` + the hb_sched slot movers ``dtl_addr_spread`` /
  ``hb_dtl_delay``: placed natively in the asm table (see test_gemm_stagger_u_asm_loop.py).
* The tuned-body levers compose on the dispatched body: ``dtl_soff0`` +
  ``dtl_addr_spread`` + ``hb_dtl_delay`` + ``kloop_unroll`` + ``hb_skew`` on
  hb_sched, and ``dtl_soff0`` + ``kloop_unroll`` on the asm_loop and sia_sched
  bodies.

Default-off, unique kernel-name flag, rejected outside its scope, refused by
the C-engine converters (Python-engine only).
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from rocke.core import backend as _be
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
# the dispatched tuned body (long-K class)
_TUNED = replace(
    _HB,
    dtl_soff0=True,
    dtl_addr_spread=True,
    hb_dtl_delay=1,
    kloop_unroll=4,
    hb_skew=1,
)


def _spec(tr, dt="bf16"):
    return UniversalGemmSpec(name="t", tile=_T22, trait=tr, data=mono_data_spec(dt))


def _ok(tr):
    return is_valid_spec(_spec(tr), arch="gfx950")


def _ir(tr, dt="bf16"):
    return lower_kernel_to_llvm(
        build_universal_gemm(_spec(tr, dt), arch="gfx950"), arch="gfx950"
    )


class HbSkewTest(unittest.TestCase):
    def test_default_off_and_spelled_out_identical(self):
        self.assertEqual(TraitSpec().hb_skew, 0)
        self.assertEqual(_ir(_HB), _ir(replace(_HB, hb_skew=0)))

    def test_name_flag(self):
        n0 = build_universal_gemm(_spec(_HB), arch="gfx950").name
        n1 = build_universal_gemm(_spec(replace(_HB, hb_skew=1)), arch="gfx950").name
        n2 = build_universal_gemm(_spec(replace(_HB, hb_skew=2)), arch="gfx950").name
        self.assertNotIn("hsk", n0)
        self.assertTrue(n1.endswith("hsk1") or "_hsk1" in n1)
        self.assertIn("_hsk2", n2)

    def test_scope(self):
        self.assertTrue(_ok(replace(_HB, hb_skew=1))[0])
        self.assertTrue(_ok(replace(_HB, hb_skew=2))[0])
        for bad in (
            replace(_HB, hb_skew=3),
            replace(_HB, hb_skew=-1),
            replace(_HB, hb_sched=False, hb_read_every=1, ilv_merge=True, hb_skew=1),
            replace(_HB, hb_read_every=1, asm_loop="tensile", hb_skew=1),
        ):
            ok, why = _ok(bad)
            self.assertFalse(ok, bad)
            self.assertIn("hb_skew", why)

    def test_emission_two_loop_copies(self):
        ir0 = _ir(_HB)
        ir1 = _ir(replace(_HB, hb_skew=1))
        ir2 = _ir(replace(_HB, hb_skew=2))
        # SIMD parity from HW_ID (inline asm) only for skew 1
        self.assertNotIn("HW_REG_HW_ID", ir0)
        self.assertEqual(ir1.count("hwreg(HW_REG_HW_ID, 4, 1)"), 1)
        self.assertNotIn("HW_REG_HW_ID", ir2)
        # a second copy of the main loop: twice the MFMAs of the unskewed loop
        # body (prologue / tail / epilogue are shared)
        m0 = ir0.count("llvm.amdgcn.mfma")
        m1 = ir1.count("llvm.amdgcn.mfma")
        self.assertGreater(m1, m0)
        self.assertEqual(m1, ir2.count("llvm.amdgcn.mfma"))
        # the odd copy is a different instruction order, not a duplicate
        self.assertNotEqual(ir1, ir2)

    def test_odd_copy_is_shifted(self):
        # the two main-loop bodies (first two ``for.body`` blocks) issue the
        # same instructions, but the odd copy places its reads / DTL issues
        # one MFMA later: the call sequences differ, the call multisets match
        import re

        ir1 = _ir(replace(_HB, hb_skew=1))
        blocks = re.split(r"^([^\s:;]+):", ir1, flags=re.M)
        bodies = [
            blocks[i + 1]
            for i in range(1, len(blocks) - 1, 2)
            if blocks[i].startswith("for.body")
        ]
        self.assertGreaterEqual(len(bodies), 2)
        seq = [re.findall(r"call [^@]*@([A-Za-z0-9_.]+)", t) for t in bodies[:2]]
        self.assertEqual(sorted(seq[0]), sorted(seq[1]))
        self.assertNotEqual(seq[0], seq[1])


class AsmLoopExclusionTest(unittest.TestCase):
    def test_asm_loop_admits_hb_slot_movers_round3(self):
        # the asm table places them natively
        asm = replace(_HB, hb_read_every=1, asm_loop="tensile", asm_loop_skew=1)
        self.assertTrue(_ok(asm)[0], _ok(asm)[1])
        for kw in (dict(dtl_addr_spread=True), dict(hb_dtl_delay=1)):
            ok, why = _ok(replace(asm, **kw))
            self.assertTrue(ok, why)
        # the hb_sched slot knobs proper stay excluded
        ok, why = _ok(replace(asm, hb_rel_slot=40))
        self.assertFalse(ok)
        self.assertIn("asm_loop", why)


class ComposeTest(unittest.TestCase):
    def test_dispatched_body_valid_and_builds(self):
        for dt in ("bf16", "fp16"):
            self.assertTrue(is_valid_spec(_spec(_TUNED, dt), arch="gfx950")[0])
            self.assertIn("llvm.amdgcn.mfma", _ir(_TUNED, dt))

    def test_round2_combinations_valid(self):
        asm = replace(_HB, hb_read_every=1, asm_loop="tensile", asm_loop_skew=1)
        sia = replace(
            _HB,
            hb_sched=False,
            hb_read_every=1,
            sia_sched=True,
            sia_plr=True,
            sia_dtl_spread=True,
            sia_acq_gap=12,
        )
        for tr in (
            replace(asm, dtl_soff0=True),
            replace(asm, dtl_soff0=True, kloop_unroll=2),
            replace(asm, dtl_soff0=True, kloop_unroll=4),
            replace(sia, dtl_soff0=True, kloop_unroll=2),
            replace(
                _TUNED,
                persistent_ctas=256,
                chiplet_wgm=8,
                chiplet_chunk_size=32,
                kloop_unroll=2,
            ),
            replace(_TUNED, hb_skew=2, hb_dtl_delay=2),
        ):
            ok, why = _ok(tr)
            self.assertTrue(ok, why)
            self.assertIn("llvm.amdgcn.mfma", _ir(tr))


class PythonOnlyTest(unittest.TestCase):
    def test_engine_refuses_hb_skew(self):
        self.assertIn(("hb_skew", 0), _be._PYTHON_ONLY_GEMM_LEVERS)
        with self.assertRaises(Exception):
            _be._refuse_python_only_levers(replace(_HB, hb_skew=1))
        # hb_skew alone (on an otherwise engine-supported trait) is refused too
        with self.assertRaises(NotImplementedError) as cm:
            _be._refuse_python_only_levers(replace(TraitSpec(), hb_skew=1))
        self.assertIn("hb_skew", str(cm.exception))
        _be._refuse_python_only_levers(TraitSpec())


if __name__ == "__main__":
    unittest.main()
