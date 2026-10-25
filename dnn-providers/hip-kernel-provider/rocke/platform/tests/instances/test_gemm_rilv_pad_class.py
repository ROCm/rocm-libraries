# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the row-interleave ragged / pad levers.

* ``dtl_addr_spread`` now combines with ``dtl_tight_rsrc``: the deferred
  address steps build the exact-bounds resource (bytes left from the tile
  origin, zero past K).
* ``dtl_tight_ragged_only``: exact DTL bounds only on the operand whose row
  dimension is padded (A with pad_m, B with pad_n); the other operand takes the
  unbounded base-origin resource.
* ``k_stagger_rc``: a StaggerU-style offset (each output tile starts its K walk at a
  tile-dependent K tile and wraps around).
* The dispatched pad bodies: the tuned row-interleave body (hb_sched + its
  levers) on every pad class, with wgm 32 / chunk 2 on the long-K pad class.

Default-off, unique kernel-name flag, rejected outside its scope, refused by
the C-engine converters (Python-engine only).
"""

from __future__ import annotations

import re
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
# the padded row-interleave body (pad on N, exact DTL bounds, branch-free epilogue bounds)
_PAD = replace(
    _HB,
    pad_n=True,
    dtl_tight_rsrc=True,
    pad_fast_tile=True,
    rve_pad=2,
)


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


class TightSpreadTest(unittest.TestCase):
    def test_das_with_tight_bounds_valid_and_builds(self):
        tr = replace(_PAD, dtl_addr_spread=True)
        ok, why = _ok(tr)
        self.assertTrue(ok, why)
        self.assertIn("llvm.amdgcn.mfma", _ir(tr))
        # the deferred path changes the emission (address steps pinned to MFMA
        # slots), it is not a no-op on the pad body
        self.assertNotEqual(_ir(tr), _ir(_PAD))

    def test_das_still_needs_hb_sched_and_hoist(self):
        for bad in (
            replace(_PAD, dtl_addr_spread=True, dtl_hoist=False),
            replace(
                _PAD,
                dtl_addr_spread=True,
                hb_sched=False,
                hb_read_every=1,
                ilv_merge=True,
            ),
        ):
            ok, why = _ok(bad)
            self.assertFalse(ok)
            self.assertIn("dtl_addr_spread", why)


class TightRaggedOnlyTest(unittest.TestCase):
    def test_default_off_and_name_flag(self):
        self.assertFalse(TraitSpec().dtl_tight_ragged_only)
        self.assertEqual(_ir(_PAD), _ir(replace(_PAD, dtl_tight_ragged_only=False)))
        self.assertNotIn("_tro", _name(_PAD))
        self.assertIn("_tro", _name(replace(_PAD, dtl_tight_ragged_only=True)))

    def test_scope(self):
        for tr in (
            replace(_PAD, dtl_tight_ragged_only=True),
            replace(_PAD, dtl_tight_ragged_only=True, dtl_addr_spread=True),
            replace(_PAD, dtl_tight_ragged_only=True, pad_m=True),
        ):
            ok, why = _ok(tr)
            self.assertTrue(ok, why)
        for bad in (
            replace(_HB, dtl_tight_ragged_only=True),  # no tight bounds
            replace(_PAD, dtl_tight_ragged_only=True, dtl_hoist=False),
            replace(
                _PAD,
                dtl_tight_ragged_only=True,
                pad_n=False,
                rve_pad=0,
                pad_fast_tile=False,
            ),
        ):
            ok, why = _ok(bad)
            self.assertFalse(ok, bad)
            self.assertIn("dtl_tight_ragged_only", why)

    def test_fewer_bound_computations(self):
        # pad_n only: A (unpadded) drops its bytes-left smax / select, so the
        # IR has fewer smax calls than with exact bounds on both operands
        both = _ir(_PAD)
        one = _ir(replace(_PAD, dtl_tight_ragged_only=True))
        self.assertLess(one.count("llvm.smax.i32"), both.count("llvm.smax.i32"))
        # pad_m + pad_n: both operands stay exact -> same IR as without the knob
        pmn = replace(_PAD, pad_m=True)
        a = _ir(pmn).replace(_name(pmn), "K")
        tro = replace(pmn, dtl_tight_ragged_only=True)
        self.assertEqual(a, _ir(tro).replace(_name(tro), "K"))


class KStaggerTest(unittest.TestCase):
    def test_default_off_and_name_flag(self):
        self.assertEqual(TraitSpec().k_stagger_rc, 0)
        self.assertEqual(_ir(_HB), _ir(replace(_HB, k_stagger_rc=0)))
        self.assertNotIn("_ksr", _name(_HB))
        self.assertIn("_ksr16", _name(replace(_HB, k_stagger_rc=16)))

    def test_scope(self):
        for tr in (
            replace(_HB, k_stagger_rc=2),
            replace(_HB, k_stagger_rc=64),
            replace(_PAD, k_stagger_rc=16, dtl_addr_spread=True),
            replace(
                _HB,
                k_stagger_rc=8,
                persistent_ctas=256,
                chiplet_wgm=8,
                chiplet_chunk_size=32,
            ),
        ):
            ok, why = _ok(tr)
            self.assertTrue(ok, why)
        for bad in (
            replace(_HB, k_stagger_rc=3),
            replace(_HB, k_stagger_rc=128),
            replace(_HB, k_stagger_rc=16, streamk=True, persistent_ctas=256),
            replace(_HB, k_stagger_rc=16, hb_read_every=1, asm_loop="tensile"),
        ):
            ok, why = _ok(bad)
            self.assertFalse(ok, bad)
            self.assertIn("k_stagger_rc", why)

    def test_emission_wraps_k(self):
        ir0 = _ir(_HB)
        ir1 = _ir(replace(_HB, k_stagger_rc=16))
        self.assertNotEqual(ir0, ir1)

        # the per-tile stagger ((r + c) mod 16 mod T: two srem) and the per-DTL
        # group wrap (compare + select) exist
        def ops(ir, op):
            return len(re.findall(r"= " + op + r" ", ir))

        self.assertEqual(ops(ir1, "srem"), ops(ir0, "srem") + 2)
        self.assertGreater(ops(ir1, "select"), ops(ir0, "select"))
        self.assertGreater(ops(ir1, "icmp"), ops(ir0, "icmp"))


class DispatchTest(unittest.TestCase):
    def test_pad_bodies(self):
        from rocke.dispatch import dispatch_gemm_bf16, dispatch_gemm_fp16
        from rocke.dispatch.gemm import GemmRequest
        from rocke.dispatch.gemm import common as C

        for fn, dt in ((dispatch_gemm_fp16, "fp16"), (dispatch_gemm_bf16, "bf16")):
            # long-K pad class: the tuned body on the wgm 32 / chunk 2 grid
            r = fn(GemmRequest(M=30720, N=20160, K=7680, arch="gfx950", dtype=dt))
            tr = r.spec.trait
            self.assertTrue(is_valid_spec(r.spec, arch="gfx950")[0])
            self.assertFalse(tr.sia_sched)
            self.assertTrue(tr.hb_sched and tr.dtl_addr_spread)
            self.assertTrue(tr.dtl_tight_rsrc and tr.dtl_tight_ragged_only)
            self.assertEqual((tr.kloop_unroll, tr.hb_skew, tr.hb_dtl_delay), (4, 1, 1))
            self.assertEqual(
                (tr.chiplet_wgm, tr.chiplet_chunk_size), tuple(C.RILV_PAD_HB_GRID)
            )
            self.assertEqual(tr.persistent_ctas, 0)
            # persistent pad class: the tuned-body levers, no dtl_soff0
            for (M, N, K), ku in (
                ((7680, 6720, 1536), 2),
                ((15360, 13440, 1536), 2),
                ((38400, 30240, 3072), 4),
            ):
                with self.subTest(dtype=dt, shape=(M, N, K)):
                    r = fn(GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt))
                    tr = r.spec.trait
                    self.assertTrue(is_valid_spec(r.spec, arch="gfx950")[0])
                    self.assertEqual(tr.persistent_ctas, 256)
                    self.assertTrue(tr.hb_sched and tr.dtl_tight_rsrc)
                    self.assertTrue(tr.dtl_addr_spread and tr.dtl_tight_ragged_only)
                    self.assertFalse(tr.dtl_soff0)
                    self.assertEqual(tr.kloop_unroll, ku)
                    self.assertEqual((tr.hb_dtl_delay, tr.hb_skew), (1, 1))

    def test_round2_bodies_selectable(self):
        from rocke.dispatch import dispatch_gemm_bf16
        from rocke.dispatch.gemm import GemmRequest
        from rocke.dispatch.gemm import common as C

        old = (C.RILV_PAD_HB_GRID, C.RILV_PAD_PERSIST_TUNED, C.RILV_PAD_HB_TUNED)
        try:
            C.RILV_PAD_HB_GRID, C.RILV_PAD_PERSIST_TUNED = (), False
            C.RILV_PAD_HB_TUNED = False
            r = dispatch_gemm_bf16(
                GemmRequest(M=30720, N=20160, K=7680, arch="gfx950", dtype="bf16")
            )
            self.assertEqual(
                (r.spec.trait.chiplet_wgm, r.spec.trait.chiplet_chunk_size), (16, 8)
            )
            self.assertTrue(r.spec.trait.sia_sched)
            r = dispatch_gemm_bf16(
                GemmRequest(M=38400, N=30240, K=3072, arch="gfx950", dtype="bf16")
            )
            self.assertFalse(r.spec.trait.dtl_addr_spread)
            self.assertEqual(r.spec.trait.kloop_unroll, 1)
        finally:
            C.RILV_PAD_HB_GRID, C.RILV_PAD_PERSIST_TUNED, C.RILV_PAD_HB_TUNED = old


class PythonOnlyTest(unittest.TestCase):
    def test_engine_refuses_round3_levers(self):
        for k, off in (("dtl_tight_ragged_only", False), ("k_stagger_rc", 0)):
            self.assertIn((k, off), _be._PYTHON_ONLY_GEMM_LEVERS)
        for kw in (dict(dtl_tight_ragged_only=True), dict(k_stagger_rc=16)):
            with self.assertRaises(NotImplementedError) as cm:
                _be._refuse_python_only_levers(replace(TraitSpec(), **kw))
            self.assertIn(next(iter(kw)), str(cm.exception))


if __name__ == "__main__":
    unittest.main()
