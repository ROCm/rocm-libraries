# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the instruction-economy levers.

* ``hb_tile_lgkm0``: one explicit lgkmcnt(0) at the end of every hb_sched
  tile (and after the pre-loop fragments), so LLVM drops its per-MFMA waits.
* ``dtl_base_fold``: per-operand base pointer at the output tile origin.
* ``hb_skew_dtl``: DTL-issue shift of the odd hb_skew loop copy.
* ``das_anchor`` / ``das_late``: placement of the dtl_addr_spread steps.
* ``k_stagger`` / ``k_stagger_tiles`` / ``k_stagger_map``: StaggerU-style
  (per-CTA K start offset with wrap-around).
* ``dtl_clamp_k``: constant num_records; prefetches past K re-load the last
  tile.

Default-off (spelled-out defaults emit the shipped IR), unique kernel-name
flag, rejected outside its scope, refused by the C-engine converters.
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
# the dispatched tuned body (long-K class)
_D = replace(
    _HB,
    dtl_soff0=True,
    dtl_addr_spread=True,
    hb_dtl_delay=1,
    kloop_unroll=4,
    hb_skew=1,
)

_KNOBS = dict(
    hb_tile_lgkm0=(0, 1, "htl1"),
    dtl_base_fold=(False, True, "dbf"),
    hb_skew_dtl=(0, 3, "hsd3"),
    das_anchor=(False, True, "dsa"),
    das_late=(False, True, "dsl"),
    k_stagger=(0, 8, "kst8"),
    dtl_clamp_k=(False, True, "dck"),
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


class DefaultsTest(unittest.TestCase):
    def test_default_off_and_spelled_out_identical(self):
        base = _ir(_D)
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
        )
        self.assertEqual(base, _ir(spelled))
        for k, (off, _on, _flag) in _KNOBS.items():
            self.assertEqual(getattr(TraitSpec(), k), off, k)
        self.assertEqual(TraitSpec().k_stagger_tiles, 0)
        self.assertEqual(TraitSpec().k_stagger_map, 0)

    def test_name_flags_unique(self):
        n0 = _name(_D)
        names = {n0}
        for k, (_off, on, flag) in _KNOBS.items():
            self.assertNotIn("_" + flag, n0)
            n = _name(replace(_D, **{k: on}))
            self.assertIn("_" + flag, n, k)
            names.add(n)
        n = _name(replace(_D, k_stagger=8, k_stagger_tiles=2))
        self.assertIn("_kss2", n)
        names.add(n)
        n = _name(replace(_D, k_stagger=8, k_stagger_map=1))
        self.assertIn("_ksm1", n)
        names.add(n)
        self.assertEqual(len(names), len(_KNOBS) + 3)


class ScopeTest(unittest.TestCase):
    def test_valid_on_dispatched_body(self):
        for k, (_off, on, _flag) in _KNOBS.items():
            ok, why = _ok(replace(_D, **{k: on}))
            self.assertTrue(ok, (k, why))
        for tr in (
            replace(_D, hb_tile_lgkm0=1, das_late=True, dtl_base_fold=True),
            replace(_D, hb_tile_lgkm0=1, das_late=True, k_stagger=8),
            replace(_D, k_stagger=32, k_stagger_tiles=2),
            replace(_D, k_stagger=8, k_stagger_map=1),
            replace(
                _D, hb_tile_lgkm0=1, das_late=True, dtl_clamp_k=True, dtl_base_fold=True
            ),
            replace(
                _D,
                persistent_ctas=256,
                chiplet_wgm=8,
                chiplet_chunk_size=32,
                kloop_unroll=2,
                hb_tile_lgkm0=1,
                das_late=True,
                k_stagger=8,
            ),
        ):
            ok, why = _ok(tr)
            self.assertTrue(ok, why)

    def test_rejections(self):
        asm = replace(_HB, hb_read_every=1, asm_loop="tensile", asm_loop_skew=1)
        cases = (
            (replace(_D, hb_tile_lgkm0=2), "hb_tile_lgkm0"),
            (replace(asm, hb_tile_lgkm0=1), "hb_tile_lgkm0"),
            (replace(_D, dtl_base_fold=True, dtl_soff0=False), "dtl_base_fold"),
            (
                replace(_D, dtl_base_fold=True, persistent_ctas=256),
                "dtl_base_fold",
            ),
            (replace(_D, hb_skew_dtl=3, hb_skew=0), "hb_skew_dtl"),
            (replace(_D, hb_skew_dtl=9), "hb_skew_dtl"),
            (replace(_D, das_anchor=True, dtl_addr_spread=False), "das_anchor"),
            (replace(_D, das_late=True, dtl_addr_spread=False), "das_late"),
            (replace(_D, k_stagger=3), "k_stagger"),
            (replace(_D, k_stagger=8, pad_k=True), "k_stagger"),
            (replace(_D, k_stagger_tiles=2), "k_stagger_tiles"),
            (replace(_D, k_stagger_map=1), "k_stagger_map"),
            (replace(_D, k_stagger=8, k_stagger_map=2), "k_stagger_map"),
            (replace(_D, dtl_clamp_k=True, dtl_soff0=False), "dtl_clamp_k"),
            (replace(_D, dtl_clamp_k=True, persistent_ctas=256), "dtl_clamp_k"),
        )
        for tr, key in cases:
            ok, why = _ok(tr)
            self.assertFalse(ok, key)
            self.assertIn(key, why)
        # split-K is rejected too (here already by the vector epilogue's rule)
        self.assertFalse(_ok(replace(_D, k_stagger=8, split_k=2))[0])


class EmissionTest(unittest.TestCase):
    def test_hb_tile_lgkm0_adds_one_wait_per_tile(self):
        w = r"llvm\.amdgcn\.s\.waitcnt"
        n0 = len(re.findall(w, _ir(_D)))
        n1 = len(re.findall(w, _ir(replace(_D, hb_tile_lgkm0=1))))
        self.assertGreater(n1, n0)

    def test_levers_change_ir_but_not_mfma_count(self):
        ir0 = _ir(_D)
        m0 = ir0.count("llvm.amdgcn.mfma")
        for k, (_off, on, _flag) in _KNOBS.items():
            ir1 = _ir(replace(_D, **{k: on}))
            self.assertNotEqual(ir0, ir1, k)
            self.assertEqual(ir1.count("llvm.amdgcn.mfma"), m0, k)

    def test_das_late_reorders_only(self):
        # the same calls in another order (the steps move, nothing is added)
        def calls(t):
            return re.findall(r"call [^@]*@(llvm\.[A-Za-z0-9_.]+)", t)

        ir0, ir1 = _ir(_D), _ir(replace(_D, das_late=True))
        self.assertEqual(sorted(calls(ir0)), sorted(calls(ir1)))
        self.assertNotEqual(ir0, ir1)

    def test_k_stagger_values_differ(self):
        self.assertNotEqual(
            _ir(replace(_D, k_stagger=8)), _ir(replace(_D, k_stagger=16))
        )
        self.assertNotEqual(
            _ir(replace(_D, k_stagger=8)),
            _ir(replace(_D, k_stagger=8, k_stagger_tiles=2)),
        )


# mirrored by the C++ engine (parity: tests/instances/parity/gemm_hb_emit)
_MIRRORED = ("hb_tile_lgkm0", "hb_skew_dtl", "das_anchor", "das_late")


class PythonOnlyTest(unittest.TestCase):
    def test_engine_receives_mirrored(self):
        keys = {k for k, _ in _be._PYTHON_ONLY_GEMM_LEVERS}
        for k in _MIRRORED:
            self.assertNotIn(k, keys)
            _off, on, _flag = _KNOBS[k]
            _be._refuse_python_only_levers(replace(TraitSpec(), **{k: on}))

    def test_engine_refuses(self):
        for k, (off, on, _flag) in _KNOBS.items():
            if k in _MIRRORED:
                continue
            self.assertIn((k, off), _be._PYTHON_ONLY_GEMM_LEVERS)
            with self.assertRaises(NotImplementedError) as cm:
                _be._refuse_python_only_levers(replace(TraitSpec(), **{k: on}))
            self.assertIn(k, str(cm.exception))
        self.assertIn(("k_stagger_tiles", 0), _be._PYTHON_ONLY_GEMM_LEVERS)
        self.assertIn(("k_stagger_map", 0), _be._PYTHON_ONLY_GEMM_LEVERS)
        _be._refuse_python_only_levers(TraitSpec())


if __name__ == "__main__":
    unittest.main()
