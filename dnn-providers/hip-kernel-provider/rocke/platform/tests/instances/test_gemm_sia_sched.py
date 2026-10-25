# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the universal-GEMM ``sia_sched`` family (Tensile-SIA3-style
K-tile schedule) and its sub-levers.

They cover validity (the family needs the rilv body and is its own schedule
family), the slot table invariants every sub-lever combination must keep
(each DTL issue after its operand's release, every read inside the tile, the
next-tile reads after their acquire, the last DTL issue of a tile belongs to
the second operand so the prologue order keeps the acquire counts exact),
kernel-name flags, the barrier / waitcnt structure in the LLVM IR, the C++
engine guard and that every lever is default-off.
"""

from __future__ import annotations

import itertools
import re
import unittest
from dataclasses import replace

from rocke.core.backend import universal_gemm_spec_to_dict
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    mono_data_spec,
    sia_acq_vmcnt,
    sia_table,
)

_W22X16 = TileSpec(256, 256, 64, 2, 2, 1, 16, 16, 32)
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
    lds_row_interleave=True,
    rilv_vec_epilogue=True,
    rve_store_nt=True,
)
_SIA = replace(_BASE, sia_sched=True)
_SUBS = (
    "sia_b_first",
    "sia_plr",
    "sia_single_lgkm",
    "sia_split_rel",
    "sia_split_acq",
    "sia_dtl_spread",
)
_FULL = replace(_SIA, **{k: True for k in _SUBS})
_MRG = replace(
    _SIA,
    sia_merge_sync=True,
    sia_use_order=True,
    sia_plr=True,
    sia_read_every=4,
    sia_acq_gap=-2,
)


def _spec(trait, tile=_W22X16, dtype="fp16"):
    return UniversalGemmSpec(
        name="t", tile=tile, trait=trait, data=mono_data_spec(dtype), wave_size=64
    )


def _ir(spec):
    return lower_kernel_to_llvm(
        build_universal_gemm(spec, arch="gfx950"), arch="gfx950"
    )


def _variants():
    for bits in itertools.product((False, True), repeat=len(_SUBS)):
        yield replace(_SIA, **dict(zip(_SUBS, bits)))
    for re_, rg, ag in ((1, 1, 0), (3, 1, 0), (4, 1, 0), (2, 6, 8)):
        yield replace(_FULL, sia_read_every=re_, sia_rel_gap=rg, sia_acq_gap=ag)
    yield replace(_FULL, sia_tensile=True)
    yield replace(_FULL, sia_tensile=True, sia_b_first=False)
    for ag in (-4, -1, 0, 3):
        yield replace(_MRG, sia_acq_gap=ag)
    yield replace(_MRG, sia_use_order=False, sia_b_first=True, sia_single_lgkm=True)


class TestValidity(unittest.TestCase):
    def test_bodies_valid(self):
        n = 0
        for tr in _variants():
            for dt in ("fp16", "bf16"):
                ok, why = is_valid_spec(_spec(tr, dtype=dt), "gfx950")
                self.assertTrue(ok, (why, tr))
            n += 1
        self.assertGreater(n, 64)

    def test_rejections(self):
        bad = [
            replace(
                _SIA,
                lds_row_interleave=False,
                rilv_vec_epilogue=False,
                rve_store_nt=False,
            ),
            replace(_SIA, ilv_merge=True),
            replace(_SIA, hb_sched=True),  # one schedule family
            replace(_SIA, iglp_opt=1),
            replace(_SIA, ilv_read_span=50),
            replace(_SIA, persistent_ctas=256, streamk=True),
            replace(_BASE, sia_b_first=True),  # sub-lever without the family
            replace(_BASE, sia_read_every=3),
            replace(_SIA, sia_read_every=0),
            replace(_SIA, sia_rel_gap=0),
            replace(_SIA, sia_acq_gap=-1),  # negative only with merge_sync
            replace(_MRG, sia_acq_gap=-9),
            replace(_MRG, sia_split_rel=True),
            replace(_SIA, sia_use_order=True, sia_split_acq=True),
            replace(_SIA, sia_tensile=True, sia_merge_sync=True),
        ]
        for tr in bad:
            ok, _ = is_valid_spec(_spec(tr), "gfx950")
            self.assertFalse(ok, tr)
        # gfx942 has no 16-byte DTL (rilv)
        self.assertFalse(is_valid_spec(_spec(_FULL), "gfx942")[0])


class TestTable(unittest.TestCase):
    def test_invariants(self):
        for tr in _variants():
            t = sia_table(_spec(tr))
            ntot, n = t["ntot"], t["n"]
            self.assertEqual((ntot, n), (128, 64))
            X, Y = t["order"]
            self.assertEqual(t["order"], "BA" if tr.sia_b_first else "AB")
            # every fragment of both atoms read exactly once, inside the tile
            for key in ("rd1", "rd0"):
                got = sorted((op, i) for _s, op, i in t[key])
                self.assertEqual(
                    got, [("A", i) for i in range(8)] + [("B", i) for i in range(8)]
                )
                self.assertTrue(all(0 <= s < ntot for s, _o, _i in t[key]), (key, tr))
            # next-atom reads finish in the first atom
            self.assertLess(max(s for s, _o, _i in t["rd1"]), n)
            rel = {op: s for s, ops in t["rel"] for op in ops}
            if t["merged"]:
                self.assertEqual(rel, {})
                rel = {op: t["acq"][0][0] for op in "AB"}
            # releases after the operand's last next-atom read
            for op in "AB":
                last = max(s for s, o, _i in t["rd1"] if o == op)
                self.assertGreater(rel[op], last, tr)
            # next-tile reads after their operand's acquire
            acq = {op: s for s, ops in t["acq"] for op in ops}
            for s, op, _i in t["rd0"]:
                self.assertGreaterEqual(s, acq[op], tr)
            # DTL issues: 8 + 8, after their operand's release, issue order
            # X then Y, the tile's last issue is a Y load
            dtl = t["dtl"]
            self.assertEqual(len(dtl), 16)
            self.assertEqual([o for _s, o, _p in dtl], [X] * 8 + [Y] * 8)
            for s, op, _p in dtl:
                self.assertGreater(s, rel[op], tr)
                self.assertLess(s, ntot)
            self.assertEqual(max(range(16), key=lambda j: (dtl[j][0], j)), 15)
            self.assertEqual(t["lgkm"], 0 if tr.sia_single_lgkm else -1)

    def test_tensile_table(self):
        t = sia_table(_spec(replace(_FULL, sia_tensile=True)))
        self.assertEqual(t["rel"], [(19, "B"), (50, "A")])
        self.assertEqual(t["acq"], [(67, "B"), (104, "A")])
        self.assertEqual([s for s, _o, _p in t["dtl"]][:5], [22, 25, 28, 31, 34])
        # the vmcnt immediates of the Tensile SIA3 acquires
        self.assertEqual(sia_acq_vmcnt(t), [18, 15])
        self.assertEqual(sia_acq_vmcnt(t, with_load=False), [8, 0])

    def test_acq_vmcnt(self):
        # computed tables: the first acquire leaves the previous tile's
        # second-operand loads and this tile's earlier issues in flight
        t = sia_table(_spec(_FULL))
        self.assertEqual(sia_acq_vmcnt(t), [8 + 8, 0 + 9])
        t = sia_table(_spec(_MRG))
        self.assertEqual(sia_acq_vmcnt(t), [0])  # the merged sync drains
        t = sia_table(_spec(replace(_SIA, sia_plr=True)))
        self.assertEqual(sia_acq_vmcnt(t), [sum(1 for s, _o, _p in t["dtl"] if s < 67)])


class TestEmission(unittest.TestCase):
    def test_default_off(self):
        name = _spec(_BASE).kernel_name()
        for flag in (
            "sia",
            "sbf",
            "splr",
            "slg",
            "ssr",
            "ssq",
            "sds",
            "stt",
            "suo",
            "sms",
        ):
            self.assertNotIn(f"_{flag}_", name + "_")
        self.assertEqual(_ir(_spec(_BASE)), _ir(_spec(replace(_BASE))))

    def test_names(self):
        n = _spec(replace(_FULL, sia_read_every=3, sia_acq_gap=5)).kernel_name()
        for flag in ("sia", "sbf", "splr", "slg", "ssr", "ssq", "sds", "sre3", "sag5"):
            self.assertIn(f"_{flag}", n)
        self.assertIn("_sms", _spec(_MRG).kernel_name())

    def test_barriers(self):
        # main loop is unrolled by the 2 stages: split release + split acquire
        # = 4 barriers per tile; one merged sync = 1 per tile
        full = _ir(_spec(_FULL))
        mrg = _ir(_spec(_MRG))
        self.assertGreaterEqual(full.count("@llvm.amdgcn.s.barrier"), 8)
        self.assertLess(
            mrg.count("@llvm.amdgcn.s.barrier"), full.count("@llvm.amdgcn.s.barrier")
        )

    def test_rilv_pad(self):
        # the 32-byte pad per KiB: 2 stages x (A + B) x 32 padded rows
        # of 1056 bytes = the 135168-byte pool of the Tensile kernels; the
        # default 16-byte pad keeps the historical 133120 bytes and name
        self.assertTrue("[133120 x i8]" in _ir(_spec(_BASE)))
        ir32 = _ir(_spec(replace(_BASE, rilv_pad=32)))
        self.assertTrue("[135168 x i8]" in ir32)
        # the DTL side writes the same layout: per-wave block 1024 + pad bytes,
        # per-pass stride 4 waves x that
        self.assertTrue(re.search(r"\D1056\D", ir32) and re.search(r"\D4224\D", ir32))
        self.assertFalse(re.search(r"\D4160\D", ir32))
        self.assertIn("_rp32", _spec(replace(_BASE, rilv_pad=32)).kernel_name())
        self.assertNotIn("_rp", _spec(_BASE).kernel_name())
        for pad in (32, 48, 64):
            self.assertTrue(
                is_valid_spec(_spec(replace(_FULL, rilv_pad=pad)), "gfx950")[0]
            )
        for tr in (
            replace(_BASE, rilv_pad=24),
            replace(_BASE, rilv_pad=0),
            replace(
                _BASE,
                lds_row_interleave=False,
                rilv_vec_epilogue=False,
                rve_store_nt=False,
                rilv_pad=32,
            ),
        ):
            self.assertFalse(is_valid_spec(_spec(tr), "gfx950")[0], tr)
        with self.assertRaises(NotImplementedError):
            universal_gemm_spec_to_dict(_spec(replace(_BASE, rilv_pad=32)))

    def test_engine_guard(self):
        for tr in (_FULL, _MRG, replace(_SIA, sia_tensile=True)):
            with self.assertRaises(NotImplementedError):
                universal_gemm_spec_to_dict(_spec(tr))


if __name__ == "__main__":
    unittest.main()
