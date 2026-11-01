# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the universal-GEMM hand-ordered main loop levers
(``asm_loop`` with its event tables, ``asm_loop_prio``, ``asm_loop_skew``)
and ``fr0_loop_order``.

They cover the event-table checker (every built-in table is hazard-free,
broken tables are rejected), validity (each lever only on the body it is
written for), kernel-name flags, the emitted K-loop structure in the LLVM IR
(barrier / wait counts per K tile, the second loop copy and the SIMD-id read
of the skew), the C++ converter guard, and that every lever is default-off.
"""

from __future__ import annotations

import re
import unittest
from dataclasses import replace

from rocke.core.backend import universal_gemm_spec_to_dict
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common import gemm_asm_loop as A
from rocke.instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    mono_data_spec,
)

_W22X16 = TileSpec(256, 256, 64, 2, 2, 1, 16, 16, 32)
_W22X32 = TileSpec(256, 256, 64, 2, 2, 1, 32, 32, 16)
_HB = TraitSpec(
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
    hb_sched=True,
    rilv_vec_epilogue=True,
    rve_store_nt=True,
)
_ASMT = replace(_HB, asm_loop="tensile")


def _spec(trait, tile=_W22X16, dtype="fp16"):
    return UniversalGemmSpec(
        name="t", tile=tile, trait=trait, data=mono_data_spec(dtype), wave_size=64
    )


def _ir(spec):
    return lower_kernel_to_llvm(
        build_universal_gemm(spec, arch="gfx950"), arch="gfx950"
    )


class TestTables(unittest.TestCase):
    def test_builtin_tables_hazard_free(self):
        for name, text in A.BUILTIN_TABLES.items():
            self.assertIsNone(A.check_table(A.parse_table(text)), name)

    def test_skew_pairs(self):
        for mode in ("tensile", "custom", "hbr"):
            tr = replace(_HB, asm_loop=mode, asm_loop_skew=1)
            self.assertIsNone(A.check_skew(tr), mode)

    def test_tensile_matches_disassembly(self):
        # the Tensile SIA3 loop order: 4 barriers, acquires vmcnt(18) / vmcnt(15)
        evs = A.parse_table(A.BUILTIN_TABLES["tensile"])
        self.assertEqual(A.barrier_signature(evs), ["bar"] * 4)
        k = {(kind, op): kk for kk, kind, op, _i in evs if kind == "v"}
        self.assertEqual(k, {("v", "B"): 68, ("v", "A"): 105})

    def test_broken_tables_rejected(self):
        good = A.BUILTIN_TABLES["tensile"].split()
        cases = {
            "missing read": [t for t in good if t != "1/rB1.0"],
            "duplicate": good + ["2/rB1.0"],
            "DTL before release": [t if t != "23/dB.0" else "19/dB.0" for t in good],
            "read before acquire": [t if t != "70/rB0.0" else "67/rB0.0" for t in good],
            "read after release": [t if t != "43/rA1.7" else "53/rA1.7" for t in good],
            "late second-atom read": [
                t if t != "1/rB1.0" else "65/rB1.0" for t in good
            ],
        }
        for why, toks in cases.items():
            self.assertIsNotNone(A.check_table(A.parse_table(" ".join(toks))), why)
        with self.assertRaises(ValueError):
            A.parse_table("3/zz.1")

    def test_skew_mismatch_rejected(self):
        odd = A.BUILTIN_TABLES["custom"]  # 3 barriers vs tensile's 4
        tr = replace(
            _HB,
            asm_loop="table",
            asm_loop_table=A.BUILTIN_TABLES["tensile"] + " || " + odd,
            asm_loop_skew=2,
        )
        self.assertIsNotNone(A.check_skew(tr))
        self.assertFalse(is_valid_spec(_spec(tr), "gfx950")[0])


class TestValidity(unittest.TestCase):
    def test_bodies_valid(self):
        for tr in (
            _ASMT,
            replace(_HB, asm_loop="custom"),
            replace(_HB, asm_loop="hbr"),
            replace(_ASMT, asm_loop_prio=3),
            replace(_ASMT, asm_loop_skew=1),
            replace(_HB, asm_loop="custom", asm_loop_skew=2),
            replace(_ASMT, persistent_ctas=256),
            replace(_HB, asm_loop="table", asm_loop_table=A.BUILTIN_TABLES["custom"]),
            replace(_HB, fr0_loop_order=True),
        ):
            for dt in ("fp16", "bf16"):
                ok, why = is_valid_spec(_spec(tr, dtype=dt), "gfx950")
                self.assertTrue(ok, why)

    def test_rejections(self):
        bad = [
            (replace(_ASMT, hb_sched=False), _W22X16),
            (
                replace(
                    _ASMT,
                    lds_row_interleave=False,
                    rilv_vec_epilogue=False,
                    rve_store_nt=False,
                ),
                _W22X16,
            ),
            (replace(_ASMT, hb_read_every=2), _W22X16),
            (replace(_ASMT, hb_split_rel=True), _W22X16),
            (_ASMT, _W22X32),
            (replace(_ASMT, asm_loop="nope"), _W22X16),
            (replace(_ASMT, asm_loop_prio=4), _W22X16),
            (replace(_ASMT, asm_loop_skew=3), _W22X16),
            (replace(_ASMT, interwave_setprio=True), _W22X16),
            (replace(_ASMT, asm_loop_table="1/wl"), _W22X16),
            (replace(_HB, asm_loop="table", asm_loop_table="1/wl"), _W22X16),
            (replace(_HB, asm_loop_prio=1), _W22X16),
            (replace(_HB, asm_loop_skew=1), _W22X16),
            (replace(_ASMT, fr0_loop_order=True), _W22X16),
        ]
        for tr, tile in bad:
            ok, _why = is_valid_spec(_spec(tr, tile=tile), "gfx950")
            self.assertFalse(ok, tr)

    def test_python_only_guard(self):
        for tr in (_ASMT, replace(_HB, fr0_loop_order=True)):
            with self.assertRaises(NotImplementedError):
                universal_gemm_spec_to_dict(_spec(tr))


class TestEmission(unittest.TestCase):
    def test_default_off(self):
        tr = TraitSpec()
        self.assertEqual(
            (tr.asm_loop, tr.asm_loop_table, tr.asm_loop_prio, tr.asm_loop_skew),
            ("", "", 0, 0),
        )
        self.assertFalse(tr.fr0_loop_order)
        # spelled-out defaults emit the same kernel as the plain HB body
        a = _ir(_spec(_HB))
        b = _ir(_spec(replace(_HB, asm_loop="", asm_loop_prio=0, asm_loop_skew=0)))
        self.assertEqual(a, b)

    def test_name_flags(self):
        self.assertTrue(_spec(_ASMT).kernel_name().endswith("_asmtensile"))
        n = _spec(replace(_ASMT, asm_loop_prio=3, asm_loop_skew=1)).kernel_name()
        self.assertTrue(n.endswith("_asmtensile_ap3_ask1"), n)
        t = replace(_HB, asm_loop="table", asm_loop_table=A.BUILTIN_TABLES["custom"])
        self.assertRegex(_spec(t).kernel_name(), r"_asmt[0-9a-f]{8}$")
        self.assertTrue(
            _spec(replace(_HB, fr0_loop_order=True)).kernel_name().endswith("_f0o")
        )
        self.assertNotIn("asm", _spec(_HB).kernel_name())

    def test_loop_structure(self):
        ir = _ir(_spec(_ASMT))
        # tensile: 4 barriers per K tile, 2 tiles per main-loop iteration
        self.assertIn("asmtensile", ir)
        bars = len(re.findall(r"call void @llvm\.amdgcn\.s\.barrier\(\)", ir))
        self.assertGreaterEqual(bars, 8)
        # the per-operand acquires compile to the vmcnt(18) / vmcnt(15)
        # (gfx9 s_waitcnt encoding, lgkmcnt / expcnt fields at "no wait")
        w = re.findall(r"llvm\.amdgcn\.s\.waitcnt\(i32 (\d+)\)", ir)
        vm18 = 2 | (7 << 4) | (15 << 8) | (1 << 14)
        vm15 = 15 | (7 << 4) | (15 << 8)
        self.assertGreaterEqual(w.count(str(vm18)), 2)
        self.assertGreaterEqual(w.count(str(vm15)), 2)
        # DTL issue order: every 16-load group (a prologue tile / a K tile) is
        # 8 loads of one operand (the inner B, released first) then 8 of the other
        rs = re.findall(
            r"call void @llvm\.amdgcn\.raw\.ptr\.buffer\.load\.lds\(ptr addrspace\(8\) (%[\w.]+)",
            ir,
        )
        self.assertEqual(len(rs) % 16, 0)
        for g in range(0, len(rs), 16):
            grp = rs[g : g + 16]
            self.assertEqual(len(set(grp[:8])), 1, grp)
            self.assertEqual(len(set(grp[8:])), 1, grp)
            self.assertNotEqual(grp[0], grp[8])
        sk = _ir(_spec(replace(_ASMT, asm_loop_skew=1)))
        self.assertIn("hwreg(HW_REG_HW_ID, 4, 1)", sk)
        self.assertGreater(
            len(re.findall(r"llvm\.amdgcn\.mfma", sk)),
            len(re.findall(r"llvm\.amdgcn\.mfma", ir)),
        )
        sk2 = _ir(_spec(replace(_ASMT, asm_loop_skew=2)))
        self.assertNotIn("HW_REG_HW_ID", sk2)


if __name__ == "__main__":
    unittest.main()
