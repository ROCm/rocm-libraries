# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the StaggerU and native asm-loop address levers.

* ``stagger_u`` / ``stagger_u_stride`` / ``stagger_u_map``: Tensile's StaggerU
  (each workgroup starts its K loop at its own K tile and wraps at K; only the
  DTL address moves).
* ``asm_loop`` + ``dtl_addr_spread`` / ``hb_dtl_delay`` natively: the address
  steps go to the table's free MFMA slots before its first DTL issue, the
  delay moves every DTL issue of the table later.

Default-off, unique kernel-name flags, rejected outside their scope, refused
by the C-engine converters (Python-engine only).
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
# the dispatched long-K body (tuned row-interleave body)
_TUNED = replace(
    _HB,
    dtl_soff0=True,
    dtl_addr_spread=True,
    hb_dtl_delay=1,
    kloop_unroll=4,
    hb_skew=1,
)
_ASM = replace(
    _HB, hb_read_every=1, asm_loop="tensile", asm_loop_skew=1, dtl_soff0=True
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


def _first_loop_body(ir):
    blocks = re.split(r"^([^\s:;]+):", ir, flags=re.M)
    for i in range(1, len(blocks) - 1, 2):
        if blocks[i].startswith("for.body"):
            return blocks[i + 1]
    raise AssertionError("no for.body block")


class StaggerUTest(unittest.TestCase):
    def test_default_off_and_spelled_out_identical(self):
        tr = TraitSpec()
        self.assertEqual(
            (tr.stagger_u, tr.stagger_u_stride, tr.stagger_u_map), (0, 1, 0)
        )
        self.assertEqual(
            _ir(_TUNED),
            _ir(replace(_TUNED, stagger_u=0, stagger_u_stride=1, stagger_u_map=0)),
        )

    def test_name_flag(self):
        self.assertNotIn("_su", _name(_TUNED))
        self.assertTrue(
            _name(replace(_TUNED, stagger_u=32, stagger_u_stride=2)).endswith(
                "_su32s2m0"
            )
        )
        self.assertIn("_su8s1m2", _name(replace(_TUNED, stagger_u=8, stagger_u_map=2)))

    def test_scope(self):
        for good in (
            replace(_TUNED, stagger_u=32, stagger_u_stride=2),
            replace(_TUNED, stagger_u=2, stagger_u_map=1),
            replace(_TUNED, stagger_u=64, stagger_u_map=2),
            replace(_ASM, stagger_u=16, stagger_u_stride=4),
        ):
            ok, why = _ok(good)
            self.assertTrue(ok, why)
        for bad in (
            replace(_TUNED, stagger_u=3),
            replace(_TUNED, stagger_u=128),
            replace(_TUNED, stagger_u_stride=2),  # stride without stagger
            replace(_TUNED, stagger_u_map=1),  # map without stagger
            replace(_TUNED, stagger_u=8, stagger_u_stride=0),
            replace(_TUNED, stagger_u=8, stagger_u_stride=65),
            replace(_TUNED, stagger_u=8, stagger_u_map=3),
            replace(_TUNED, stagger_u=8, stagger_u_map=2, chiplet_swizzle=False),
            replace(
                _TUNED,
                stagger_u=8,
                persistent_ctas=256,
                chiplet_wgm=8,
                chiplet_chunk_size=32,
            ),
        ):
            ok, why = _ok(bad)
            self.assertFalse(ok, bad)
            self.assertIn("stagger_u", why)
        # split-K (rejected by rve already; stagger has its own rule too)
        self.assertFalse(_ok(replace(_TUNED, stagger_u=8, split_k=2))[0])

    def test_emission_wraps_dtl_address_only(self):
        ir0 = _ir(_TUNED)
        ir1 = _ir(replace(_TUNED, stagger_u=32, stagger_u_stride=2))
        # same instruction mix in the main loop: MFMAs, DTLs, barriers, reads
        b0, b1 = _first_loop_body(ir0), _first_loop_body(ir1)
        for pat in (
            "llvm.amdgcn.mfma",
            "llvm.amdgcn.raw.ptr.buffer.load.lds",
            "llvm.amdgcn.s.barrier",
            "load <",
        ):
            self.assertEqual(b0.count(pat), b1.count(pat), pat)
        # the wrap: one compare-against-K select more per DTL tile address
        self.assertGreater(b1.count("select i1"), b0.count("select i1"))
        self.assertNotEqual(ir0, ir1)
        # the K-start offset uses the tile row index modulo SU (urem / and)
        self.assertNotEqual(_ir(replace(_TUNED, stagger_u=32, stagger_u_map=1)), ir1)


class AsmNativeSpreadDelayTest(unittest.TestCase):
    def test_round2_asm_body_unchanged(self):
        # without the two knobs the asm body is the plain asm-loop body
        self.assertEqual(
            _ir(_ASM), _ir(replace(_ASM, dtl_addr_spread=False, hb_dtl_delay=0))
        )

    def test_valid_and_distinct(self):
        irs = set()
        for kw in (
            {},
            dict(dtl_addr_spread=True),
            dict(hb_dtl_delay=1),
            dict(dtl_addr_spread=True, hb_dtl_delay=2),
            dict(dtl_addr_spread=True, hb_dtl_delay=1, kloop_unroll=4),
            dict(dtl_addr_spread=True, hb_dtl_delay=1, asm_loop="custom"),
        ):
            tr = replace(_ASM, **kw)
            ok, why = _ok(tr)
            self.assertTrue(ok, why)
            ir = _ir(tr)
            self.assertIn("llvm.amdgcn.mfma", ir)
            irs.add(ir)
        self.assertEqual(len(irs), 6)

    def test_delay_moves_dtl_issues_later(self):
        # MFMA index of each DTL issue in the first loop body
        def dtl_slots(tr):
            body = _first_loop_body(_ir(tr))
            n, out = 0, []
            for ln in body.splitlines():
                if "llvm.amdgcn.mfma" in ln:
                    n += 1
                elif "buffer.load.lds" in ln:
                    out.append(n)
            return out

        s0 = dtl_slots(_ASM)
        s1 = dtl_slots(replace(_ASM, hb_dtl_delay=1))
        self.assertEqual(len(s0), len(s1))
        self.assertGreater(len(s0), 0)
        # each issue one MFMA later (an issue already behind the tile's last
        # MFMA stays there)
        self.assertTrue(all(b - a in (0, 1) for a, b in zip(s0, s1)), (s0, s1))
        self.assertGreater(sum(s1), sum(s0))

    def test_spread_keeps_dtl_slots(self):
        # dtl_addr_spread only moves the address math, not the DTL issues
        def dtl_slots(tr):
            body = _first_loop_body(_ir(tr))
            n, out = 0, []
            for ln in body.splitlines():
                if "llvm.amdgcn.mfma" in ln:
                    n += 1
                elif "buffer.load.lds" in ln:
                    out.append(n)
            return out

        self.assertEqual(
            dtl_slots(_ASM), dtl_slots(replace(_ASM, dtl_addr_spread=True))
        )

    def test_spread_builds_resource_before_first_dtl(self):
        # the B resource of the first DTL issue is built in an earlier MFMA
        # slot (pinned address steps), not right in front of the issue
        def first_rsrc_and_dtl(tr):
            body = _first_loop_body(_ir(tr))
            n, defs, first = 0, {}, None
            for ln in body.splitlines():
                if "llvm.amdgcn.mfma" in ln:
                    n += 1
                m = re.match(r"\s*(%[\w.]+) = ", ln)
                if m:
                    defs.setdefault(m.group(1), n)
                if first is None and "buffer.load.lds" in ln:
                    first = (re.search(r"ptr addrspace\(8\) (%[\w.]+)", ln).group(1), n)
            return defs.get(first[0], -1), first[1]

        d0, i0 = first_rsrc_and_dtl(_ASM)
        d1, i1 = first_rsrc_and_dtl(replace(_ASM, dtl_addr_spread=True))
        self.assertEqual(i0, i1)
        self.assertGreaterEqual(d1, 0)
        self.assertLess(d1, i1)


class SerpentineTest(unittest.TestCase):
    """``mfma_serpentine``: odd A rows of each K atom walk the B fragments
    backwards (same MFMAs, same event slots, different issue order)."""

    @staticmethod
    def _mfma_args(tr):
        body = _first_loop_body(_ir(tr))
        return [
            ln.split("llvm.amdgcn.mfma", 1)[1]
            for ln in body.splitlines()
            if "llvm.amdgcn.mfma" in ln
        ]

    def test_default_off_identical_and_flag(self):
        self.assertFalse(TraitSpec().mfma_serpentine)
        self.assertEqual(_ir(_TUNED), _ir(replace(_TUNED, mfma_serpentine=False)))
        self.assertTrue(_name(replace(_TUNED, mfma_serpentine=True)).endswith("_msp"))
        self.assertNotIn("_msp", _name(_TUNED))

    def test_scope(self):
        for good in (
            replace(_TUNED, mfma_serpentine=True),
            replace(_ASM, mfma_serpentine=True),
        ):
            self.assertTrue(_ok(good)[0], _ok(good)[1])
        ok, why = _ok(
            replace(
                _TUNED,
                hb_sched=False,
                hb_read_every=1,
                hb_skew=0,
                dtl_addr_spread=False,
                hb_dtl_delay=0,
                mfma_interleave=False,
                dtl_frag_prefetch=False,
                mfma_serpentine=True,
            )
        )
        # no hb_sched / sia_sched schedule and no mfma_interleave atom emitter
        self.assertFalse(ok)
        self.assertIn("mfma_serpentine", why)

    def test_same_mfmas_reordered(self):
        # the (A fragment, B fragment) operand pairs: same multiset, other order
        # (accumulator SSA names follow creation order, so they are not compared)
        def pairs(tr):
            return [tuple(re.findall(r"(%[\w.]+)", a)[:2]) for a in self._mfma_args(tr)]

        for base in (_TUNED, _ASM):
            p0 = pairs(base)
            p1 = pairs(replace(base, mfma_serpentine=True))
            self.assertEqual(len(p0), len(p1))
            self.assertEqual(sorted(p0), sorted(p1))
            self.assertNotEqual(p0, p1)

    def test_no_double_operand_switch(self):
        # count consecutive MFMA pairs whose B operand (2nd value arg) differs
        # AND whose A operand differs: 7 per K atom without, 0 with serpentine
        def doubles(tr):
            args = [re.findall(r"(%[\w.]+)", a)[:2] for a in self._mfma_args(tr)]
            n = 0
            for (a0, b0), (a1, b1) in zip(args, args[1:]):
                n += a0 != a1 and b0 != b1
            return n

        n_atoms = len(self._mfma_args(_TUNED)) // 64
        self.assertGreater(doubles(_TUNED), 0)
        # only the atom boundaries (new fragments) switch both operands
        self.assertLessEqual(doubles(replace(_TUNED, mfma_serpentine=True)), n_atoms)
        self.assertGreater(doubles(_TUNED), 2 * n_atoms)


class DispatchKnobTest(unittest.TestCase):
    """``RILV_HB_LOOP``: default "ir" (the tuned row-interleave body, unchanged);
    "asm_loop" swaps in the Tensile asm table on the same grid."""

    def _pick(self, M, N, K, dt="bf16"):
        from rocke.dispatch.gemm import (
            GemmRequest,
            dispatch_gemm_bf16,
            dispatch_gemm_fp16,
        )

        f = dispatch_gemm_bf16 if dt == "bf16" else dispatch_gemm_fp16
        return f(GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt)).spec

    def test_default_and_asm_loop(self):
        from rocke.dispatch.gemm import common as C

        self.assertEqual(C.RILV_HB_LOOP, "ir")
        self.assertEqual(C.RILV_ASM_LOOP_STAGGER_U, 0)
        x = self._pick(8192, 8192, 8192)
        xp = self._pick(15360, 15360, 768)
        self.assertEqual(x.trait.asm_loop, "")
        self.assertEqual(x.trait.hb_skew, 1)
        # the serpentine MFMA order on both shipped classes
        self.assertTrue(C.RILV_SERPENTINE)
        self.assertTrue(x.trait.mfma_serpentine and xp.trait.mfma_serpentine)
        self.assertTrue(x.kernel_name().endswith("_msp"))
        self.assertGreater(xp.trait.persistent_ctas, 0)
        old_s = C.RILV_SERPENTINE
        try:
            C.RILV_SERPENTINE = False
            x0 = self._pick(8192, 8192, 8192)
        finally:
            C.RILV_SERPENTINE = old_s
        self.assertFalse(x0.trait.mfma_serpentine)
        self.assertEqual(replace(x0.trait, mfma_serpentine=True), x.trait)

        old = (C.RILV_HB_LOOP, C.RILV_ASM_LOOP_STAGGER_U)
        try:
            C.RILV_HB_LOOP = "asm_loop"
            t = self._pick(8192, 8192, 8192)
            tp = self._pick(15360, 15360, 768)
            C.RILV_ASM_LOOP_STAGGER_U = 32
            ts = self._pick(8192, 8192, 8192)
            tps = self._pick(15360, 15360, 768)
        finally:
            C.RILV_HB_LOOP, C.RILV_ASM_LOOP_STAGGER_U = old
        for a, b in ((x, t), (xp, tp)):
            self.assertEqual(
                (b.trait.asm_loop, b.trait.asm_loop_skew, b.trait.hb_skew),
                ("tensile", 1, 0),
            )
            self.assertEqual(
                (
                    a.trait.persistent_ctas,
                    a.trait.chiplet_wgm,
                    a.trait.chiplet_chunk_size,
                    a.trait.kloop_unroll,
                ),
                (
                    b.trait.persistent_ctas,
                    b.trait.chiplet_wgm,
                    b.trait.chiplet_chunk_size,
                    b.trait.kloop_unroll,
                ),
            )
            self.assertTrue(is_valid_spec(b, arch="gfx950")[0])
        self.assertEqual(ts.trait.stagger_u, 32)
        self.assertEqual(tps.trait.stagger_u, 0)  # persistent grid: no stagger
        self.assertTrue(is_valid_spec(ts, arch="gfx950")[0])

    def test_pad_class_serpentine(self):
        # the ragged (pad) row-interleave bodies: hb_sched (the long-K class, RILV_PAD_HB_TUNED) and hb_sched (persistent)
        from rocke.dispatch.gemm import common as _cm

        long_sia = not _cm.RILV_PAD_HB_TUNED
        for M, N, K, sia in ((30720, 20160, 7680, long_sia), (7680, 6720, 1536, False)):
            sp = self._pick(M, N, K)
            self.assertTrue(sp.trait.pad_n and sp.trait.mfma_serpentine, (M, N, K))
            self.assertEqual(sp.trait.sia_sched, sia)
            self.assertTrue(is_valid_spec(sp, arch="gfx950")[0])
        # the short-K merged-interleave and small-ring bodies are unchanged
        for M, N, K in ((4608, 4608, 384), (3072, 1920, 576)):
            self.assertFalse(self._pick(M, N, K).trait.mfma_serpentine, (M, N, K))


class PythonOnlyTest(unittest.TestCase):
    def test_engine_refuses_stagger(self):
        for f in (
            ("stagger_u", 0),
            ("stagger_u_stride", 1),
            ("stagger_u_map", 0),
            ("mfma_serpentine", False),
        ):
            self.assertIn(f, _be._PYTHON_ONLY_GEMM_LEVERS)
        with self.assertRaises(NotImplementedError) as cm:
            _be._refuse_python_only_levers(replace(TraitSpec(), stagger_u=8))
        self.assertIn("stagger_u", str(cm.exception))
        _be._refuse_python_only_levers(TraitSpec())


if __name__ == "__main__":
    unittest.main()
