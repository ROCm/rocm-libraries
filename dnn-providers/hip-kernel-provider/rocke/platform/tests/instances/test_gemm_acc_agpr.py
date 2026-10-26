# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the universal-GEMM ``acc_agpr`` lever (MFMA accumulators
in AGPRs).

``TraitSpec.acc_agpr`` sets the kernel attr ``amdgpu-agpr-alloc`` to exactly the
per-lane accumulator registers and seeds the accumulators from an opaque
``asm "", "=a,0"`` wrapper of the zero vector instead of the zero constant (the
constant makes LLVM's CodeGenPrepare split the loop-carried accumulator vectors
into scalars). These tests cover validity, the register count, the kernel name,
the emitted attribute and seed (once per kernel, feeding every accumulator), and
that the lever is default-off (the default trait keeps the historical emission).
"""

from __future__ import annotations

import re
import unittest
import warnings
from dataclasses import replace

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    acc_agpr_regs,
    build_universal_gemm,
    is_valid_spec,
    mono_data_spec,
)

# 4-wave 2x2 body with 128x128 per wave (32x32x16 and 16x16x32 atoms), the
# 16-wave ping-pong geometry, an 8-wave 2x4 body and a gfx942 2x2 body
_W22X32 = TileSpec(256, 256, 64, 2, 2, 1, 32, 32, 16)
_W22X16 = TileSpec(256, 256, 64, 2, 2, 1, 16, 16, 32)
_F_TILE = TileSpec(256, 256, 64, 4, 4, 1, 16, 16, 32)
_W24X32 = TileSpec(256, 256, 64, 2, 4, 1, 32, 32, 16)
_F942_W22 = TileSpec(256, 256, 32, 2, 2, 1, 32, 32, 8)
_F_TRAIT = TraitSpec(
    pipeline="compv4",
    epilogue="default",
    direct_to_lds=True,
    dtl_prefetch=True,
    lds_swizzle=True,
    dtl_cache_a=0,
    dtl_cache_b=0,
    chiplet_swizzle=True,
)

_ASM_SEED = re.compile(
    r'(%\S+) = call <(\d+) x float> asm "", "=a,0"\(<\d+ x float> (%\S+)\)'
)
_ATTR = re.compile(r'"amdgpu-agpr-alloc"="(\d+),(\d+)"')


def _spec(tile=_W22X32, dtype="fp16", **trait):
    return UniversalGemmSpec(
        name="t",
        tile=tile,
        trait=replace(_F_TRAIT, **trait),
        data=mono_data_spec(dtype),
        wave_size=64,
    )


def _ll(s, arch="gfx950"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lower_kernel_to_llvm(build_universal_gemm(s, arch=arch), arch=arch)


class TestAccAgprValidity(unittest.TestCase):
    def test_default_off(self):
        self.assertFalse(TraitSpec().acc_agpr)

    def test_regs(self):
        # mfmas_m * mfmas_n * warp_tile_m * warp_tile_n / 64
        self.assertEqual(acc_agpr_regs(_spec(_W22X32)), 4 * 4 * 16)
        self.assertEqual(acc_agpr_regs(_spec(_W22X16)), 8 * 8 * 4)
        self.assertEqual(acc_agpr_regs(_spec(_F_TILE)), 4 * 4 * 4)
        self.assertEqual(acc_agpr_regs(_spec(_W24X32)), 4 * 2 * 16)

    def test_valid_on_cdna(self):
        for tile, arch, kw in (
            (_W22X32, "gfx950", {"dtl_stages": 2}),
            (_W22X32, "gfx950", {}),
            (_W22X16, "gfx950", {"dtl_stages": 2, "interwave_setprio": True}),
            (_F_TILE, "gfx950", {}),
            (_W24X32, "gfx950", {"dtl_stages": 2, "dtl_frag_prefetch": True}),
            (_F942_W22, "gfx942", {}),
            (_F942_W22, "gfx942", {"dtl_stages": 2}),
        ):
            with self.subTest(tile=tile, arch=arch, **kw):
                ok, why = is_valid_spec(_spec(tile, acc_agpr=True, **kw), arch)
                self.assertTrue(ok, why)

    def test_arch_admission_follows_gpu_numerics(self):
        # every AGPR / interleave lever: gfx942 + gfx950; gfx90a: none (same bodies valid
        # without them)
        ilv = dict(dtl_stages=2, dtl_frag_prefetch=True, mfma_interleave=True)
        for kw in (
            dict(acc_agpr=True),
            ilv,
            dict(ilv, acc_agpr=True),
            dict(lds_swizzle_r=1),
            dict(dtl_stages=2, dtl_hoist=True),
            dict(ilv, acc_agpr=True, lds_swizzle_r=2, dtl_hoist=True),
        ):
            with self.subTest(**kw):
                ok, why = is_valid_spec(_spec(_F942_W22, **kw), "gfx942")
                self.assertTrue(ok, why)
        # gfx90a: no DTL (and no swizzle for its 32x32x8 atom)
        no_dtl = dict(
            direct_to_lds=False,
            dtl_prefetch=False,
            lds_swizzle=False,
            chiplet_swizzle=False,
        )
        self.assertTrue(is_valid_spec(_spec(_F942_W22, **no_dtl), "gfx90a")[0])
        ok, why = is_valid_spec(_spec(_F942_W22, acc_agpr=True, **no_dtl), "gfx90a")
        self.assertFalse(ok)
        self.assertIn("admitted on gfx942 / gfx950 only", why)
        # gfx90a has no DTL ring (dtl_hoist, interleave) and no lds_swizzle for
        # its f16 atoms, so those gates are not reachable there; and on a
        # target without MFMA every lever is rejected
        ok, why = is_valid_spec(_spec(_F942_W22, lds_swizzle_r=1, **no_dtl), "gfx90a")
        self.assertFalse(ok)

    def test_rejected_without_agprs(self):
        # RDNA (WMMA, wave32) has no AGPR file
        trait = TraitSpec(pipeline="mem", epilogue="default")
        s = UniversalGemmSpec(
            name="t",
            tile=TileSpec(64, 64, 32, 2, 2, 1, 16, 16, 16),
            trait=replace(trait, acc_agpr=True),
            data=mono_data_spec("fp16"),
            wave_size=32,
        )
        ok, why = is_valid_spec(s, "gfx1151")
        self.assertFalse(ok)
        self.assertIn("acc_agpr needs MFMA accumulators in AGPRs", why)
        ok, why = is_valid_spec(replace(s, trait=trait), "gfx1151")
        self.assertTrue(ok, why)

    def test_rejected_over_256_regs(self):
        # 2x2 waves of 256x128 with 32x32 atoms: 8 x 4 x 16 = 512 registers
        big = TileSpec(512, 256, 32, 2, 2, 1, 32, 32, 16)
        s = _spec(
            big,
            direct_to_lds=False,
            dtl_prefetch=False,
            lds_swizzle=False,
            chiplet_swizzle=False,
        )
        self.assertEqual(acc_agpr_regs(s), 512)
        ok, why = is_valid_spec(
            replace(s, trait=replace(s.trait, acc_agpr=True)), "gfx950"
        )
        self.assertFalse(ok)
        self.assertIn("256 accumulator registers", why)

    def test_agpr_form_compile_option(self):
        from unittest import mock

        from rocke.helpers import compile as hc
        from rocke.runtime import comgr as rc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = build_universal_gemm(_spec(dtl_stages=2, acc_agpr=True), arch="gfx950")
            k0 = build_universal_gemm(_spec(dtl_stages=2), arch="gfx950")
        self.assertIs(k.attrs.get("mfma_agpr_form"), True)
        self.assertNotIn("mfma_agpr_form", k0.attrs)
        opt = ["-mllvm", "-amdgpu-mfma-vgpr-form=0"]
        # a pure function of the kernel and the lowering flavor: no compiler is
        # asked (offline emission gives the same options as a live compile)
        with mock.patch.object(
            rc, "_require_compiler_info", side_effect=AssertionError("asked")
        ):
            for fl in ("llvm22", "llvm23"):
                self.assertEqual(hc._comgr_options_for_kernel(k, fl), ["-O3"] + opt)
                self.assertEqual(hc._comgr_options_for_kernel(k0, fl), ["-O3"])
            # LLVM 20 has no such option (and selects the AGPR form anyway)
            self.assertEqual(hc._comgr_options_for_kernel(k, "llvm20"), ["-O3"])
            with self.assertRaises(ValueError):
                hc._comgr_options_for_kernel(k, "llvm99")
            # default flavor: the ROCKE_LLVM_FLAVOR override, as for the
            # lowering
            with mock.patch.dict("os.environ", {"ROCKE_LLVM_FLAVOR": "llvm23"}):
                self.assertEqual(hc._comgr_options_for_kernel(k), ["-O3"] + opt)
            with mock.patch.dict("os.environ", {"ROCKE_LLVM_FLAVOR": "llvm20"}):
                self.assertEqual(hc._comgr_options_for_kernel(k), ["-O3"])
            # a kernel without the attr never resolves a flavor
            with mock.patch.dict("os.environ", {"ROCKE_LLVM_FLAVOR": ""}):
                self.assertEqual(hc._comgr_options_for_kernel(k0), ["-O3"])
        # the COMGR input carries the options of the lowering flavor
        with mock.patch.dict("os.environ", {"ROCKE_LLVM_FLAVOR": "llvm22"}):
            with mock.patch.object(
                rc, "_require_compiler_info", side_effect=AssertionError("asked")
            ):
                inp = hc.lower_kernel_for_comgr(k, arch="gfx950", backend="python")
        self.assertEqual(inp.options, tuple(["-O3"] + opt))

    def test_exp_agpr_alloc_env_conflicts(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"ROCKE_EXP_AGPR_ALLOC": "0,0"}):
            with self.assertRaises(ValueError) as cm:
                build_universal_gemm(_spec(dtl_stages=2, acc_agpr=True), arch="gfx950")
            self.assertIn("conflicts with TraitSpec.acc_agpr", str(cm.exception))
            # the experiment itself still works on a kernel without acc_agpr
            k = build_universal_gemm(_spec(dtl_stages=2), arch="gfx950")
            self.assertEqual(k.attrs.get("agpr_alloc"), "0,0")

    def test_name_flag(self):
        self.assertTrue(_spec(acc_agpr=True).kernel_name().endswith("_agpr"))
        self.assertNotIn("agpr", _spec().kernel_name())


class TestAccAgprEmission(unittest.TestCase):
    def _check(self, s, arch="gfx950"):
        ll = _ll(s, arch)
        n = acc_agpr_regs(s)
        self.assertEqual(_ATTR.findall(ll), [(str(n), str(n))])
        zeros = set(
            re.findall(
                r"(%\S+) = select i1 true, <\d+ x float> zeroinitializer, "
                r"<\d+ x float> zeroinitializer",
                ll,
            )
        )
        # the seed wraps the zero; the other "=a,0" asms pin MFMA results
        seeds = [x for x in _ASM_SEED.findall(ll) if x[2] in zeros]
        self.assertEqual(len(seeds), 1, "one opaque zero seed per kernel")
        name, width, src = seeds[0]
        self.assertEqual(int(width), (s.tile.warp_tile_m * s.tile.warp_tile_n) // 64)
        # it wraps the zero accumulator
        self.assertIn(
            f"{src} = select i1 true, <{width} x float> zeroinitializer, "
            f"<{width} x float> zeroinitializer",
            ll,
        )
        # the seed feeds every accumulator phi of the first K loop, and no
        # accumulator phi starts from the zero constant or the unwrapped zero
        phis = re.findall(r"= phi <\d+ x float> \[ (\S+), ", ll)
        n_acc = s.tile.mfmas_per_warp_m * s.tile.mfmas_per_warp_n
        self.assertGreaterEqual(phis.count(name), n_acc)
        self.assertNotIn("zeroinitializer", phis)
        self.assertNotIn(src, phis)
        return ll

    def test_ring_body(self):
        self._check(_spec(dtl_stages=2, interwave_setprio=True, acc_agpr=True))

    def test_ring_body_16x16(self):
        self._check(_spec(_W22X16, dtype="bf16", dtl_stages=2, acc_agpr=True))

    def test_persistent_body(self):
        self._check(_spec(_W22X16, dtl_stages=2, persistent_ctas=256, acc_agpr=True))

    def test_pingpong_body(self):
        self._check(_spec(acc_agpr=True))

    def test_gfx942(self):
        self._check(_spec(_F942_W22, acc_agpr=True), arch="gfx942")

    def test_gfx942(self):
        self._check(_spec(_F942_W22, acc_agpr=True), arch="gfx942")

    def test_split_k_epilogue_hands_out_one_atom_at_a_time(self):
        # split-K atomic epilogue: one ordered hand-out per accumulator atom,
        # each feeding that atom's f32 atomic adds (no bulk AGPR copy at the
        # loop exit)
        s = _spec(_W22X32, dtl_stages=2, acc_agpr=True, split_k=2)
        ok, why = is_valid_spec(s, "gfx950")
        self.assertTrue(ok, why)
        ll = self._check(s)
        n_acc = s.tile.mfmas_per_warp_m * s.tile.mfmas_per_warp_n
        outs = re.findall(r'(%\S+) = call <\d+ x float> asm sideeffect "", "=a,0"', ll)
        self.assertEqual(len(outs), n_acc)
        for v in outs:
            self.assertRegex(ll, r"extractelement <\d+ x float> " + re.escape(v) + ",")
        self.assertIn("atomicrmw fadd", ll)
        base = _ll(_spec(_W22X32, dtl_stages=2, split_k=2))
        self.assertNotIn("asm sideeffect", base)

    def test_default_has_no_agpr_attr_or_seed(self):
        for s in (_spec(dtl_stages=2), _spec(_F_TILE), _spec()):
            ll = _ll(s)
            self.assertNotIn("amdgpu-agpr-alloc", ll)
            self.assertNotIn('"=a,0"', ll)

    def test_every_mfma_result_is_pinned(self):
        for tile, kw in (
            (_W22X32, dict(dtl_stages=2)),
            (_W22X16, dict(dtl_stages=2, dtl_frag_prefetch=True, mfma_interleave=True)),
            (_W22X32, {}),
        ):
            with self.subTest(tile=tile, **kw):
                ll = _ll(_spec(tile, acc_agpr=True, acc_agpr_pin=True, **kw))
                mf = re.findall(
                    r"(%\S+) = call <\d+ x float> @llvm\.amdgcn\.mfma\.", ll
                )
                self.assertGreater(len(mf), 0)
                pins = set(
                    re.findall(
                        r'= call <\d+ x float> asm "", "=a,0"\(<\d+ x float> (%\S+)\)',
                        ll,
                    )
                )
                self.assertEqual([v for v in mf if v not in pins], [])
        # acc_agpr alone: only the seed (no per-MFMA pins); no AGPR asm by default
        ll = _ll(_spec(dtl_stages=2, acc_agpr=True))
        self.assertEqual(len(re.findall(r'asm "", "=a,0"', ll)), 1)
        ll = _ll(_spec(dtl_stages=2))
        self.assertNotIn('"=a,0"', ll)
        ok, why = is_valid_spec(_spec(dtl_stages=2, acc_agpr_pin=True), "gfx950")
        self.assertFalse(ok)
        self.assertIn("acc_agpr_pin requires acc_agpr", why)
        self.assertIn(
            "_agpr_apin",
            _spec(dtl_stages=2, acc_agpr=True, acc_agpr_pin=True).kernel_name(),
        )

    def test_epilogue_hands_out_one_atom_at_a_time(self):
        # one ordered (sideeffect) "=a,0" asm per accumulator atom, each right
        # in front of that atom's f32 -> f16 conversion
        for tile in (_W22X32, _W22X16):
            s = _spec(tile, dtl_stages=2, acc_agpr=True)
            ll = _ll(s)
            n_acc = s.tile.mfmas_per_warp_m * s.tile.mfmas_per_warp_n
            outs = re.findall(
                r'(%\S+) = call <\d+ x float> asm sideeffect "", "=a,0"', ll
            )
            self.assertEqual(len(outs), n_acc)
            for v in outs:
                self.assertRegex(ll, r"fptrunc <\d+ x float> " + re.escape(v) + r" ")
        ll = _ll(_spec(dtl_stages=2))
        self.assertNotIn("asm sideeffect", ll)

    def test_only_the_seed_and_handouts_are_added(self):
        # Same body: the agpr build has exactly the seed and one hand-out asm
        # per accumulator atom extra, and the same multiset of opcodes / callees.
        base = _ll(_spec(dtl_stages=2))
        agpr = _ll(_spec(dtl_stages=2, acc_agpr=True))

        def ops(ll):
            out = []
            for ln in ll.splitlines():
                m = re.match(r"\s+(?:%\S+ = )?(\w+)(?: [^@]*@([\w.]+))?", ln)
                if m:
                    callee = m.group(2) or ""
                    # only intrinsics: other globals embed the kernel name
                    out.append(
                        (m.group(1), callee if callee.startswith("llvm.") else "")
                    )
            return out

        ob, oa = ops(base), ops(agpr)
        # seed + one hand-out per 32x32 atom (4 x 4)
        n_asm = 1 + 16
        self.assertEqual(len(oa), len(ob) + n_asm)
        for op in sorted(set(ob) | set(oa)):
            extra = n_asm if op == ("call", "") else 0  # asm calls have no @callee
            with self.subTest(op=op):
                self.assertEqual(oa.count(op), ob.count(op) + extra)


class TestMfmaInterleave(unittest.TestCase):
    """``mfma_interleave``: the fragment-prefetch body with every next-atom
    ds_read and DTL issue placed between two MFMAs and pinned by
    ``sched_barrier(0)``."""

    _MFMA = re.compile(r"call <\d+ x float> @llvm\.amdgcn\.mfma\.")
    _LDS = re.compile(r"= load <\d+ x (?:half|bfloat)>, ptr addrspace\(3\)")
    _DTL = re.compile(r"call void @llvm\.amdgcn\.raw\.ptr\.buffer\.load\.lds")
    _SB = re.compile(r"call void @llvm\.amdgcn\.sched\.barrier\(i32 0\)")

    def _events(self, ll):
        ev = []
        for ln in ll.splitlines():
            for tag, rx in (
                ("M", self._MFMA),
                ("L", self._LDS),
                ("D", self._DTL),
                ("B", self._SB),
            ):
                if rx.search(ln):
                    ev.append(tag)
        return "".join(ev)

    def test_validity(self):
        base = dict(dtl_stages=2, acc_agpr=True)
        ok, why = is_valid_spec(_spec(mfma_interleave=True, **base), "gfx950")
        self.assertFalse(ok)
        self.assertIn("requires dtl_frag_prefetch", why)
        ok, why = is_valid_spec(
            _spec(
                mfma_interleave=True,
                dtl_frag_prefetch=True,
                dtl_issue_spread=True,
                **base,
            ),
            "gfx950",
        )
        self.assertFalse(ok)
        self.assertIn("excludes dtl_issue_spread", why)
        for tile, kw in (
            (_W22X32, base),
            (_W22X16, base),
            (_F_TILE, dict(dtl_stages=2)),
            (TileSpec(256, 256, 32, 2, 2, 1, 16, 16, 32), dict(dtl_stages=3)),
        ):
            with self.subTest(tile=tile, **kw):
                ok, why = is_valid_spec(
                    _spec(tile, mfma_interleave=True, dtl_frag_prefetch=True, **kw),
                    "gfx950",
                )
                self.assertTrue(ok, why)
        self.assertFalse(TraitSpec().mfma_interleave)

    def test_name_flag(self):
        s = _spec(dtl_stages=2, dtl_frag_prefetch=True, mfma_interleave=True)
        self.assertIn("_frp_ilv", s.kernel_name())
        self.assertNotIn(
            "ilv", _spec(dtl_stages=2, dtl_frag_prefetch=True).kernel_name()
        )

    def test_each_mfma_is_fenced(self):
        for tile, dtype in ((_W22X32, "fp16"), (_W22X16, "bf16")):
            with self.subTest(tile=tile):
                s = _spec(
                    tile,
                    dtype=dtype,
                    dtl_stages=2,
                    dtl_frag_prefetch=True,
                    mfma_interleave=True,
                    acc_agpr=True,
                )
                ev = self._events(_ll(s))
                n_mfma = ev.count("M")
                # every MFMA is followed by its slot loads / DTL issues and
                # then a sched_barrier(0), before the next MFMA
                self.assertEqual(len(re.findall(r"M[LD]*B", ev)), n_mfma)
                self.assertNotIn("MM", ev)
                # same loads as the plain frag-prefetch body
                plain = self._events(
                    _ll(replace(s, trait=replace(s.trait, mfma_interleave=False)))
                )
                self.assertEqual(ev.count("L"), plain.count("L"))
                self.assertEqual(ev.count("D"), plain.count("D"))
                self.assertEqual(n_mfma, plain.count("M"))
                # loads are spread: no two ds_reads of a cluster back to back
                # in the K loop (each sits behind its own MFMA when a cluster
                # has at least as many MFMAs as fragments)
                m = tile.tile_m // (tile.warp_m * tile.warp_tile_m)
                n = tile.tile_n // (tile.warp_n * tile.warp_tile_n)
                if m * n >= m + n:
                    body = ev[ev.index("MLB") :]
                    self.assertNotIn("LL", body)


class TestIlvDtlEvery(unittest.TestCase):
    """``ilv_dtl_every``: DTL issues of an interleaved cluster one every k MFMAs."""

    def test_validity_and_name(self):
        ok, why = is_valid_spec(_spec(dtl_stages=2, ilv_dtl_every=2), "gfx950")
        self.assertFalse(ok)
        self.assertIn("requires mfma_interleave", why)
        base = dict(dtl_stages=2, dtl_frag_prefetch=True, mfma_interleave=True)
        self.assertFalse(is_valid_spec(_spec(ilv_dtl_every=-1, **base), "gfx950")[0])
        self.assertTrue(is_valid_spec(_spec(ilv_dtl_every=3, **base), "gfx950")[0])
        self.assertIn("_de3", _spec(ilv_dtl_every=3, **base).kernel_name())
        self.assertNotRegex(_spec(**base).kernel_name(), r"_de\d")

    def test_placement(self):
        ev_rx = TestMfmaInterleave._MFMA, TestMfmaInterleave._DTL

        def dtl_gaps(every):
            s = _spec(
                TileSpec(256, 256, 32, 2, 2, 1, 16, 16, 32),
                dtl_stages=3,
                dtl_frag_prefetch=True,
                mfma_interleave=True,
                acc_agpr=True,
                ilv_dtl_every=every,
            )
            seq = []
            for ln in _ll(s).splitlines():
                if ev_rx[0].search(ln):
                    seq.append("M")
                elif ev_rx[1].search(ln):
                    seq.append("D")
            # the first interleaved cluster with DTL issues (after the prologue)
            body = "".join(seq)
            run = body[body.index("MD") :][: 2 * 64 + 16]
            return [len(x) for x in run.split("D")[1:-1]]

        # every 1: one DTL behind each of the first L MFMAs (gaps of one MFMA)
        self.assertTrue(all(g == 1 for g in dtl_gaps(1)[:7]))
        # every 4: four MFMAs between consecutive DTL issues
        self.assertTrue(all(g == 4 for g in dtl_gaps(4)[:7]))

    def test_first_issue_behind_first_mfma(self):
        # the issue slots start at the cluster's first MFMA: right after the
        # in-loop barrier comes exactly one MFMA, then the first DTL issue
        rx = TestIlvBarrierAfter._rx
        for every in (1, 4):
            with self.subTest(every=every):
                s = _spec(
                    TileSpec(256, 256, 32, 2, 2, 1, 16, 16, 32),
                    dtl_stages=3,
                    dtl_frag_prefetch=True,
                    mfma_interleave=True,
                    acc_agpr=True,
                    ilv_dtl_every=every,
                )
                seq = []
                for ln in _ll(s).splitlines():
                    for tag, r in rx:
                        if tag in "MDW" and r.search(ln):
                            seq.append(tag)
                body = "".join(seq)
                body = body[body.index("M") :]
                self.assertIn("WMD", body)
                self.assertNotIn("WMMD", body)


class TestIlvMerge(unittest.TestCase):
    """``ilv_merge``: one slot schedule for the fragment reads and DTL issues."""

    def _spec(self, **kw):
        return _spec(
            TileSpec(256, 256, 32, 2, 2, 1, 16, 16, 32),
            dtl_stages=3,
            dtl_frag_prefetch=True,
            mfma_interleave=True,
            acc_agpr=True,
            **kw,
        )

    def test_validity_and_name(self):
        self.assertFalse(TraitSpec().ilv_merge)
        ok, why = is_valid_spec(_spec(dtl_stages=2, ilv_merge=True), "gfx950")
        self.assertFalse(ok)
        self.assertIn("ilv_merge requires mfma_interleave", why)
        ok, why = is_valid_spec(self._spec(ilv_merge=True, ilv_dtl_every=2), "gfx950")
        self.assertFalse(ok)
        self.assertIn("excludes ilv_dtl_every", why)
        self.assertTrue(is_valid_spec(self._spec(ilv_merge=True), "gfx950")[0])
        self.assertIn("_mrg", self._spec(ilv_merge=True).kernel_name())

    def test_no_shared_slots(self):
        rx = TestIlvBarrierAfter._rx

        def body(**kw):
            seq = []
            for ln in _ll(self._spec(**kw)).splitlines():
                for tag, r in rx:
                    if r.search(ln):
                        seq.append(tag)
            b_ = "".join(seq)
            return b_[b_.index("W", b_.index("M")) :]

        # 16 reads + 8 DTL issues in a 64-MFMA cluster: the separate spreads
        # stack a read and a DTL issue behind the same MFMA; the merged
        # schedule gives every op its own MFMA
        self.assertIn("LD", body())
        merged = body(ilv_merge=True)
        self.assertNotIn("LD", merged)
        self.assertNotIn("DL", merged)
        self.assertNotIn("DD", merged)
        self.assertNotIn("LL", merged)
        self.assertEqual(merged.count("D"), body().count("D"))
        self.assertEqual(merged.count("L"), body().count("L"))


class TestIlvBarrierAfter(unittest.TestCase):
    """``ilv_barrier_after``: the first k MFMAs of a K tile's last cluster go in
    front of its wait + barrier."""

    _rx = (
        ("M", TestMfmaInterleave._MFMA),
        ("L", TestMfmaInterleave._LDS),
        ("D", TestMfmaInterleave._DTL),
        ("W", re.compile(r"@llvm\.amdgcn\.s\.barrier\(")),
    )

    def _seq(self, s):
        out = []
        for ln in _ll(s).splitlines():
            for tag, rx in self._rx:
                if rx.search(ln):
                    out.append(tag)
        return "".join(out)

    def test_validity_and_name(self):
        base = dict(dtl_stages=2, dtl_frag_prefetch=True, mfma_interleave=True)
        ok, why = is_valid_spec(_spec(dtl_stages=2, ilv_barrier_after=1), "gfx950")
        self.assertFalse(ok)
        self.assertIn("ilv_barrier_after requires mfma_interleave", why)
        # 32x32 atoms on 2x2 waves: 16 MFMAs per K atom
        self.assertTrue(is_valid_spec(_spec(ilv_barrier_after=15, **base), "gfx950")[0])
        self.assertFalse(
            is_valid_spec(_spec(ilv_barrier_after=16, **base), "gfx950")[0]
        )
        self.assertFalse(
            is_valid_spec(_spec(ilv_barrier_after=-1, **base), "gfx950")[0]
        )
        ok, why = is_valid_spec(_spec(ilv_barrier_after=16, **base), "gfx950")
        self.assertIn("0 < k < the MFMAs of one K atom (16)", why)
        self.assertIn("_ba3", _spec(ilv_barrier_after=3, **base).kernel_name())
        self.assertNotRegex(_spec(**base).kernel_name(), r"_ba\d")

    def test_mfmas_before_barrier(self):
        base = dict(
            dtl_stages=2, dtl_frag_prefetch=True, mfma_interleave=True, acc_agpr=True
        )

        def runs(k):
            seq = self._seq(_spec(ilv_barrier_after=k, **base))
            body = seq[seq.index("M") :]
            out = []
            for pos in [i for i, c in enumerate(body) if c == "W"]:
                run = 0
                for c in reversed(body[:pos]):
                    if c != "M":
                        break
                    run += 1
                out.append(run)
            # the loads and DTL issues of the barrier cluster all follow it
            self.assertNotIn("LW", body)
            self.assertNotIn("DW", body)
            return out

        r0 = runs(0)
        self.assertTrue(r0)
        for k in (1, 3):
            with self.subTest(k=k):
                # k more MFMAs between the previous cluster's last slot and
                # every in-loop barrier
                self.assertEqual([r - k for r in runs(k)], r0)


class TestLdsSwizzleR(unittest.TestCase):
    """``lds_swizzle_r``: the row shift of the lds_swizzle XOR key."""

    def test_validity_and_name(self):
        self.assertEqual(TraitSpec().lds_swizzle_r, -1)
        ok, why = is_valid_spec(_spec(lds_swizzle=False, lds_swizzle_r=1), "gfx950")
        self.assertFalse(ok)
        self.assertIn("lds_swizzle_r", why)
        self.assertFalse(is_valid_spec(_spec(lds_swizzle_r=5), "gfx950")[0])
        self.assertFalse(is_valid_spec(_spec(lds_swizzle_r=-2), "gfx950")[0])
        self.assertTrue(is_valid_spec(_spec(lds_swizzle_r=1), "gfx950")[0])
        self.assertIn("_swr1", _spec(lds_swizzle_r=1).kernel_name())
        self.assertNotIn("_swr", _spec().kernel_name())

    def test_shift_is_emitted(self):
        def shifts(ll):
            return sorted(set(re.findall(r"= lshr i32 %\S+, (\d+)", ll)))

        tile = TileSpec(256, 256, 32, 2, 2, 1, 16, 16, 32)
        base = _ll(_spec(tile, dtl_stages=3, acc_agpr=True))
        r1 = _ll(_spec(tile, dtl_stages=3, acc_agpr=True, lds_swizzle_r=1))
        # R = 0 emits no shift of the key (lshr by 0 folds away or stays 0);
        # R = 1 shifts the row by one in every swizzle key
        self.assertNotIn("1", shifts(base))
        self.assertIn("1", shifts(r1))
        # R = 0 explicitly is the geometry default (identical IR modulo the name)
        r0 = _ll(_spec(tile, dtl_stages=3, acc_agpr=True, lds_swizzle_r=0))
        n0 = _spec(tile, dtl_stages=3, acc_agpr=True, lds_swizzle_r=0).kernel_name()
        nb = _spec(tile, dtl_stages=3, acc_agpr=True).kernel_name()
        self.assertEqual(r0.replace(n0, nb), base)

    def test_conflict_free_reads(self):
        # the R the case study recommends removes the ds_read_b128 conflicts of
        # the 16x16x32 fragment reads at tile_k 32 (production predictor)
        from rocke.analysis.lds import LdsAccess, predict_lds_conflicts
        from rocke.instances.common.gemm_universal import lds_swizzle_params

        def mult(r):
            _r0, w, l_ = lds_swizzle_params(8, 8, 32)
            acc = []
            for lane in range(64):
                m_in, kb = lane % 16, lane // 16
                col = (kb * 8) ^ (((m_in >> r) % (1 << w)) << l_)
                acc.append(
                    LdsAccess(
                        access_id=lane,
                        lane=lane,
                        lds_byte_address=(m_in * 32 + col) * 2,
                        access_width_bytes=16,
                        active=True,
                    )
                )
            res = predict_lds_conflicts(
                target="gfx950", opcode="ds_read_b128", wave_size=64, accesses=acc
            )
            return res.summary.maximum_multiplicity

        self.assertEqual(mult(0), 2)
        self.assertEqual(mult(1), 0)


class TestDtlHoist(unittest.TestCase):
    """``dtl_hoist``: one buffer resource + soffset per operand and K tile."""

    def test_validity_and_name(self):
        self.assertFalse(TraitSpec().dtl_hoist)
        ok, why = is_valid_spec(_spec(dtl_hoist=True), "gfx950")
        self.assertFalse(ok)
        self.assertIn("dtl_hoist requires dtl_stages", why)
        self.assertIn("_dh", _spec(dtl_stages=2, dtl_hoist=True).kernel_name())

    def test_fewer_resources(self):
        def n_rsrc(ll):
            return len(
                re.findall(
                    r"= call ptr addrspace\(8\) @llvm\.amdgcn\.make\.buffer\.rsrc", ll
                )
            )

        s = _spec(_W22X16, dtl_stages=2, acc_agpr=True)
        base, hoist = _ll(s), _ll(replace(s, trait=replace(s.trait, dtl_hoist=True)))
        nb, nh = n_rsrc(base), n_rsrc(hoist)
        self.assertGreater(nb, 0)
        # per K tile: one resource per DTL pass without the hoist, one per
        # operand with it (2 operands, 16 passes per tile at 4 waves)
        self.assertLess(nh, nb)
        self.assertLessEqual(4 * nh, nb)


class TestLaunchShape(unittest.TestCase):
    """``launch_shape_error`` / ``check_launch_shape``: the unpadded-body launch
    contract (K % tile_k == 0 without pad_k; split-K slices whole tiles)."""

    def test_contract(self):
        from rocke.instances.common.gemm_universal import (
            check_launch_shape,
            launch_shape_error,
        )

        s = _spec(_W22X32, dtl_stages=2, acc_agpr=True)  # 256x256x64 tile
        self.assertIsNone(launch_shape_error(s, 8192, 8192, 8192))
        self.assertIsNone(launch_shape_error(s, 256, 512, 4160))
        for m, n, k, dim in (
            (8192, 8192, 4128, "K="),  # K % 64 == 32: silent wrong results
            (8192, 8192, 96, "K="),
            (8200, 8192, 8192, "M="),
            (8192, 8200, 8192, "N="),
        ):
            with self.subTest(m=m, n=n, k=k):
                err = launch_shape_error(s, m, n, k)
                self.assertIsNotNone(err)
                self.assertTrue(err.startswith(dim), err)
                with self.assertRaises(ValueError):
                    check_launch_shape(s, m, n, k)
        # the DTL body has no K mask, so pad_k alone does not make K % 64 != 0
        # safe (the last tile would read the next row); dtl_k_tail does
        pk = replace(s, trait=replace(s.trait, pad_k=True))
        self.assertIsNone(launch_shape_error(pk, 8192, 8192, 4160))
        # pad_k does not cover a K tail on the direct-to-LDS path (the
        # DRAM->LDS loads copy whole K tiles), so the launch is refused
        self.assertIn("direct_to_lds", launch_shape_error(pk, 8192, 8192, 4128))
        self.assertIn("dtl_k_tail", launch_shape_error(pk, 8192, 8192, 4128))
        kt = replace(s, trait=replace(s.trait, pad_k=True, dtl_k_tail=True))
        self.assertIsNone(launch_shape_error(kt, 8192, 8192, 4128))
        self.assertIn("multiple of 8", launch_shape_error(kt, 8192, 8192, 4132))
        sk = replace(s, trait=replace(s.trait, split_k=2))
        self.assertIsNone(launch_shape_error(sk, 8192, 8192, 8192))
        self.assertIn("split_k * tile_k", launch_shape_error(sk, 8192, 8192, 8256))
        pks = replace(s, trait=replace(s.trait, pad_k=True, split_k=2))
        self.assertIn("direct_to_lds", launch_shape_error(pks, 8192, 8192, 8193))
        # the split-K K-multiple rule, on the global-load (masked) path
        gks = replace(pks, trait=replace(pks.trait, direct_to_lds=False))
        self.assertIn("split_k=2", launch_shape_error(gks, 8192, 8192, 8193))


def _engine():
    try:
        import rocke_engine  # type: ignore
    except ImportError:
        return None
    return rocke_engine


_T32 = TileSpec(256, 256, 32, 2, 2, 1, 16, 16, 32)
_AGPR_ILV = dict(
    chiplet_wgm=8,
    chiplet_chunk_size=64,
    dtl_stages=3,
    interwave_setprio=True,
    dtl_frag_prefetch=True,
    acc_agpr=True,
    mfma_interleave=True,
    ilv_barrier_after=2,
    ilv_merge=True,
    lds_swizzle_r=1,
    dtl_hoist=True,
)


@unittest.skipIf(_engine() is None, "rocke_engine (C++ pybind engine) not built")
class TestEngineMirror(unittest.TestCase):
    """The C++ engine's spec-dict path (rocke_engine.cpp spec parse + the C++
    builder) builds the same kernel as Python for every AGPR / interleave lever, and
    rejects the same gfx942 specs."""

    _SPECS = (
        ("agpr_ilv", _spec(_T32, **_AGPR_ILV)),
        (
            "apin_swr0",
            _spec(
                _T32,
                dtype="bf16",
                **dict(_AGPR_ILV, lds_swizzle_r=0, acc_agpr_pin=True),
            ),
        ),
        (
            "csh_swr2",
            _spec(_T32, epilogue="cshuffle", **dict(_AGPR_ILV, lds_swizzle_r=2)),
        ),
        ("spk2", _spec(_W22X32, dtl_stages=2, acc_agpr=True, split_k=2)),
        (
            "de2_apin_swr3",
            _spec(
                _T32,
                dtype="bf16",
                dtl_stages=4,
                dtl_frag_prefetch=True,
                acc_agpr=True,
                acc_agpr_pin=True,
                mfma_interleave=True,
                ilv_dtl_every=2,
                lds_swizzle_r=3,
                dtl_hoist=True,
            ),
        ),
        (
            "s2_swr1_dh",
            _spec(
                _F_TILE,
                dtl_stages=2,
                interwave_setprio=True,
                lds_swizzle_r=1,
                dtl_hoist=True,
            ),
        ),
    )

    def test_both_engines_agree(self):
        from rocke.core.backend import lower_universal_gemm

        for name, s in self._SPECS:
            with self.subTest(name=name):
                self.assertTrue(is_valid_spec(s, "gfx950")[0])
                r = lower_universal_gemm(s, arch="gfx950", backend="both", want_ir=True)
                self.assertIn(s.kernel_name(), r.llvm_text)

    def test_same_validity(self):
        from rocke.core.backend import universal_gemm_spec_to_dict

        eng = _engine()
        cases = [(s, "gfx950") for _n, s in self._SPECS]
        cases.append((_spec(_T32, **dict(_AGPR_ILV, lds_swizzle_r=5)), "gfx950"))
        cases += [
            (_spec(_F942_W22, acc_agpr=True), "gfx942"),
            (
                _spec(
                    _F942_W22,
                    acc_agpr=True,
                    direct_to_lds=False,
                    dtl_prefetch=False,
                    lds_swizzle=False,
                    chiplet_swizzle=False,
                ),
                "gfx90a",
            ),
            (_spec(_F942_W22, dtl_stages=2, dtl_hoist=True), "gfx942"),
            (_spec(_F942_W22, lds_swizzle_r=1), "gfx942"),
            (
                _spec(
                    _F942_W22,
                    dtl_stages=2,
                    dtl_frag_prefetch=True,
                    mfma_interleave=True,
                ),
                "gfx942",
            ),
            (
                _spec(
                    dtl_stages=2,
                    dtl_frag_prefetch=True,
                    mfma_interleave=True,
                    ilv_barrier_after=16,
                ),
                "gfx950",
            ),
        ]
        for s, arch in cases:
            with self.subTest(name=s.kernel_name(), arch=arch):
                py_ok, py_why = is_valid_spec(s, arch)
                c_ok, c_why = eng.gemm_is_valid(universal_gemm_spec_to_dict(s), arch)
                self.assertEqual(bool(c_ok), py_ok, (py_why, c_why))
                if not py_ok:
                    self.assertEqual(c_why, py_why)


if __name__ == "__main__":
    unittest.main()
