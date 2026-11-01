# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the gfx942 main-loop levers of the single-LDS-buffer
``vgpr_prefetch`` (mode 4) body and of the register-staged plan admission /
dispatch fixes.

* ``vp_release_atoms`` / ``vp_tail_pack`` / ``vp_pair_reads``: validity scope,
  kernel-name tags, Python-engine-only levers, and the LDS race freedom of the
  emitted schedule (every LDS read phase and write phase separated by a
  workgroup barrier).
* ``vgpr_prefetch_depth > 1`` with AGPR accumulators is refused (spill-heavy
  kernels with measured wrong results).
* ``g5_plan.G5P_LAUNCHABLE_TOP``: the dispatcher takes the best plan the launch
  contract accepts (a padded top plan with operand bytes >= 2^31 refused the
  whole register-staged candidate before).
"""

from __future__ import annotations

import re
import unittest
from dataclasses import replace

from rocke.core import backend
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    launch_shape_error,
)


def _spec(geo=(256, 128, 64, 4, 2, 1, 16, 16, 16), dt="fp16", layout="RCR", **trait):
    base = dict(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        ab_double_buffer=False,
        chiplet_swizzle=True,
        lds_swizzle=True,
        lds_swizzle_r=0,
        vgpr_prefetch=4,
    )
    base.update(trait)
    return UniversalGemmSpec(
        name="r3g5",
        tile=TileSpec(*geo),
        trait=TraitSpec(**base),
        data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt, layout=layout),
    )


def _ir(spec, arch="gfx942"):
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    return lower_kernel_to_llvm(build_universal_gemm(spec, arch=arch), arch=arch)


_EV = re.compile(
    r"(call void @llvm\.amdgcn\.s\.barrier\(\))"
    r"|(= load <?[^\n]*addrspace\(3\))"
    r"|(store <?[^\n]*addrspace\(3\))"
    r"|(call <[^\n]*@llvm\.amdgcn\.mfma)"
)


def _events(ir: str) -> str:
    """The kernel's LDS / MFMA events in program order: B (barrier), L (LDS
    load), S (LDS store), M (MFMA)."""
    body = ir.split("define", 1)[1]
    out = []
    for m in _EV.finditer(body):
        out.append(
            "B" if m.group(1) else "L" if m.group(2) else "S" if m.group(3) else "M"
        )
    return "".join(out)


def _lds_events(ir: str) -> str:
    return _events(ir).replace("M", "")


_EV2 = re.compile(
    r"(call void @llvm\.amdgcn\.s\.barrier\(\))"
    r"|(= load <?[^\n]*addrspace\(3\))"
    r"|(store <?[^\n]*addrspace\(3\))"
    r"|(call <[^\n]*@llvm\.amdgcn\.mfma)"
    r"|(call [^\n]*@llvm\.amdgcn\.raw\.ptr\.buffer\.load)"
    r"|(call void @llvm\.amdgcn\.sched\.barrier\(i32 0\))"
)


def _events2(ir: str) -> str:
    """As _events, plus G (buffer load) and X (sched_barrier(0))."""
    body = ir.split("define", 1)[1]
    return "".join(
        "BLSMGX"[[g is not None for g in m.groups()].index(True)]
        for m in _EV2.finditer(body)
    )


class TestValidity(unittest.TestCase):
    def test_defaults_unchanged(self):
        s = _spec()
        self.assertEqual(s.trait.vp_release_atoms, 1)
        self.assertEqual(s.trait.vp_tail_pack, 0)
        self.assertEqual(s.trait.vp_pair_reads, 0)
        self.assertEqual(s.trait.vp_mfma_first, 0)
        name = s.kernel_name()
        for tag in ("_vpr", "_vtp", "_vpp", "_vmf"):
            self.assertNotIn(tag, name)

    def test_name_tags(self):
        s = _spec(vp_release_atoms=3, vp_tail_pack=3, vp_pair_reads=1, vp_mfma_first=2)
        name = s.kernel_name()
        for tag in ("_vpr3", "_vtp3", "_vpp1", "_vmf2"):
            self.assertIn(tag, name)

    def test_release_atoms_scope(self):
        ok = [2, 3]  # tile_k 64 with 16x16x16 atoms: 4 K atoms
        for r in ok:
            self.assertEqual(
                is_valid_spec(_spec(vp_release_atoms=r), arch="gfx942"), (True, "ok")
            )
        for bad in (
            _spec(vp_release_atoms=4),  # >= K atoms per tile
            _spec(vp_release_atoms=0),
            _spec(vp_release_atoms=2, vp_late_reads=1),
            _spec(
                (256, 256, 32, 2, 4, 1, 16, 16, 16),
                vgpr_prefetch=3,
                ab_double_buffer=True,
                vp_release_atoms=2,
            ),
        ):
            with self.subTest(trait=bad.trait.vp_release_atoms):
                self.assertFalse(is_valid_spec(bad, arch="gfx942")[0])
        # the double-LDS-buffer body (mode 3) with 4 K atoms: still refused
        m3 = _spec(
            (64, 64, 64, 2, 2, 1, 16, 16, 16), vgpr_prefetch=3, ab_double_buffer=True
        )
        self.assertEqual(is_valid_spec(m3, arch="gfx942"), (True, "ok"))
        m3r = replace(m3, trait=replace(m3.trait, vp_release_atoms=2))
        self.assertFalse(is_valid_spec(m3r, arch="gfx942")[0])

    def test_tail_pack_scope(self):
        for tp in (1, 2, 3):
            self.assertEqual(
                is_valid_spec(_spec(vp_tail_pack=tp), arch="gfx942"), (True, "ok")
            )
        self.assertFalse(is_valid_spec(_spec(vp_tail_pack=4), arch="gfx942")[0])
        m3 = _spec(
            (256, 256, 32, 2, 4, 1, 16, 16, 16),
            vgpr_prefetch=3,
            ab_double_buffer=True,
            vp_tail_pack=1,
        )
        self.assertFalse(is_valid_spec(m3, arch="gfx942")[0])

    def test_pair_reads_scope(self):
        for pr in (1, 2):
            for extra in ({}, {"vp_release_atoms": 3}, {"vp_tail_pack": 3}):
                s = _spec(vp_pair_reads=pr, **extra)
                with self.subTest(pr=pr, extra=extra):
                    self.assertEqual(is_valid_spec(s, arch="gfx942"), (True, "ok"))
        for bad in (
            dict(vp_pair_reads=3),
            dict(vp_pair_reads=1, vgpr_prefetch_depth=2),
            dict(vp_pair_reads=1, vp_late_reads=1),
            dict(vp_pair_reads=1, vp_tail_pack=2),
        ):
            with self.subTest(bad=bad):
                self.assertFalse(is_valid_spec(_spec(**bad), arch="gfx942")[0])
        # the native MN-contiguous forms keep their own loader: a valid native
        # form (the dispatcher's NN pick) with vp_pair_reads is refused
        from rocke.dispatch import dispatch_gemm_fp16
        from rocke.dispatch.gemm.common import GemmRequest

        nn = dispatch_gemm_fp16(
            GemmRequest(
                M=8192,
                N=8192,
                K=8192,
                arch="gfx942",
                dtype="fp16",
                layout="RRR",
                trans_a=False,
                trans_b=False,
            )
        ).spec
        self.assertEqual(is_valid_spec(nn, arch="gfx942"), (True, "ok"))
        self.assertEqual(int(nn.trait.vgpr_prefetch), 4)
        bad = replace(nn, trait=replace(nn.trait, vp_pair_reads=1))
        ok, why = is_valid_spec(bad, arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("vp_pair_reads", why)

    def test_mfma_first_scope(self):
        for v in (1, 2):
            self.assertEqual(
                is_valid_spec(_spec(vp_mfma_first=v), arch="gfx942"), (True, "ok")
            )
            m3 = _spec(
                (256, 256, 32, 2, 4, 1, 16, 16, 16),
                vgpr_prefetch=3,
                ab_double_buffer=True,
                vp_mfma_first=v,
            )
            self.assertEqual(is_valid_spec(m3, arch="gfx942"), (True, "ok"))
        self.assertFalse(is_valid_spec(_spec(vp_mfma_first=3), arch="gfx942")[0])
        self.assertFalse(
            is_valid_spec(_spec(vgpr_prefetch=0, vp_mfma_first=1), arch="gfx942")[0]
        )

    def test_mfma_first_adds_scheduling_barriers(self):
        # vp_mfma_first 1: one sched_barrier more behind the first MFMA of each
        # emitted K tile (loop tile + last tile); 2: behind every MFMA
        def n_sb(**kv):
            return _ir(_spec(**kv)).count("@llvm.amdgcn.sched.barrier(i32 0)")

        def n_mfma(**kv):
            return _events(_ir(_spec(**kv))).count("M")

        base = n_sb()
        self.assertEqual(n_sb(vp_mfma_first=1), base + 2)
        self.assertEqual(n_sb(vp_mfma_first=2), base + n_mfma())

    def test_deep_prefetch_with_agpr_refused(self):
        big = (224, 256, 64, 2, 2, 1, 16, 16, 16)  # 224 accumulators per lane
        for d in (2, 3):
            s = _spec(big, acc_agpr=True, vgpr_prefetch_depth=d)
            with self.subTest(depth=d):
                ok, why = is_valid_spec(s, arch="gfx942")
                self.assertFalse(ok)
                self.assertIn("acc_agpr", why)
        self.assertEqual(
            is_valid_spec(_spec(big, acc_agpr=True), arch="gfx942"), (True, "ok")
        )
        # VGPR accumulators keep depth 2
        self.assertEqual(
            is_valid_spec(_spec(vgpr_prefetch_depth=2), arch="gfx942"), (True, "ok")
        )

    def test_converted(self):
        # the C++ engine mirrors the main-loop levers (gemm_build_vp.cpp): the
        # converter passes each one through, none is on the Python-only list
        for kv in (
            dict(vp_release_atoms=2),
            dict(vp_tail_pack=1),
            dict(vp_pair_reads=1),
            dict(vp_mfma_first=1),
        ):
            with self.subTest(kv=kv):
                d = backend.universal_gemm_spec_to_dict(_spec(**kv))
                for k, v in kv.items():
                    self.assertEqual(d[k], v)
        levers = dict(backend._PYTHON_ONLY_GEMM_LEVERS)
        for k, off in (
            ("vp_release_atoms", 1),
            ("vp_tail_pack", 0),
            ("vp_pair_reads", 0),
            ("vp_mfma_first", 0),
        ):
            with self.subTest(lever=k):
                self.assertNotIn(k, levers)
                self.assertEqual(getattr(TraitSpec(), k), off)


class TestLdsPhases(unittest.TestCase):
    """The single LDS buffer is shared by the fragment reads of tile i and the
    writes of tile i + 1: a read phase and a write phase must never touch
    without a barrier between them (in either order)."""

    def _check(self, s):
        self.assertEqual(is_valid_spec(s, arch="gfx942"), (True, "ok"))
        ev = _lds_events(_ir(s))
        self.assertIn("S", ev)
        self.assertIn("L", ev)
        self.assertIsNone(re.search(r"SL|LS", ev), ev)
        return ev

    def test_default_and_knobs(self):
        geos = [
            (256, 128, 64, 4, 2, 1, 16, 16, 16),
            (256, 224, 64, 2, 2, 1, 16, 16, 16),
            (64, 64, 64, 4, 1, 1, 16, 16, 16),
            (128, 128, 128, 2, 4, 1, 16, 16, 16),
        ]
        knobs = [
            {},
            {"vp_release_atoms": 2},
            {"vp_release_atoms": 3},
            {"vp_tail_pack": 1},
            {"vp_tail_pack": 2},
            {"vp_tail_pack": 3},
            {"vp_pair_reads": 1},
            {"vp_pair_reads": 2},
            {"vp_pair_reads": 1, "vp_release_atoms": 3, "vp_tail_pack": 3},
            {"vp_pair_reads": 1, "vp_release_atoms": 2, "vp_unroll": 1},
            {"vp_mfma_first": 1},
            {"vp_mfma_first": 2, "vp_tail_pack": 3},
            {
                "vp_pair_reads": 1,
                "vp_release_atoms": 3,
                "vp_tail_pack": 3,
                "vp_mfma_first": 1,
            },
        ]
        for geo in geos:
            acc = geo[0] * geo[1] // (64 * geo[3] * geo[4])
            for kv in knobs:
                kv = dict(kv)
                if kv.get("vp_release_atoms", 1) >= geo[2] // 16:
                    continue
                s = _spec(geo, acc_agpr=acc > 128, pad_m=True, pad_n=True, **kv)
                with self.subTest(geo=geo, kv=kv):
                    self._check(s)

    def test_release_moves_the_first_barrier(self):
        # the loop's first barrier (LDS release) comes after (KA - R) atoms of
        # MFMAs of the first loop tile: R atoms earlier than the end of the tile
        for geo, mpa, agpr in (
            ((256, 128, 64, 4, 2, 1, 16, 16, 16), 16, False),
            ((256, 224, 64, 2, 2, 1, 16, 16, 16), 56, True),
        ):
            for r in (1, 2, 3):
                for extra in ({}, {"vp_pair_reads": 1}):
                    s = _spec(geo, vp_release_atoms=r, acc_agpr=agpr, **extra)
                    self._check(s)
                    ev = _events(_ir(s))
                    b = [i for i, c in enumerate(ev) if c == "B"]
                    with self.subTest(geo=geo, r=r, extra=extra):
                        self.assertEqual(ev[b[0] : b[1]].count("M"), (4 - r) * mpa)

    def _window(self, ev, strip="MX"):
        """Events of the loop tile's refill window (between its two barriers)
        without the given classes."""
        b = [i for i, c in enumerate(ev) if c == "B"]
        w = ev[b[1] + 1 : b[2]]
        for c in strip:
            w = w.replace(c, "")
        return w

    def test_default_window_unchanged(self):
        # R = 1 keeps the default window: the writes spread over MPA // 2 slots
        # (odd MPA: 7 x 5 = 35 MFMAs per atom), the second barrier behind the
        # MFMA of the slot after them
        ev = _events(_ir(_spec((112, 80, 64, 1, 1, 1, 16, 16, 16), acc_agpr=True)))
        b = [i for i, c in enumerate(ev) if c == "B"]
        w = ev[b[1] : b[2]]
        self.assertGreaterEqual(w.count("S"), 35 // 2)
        self.assertEqual(w.count("M"), 35 // 2 + 1)

    def test_writes_start_at_the_release(self):
        # the refill window opens with the LDS writes (no MFMA-only gap after
        # the first barrier beyond the slot of the barrier itself)
        for r in (1, 2, 3):
            ev = _events2(_ir(_spec(vp_release_atoms=r)))
            b = [i for i, c in enumerate(ev) if c == "B"]
            w = ev[b[1] + 1 : b[2]].replace("X", "")
            with self.subTest(r=r):
                self.assertLessEqual(w.index("S"), 2, w[:20])

    def test_pair_reads_window_order(self):
        # vp_pair_reads: the window holds every global read of the next tile,
        # each behind the LDS writes of the vector whose registers it reuses
        for pr in (1, 2):
            for r in (1, 3):
                ev = _events2(_ir(_spec(vp_pair_reads=pr, vp_release_atoms=r)))
                w = self._window(ev)
                with self.subTest(pr=pr, r=r):
                    self.assertRegex(w, r"^(S+G)+$")
        # default: the global reads are in the first atoms' free slots
        w = self._window(_events2(_ir(_spec())))
        self.assertNotIn("G", w)

    def test_tail_pack_density(self):
        # after the loop's second barrier: vp_tail_pack 3 issues one fragment
        # read per MFMA slot, 1 two per slot (one read = a fixed number of
        # LDS load ops, the first run of the prologue's read phase / reads)
        def runs(**kv):
            ev = _events2(_ir(_spec(**kv))).replace("X", "")
            b = [i for i, c in enumerate(ev) if c == "B"]
            tail = ev[b[2] + 1 :]
            return [len(x) for x in re.findall(r"L+", tail)[:4]]

        r3, r1 = runs(vp_tail_pack=3), runs(vp_tail_pack=1)
        self.assertEqual(len(set(r3)), 1, r3)
        self.assertEqual([2 * x for x in r3], r1)

    def test_mfma_first_barrier_position(self):
        # vp_mfma_first 1: the loop tile's first MFMA is followed by a
        # scheduling barrier before its slot instructions
        def after_first_mfma(**kv):
            ev = _events2(_ir(_spec(**kv)))
            b0 = ev.index("B")
            m0 = ev.index("M", b0)
            return ev[m0 + 1]

        self.assertEqual(after_first_mfma(vp_mfma_first=1), "X")
        self.assertNotEqual(after_first_mfma(), "X")


class TestMainloopDispatch(unittest.TestCase):
    """``g5_plan.G5P_MAINLOOP``: the measured main-loop traits applied to the
    dispatched single-buffer RCR plan within its scope (4-wave AGPR bodies),
    and the ranking scale of the plans they apply to."""

    KNOBS = (
        "G5P_MAINLOOP",
        "G5P_MAINLOOP_MIN_KTILES",
        "G5P_MAINLOOP_MIN_ACC",
        "G5P_MAINLOOP_MAX_ACC",
        "G5P_MAINLOOP_WAVES",
        "G5P_MAINLOOP_MIN_KATOMS",
        "G5P_MAINLOOP_SCALE",
    )

    def setUp(self):
        from rocke.dispatch.gemm import g5_plan as gp

        self._saved = {k: getattr(gp, k) for k in self.KNOBS}

    def tearDown(self):
        from rocke.dispatch.gemm import g5_plan as gp

        for k, v in self._saved.items():
            setattr(gp, k, v)
        self._clear()

    def _clear(self):
        from rocke.dispatch.gemm import common as C

        for nm in ("_G5P_CACHE", "_R2_PLAN_CACHE"):
            if hasattr(C, nm):
                getattr(C, nm).clear()

    def _pick(self, M, N, K, layout="RCR"):
        from rocke.dispatch import dispatch_gemm_fp16
        from rocke.dispatch.gemm.common import GemmRequest

        self._clear()
        return dispatch_gemm_fp16(
            GemmRequest(
                M=M,
                N=N,
                K=K,
                arch="gfx942",
                dtype="fp16",
                layout=layout,
                trans_a=layout[0] == "C",
                trans_b=layout[1] == "C",
            )
        )

    @staticmethod
    def _in_scope(s):
        acc = s.tile.tile_m * s.tile.tile_n // (64 * s.tile.warp_m * s.tile.warp_n)
        return (
            int(s.trait.vgpr_prefetch) == 4
            and s.tile.warp_m * s.tile.warp_n == 4
            and acc > 128
            and s.tile.tile_k // s.tile.warp_tile_k >= 4
        )

    def test_defaults(self):
        from rocke.dispatch.gemm import g5_plan as gp

        self.assertEqual(
            dict(gp.G5P_MAINLOOP),
            {
                "vp_pair_reads": 1,
                "vp_release_atoms": 3,
                "vp_tail_pack": 3,
                "vp_mfma_first": 1,
            },
        )
        self.assertEqual(
            (gp.G5P_MAINLOOP_MIN_ACC, tuple(gp.G5P_MAINLOOP_WAVES)), (129, (4,))
        )
        self.assertEqual(gp.G5P_MAINLOOP_MIN_KATOMS, 4)
        self.assertLess(gp.G5P_MAINLOOP_SCALE, 1.0)

    def test_scope_of_the_picks(self):
        # every dispatched register-staged RCR pick carries the traits iff it is in scope
        for M, N, K in (
            (8192, 8192, 8192),
            (16696, 6872, 19080),
            (1000, 3000, 512),
            (256, 256, 16384),
            (4096, 1024, 2048),
        ):
            d = self._pick(M, N, K)
            s = d.spec
            with self.subTest(M=M, N=N, K=K):
                self.assertEqual(is_valid_spec(s, arch="gfx942"), (True, "ok"))
                self.assertIsNone(launch_shape_error(s, M, N, K))
                if "vgpr" not in d.candidate.name:
                    continue
                on = self._in_scope(s)
                self.assertEqual(s.trait.vp_pair_reads, int(on))
                self.assertEqual(s.trait.vp_release_atoms, 3 if on else 1)
                self.assertEqual(s.trait.vp_tail_pack, 3 if on else 0)
                self.assertEqual(s.trait.vp_mfma_first, int(on))
        # a large problem picks an in-scope body
        self.assertTrue(self._in_scope(self._pick(8192, 8192, 8192).spec))

    def test_off_restores_round2(self):
        from rocke.dispatch.gemm import g5_plan as gp

        gp.G5P_MAINLOOP = {}
        s = self._pick(8192, 8192, 8192).spec
        for tag in ("_vpr", "_vtp", "_vpp", "_vmf"):
            self.assertNotIn(tag, s.kernel_name())
        # without the main-loop traits the pick for 8192^3 is the 8-wave 256x128 body
        self.assertEqual((s.tile.tile_m, s.tile.tile_n, s.tile.warp_m), (256, 128, 4))

    def test_scale_moves_the_ranking(self):
        from rocke.dispatch.gemm import g5_plan as gp

        gp.G5P_MAINLOOP_SCALE = 1.0
        s1 = self._pick(8192, 8192, 8192).spec
        gp.G5P_MAINLOOP_SCALE = 0.95
        s2 = self._pick(8192, 8192, 8192).spec
        self.assertFalse(self._in_scope(s1))
        self.assertTrue(self._in_scope(s2))

    def test_plan_mainloop_matches_mainloop_applies(self):
        from rocke.dispatch.gemm import g5_plan as gp

        menu = gp.load_menu("gfx942")
        for geo in sorted(menu):
            for M, N, K in ((8192, 8192, 8192), (256, 512, 256)):
                p = gp.G5Plan(geo, 1, 0)
                tile, trait = gp.plan_tile_trait(p, M, N, K)
                with self.subTest(geo=geo, K=K):
                    self.assertEqual(
                        gp.plan_mainloop(p, M, N, K),
                        gp.mainloop_applies(tile, trait, M, N, K),
                    )
        self.assertFalse(gp.plan_mainloop(gp.G5Plan("skinny"), 1, 1, 1))

    def test_mode3_plans_never_in_scope(self):
        from rocke.dispatch.gemm import g5_plan as gp

        gp.G5P_MAINLOOP_MIN_ACC = 0
        gp.G5P_MAINLOOP_WAVES = ()
        gp.G5P_MAINLOOP_MIN_KATOMS = 0
        m3 = [g for g in gp.load_menu("gfx942") if gp.parse_geo(g)[5] == 3]
        self.assertTrue(m3)
        for g in m3:
            with self.subTest(geo=g):
                self.assertFalse(gp.plan_mainloop(gp.G5Plan(g, 1, 0), 8192, 8192, 8192))

    def test_ranking_scale_is_rcr_only(self):
        from dataclasses import replace as _r

        from rocke.dispatch.gemm import common as C
        from rocke.dispatch.gemm.common import GemmRequest

        rcr = GemmRequest(M=8192, N=8192, K=8192, arch="gfx942", dtype="fp16")
        nn = _r(rcr, layout="RRR", trans_a=False, trans_b=False)
        self._clear()
        a = [(round(t, 6), p.geo) for t, p in C._g5p_ranked(nn)]
        b = [(round(t, 6), p.geo) for t, p in C._g5p_ranked(rcr, mainloop=False)]
        c = [(round(t, 6), p.geo) for t, p in C._g5p_ranked(rcr)]
        self.assertEqual(a, b)
        self.assertNotEqual(b, c)

    def test_cache_key_tracks_the_knobs(self):
        from rocke.dispatch import dispatch_gemm_fp16
        from rocke.dispatch.gemm import g5_plan as gp
        from rocke.dispatch.gemm.common import GemmRequest

        req = GemmRequest(M=8192, N=8192, K=8192, arch="gfx942", dtype="fp16")
        self._clear()
        on = dispatch_gemm_fp16(req).spec  # cached under the default knobs
        gp.G5P_MAINLOOP = {}
        off = dispatch_gemm_fp16(req).spec  # no cache clear
        self.assertTrue(self._in_scope(on))
        self.assertNotEqual(on.kernel_name(), off.kernel_name())
        self._clear()
        fresh = dispatch_gemm_fp16(req).spec
        self.assertEqual(off.kernel_name(), fresh.kernel_name())
        # the scale alone too
        gp.G5P_MAINLOOP = dict(self._saved["G5P_MAINLOOP"])
        self._clear()
        dispatch_gemm_fp16(req)
        gp.G5P_MAINLOOP_SCALE = 1.0
        stale = dispatch_gemm_fp16(req).spec
        self._clear()
        self.assertEqual(
            stale.kernel_name(), dispatch_gemm_fp16(req).spec.kernel_name()
        )

    def test_plan_module_calibration_is_unscaled(self):
        # the fitted RCR estimate is of the body without main-loop traits: its
        # calibration ratio must not move with the main-loop scale, while the
        # plan model's time of a main-loop body does (and a native form's not)
        from dataclasses import replace as _r

        from rocke.dispatch import dispatch_gemm_fp16
        from rocke.dispatch.gemm import g5_plan as gp
        from rocke.dispatch.gemm import plan as P
        from rocke.dispatch.gemm.common import GemmRequest

        req = GemmRequest(M=8192, N=8192, K=8192, arch="gfx942", dtype="fp16")
        self._clear()
        spec = dispatch_gemm_fp16(req).spec  # an in-scope body (default knobs)
        self.assertTrue(self._in_scope(spec))
        plain = _r(spec, trait=_r(spec.trait, vp_pair_reads=0, vp_mfma_first=0))
        cals, preds, npreds = [], [], []
        for sc in (1.0, 0.8):
            gp.G5P_MAINLOOP_SCALE = sc
            self._clear()
            cals.append(P._g5p_cal(req))
            preds.append(P._g5p_plan_pred(req, spec))
            npreds.append(P._g5p_plan_pred(req, plain))
        self.assertIsNotNone(cals[0])
        self.assertAlmostEqual(cals[0], cals[1], places=9)
        self.assertIsNotNone(preds[0])
        self.assertLess(preds[1], preds[0] * 0.9)
        self.assertIsNotNone(npreds[0])
        self.assertAlmostEqual(npreds[0], npreds[1], places=6)

    def test_applied_valid_and_clipped(self):
        from rocke.dispatch.gemm import g5_plan as gp

        # scope wide open: every mode-4 RCR pick gets the traits, the release
        # count clipped to the tile's K atoms - 1
        gp.G5P_MAINLOOP = {
            "vp_pair_reads": 1,
            "vp_release_atoms": 7,
            "vp_tail_pack": 3,
            "vp_mfma_first": 1,
        }
        gp.G5P_MAINLOOP_MIN_ACC = 0
        gp.G5P_MAINLOOP_WAVES = ()
        gp.G5P_MAINLOOP_MIN_KATOMS = 0
        for M, N, K in ((8192, 8192, 8192), (1000, 3000, 512), (256, 256, 16384)):
            d = self._pick(M, N, K)
            s = d.spec
            with self.subTest(M=M, N=N, K=K):
                self.assertEqual(is_valid_spec(s, arch="gfx942"), (True, "ok"))
                self.assertIsNone(launch_shape_error(s, M, N, K))
                if "vgpr" in d.candidate.name and s.trait.vgpr_prefetch == 4:
                    ka = s.tile.tile_k // s.tile.warp_tile_k
                    self.assertEqual(s.trait.vp_pair_reads, 1)
                    self.assertEqual(s.trait.vp_release_atoms, ka - 1)
        # the native forms keep their loader (vp_pair_reads is RCR only)
        n = self._pick(8192, 8192, 8192, layout="RRR").spec
        self.assertEqual(n.trait.vp_pair_reads, 0)
        # scope bounds
        gp.G5P_MAINLOOP_MIN_KTILES = 10**6
        self.assertEqual(self._pick(8192, 8192, 8192).spec.trait.vp_pair_reads, 0)
        gp.G5P_MAINLOOP_MIN_KTILES = 0
        gp.G5P_MAINLOOP_MAX_ACC = 1
        self.assertEqual(self._pick(8192, 8192, 8192).spec.trait.vp_pair_reads, 0)
        gp.G5P_MAINLOOP_MAX_ACC = 0
        gp.G5P_MAINLOOP_MIN_ACC = 10**6
        self.assertEqual(self._pick(8192, 8192, 8192).spec.trait.vp_pair_reads, 0)
        gp.G5P_MAINLOOP_MIN_ACC = 0
        gp.G5P_MAINLOOP_WAVES = (3,)
        self.assertEqual(self._pick(8192, 8192, 8192).spec.trait.vp_pair_reads, 0)
        gp.G5P_MAINLOOP_WAVES = ()
        gp.G5P_MAINLOOP_MIN_KATOMS = 99
        self.assertEqual(self._pick(8192, 8192, 8192).spec.trait.vp_pair_reads, 0)
        gp.G5P_MAINLOOP_MIN_KATOMS = 0
        # an invalid combination (pair + tail pack 2) leaves the plan as is
        gp.G5P_MAINLOOP = {"vp_pair_reads": 1, "vp_tail_pack": 2}
        d = self._pick(8192, 8192, 8192)
        s = d.spec
        self.assertIn("vgpr", d.candidate.name)
        self.assertEqual(is_valid_spec(s, arch="gfx942"), (True, "ok"))
        self.assertEqual((s.trait.vp_pair_reads, s.trait.vp_tail_pack), (0, 0))


class TestLaunchablePlan(unittest.TestCase):
    def _pick(self, dt, M, N, K):
        from rocke.dispatch import dispatch_gemm_bf16, dispatch_gemm_fp16
        from rocke.dispatch.gemm.common import GemmRequest

        fn = dispatch_gemm_fp16 if dt == "fp16" else dispatch_gemm_bf16
        return fn(GemmRequest(M=M, N=N, K=K, arch="gfx942", dtype=dt))

    def _clear(self):
        from rocke.dispatch.gemm import common as C

        for nm in ("_G5P_CACHE", "_R2_PLAN_CACHE"):
            if hasattr(C, nm):
                getattr(C, nm).clear()

    def test_large_operand_takes_a_launchable_plan(self):
        from rocke.dispatch.gemm import g5_plan as gp

        for dt in ("fp16", "bf16"):
            for M, N in ((128, 32768), (32768, 128)):
                self._clear()
                d = self._pick(dt, M, N, 32768)
                with self.subTest(dt=dt, M=M, N=N):
                    self.assertIn("vgpr", d.candidate.name)
                    self.assertIsNone(launch_shape_error(d.spec, M, N, 32768))
        old = gp.G5P_LAUNCHABLE_TOP
        try:
            gp.G5P_LAUNCHABLE_TOP = 1
            self._clear()
            d = self._pick("fp16", 128, 32768, 32768)
            self.assertNotIn("vgpr", d.candidate.name)
        finally:
            gp.G5P_LAUNCHABLE_TOP = old
            self._clear()

    def test_launchable_top_keeps_the_top_plan(self):
        # where the top plan launches, the knob changes nothing
        from rocke.dispatch.gemm import g5_plan as gp

        old = gp.G5P_LAUNCHABLE_TOP
        try:
            names = []
            for top in (1, old):
                gp.G5P_LAUNCHABLE_TOP = top
                self._clear()
                names.append(self._pick("fp16", 4096, 4096, 4096).spec.kernel_name())
            self.assertEqual(names[0], names[1])
        finally:
            gp.G5P_LAUNCHABLE_TOP = old
            self._clear()


if __name__ == "__main__":
    unittest.main()
