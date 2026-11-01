# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``TraitSpec.sab_vec16``: the 16-byte swapped-accumulator epilogue of an
N-contiguous B on the gfx950 DTL ring.

CPU only. The permuted transpose-read quads (``MnLayout.sub_perm``) keep each
16-lane group's address set (same banks); a model of the data flow (fragment
lanes -> swapped MFMA output -> ``v_permlane32_swap`` -> 16-byte stores)
writes every C element of a warp tile exactly once with the right value; the
validity rules; the emitted IR; the C++ converter refusal; the native form.
"""

from __future__ import annotations

import dataclasses
import unittest

from rocke.dispatch.gemm import GemmRequest, dispatch_gemm_fp16
from rocke.dispatch.gemm import common as GC
from rocke.instances.common import gemm_mn_operand as MNO
from rocke.instances.common.gemm_universal import build_universal_gemm, is_valid_spec


def _beat_specs(M=8192, N=8192, K=8192, lay="RRR"):
    rr = dispatch_gemm_fp16(GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype="fp16"))
    req = GemmRequest(
        M=M,
        N=N,
        K=K,
        arch="gfx950",
        dtype="fp16",
        layout=lay,
        trans_a=lay[0] == "C",
        trans_b=lay[1] == "C",
    )
    return dict(GC.native_layout_alternatives(req, rr.spec)[0])


class TestQuadPerm(unittest.TestCase):
    def test_involution(self):
        self.assertEqual(sorted(MNO.QUAD_PERM), [0, 1, 2, 3])
        for s in range(4):
            self.assertEqual(MNO.QUAD_PERM[MNO.QUAD_PERM[s]], s)
            self.assertEqual(MNO.QUAD_PERM[s], 2 * (s % 2) + s // 2)

    def test_same_address_set_per_group(self):
        kw = dict(tile_mn=256, tile_k=64, block_size=256, wave_size=64, warp_ext=128)
        a = MNO.MnLayout(**kw)
        b = MNO.MnLayout(**kw, sub_perm=True)
        for kk in range(2):
            for r in range(2):
                for g0 in range(0, 64, 16):
                    sa = sorted(a.read_addr(l, 0, 0, r, kk) for l in range(g0, g0 + 16))
                    sb = sorted(b.read_addr(l, 0, 0, r, kk) for l in range(g0, g0 + 16))
                    self.assertEqual(sa, sb)
        for lane in range(64):
            k0, s0 = a.read_lane_k_mn(lane, 0, 0)
            k1, s1 = b.read_lane_k_mn(lane, 0, 0)
            self.assertEqual(k0, k1)
            self.assertEqual(s1, 4 * MNO.QUAD_PERM[s0 // 4])

    def test_spec_layout_sub_perm_only_on_b(self):
        sv = _beat_specs()["no_rve_sv16"]
        la, lb = MNO.spec_mn_layouts(sv)
        self.assertIsNone(la)
        self.assertTrue(lb.sub_perm)
        ca, cb = MNO.spec_mn_layouts(_beat_specs(lay="CRR")["b3"])
        for l_ in (ca, cb):
            if l_ is not None:
                self.assertFalse(l_.sub_perm)


class _IntBuilder:
    """Integer stand-in for the IRBuilder arithmetic the MN reader emitters use."""

    def const_i32(self, x):
        return int(x)

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def div(self, a, b):
        return a // b

    def mod(self, a, b):
        return a % b


class TestEmitterMatchesGeometry(unittest.TestCase):
    """The emitted lane-address arithmetic (``emit_read_lane_elems``) equals
    the Python geometry (``read_lane_bytes`` + the wave origin), with and
    without the quad permutation."""

    def test_lane_elems(self):
        for chunk in (0, 64, 128, 256):
            for perm in (False, True):
                try:
                    lay = MNO.MnLayout(
                        tile_mn=256,
                        tile_k=64,
                        block_size=256,
                        wave_size=64,
                        warp_ext=128,
                        chunk=chunk,
                        sub_perm=perm,
                    )
                except ValueError:
                    continue
                for w in range(2):
                    for lane in range(64):
                        got = MNO.emit_read_lane_elems(_IntBuilder(), lay, lane, w)
                        want = (lay.read_lane_bytes(lane) + lay.warp_bytes(w)) // 2
                        self.assertEqual(got, want, (chunk, perm, w, lane))


class _LaneBuilder:
    """64-lane stand-in for the IRBuilder ops of ``_emit_store_swapped_acc_vec16``:
    scalars are per-lane integer arrays, vectors lists of per-lane object arrays
    (element tags), ``permlane32_swap`` follows ``v_permlane32_swap_b32`` (vdst
    lanes 32..63 <-> src0 lanes 0..31), stores are recorded per lane."""

    def __init__(self):
        import numpy as np

        self.np = np
        self.stores = []

    def const_i32(self, x):
        return int(x)

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def div(self, a, b):
        return a // b

    def mod(self, a, b):
        return a % b

    def cmp_ge(self, a, b):
        return a >= b

    def cmp_lt(self, a, b):
        return a < b

    def select(self, c, x, y):
        return self.np.where(c, x, y)

    def vec_cast_f32_to(self, v, dt):
        return v

    def vec_extract(self, v, j):
        return v[j]

    def vec_bitcast(self, v, ty):
        np = self.np
        n = ty.n if hasattr(ty, "n") else ty.count
        if len(v) > n:  # 2-byte elements -> dwords
            k = len(v) // n
            out = []
            for j in range(n):
                a = np.empty(64, dtype=object)
                for l_ in range(64):
                    a[l_] = tuple(v[j * k + i][l_] for i in range(k))
                out.append(a)
            return out
        out = []  # dwords -> 2-byte elements
        for e in v:
            k = len(e[0])
            for i in range(k):
                a = np.empty(64, dtype=object)
                for l_ in range(64):
                    a[l_] = e[l_][i]
                out.append(a)
        return out

    def vec_pack(self, comps, ty):
        return list(comps)

    def permlane32_swap(self, a, b):
        na, nb = a.copy(), b.copy()
        na[32:] = b[:32]
        nb[:32] = a[32:]
        return na, nb

    def global_store_vN(self, C, off, vec, n, align=0, temporal_hint=None):
        for l_ in range(64):
            self.stores.append(
                (
                    int(off[l_]) if hasattr(off, "__len__") else int(off),
                    [vec[j][l_] for j in range(n)],
                )
            )


class TestEpilogueEmitter(unittest.TestCase):
    """Runs ``_emit_store_swapped_acc_vec16`` on the lane builder: every C
    element of the warp tile is stored once, with the accumulator slot the
    swapped MFMA layout + permuted B quads put there."""

    def test_stores(self):
        from unittest import mock

        import numpy as np
        from rocke.instances.common import gemm_universal as GU

        sv = _beat_specs()["no_rve_sv16"]
        t = sv.tile
        mm, mn = t.mfmas_per_warp_m, t.mfmas_per_warp_n
        N = 16 * mn
        accs = []
        for i in range(mm * mn):
            vec = []
            for v in range(4):
                a = np.empty(64, dtype=object)
                for l_ in range(64):
                    a[l_] = (i, v, l_)
                vec.append(a)
            accs.append(vec)
        lb = _LaneBuilder()
        lane = np.arange(64)
        with mock.patch.object(GU, "_acc_agpr_out", lambda b, spec, acc: acc):
            GU._emit_store_swapped_acc_vec16(
                lb, sv, lane, lane % 16, accs, 0, 0, 4, GU.F16, 16 * mm, N, None
            )
        seen = {}
        for off, vals in lb.stores:
            row, col = divmod(off, N)
            for j, (i, v, l_) in enumerate(vals):
                mi, ni = divmod(i, mn)
                # swapped layout: lane l_ holds C row 16 mi + l_ % 16, columns
                # 16 ni + 4 QUAD_PERM[l_ // 16] + v
                self.assertEqual(row, 16 * mi + l_ % 16)
                self.assertEqual(col + j, 16 * ni + 4 * MNO.QUAD_PERM[l_ // 16] + v)
                seen[(row, col + j)] = seen.get((row, col + j), 0) + 1
        self.assertEqual(len(seen), 16 * mm * N)
        self.assertTrue(all(c == 1 for c in seen.values()))


class TestDataFlowModel(unittest.TestCase):
    """Model of one warp tile: B fragment lanes from the permuted transpose
    reads, the C^T MFMA output (lane L: rows 4q..4q+3 = the fragment lanes'
    N, column L % 16 = M), the permlane32 swap and the store offsets of
    ``_emit_store_swapped_acc_vec16``."""

    def test_every_element_once(self):
        mfmas_m, mfmas_n = 4, 2
        lay = MNO.MnLayout(
            tile_mn=256,
            tile_k=64,
            block_size=256,
            wave_size=64,
            warp_ext=128,
            sub_perm=True,
        )

        def frag_n(n_idx):
            # fragment lane n (0..15) receives the MN quad of the source lanes
            # whose quad slot is n // 4 (lane % 4 == n // 4), element n % 4
            src = n_idx // 4
            return lay.read_lane_k_mn(src, 0, 0)[1] + n_idx % 4

        def acc(mi, ni, lane):
            q, c = lane // 16, lane % 16
            return [(16 * mi + c, 16 * ni + frag_n(4 * q + v)) for v in range(4)]

        written = {}
        for mp in range(0, mfmas_m, 2):
            for ni in range(mfmas_n):
                lo = {L: acc(mp, ni, L) for L in range(64)}
                hi = {L: acc(mp + 1, ni, L) for L in range(64)}
                for L in range(64):
                    # dwords: (v0, v1), (v2, v3); vdst = atom mp, src0 = atom mp+1
                    if L < 32:
                        x = lo[L]  # own
                        y = lo[L + 32]  # high lane's atom-mp quad
                    else:
                        x = hi[L - 32]  # low lane's atom-(mp+1) quad
                        y = hi[L]
                    # vec = [x.dword0 (v0 v1), x.dword1 (v2 v3), y.dword0, y.dword1]
                    vals = [x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]]
                    c_m = 16 * (mp + (1 if L >= 32 else 0)) + L % 16
                    c_n0 = 16 * ni + 8 * ((L // 16) % 2)
                    for j, (m_, n_) in enumerate(vals):
                        self.assertEqual((m_, n_), (c_m, c_n0 + j), (mp, ni, L, j))
                        written[(c_m, c_n0 + j)] = written.get((c_m, c_n0 + j), 0) + 1
        self.assertEqual(len(written), 16 * mfmas_m * 16 * mfmas_n)
        self.assertTrue(all(v == 1 for v in written.values()))

    def test_without_perm_the_pairs_are_not_contiguous(self):
        lay = MNO.MnLayout(
            tile_mn=256, tile_k=64, block_size=256, wave_size=64, warp_ext=128
        )
        n_lo = lay.read_lane_k_mn(0, 0, 0)[1]  # quad of fragment lanes 0..3 (q = 0)
        n_hi = lay.read_lane_k_mn(2, 0, 0)[1]  # q = 2
        self.assertNotEqual(n_hi, n_lo + 4)


class TestValidityAndEmission(unittest.TestCase):
    def setUp(self):
        self.alts = _beat_specs()

    def test_form_present_and_valid(self):
        sv = self.alts["no_rve_sv16"]
        self.assertTrue(sv.trait.mfma_swap_ab and sv.trait.sab_vec16)
        self.assertTrue(is_valid_spec(sv, arch="gfx950")[0])
        self.assertIn("_sv16", sv.kernel_name())
        self.assertNotIn("_sv16", self.alts["no_rve_sab"].kernel_name())

    def test_rule_pick_unchanged(self):
        n_sv = 0
        for M, N, K in (
            (8192, 8192, 8192),
            (4608, 4608, 384),
            (7936, 17664, 512),
            (4096, 4096, 1024),
        ):
            req = GemmRequest(
                M=M, N=N, K=K, arch="gfx950", dtype="fp16", layout="RRR", trans_b=False
            )
            rr = dispatch_gemm_fp16(
                GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype="fp16")
            )
            a, _ = GC.native_layout_spec(req, rr.spec)
            n_sv += "no_rve_sv16" in dict(
                GC.native_layout_alternatives(req, rr.spec)[0]
            )
            old = GC.NATIVE_LAYOUT_SAB_VEC16
            try:
                GC.NATIVE_LAYOUT_SAB_VEC16 = False
                b, _ = GC.native_layout_spec(req, rr.spec)
                alts, _ = GC.native_layout_alternatives(req, rr.spec)
                self.assertNotIn("no_rve_sv16", dict(alts))
            finally:
                GC.NATIVE_LAYOUT_SAB_VEC16 = old
            self.assertEqual(a, b, (M, N, K))
        self.assertGreaterEqual(n_sv, 2)

    def test_refusals(self):
        sv = self.alts["no_rve_sv16"]
        tr = sv.trait
        bad = {
            "no swap": dataclasses.replace(
                sv, trait=dataclasses.replace(tr, mfma_swap_ab=False)
            ),
            "rcr": dataclasses.replace(
                sv, data=dataclasses.replace(sv.data, layout="RCR")
            ),
            "a only": dataclasses.replace(
                sv, data=dataclasses.replace(sv.data, layout="CCR")
            ),
        }
        for nm, s in bad.items():
            self.assertFalse(is_valid_spec(s, arch="gfx950")[0], nm)
        self.assertFalse(is_valid_spec(sv, arch="gfx942")[0])
        odd = dataclasses.replace(
            sv, tile=dataclasses.replace(sv.tile, tile_m=48 * 2, warp_m=2)
        )
        self.assertFalse(is_valid_spec(odd, arch="gfx950")[0])

    def test_ir_has_swaps_and_wide_stores(self):
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

        sab = lower(
            build_universal_gemm(self.alts["no_rve_sab"]),
            llvm_flavor="llvm22",
            arch="gfx950",
        )
        sv = lower(
            build_universal_gemm(self.alts["no_rve_sv16"]),
            llvm_flavor="llvm22",
            arch="gfx950",
        )
        self.assertEqual(sab.count("permlane32.swap("), 0)
        # one swap per dword: 2 per (M atom pair, N atom) = mfmas_m * mfmas_n
        t = self.alts["no_rve_sv16"].tile
        self.assertEqual(
            sv.count("call { i32, i32 } @llvm.amdgcn.permlane32.swap("),
            t.mfmas_per_warp_m * t.mfmas_per_warp_n,
        )
        self.assertGreater(sv.count("store <8 x half>"), 0)
        self.assertEqual(sab.count("store <8 x half>"), 0)

    def test_store_nt(self):
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

        sv = self.alts["no_rve_sv16"]
        nt = dataclasses.replace(
            sv, trait=dataclasses.replace(sv.trait, sab_store_nt=True)
        )
        self.assertTrue(is_valid_spec(nt, arch="gfx950")[0])
        self.assertIn("_svnt", nt.kernel_name())
        bad = dataclasses.replace(
            sv, trait=dataclasses.replace(sv.trait, sab_vec16=False, sab_store_nt=True)
        )
        self.assertFalse(is_valid_spec(bad, arch="gfx950")[0])
        a = lower(build_universal_gemm(sv), llvm_flavor="llvm22", arch="gfx950")
        b = lower(build_universal_gemm(nt), llvm_flavor="llvm22", arch="gfx950")
        self.assertGreater(b.count("!nontemporal"), a.count("!nontemporal"))

    def test_ring_forms(self):
        req = GemmRequest(
            M=7936,
            N=17664,
            K=512,
            arch="gfx950",
            dtype="fp16",
            layout="CRR",
            trans_a=True,
            trans_b=False,
        )
        rr = dispatch_gemm_fp16(
            GemmRequest(M=7936, N=17664, K=512, arch="gfx950", dtype="fp16")
        )
        self.assertFalse(GC.NATIVE_LAYOUT_SAB_VEC16_RINGS)  # default off
        self.assertNotIn(
            "b3_sv16", dict(GC.native_layout_alternatives(req, rr.spec)[0])
        )
        old = GC.NATIVE_LAYOUT_SAB_VEC16_RINGS, GC.NATIVE_LAYOUT_SV16_NT
        try:
            GC.NATIVE_LAYOUT_SAB_VEC16_RINGS = True
            alts = dict(GC.native_layout_alternatives(req, rr.spec)[0])
            self.assertIn("b3_sv16", alts)
            s = alts["b3_sv16"]
            self.assertTrue(s.trait.sab_vec16 and s.trait.mfma_swap_ab)
            self.assertEqual(s.trait.epilogue, "default")
            self.assertEqual(GC.native_layout_spec(req, rr.spec)[0], alts["b3"])
            GC.NATIVE_LAYOUT_SV16_NT = True
            al2 = dict(GC.native_layout_alternatives(req, rr.spec)[0])
            self.assertTrue(al2["b3_sv16"].trait.sab_store_nt)
            self.assertFalse(al2["b3"].trait.sab_store_nt)
        finally:
            GC.NATIVE_LAYOUT_SAB_VEC16_RINGS, GC.NATIVE_LAYOUT_SV16_NT = old

    def test_sub_perm_only_b_on_both_mn(self):
        req = GemmRequest(
            M=7936,
            N=17664,
            K=512,
            arch="gfx950",
            dtype="fp16",
            layout="CRR",
            trans_a=True,
            trans_b=False,
        )
        rr = dispatch_gemm_fp16(
            GemmRequest(M=7936, N=17664, K=512, arch="gfx950", dtype="fp16")
        )
        old = GC.NATIVE_LAYOUT_SAB_VEC16_RINGS
        try:
            GC.NATIVE_LAYOUT_SAB_VEC16_RINGS = True
            s = dict(GC.native_layout_alternatives(req, rr.spec)[0])["b3_sv16"]
        finally:
            GC.NATIVE_LAYOUT_SAB_VEC16_RINGS = old
        la, lb = MNO.spec_mn_layouts(s)
        self.assertFalse(la.sub_perm)
        self.assertTrue(lb.sub_perm)

    def test_error_rules_direct(self):
        from types import SimpleNamespace

        from rocke.instances.common.gemm_universal import _sab_vec16_error

        sv = self.alts["no_rve_sv16"]
        self.assertIsNone(_sab_vec16_error(sv, "gfx950"))
        odd_tile = SimpleNamespace(
            warp_tile_m=16, warp_tile_n=16, mfmas_per_warp_m=3, mfmas_per_warp_n=8
        )
        fake = SimpleNamespace(
            tile=odd_tile, trait=sv.trait, wave_size=64, data=sv.data, batched=False
        )
        self.assertIn("even", _sab_vec16_error(fake, "gfx950"))
        fake2 = SimpleNamespace(
            tile=sv.tile, trait=sv.trait, wave_size=64, data=sv.data, batched=True
        )
        self.assertIn("non-batched", _sab_vec16_error(fake2, "gfx950"))

    def test_cpp_converter_refuses(self):
        from rocke.core.backend import _refuse_python_only_levers

        from rocke.instances.common.gemm_universal import TraitSpec

        with self.assertRaises(NotImplementedError) as cm:
            _refuse_python_only_levers(TraitSpec(mfma_swap_ab=True, sab_vec16=True))
        self.assertIn("sab_vec16", str(cm.exception))
        _refuse_python_only_levers(TraitSpec(mfma_swap_ab=True))


if __name__ == "__main__":
    unittest.main()
