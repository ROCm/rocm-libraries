# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gemm_skinny (register-direct split-K GEMM) spec rules and emission (CPU only).

GPU numerics (random and integer-exact data, odd K, empty split slices, the
self-reset of the tile counters, run-to-run determinism) are covered by the
GPU harness on gfx950; these tests pin the static contract.
"""

from __future__ import annotations

import dataclasses
import unittest

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_skinny import (
    SkinnyGemmSpec,
    build_gemm_skinny,
    is_valid_spec,
    skinny_counter_ints,
    skinny_grid,
    skinny_kps,
    skinny_launch_error,
    skinny_signature,
    skinny_steps,
    skinny_workspace_floats,
)


def _llvm(spec, arch="gfx950"):
    # the run's backend (ROCKE_BACKEND): under "both" every emission checked
    # here is also compared byte for byte with the C++ engine (the
    # engine lowers the scalar ``tile.buffer_load`` of the kalign-1 loads)
    return lower_kernel_to_llvm(build_gemm_skinny(spec, arch), arch=arch)


class TestSkinnyValidity(unittest.TestCase):
    def test_default_is_valid_on_cdna(self):
        for arch in ("gfx942", "gfx950"):
            ok, why = is_valid_spec(SkinnyGemmSpec(), arch)
            self.assertTrue(ok, why)

    def test_rejects_other_arches(self):
        for arch in ("gfx90a", "gfx1151", "gfx1201"):
            self.assertFalse(is_valid_spec(SkinnyGemmSpec(), arch)[0])

    def test_rejects_bad_fields(self):
        bad = (
            dict(dtype="fp32"),
            dict(kc=24),
            dict(prefetch=0),
            dict(prefetch=5),
            dict(red_unroll=3),
            dict(red_sync="system"),
            dict(kmap="row"),
            dict(kalign=4),
            dict(b_load="lds"),
            dict(split=0),
            dict(warp_m=4, warp_n=4, warp_k=2),  # 32 waves > 1024 threads
            dict(wave_m_atoms=0),
        )
        for kw in bad:
            with self.subTest(kw=kw):
                self.assertFalse(is_valid_spec(SkinnyGemmSpec(**kw), "gfx950")[0])

    def test_register_and_lds_budgets(self):
        # 8x8 atoms per wave: 256 accumulator registers plus fragments
        self.assertFalse(
            is_valid_spec(SkinnyGemmSpec(wave_m_atoms=8, wave_n_atoms=8), "gfx950")[0]
        )
        # 8 K-waves of a 256x256 tile do not fit the LDS reduce buffer
        spec = SkinnyGemmSpec(
            wave_m_atoms=4,
            wave_n_atoms=4,
            warp_m=2,
            warp_n=2,
            warp_k=4,
            kc=8,
            prefetch=1,
        )
        ok, why = is_valid_spec(spec, "gfx942")
        self.assertFalse(ok)
        self.assertIn("LDS", why)

    def test_warp_k_reduce_needs_no_divisibility(self):
        # fewer tile float4s than threads (16x16 tile, 4 K-waves) is valid
        ok, why = is_valid_spec(SkinnyGemmSpec(wave_n_atoms=1, warp_k=4), "gfx950")
        self.assertTrue(ok, why)


class TestSkinnyLaunchContract(unittest.TestCase):
    def test_kalign(self):
        s8 = SkinnyGemmSpec()
        self.assertIsNone(skinny_launch_error(s8, 7, 192, 4096, 1))
        self.assertIn("kalign", skinny_launch_error(s8, 7, 192, 4098, 1))
        s2 = dataclasses.replace(s8, kalign=2)
        self.assertIsNone(skinny_launch_error(s2, 7, 192, 4098, 1))
        self.assertIn("kalign", skinny_launch_error(s2, 7, 192, 4097, 1))
        s1 = dataclasses.replace(s8, kalign=1)
        self.assertIsNone(skinny_launch_error(s1, 1, 1, 1, 1))

    def test_split_bounds_and_grid(self):
        s = SkinnyGemmSpec(wave_m_atoms=2, wave_n_atoms=4, warp_n=2)
        self.assertEqual(
            skinny_grid(s, 33, 1000, 7), (-(-1000 // 128), -(-33 // 32), 7)
        )
        self.assertIn("split", skinny_launch_error(s, 33, 1000, 4096, 0))
        self.assertIn("split", skinny_launch_error(s, 33, 1000, 4096, 70000))
        self.assertIn("M, N, K", skinny_launch_error(s, 0, 1000, 4096, 1))

    def test_workspace_and_counters(self):
        s = SkinnyGemmSpec(wave_n_atoms=2)  # 16 x 32 tile
        self.assertEqual(skinny_workspace_floats(s, 5, 100, 1), 0)
        gx, gy = -(-100 // 32), -(-5 // 16)
        self.assertEqual(skinny_workspace_floats(s, 5, 100, 6), 6 * gx * gy * 16 * 32)
        self.assertEqual(skinny_counter_ints(s, 5, 100), gx * gy)

    def test_steps_per_slice_cover_k(self):
        s = SkinnyGemmSpec(kc=16)  # 64 K per step
        for K, split in ((1, 1), (64, 3), (65, 2), (26816, 13), (4096, 1000)):
            with self.subTest(K=K, split=split):
                self.assertGreaterEqual(
                    skinny_kps(s, K, split) * split, skinny_steps(s, K)
                )

    def test_signature_matches_kernel_params(self):
        for dt in ("fp16", "bf16"):
            spec = SkinnyGemmSpec(dtype=dt)
            sig = [a["name"] for a in skinny_signature(spec)]
            params = [p.name for p in build_gemm_skinny(spec, "gfx950").params]
            self.assertEqual(sig, params)
            self.assertEqual(
                sig, ["A", "B", "C", "W", "Cnt", "M", "N", "K", "SPLIT", "KPS"]
            )


class TestSkinnyEmission(unittest.TestCase):
    def test_split_is_launch_only(self):
        base = SkinnyGemmSpec(wave_n_atoms=2, warp_k=2, red_sync="sc1")
        ir1 = _llvm(base)
        ir7 = _llvm(dataclasses.replace(base, split=7))
        self.assertEqual(ir1, ir7)
        self.assertEqual(
            base.kernel_name(), dataclasses.replace(base, split=7).kernel_name()
        )

    def test_kernel_names_distinguish_codegen_fields(self):
        base = SkinnyGemmSpec()
        variants = [
            dict(dtype="fp16"),
            dict(wave_m_atoms=2),
            dict(wave_n_atoms=2),
            dict(warp_m=2),
            dict(warp_n=2),
            dict(warp_k=2),
            dict(kc=32),
            dict(prefetch=3),
            dict(waves_per_eu=2),
            dict(nt_ws=True),
            dict(red_unroll=8),
            dict(kalign=1),
            dict(kmap="lane"),
            dict(red_sync="sc1"),
            dict(b_load="global"),
            dict(b_load="global_nt"),
        ]
        names = {base.kernel_name()}
        for kw in variants:
            n = dataclasses.replace(base, **kw).kernel_name()
            self.assertNotIn(n, names, kw)
            names.add(n)

    def test_mfma_atom_per_arch(self):
        for dt, a950, a942 in (
            ("bf16", "mfma.f32.16x16x32.bf16", "mfma.f32.16x16x16bf16"),
            ("fp16", "mfma.f32.16x16x32.f16", "mfma.f32.16x16x16f16"),
        ):
            spec = SkinnyGemmSpec(dtype=dt)
            self.assertIn(a950, _llvm(spec, "gfx950"))
            self.assertIn(a942, _llvm(spec, "gfx942"))

    def test_reduction_sync_modes(self):
        agent = _llvm(SkinnyGemmSpec(red_sync="agent"))
        self.assertIn('fence syncscope("agent") release', agent)
        self.assertIn('fence syncscope("agent") acquire', agent)
        sc1 = _llvm(SkinnyGemmSpec(red_sync="sc1"))
        self.assertNotIn('syncscope("agent")', sc1)
        self.assertIn("global_store_dwordx4 $0, $1, off sc1", sc1)
        self.assertIn("global_load_dwordx4 $0, $", sc1)
        # every asm group of partial stores / loads ends with its own wait
        for blk in sc1.split("asm sideeffect")[1:]:
            if "off sc1" in blk.split('"')[1]:
                self.assertIn("s_waitcnt vmcnt(0)", blk.split('"')[1])

    def test_kalign1_uses_byte_realign(self):
        self.assertIn("alignbyte", _llvm(SkinnyGemmSpec(kalign=1)))
        self.assertNotIn("alignbyte", _llvm(SkinnyGemmSpec(kalign=8)))

    def test_b_load_global_nt(self):
        ir = _llvm(SkinnyGemmSpec(b_load="global_nt"))
        self.assertIn("!nontemporal", ir)
        self.assertNotIn("!nontemporal", _llvm(SkinnyGemmSpec()))


if __name__ == "__main__":
    unittest.main()
