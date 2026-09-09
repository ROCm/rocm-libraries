# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU spec-validation tests for the fp8 single-launch MoE mega-kernel.

These pin the three ways an unbuildable fp8 mega spec must fail at build time
instead of inside comgr or at kernel load: an arch without the scaled f8f6f4
instruction, a wave32 arch, and a tiling whose total LDS overruns the per-WG
budget. On-device numeric verification lives under ``examples/gfx950``.
"""

from __future__ import annotations

import unittest


def _spec(**kw):
    from rocke.instances.common.moe_fused_mega_fp8 import FusedMegaKernelSpecFp8

    kw.setdefault("name", "mega_fp8")
    return FusedMegaKernelSpecFp8(**kw)


class TestMoeFusedMegaFp8ArchGuards(unittest.TestCase):
    def test_rejects_scaled_hero_atom_on_gfx942(self):
        from rocke.instances.common.moe_fused_mega_fp8 import (
            build_moe_fused_mega_gemm_fp8,
        )

        # The K=128 hero atom lowers to mfma.scale.f32.16x16x128.f8f6f4, which
        # gfx942 does not have. It is not a catalog shape anywhere, so the
        # shared catalog guard is skipped and this needs its own rejection --
        # otherwise the CDNA4-only instruction reaches comgr as an uncatchable
        # LLVM abort.
        with self.assertRaises(NotImplementedError) as cm:
            build_moe_fused_mega_gemm_fp8(_spec(), arch="gfx942")
        msg = str(cm.exception)
        self.assertIn("16x16x128", msg)
        self.assertIn("gfx942", msg)

    def test_accepts_catalog_atom_on_gfx942(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.moe_fused_mega_fp8 import (
            build_moe_fused_mega_gemm_fp8,
        )

        # The legacy K=32 path IS a gfx942 catalog shape, so it must still build.
        kd = build_moe_fused_mega_gemm_fp8(
            _spec(gate_up_k=32, down_k=32), arch="gfx942"
        )
        self.assertIn("atomicrmw", lower_kernel_to_llvm(kd, arch="gfx942"))

    def test_rejects_wave32_target(self):
        from rocke.instances.common.moe_fused_mega_fp8 import (
            build_moe_fused_mega_gemm_fp8,
        )

        # Every lane map here is wave64 (the amax butterfly is a hardcoded
        # 6-stage xor over lanes 1..32), so a wave32 target is silently wrong.
        for arch in ("gfx1250", "gfx1151"):
            with self.subTest(arch=arch):
                with self.assertRaises(ValueError) as cm:
                    build_moe_fused_mega_gemm_fp8(_spec(), arch=arch)
                self.assertIn(
                    f"spec wave_size 64 != {arch} wave_size 32", str(cm.exception)
                )

    def test_gfx950_default_still_builds(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.moe_fused_mega_fp8 import (
            build_moe_fused_mega_gemm_fp8,
        )

        ll = lower_kernel_to_llvm(
            build_moe_fused_mega_gemm_fp8(_spec(), arch="gfx950"), arch="gfx950"
        )
        self.assertIn("mfma.scale.f32.16x16x128.f8f6f4", ll)
        self.assertIn("atomicrmw", ll)


class TestMoeFusedMegaFp8LdsBudget(unittest.TestCase):
    def test_accounting_matches_emitted_pool(self):
        # ``BStage_smem`` is allocated unconditionally but only referenced under
        # ``use_dtla``; the packer dead-strips it otherwise, so the accounting
        # has to track that or it over-counts 32 KiB.
        from rocke.core import lower_llvm
        from rocke.instances.common._moe_fused_mega_lds import mega_lds_pool_bytes
        from rocke.instances.common.moe_fused_mega_fp8 import (
            _lds_allocs,
            build_moe_fused_mega_gemm_fp8,
        )

        cases = (
            {},
            {"use_dtla": False},
            {"gate_up_k": 32, "down_k": 32},
            {"tile_m": 32},
            {"tile_n_inter": 512, "warp_n": 4},
        )
        for kw in cases:
            with self.subTest(**kw):
                spec = _spec(**kw)
                kd = build_moe_fused_mega_gemm_fp8(spec, arch="gfx950")
                low = lower_llvm._Lowerer(kd, arch="gfx950")
                low._collect_smem(kd.body)
                low._compute_smem_layout()
                self.assertEqual(
                    mega_lds_pool_bytes(_lds_allocs(spec)), low._smem_pool_size
                )

    def test_rejects_over_budget_spec(self):
        from rocke.core.arch import ArchTarget
        from rocke.instances.common._moe_fused_mega_lds import mega_lds_pool_bytes
        from rocke.instances.common.moe_fused_mega_fp8 import (
            _lds_allocs,
            build_moe_fused_mega_gemm_fp8,
        )

        # Widening the inter slice scales Hidden + the f32 amax scratch + the
        # DTLA landing zone together, past CDNA4's 160 KiB.
        spec = _spec(tile_m=32, tile_n_inter=2048, warp_n=16)
        total = mega_lds_pool_bytes(_lds_allocs(spec))
        cap = ArchTarget.from_gfx("gfx950").lds_capacity_bytes
        self.assertGreater(total, cap)
        with self.assertRaises(ValueError) as cm:
            build_moe_fused_mega_gemm_fp8(spec, arch="gfx950")
        msg = str(cm.exception)
        self.assertIn(str(total), msg)
        self.assertIn(str(cap), msg)
        self.assertIn("Hidden_smem", msg)


if __name__ == "__main__":
    unittest.main()
