# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU structure/lowering tests for the f16/bf16 single-launch MoE mega-kernel.

These prove the kernel is a true single-launch fusion (no HBM intermediates)
and that it lowers + assembles for gfx950 in both f16 and bf16. On-device
numeric verification lives in ``examples/gfx950/moe/fused_mega_verify.py`` and
requires a gfx950 (MFMA) device.
"""

from __future__ import annotations

import unittest


class TestMoeFusedMega(unittest.TestCase):
    def _spec(self, dtype):
        from rocke.instances.common.moe_fused_mega import FusedMegaKernelSpec

        return FusedMegaKernelSpec(name=f"mega_{dtype}", dtype=dtype)

    def test_signature_is_single_launch(self):
        from rocke.instances.common.moe_fused_mega import moe_fused_mega_signature

        names = [p["name"] for p in moe_fused_mega_signature(self._spec("fp16"))]
        # Inputs + routing + single output only -- no HBM intermediates.
        self.assertIn("A", names)
        self.assertIn("WGate", names)
        self.assertIn("WUp", names)
        self.assertIn("WDown", names)
        self.assertIn("Y", names)
        for forbidden in ("Hidden", "GateOut", "UpOut", "DownOut"):
            self.assertNotIn(forbidden, names)

    def test_grid_splits_inter_and_mblocks(self):
        from rocke.instances.common.moe_fused_mega import moe_fused_mega_grid

        spec = self._spec("fp16")
        gx, gy, gz = moe_fused_mega_grid(8, 7168, spec)
        self.assertEqual((gx, gy, gz), (7168 // spec.tile_n_inter, 8, 1))

    def test_lowers_with_mfma_and_atomic_reduce(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.moe_fused_mega import build_moe_fused_mega_gemm

        for dt, tag in (("fp16", "f16"), ("bf16", "bf16")):
            ll = lower_kernel_to_llvm(
                build_moe_fused_mega_gemm(self._spec(dt), arch="gfx950"), arch="gfx950"
            )
            # gate+up+down all use the 16x16x32 MFMA atom.
            self.assertIn(f"mfma.f32.16x16x32.{tag}", ll)
            # the down stage reduces into Y via an atomic add.
            self.assertIn("atomicrmw", ll)
            # empty-expert tiles are skipped (BlockExpertIds == -1 guard).
            self.assertIn("define amdgpu_kernel", ll)

    def test_rejects_wave32_wmma_target(self):
        from rocke.instances.common.moe_fused_mega import build_moe_fused_mega_gemm

        # gfx1250 is WMMA/no-MFMA: the MFMA-atom GEMM spec validator must reject.
        with self.assertRaises(ValueError):
            build_moe_fused_mega_gemm(self._spec("bf16"), arch="gfx1250")


class TestMoeFusedMegaLdsBudget(unittest.TestCase):
    """The fused total must be validated, not just the two GEMMs separately.

    ``is_valid_spec`` runs over the gate/up and down ``UniversalGemmSpec`` s
    independently and neither models ``Hidden_smem`` or the fact that all five
    buffers coexist, so a tiling can pass both and still overrun the per-WG LDS
    budget -- a kernel-load failure rather than a spec rejection.
    """

    # Shipped tile geometry, measured from the lowered IR's addrspace(3) pool.
    SHIPPED_LDS_BYTES = 74752
    GFX942_LDS_CAP = 65536

    def _spec(self, **kw):
        from rocke.instances.common.moe_fused_mega import FusedMegaKernelSpec

        kw.setdefault("dtype", "bf16")
        return FusedMegaKernelSpec(name="mega_lds", **kw)

    def test_accounting_matches_emitted_pool(self):
        # The accounting must equal the bytes the smem packer actually reserves,
        # otherwise the gate rejects valid specs (or passes invalid ones).
        from rocke.core import lower_llvm
        from rocke.instances.common._moe_fused_mega_lds import mega_lds_pool_bytes
        from rocke.instances.common.moe_fused_mega import (
            _lds_allocs,
            build_moe_fused_mega_gemm,
        )

        for kw in ({}, {"tile_m": 32}, {"tile_n_down": 128}, {"tile_k_down": 32}):
            with self.subTest(**kw):
                spec = self._spec(**kw)
                kd = build_moe_fused_mega_gemm(spec, arch="gfx950")
                low = lower_llvm._Lowerer(kd, arch="gfx950")
                low._collect_smem(kd.body)
                low._compute_smem_layout()
                self.assertEqual(
                    mega_lds_pool_bytes(_lds_allocs(spec)), low._smem_pool_size
                )

    def test_shipped_geometry_costs_the_measured_bytes(self):
        from rocke.instances.common._moe_fused_mega_lds import mega_lds_pool_bytes
        from rocke.instances.common.moe_fused_mega import _lds_allocs

        self.assertEqual(
            mega_lds_pool_bytes(_lds_allocs(self._spec())), self.SHIPPED_LDS_BYTES
        )

    def test_gfx950_default_still_builds(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.moe_fused_mega import build_moe_fused_mega_gemm

        # The shipped geometry fits CDNA4's 160 KiB; only gfx942 is over.
        ll = lower_kernel_to_llvm(
            build_moe_fused_mega_gemm(self._spec(), arch="gfx950"), arch="gfx950"
        )
        self.assertIn("mfma.f32.16x16x32.bf16", ll)
        self.assertIn("atomicrmw", ll)

    def test_rejects_over_budget_gfx942_retile(self):
        from rocke.instances.common.moe_fused_mega import build_moe_fused_mega_gemm

        # warp_tile_k=16 is the gfx942-legal atom, so both GEMM sub-specs pass;
        # only the fused total catches this one.
        with self.assertRaises(ValueError) as cm:
            build_moe_fused_mega_gemm(self._spec(warp_tile_k=16), arch="gfx942")
        msg = str(cm.exception)
        self.assertIn(str(self.SHIPPED_LDS_BYTES), msg)
        self.assertIn(str(self.GFX942_LDS_CAP), msg)
        self.assertIn("Hidden_smem", msg)

    def test_accepts_gfx942_config_that_fits(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.moe_fused_mega import build_moe_fused_mega_gemm

        # One field off the shipped geometry brings the total under 64 KiB.
        spec = self._spec(warp_tile_k=16, tile_n_down=128)
        ll = lower_kernel_to_llvm(
            build_moe_fused_mega_gemm(spec, arch="gfx942"), arch="gfx942"
        )
        self.assertIn("mfma.f32.16x16x16bf16", ll)  # the gfx942-legal bf16 atom
        self.assertIn("atomicrmw", ll)
        # 58368 B of LDS -- 7168 B of headroom under gfx942's 64 KiB.
        self.assertIn("[58368 x i8]", ll)


if __name__ == "__main__":
    unittest.main()
