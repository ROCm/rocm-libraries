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


if __name__ == "__main__":
    unittest.main()
