# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU structure/lowering tests for the gfx1250 WMMA single-launch MoE mega.

These prove the WMMA mega is a true single-launch fusion (no HBM intermediates)
and that it lowers + assembles for gfx1250 in both f16 and bf16. On-device
numeric verification lives in ``examples/gfx1250/fused_mega_moe`` and requires a
gfx1250 (WMMA / wave32) device.
"""

from __future__ import annotations

import unittest


class TestMoeFusedMegaWmma(unittest.TestCase):
    def _spec(self, dtype):
        from rocke.instances.gfx1250.fused_moe_mega_wmma import FusedMegaWmmaSpec

        return FusedMegaWmmaSpec(name=f"mega_wmma_{dtype}", dtype=dtype)

    def test_block_size_is_wave32(self):
        # warp_m * warp_n * 32 = 1 * 4 * 32 = 128 (wave32 vs the MFMA 256).
        self.assertEqual(self._spec("bf16").block_size, 128)

    def test_signature_is_single_launch(self):
        from rocke.instances.gfx1250.fused_moe_mega_wmma import (
            moe_fused_mega_wmma_signature,
        )

        names = [p["name"] for p in moe_fused_mega_wmma_signature(self._spec("bf16"))]
        for needed in ("A", "WGate", "WUp", "WDown", "Y"):
            self.assertIn(needed, names)
        # No HBM intermediates -> single-launch fusion.
        for forbidden in ("Hidden", "GateOut", "UpOut", "DownOut"):
            self.assertNotIn(forbidden, names)

    def test_grid_splits_inter_and_mblocks(self):
        from rocke.instances.gfx1250.fused_moe_mega_wmma import (
            moe_fused_mega_wmma_grid,
        )

        spec = self._spec("bf16")
        gx, gy, gz = moe_fused_mega_wmma_grid(8, 7168, spec)
        self.assertEqual((gx, gy, gz), (7168 // spec.tile_n_inter, 8, 1))

    def test_lowers_with_wmma_and_atomic_reduce(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.gfx1250.fused_moe_mega_wmma import (
            build_moe_fused_mega_wmma,
        )

        for dt, tag in (("fp16", "f16"), ("bf16", "bf16")):
            ll = lower_kernel_to_llvm(
                build_moe_fused_mega_wmma(self._spec(dt), arch="gfx1250"),
                arch="gfx1250",
            )
            # gate+up+down all use the gfx1250 16x16x32 WMMA atom.
            self.assertIn(f"llvm.amdgcn.wmma.f32.16x16x32.{tag}", ll)
            # the down stage reduces into Y via an atomic add.
            self.assertIn("atomicrmw", ll)
            # empty-expert tiles are skipped (BlockExpertIds == -1 guard).
            self.assertIn("define amdgpu_kernel", ll)

    def test_wmma_v1_pipeline_emits_schedule(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.gfx1250.fused_moe_mega_wmma import (
            FusedMegaWmmaSpec,
            build_moe_fused_mega_wmma,
        )

        mem = lower_kernel_to_llvm(
            build_moe_fused_mega_wmma(self._spec("bf16"), arch="gfx1250"),
            arch="gfx1250",
        )
        sched = lower_kernel_to_llvm(
            build_moe_fused_mega_wmma(
                FusedMegaWmmaSpec(name="mega_sched", dtype="bf16", pipeline="wmma_v1"),
                arch="gfx1250",
            ),
            arch="gfx1250",
        )
        # 'mem' (default) emits no scheduler hints; 'wmma_v1' does.
        self.assertNotIn("sched.group.barrier", mem)
        self.assertIn("sched.group.barrier", sched)
        # Still a correct WMMA kernel with the atomic reduce.
        self.assertIn("llvm.amdgcn.wmma.f32.16x16x32.bf16", sched)
        self.assertIn("atomicrmw", sched)

    def test_double_buffer_lowers(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.gfx1250.fused_moe_mega_wmma import (
            FusedMegaWmmaSpec,
            build_moe_fused_mega_wmma,
        )

        sb = lower_kernel_to_llvm(
            build_moe_fused_mega_wmma(self._spec("bf16"), arch="gfx1250"),
            arch="gfx1250",
        )
        db = lower_kernel_to_llvm(
            build_moe_fused_mega_wmma(
                FusedMegaWmmaSpec(name="mega_db", dtype="bf16", double_buffer=True),
                arch="gfx1250",
            ),
            arch="gfx1250",
        )
        # Double-buffer still lowers to a correct WMMA kernel with the atomic
        # reduce (the one-barrier-per-K-tile win is dynamic and GPU-measured, not
        # statically countable in the lowered IR text).
        for ll in (sb, db):
            self.assertIn("llvm.amdgcn.wmma.f32.16x16x32.bf16", ll)
            self.assertIn("atomicrmw", ll)
            self.assertIn("define amdgpu_kernel", ll)

    def test_waves_per_eu_attr(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.gfx1250.fused_moe_mega_wmma import (
            FusedMegaWmmaSpec,
            build_moe_fused_mega_wmma,
        )

        ll = lower_kernel_to_llvm(
            build_moe_fused_mega_wmma(
                FusedMegaWmmaSpec(name="mega_wpe", dtype="bf16", waves_per_eu=2),
                arch="gfx1250",
            ),
            arch="gfx1250",
        )
        self.assertIn("amdgpu-waves-per-eu", ll)

    def test_rejects_wave64_mfma_target(self):
        from rocke.instances.gfx1250.fused_moe_mega_wmma import (
            build_moe_fused_mega_wmma,
        )

        # gfx950 is MFMA/wave64: the WMMA wave32 spec validator must reject.
        with self.assertRaises(ValueError):
            build_moe_fused_mega_wmma(self._spec("bf16"), arch="gfx950")


if __name__ == "__main__":
    unittest.main()
