# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for gfx1250 Qwen3-30B-A3B GEMM contracts (bf16 + low-bit)."""

from __future__ import annotations

import unittest
from dataclasses import replace


class TestGfx1250Gemm(unittest.TestCase):
    def test_bf16_qwen_gemm_shapes_validate_and_lower(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            ALL_BF16_GEMM_SHAPES,
            bf16_universal_gemm_spec,
            shape_by_name,
        )
        from rocke.instances.common.gemm_universal import (
            build_universal_gemm,
            is_valid_spec,
        )

        for shape in ALL_BF16_GEMM_SHAPES:
            with self.subTest(shape=shape.name):
                spec = bf16_universal_gemm_spec(shape)
                ok, why = is_valid_spec(spec, arch="gfx1250")
                self.assertTrue(ok, why)

        for name in ("qkv_decode_bf16", "qkv_prefill_bf16"):
            spec = bf16_universal_gemm_spec(shape_by_name(name))
            ll = lower_kernel_to_llvm(
                build_universal_gemm(spec, arch="gfx1250"), arch="gfx1250"
            )
            self.assertIn("llvm.amdgcn.wmma.f32.16x16x32.bf16", ll)
            self.assertIn("<16 x bfloat>", ll)

    def test_block_scaled_lowbit_expert_gemms_lower(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            EXPERT_BLOCK_SCALED_GEMM_SHAPES,
        )
        from rocke.instances.gfx1250.block_scaled_gemm import (
            BlockScaledGemmSpec,
            block_scaled_gemm_grid,
            block_scaled_gemm_signature,
            build_block_scaled_gemm,
            is_valid_spec,
        )

        for shape in EXPERT_BLOCK_SCALED_GEMM_SHAPES:
            with self.subTest(shape=shape.name):
                spec = BlockScaledGemmSpec(
                    name=f"gfx1250_{shape.name}",
                    M=shape.M,
                    N=shape.N,
                    K=shape.K,
                    dtype_a=shape.dtype,
                    dtype_b=shape.dtype,
                )
                ok, why = is_valid_spec(spec, arch="gfx1250")
                self.assertTrue(ok, why)
                self.assertIn("K=64 FP8/BF8 WMMA", why)
                self.assertIn("wmma", spec.kernel_name())
                self.assertEqual(block_scaled_gemm_grid(spec)[2], 1)
                sig = block_scaled_gemm_signature(spec)
                self.assertEqual(
                    [p["name"] for p in sig[:5]],
                    ["A", "B", "A_scale", "B_scale", "C"],
                )

                mfma_spec = replace(spec, matrix_path="mfma")
                ok, why = is_valid_spec(mfma_spec, arch="gfx1250")
                self.assertFalse(ok)
                self.assertIn("no MFMA block_scale path", why)

                ll = lower_kernel_to_llvm(
                    build_block_scaled_gemm(spec, arch="gfx1250"), arch="gfx1250"
                )
                self.assertIn("define amdgpu_kernel", ll)
                # The real K=64 FP8/BF8 WMMA intrinsic + per-block f32 scale.
                lowbit = "fp8" if shape.dtype == "fp8e4m3" else "bf8"
                self.assertIn(
                    f"llvm.amdgcn.wmma.f32.16x16x64.{lowbit}.{lowbit}.v8f32.v8i32",
                    ll,
                )
                self.assertIn("fmul float", ll)


if __name__ == "__main__":
    unittest.main(verbosity=2)
