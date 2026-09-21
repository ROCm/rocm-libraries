# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the gfx1250 native FP8 SCALE and SCALE16 WMMA slice."""

from __future__ import annotations

import unittest

from rocke.core.arch import ArchTarget
from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    block_scaled_gemm_signature,
    build_block_scaled_gemm,
    is_valid_spec,
)


def _build_scaled_atom(*, scale16: bool):
    b = IRBuilder("gfx1250_scaled_wmma")
    matrix = b.param("matrix", PtrType(I32, "global"), readonly=True)
    accum = b.param("accum", PtrType(F32, "global"))
    scale_ty = I64 if scale16 else I32
    scales = b.param("scales", PtrType(scale_ty, "global"), readonly=True)
    lane = b.thread_id_x()
    lo = b.global_load_vN(matrix, lane, I32, 8)
    hi = b.global_load_vN(matrix, b.add(lane, b.const_i32(8)), I32, 8)
    fragment = b.vec_concat(lo, hi)
    c = b.global_load_vN(accum, lane, F32, 8)
    scale = b.global_load(scales, lane, scale_ty)
    if scale16:
        d = b.wmma_scale16_f32_16x16x128_fp8_fp8(fragment, fragment, c, scale, scale)
    else:
        d = b.wmma_scale_f32_16x16x128_fp8_fp8(fragment, fragment, c, scale, scale)
    b.global_store(accum, lane, b.vec_extract(d, 0))
    return b.kernel


class TestGfx1250ScaledWmma(unittest.TestCase):
    def test_catalog_fragment_lengths(self):
        target = ArchTarget.from_gfx("gfx1250")
        for op_id in (
            "wmma_scale_f32_16x16x128_fp8_fp8",
            "wmma_scale16_f32_16x16x128_fp8_fp8",
        ):
            with self.subTest(op_id=op_id):
                op = target.mma.by_op_id(op_id)
                self.assertIsNotNone(op)
                self.assertEqual(
                    (op.a_frag_len, op.b_frag_len, op.c_frag_len), (16, 16, 8)
                )

    def test_scale_and_scale16_exact_llvm23_abi(self):
        cases = (
            (
                False,
                ("llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4." "v8f32.v16i32.v16i32"),
                "i32",
            ),
            (
                True,
                (
                    "llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4."
                    "v8f32.v16i32.v16i32"
                ),
                "i64",
            ),
        )
        for scale16, intrinsic, scale_ty in cases:
            with self.subTest(scale16=scale16):
                ll = lower_kernel_to_llvm(
                    _build_scaled_atom(scale16=scale16),
                    llvm_flavor="llvm23",
                    arch="gfx1250",
                )
                self.assertIn(f"declare <8 x float> @{intrinsic}", ll)
                self.assertIn(
                    "i32 0, <16 x i32>",
                    next(
                        line
                        for line in ll.splitlines()
                        if f"call <8 x float> @{intrinsic}" in line
                    ),
                )
                self.assertIn(f", {scale_ty} %", ll)

    def test_scaled_wmma_rejects_pre_llvm23_flavors(self):
        for flavor in ("llvm20", "llvm22"):
            with self.subTest(flavor=flavor), self.assertRaisesRegex(
                NotImplementedError, "requires llvm23"
            ):
                lower_kernel_to_llvm(
                    _build_scaled_atom(scale16=False),
                    llvm_flavor=flavor,
                    arch="gfx1250",
                )

    def test_scale_and_scale16_hip_builtin_lowering(self):
        for scale16, builtin in (
            (False, "__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4"),
            (True, "__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4"),
        ):
            with self.subTest(scale16=scale16):
                hip = lower_kernel_to_hip(
                    _build_scaled_atom(scale16=scale16), arch="gfx1250"
                )
                self.assertIn(builtin, hip)

    def test_native_block_scaled_gemm_uses_packed_e8m0_in_instruction(self):
        for matrix_path, block_k, scale_ty, fragment_load, load_count in (
            ("wmma_scale", 32, "i32", "load <16 x i8>", 8),
            ("wmma_scale16", 16, "i64", "load <16 x i8>", 8),
        ):
            with self.subTest(matrix_path=matrix_path):
                spec = BlockScaledGemmSpec(
                    name="native_mx",
                    M=16,
                    N=16,
                    K=128,
                    scale_dtype="e8m0",
                    block_k=block_k,
                    matrix_path=matrix_path,
                )
                ok, why = is_valid_spec(spec)
                self.assertTrue(ok, why)
                signature = block_scaled_gemm_signature(spec)
                self.assertEqual(signature[2]["type"], "ptr<i8, global>")
                ll = lower_kernel_to_llvm(
                    build_block_scaled_gemm(spec),
                    llvm_flavor="llvm23",
                    arch="gfx1250",
                )
                intrinsic_family = "wmma.scale16" if block_k == 16 else "wmma.scale"
                self.assertIn(f"llvm.amdgcn.{intrinsic_family}.", ll)
                self.assertIn(f", {scale_ty} %", ll)
                self.assertEqual(ll.count(fragment_load), load_count)
                self.assertNotIn("fmul float", ll)

    def test_native_block_scaled_gemm_hip_zero_extends_e8m0_bytes(self):
        cases = (
            ("wmma_scale", 32, "(int)(uint8_t)", 8, r" int zx\d+ = \(int\)\w+;"),
            (
                "wmma_scale16",
                16,
                "(int64_t)(uint8_t)",
                16,
                r" int64_t zx\d+ = \(int64_t\)\w+;",
            ),
        )
        for matrix_path, block_k, unsigned_cast, count, direct_signed in cases:
            with self.subTest(matrix_path=matrix_path):
                spec = BlockScaledGemmSpec(
                    name="native_mx_hip",
                    M=16,
                    N=16,
                    K=128,
                    scale_dtype="e8m0",
                    block_k=block_k,
                    matrix_path=matrix_path,
                )
                hip = lower_kernel_to_hip(build_block_scaled_gemm(spec), arch="gfx1250")
                self.assertEqual(hip.count(unsigned_cast), count)
                self.assertNotRegex(hip, direct_signed)

        scale_bytes = [0x80, 0xFF, 0x00, 0x01]
        packed = sum(byte << (8 * index) for index, byte in enumerate(scale_bytes))
        self.assertEqual(packed, 0x0100FF80)

    def test_native_slice_rejects_unsupported_contracts(self):
        bad_dtype = BlockScaledGemmSpec(
            name="bad_dtype",
            M=16,
            N=16,
            K=128,
            dtype_b="bf8",
            scale_dtype="e8m0",
            block_k=32,
            matrix_path="wmma_scale",
        )
        ok, why = is_valid_spec(bad_dtype)
        self.assertFalse(ok)
        self.assertIn("fp8 x fp8 only", why)

        bad_block = BlockScaledGemmSpec(
            name="bad_block",
            M=16,
            N=16,
            K=128,
            scale_dtype="e8m0",
            block_k=32,
            matrix_path="wmma_scale16",
        )
        ok, why = is_valid_spec(bad_block)
        self.assertFalse(ok)
        self.assertIn("requires block_k=16", why)


if __name__ == "__main__":
    unittest.main(verbosity=2)
