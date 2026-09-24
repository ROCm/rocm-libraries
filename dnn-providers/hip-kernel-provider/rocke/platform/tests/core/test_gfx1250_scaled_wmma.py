# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the gfx1250 native FP8 SCALE and SCALE16 WMMA slice."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

from rocke.core.arch import ArchTarget
from rocke.core.arch.wmma_scale import gfx1250_scaled_wmma
from rocke.core.isa.wmma_scale import ScaledWmmaLLVM
from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    block_scaled_gemm_signature,
    build_block_scaled_gemm,
    is_valid_spec,
)


def _scaled_atom(dtype="fp8", scale16=False):
    block_k = 16 if scale16 else 32
    atom = ArchTarget.from_gfx("gfx1250").mma.op_for_shape(
        family="wmma_scaled",
        a_dtype=dtype,
        b_dtype=dtype,
        c_dtype="fp32",
        scales=("e8m0", "e8m0", block_k),
        m=16,
        n=16,
        k=128,
    )
    assert atom is not None
    return atom


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
    d = b.mma(_scaled_atom(scale16=scale16), fragment, fragment, c, scale, scale)
    b.global_store(accum, lane, b.vec_extract(d, 0))
    return b.kernel


class TestGfx1250ScaledWmma(unittest.TestCase):
    def test_backend_rejects_unsupported_scale_type_on_either_input(self):
        catalog = ArchTarget.from_gfx("gfx1250").mma
        for field in ("a_scale_dtype", "b_scale_dtype"):
            atom = _scaled_atom()
            index = 0 if field == "a_scale_dtype" else 1
            srcs = list(atom.srcs)
            srcs[index] = replace(
                srcs[index], scale=replace(srcs[index].scale, dtype="e4m3")
            )
            atom = replace(atom, srcs=tuple(srcs))
            with (
                self.subTest(field=field),
                mock.patch.object(catalog, "by_op_id", return_value=atom),
                self.assertRaisesRegex(ValueError, "unsupported scaled WMMA"),
            ):
                gfx1250_scaled_wmma(atom.op_id)

    def test_scale_fragment_counts_are_checked_for_each_source(self):
        catalog = ArchTarget.from_gfx("gfx1250").mma
        for role in ("a_scale", "b_scale"):
            for count in (0, 8):
                atom = _scaled_atom()
                index = 0 if role == "a_scale" else 1
                srcs = list(atom.srcs)
                srcs[index] = replace(
                    srcs[index],
                    scale=replace(srcs[index].scale, frag_len=count, layout=None),
                )
                atom = replace(atom, srcs=tuple(srcs))
                with (
                    self.subTest(role=role, count=count),
                    mock.patch.object(catalog, "by_op_id", return_value=atom),
                    self.assertRaisesRegex(ValueError, "scale fragment lengths"),
                ):
                    gfx1250_scaled_wmma(atom.op_id)

    def test_loader_requires_both_scale_maps(self):
        for role in ("a_scale", "b_scale"):
            atom = _scaled_atom()
            index = 0 if role == "a_scale" else 1
            srcs = list(atom.srcs)
            srcs[index] = replace(
                srcs[index], scale=replace(srcs[index].scale, layout=None)
            )
            atom = replace(atom, srcs=tuple(srcs))
            spec = BlockScaledGemmSpec(
                "missing_scale_map",
                M=16,
                N=16,
                K=128,
                block_k=32,
                scale_dtype="e8m0",
                matrix_path="wmma_scale",
            )
            with (
                mock.patch(
                    "rocke.instances.gfx1250.block_scaled_gemm._native_scaled_atom",
                    return_value=atom,
                ),
                self.assertRaisesRegex(NotImplementedError, f"src{index}_scale"),
            ):
                build_block_scaled_gemm(spec, arch="gfx1250")
            # Lowering prepacked operands needs the carrier contract, not a loader map.
            with mock.patch.object(
                ArchTarget.from_gfx("gfx1250").mma, "by_op_id", return_value=atom
            ):
                self.assertEqual(gfx1250_scaled_wmma(atom.op_id).scales.count, 4)

    def test_backend_derives_each_matrix_carrier_and_selector(self):
        # Synthetic metadata exercises the lowering contract without adding atoms.
        atom = _scaled_atom()
        atom = replace(
            atom,
            srcs=(
                atom.srcs[0],
                replace(atom.srcs[1], dtype="bf8e5m2", frag_len=8),
                atom.srcs[2],
            ),
        )
        catalog = ArchTarget.from_gfx("gfx1250").mma
        with mock.patch.object(catalog, "by_op_id", return_value=atom):
            spec = gfx1250_scaled_wmma(atom.op_id)
        self.assertEqual(spec.matrix_formats, (0, 1))
        signature = ScaledWmmaLLVM(spec)
        self.assertEqual(signature.matrix_types, ("<16 x i32>", "<8 x i32>"))
        self.assertEqual(signature.scale_type, "i32")
        self.assertTrue(signature.intrinsic.endswith("v16i32.v8i32"))
        self.assertIn("wmma.scale.block32", signature.declaration_key)

    def test_matrix_formats_share_intrinsic_declarations(self):
        b = IRBuilder("shared_scaled_wmma_declarations")
        matrix = b.param("matrix", PtrType(I32, "global"), readonly=True)
        accum = b.param("accum", PtrType(F32, "global"))
        lane = b.thread_id_x()
        lo = b.global_load_vN(matrix, lane, I32, 8)
        hi = b.global_load_vN(matrix, b.add(lane, b.const_i32(8)), I32, 8)
        fragment = b.vec_concat(lo, hi)
        c = b.global_load_vN(accum, lane, F32, 8)
        for mode, scale_ty in (("scale", I32), ("scale16", I64)):
            scale_ptr = b.param(
                f"{mode}_ptr", PtrType(scale_ty, "global"), readonly=True
            )
            scale = b.global_load(scale_ptr, lane, scale_ty)
            for dtype in ("fp8", "bf8"):
                c = b.mma(
                    _scaled_atom(dtype, mode == "scale16"),
                    fragment,
                    fragment,
                    c,
                    scale,
                    scale,
                )
        b.global_store(accum, lane, b.vec_extract(c, 0))
        llvm = lower_kernel_to_llvm(b.kernel, arch="gfx1250", llvm_flavor="llvm23")
        for mode, scale_ty in (("scale", "i32"), ("scale16", "i64")):
            intrinsic = (
                f"llvm.amdgcn.wmma.{mode}.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32"
            )
            with self.subTest(mode=mode):
                self.assertEqual(llvm.count(f"declare <8 x float> @{intrinsic}("), 1)
                calls = [
                    line
                    for line in llvm.splitlines()
                    if f"call <8 x float> @{intrinsic}(" in line
                ]
                self.assertEqual(len(calls), 2)
                for selector, call in enumerate(calls):
                    self.assertEqual(call.count(f"i32 {selector}, <16 x i32>"), 2)
                    self.assertEqual(call.count(f", {scale_ty} %"), 2)

    def test_catalog_ids_survive_serialization(self):
        for scale16 in (False, True):
            kernel = _build_scaled_atom(scale16=scale16)
            text = serialize(kernel)
            self.assertIn(_scaled_atom(scale16=scale16).op_id, text)
            restored = parse(text)
            self.assertEqual(
                lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor="llvm23"),
                lower_kernel_to_llvm(restored, arch="gfx1250", llvm_flavor="llvm23"),
            )
            self.assertEqual(
                lower_kernel_to_hip(kernel, arch="gfx1250"),
                lower_kernel_to_hip(restored, arch="gfx1250"),
            )

    def test_catalog_fragment_lengths(self):
        target = ArchTarget.from_gfx("gfx1250")
        for op_id in (
            "wmma_gfx1250_f32_16x16x128_fp8_fp8_scale_e8m0_e8m0_k32",
            "wmma_gfx1250_f32_16x16x128_fp8_fp8_scale_e8m0_e8m0_k16",
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
            with (
                self.subTest(flavor=flavor),
                self.assertRaisesRegex(NotImplementedError, "requires llvm23"),
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

    def test_scaled_hip_dispatch_preserves_neutral_and_concrete_ops(self):
        for scale16 in (False, True):
            mode = "wmma_scale16" if scale16 else "wmma_scale"
            for dtype, selector in (("fp8", 0), ("bf8", 1)):
                with self.subTest(mode=mode, dtype=dtype):
                    kernel = _build_scaled_atom(scale16=scale16)
                    call = next(op for op in kernel.body.ops if op.name == "tile.mma")
                    op_id = _scaled_atom(dtype, scale16).op_id
                    call.attrs["op_id"] = op_id
                    neutral = lower_kernel_to_hip(kernel, arch="gfx1250")
                    self.assertIn(
                        f"__builtin_amdgcn_{mode}_f32_16x16x128_f8f6f4({selector},",
                        neutral,
                    )
                    call.name = f"tile.{call.attrs.pop('op_id')}"
                    self.assertEqual(
                        neutral, lower_kernel_to_hip(kernel, arch="gfx1250")
                    )
                    with self.assertRaisesRegex(NotImplementedError, "not available"):
                        lower_kernel_to_hip(kernel, arch="gfx942")

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
        self.assertIn("requested operand and scale contract", why)

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

    def test_unknown_scaled_atom_reports_supported_operations(self):
        for scale16 in (False, True):
            with self.subTest(scale16=scale16):
                kernel = _build_scaled_atom(scale16=scale16)
                call = next(op for op in kernel.body.ops if op.name == "tile.mma")
                mode = "wmma_scale16" if scale16 else "wmma_scale"
                call.attrs["op_id"] = f"{mode}_f32_16x16x128_fp16_fp16"
                with self.assertRaisesRegex(
                    NotImplementedError, "not yet wired for gfx1250"
                ) as error:
                    lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor="llvm23")
                self.assertIn(_scaled_atom("bf8", scale16).op_id, str(error.exception))
                with self.assertRaisesRegex(NotImplementedError, "no HIP lowering"):
                    lower_kernel_to_hip(kernel, arch="gfx1250")


if __name__ == "__main__":
    unittest.main(verbosity=2)
