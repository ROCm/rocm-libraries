# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the gfx1250 native FP8 SCALE and SCALE16 WMMA slice."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

import pytest

from rocke.core.arch import ArchTarget
from rocke.core.arch.wmma_scale import gfx1250_scaled_wmma
from rocke.core.backend import _cpp_strict, resolve_backend
from rocke.core.isa.wmma_scale import ScaledWmmaLLVM
from rocke.core.ir import F32, I32, I64, IRBuilder, Param, PtrType, VectorType
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import (
    _lower_kernel_to_llvm_python,
    lower_kernel_to_llvm,
)
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


@pytest.mark.parametrize("scale16", [False, True])
@pytest.mark.parametrize("role", ["src2", "dst", "both"])
@pytest.mark.parametrize("width", [0, 4, 7, 16, 8.0])
def test_scaled_backend_rejects_accumulator_widths(scale16, role, width):
    atom = _scaled_atom(scale16=scale16)
    if role in ("src2", "both"):
        atom = replace(
            atom, srcs=(*atom.srcs[:2], replace(atom.srcs[2], frag_len=width))
        )
    if role in ("dst", "both"):
        atom = replace(atom, dst=replace(atom.dst, frag_len=width))
    catalog = ArchTarget.from_gfx("gfx1250").mma
    with mock.patch.object(catalog, "by_op_id", return_value=atom):
        with pytest.raises(
            ValueError, match="unsupported scaled WMMA backend contract"
        ):
            gfx1250_scaled_wmma(atom.op_id)


def _scaled_call(scale16, concrete, dtype="fp8"):
    b = IRBuilder("scaled_operand_contract")
    a = b.param("a", VectorType(I32, 16))
    bb = b.param("b", VectorType(I32, 16))
    c = b.param("c", VectorType(F32, 8))
    s = b.param("s", I64 if scale16 else I32)
    b.mma(_scaled_atom(dtype, scale16), a, bb, c, s, s)
    call = b.kernel.body.ops[-1]
    if concrete:
        call.name = f"tile.{call.attrs.pop('op_id')}"
    b.ret()
    return b.kernel, call


def _assert_scaled_lowering_rejects(kernel, message, engine):
    # Use the normal catalog; re-resolving op_id must not hide malformed SSA.
    ir = serialize(kernel)
    neutral = any(op.name == "tile.mma" for op in kernel.body.ops)
    if engine == "native":
        if not neutral:
            pytest.skip("native LLVM uses neutral MMA dispatch")
        native = pytest.importorskip("rocke_engine")
        with pytest.raises(RuntimeError) as caught:
            native.lower_serialized_ir(ir, arch="gfx1250", flavor="llvm23")
        assert str(caught.value) == (
            "rocke_engine.lower_serialized_ir: lower failed for arch 'gfx1250' "
            f"(status 1): {message}"
        )
        return
    for candidate in (kernel, parse(ir)):
        if neutral:
            with pytest.raises(ValueError) as caught:
                _lower_kernel_to_llvm_python(
                    candidate, arch="gfx1250", llvm_flavor="llvm23"
                )
            assert str(caught.value) == message
        with pytest.raises(ValueError) as caught:
            lower_kernel_to_hip(candidate, arch="gfx1250")
        assert str(caught.value) == message


@pytest.mark.parametrize("scale16", [False, True])
@pytest.mark.parametrize("concrete", [False, True])
@pytest.mark.parametrize("role", ["src2", "dst"])
@pytest.mark.parametrize(
    "bad_type",
    [
        VectorType(F32, 4),
        VectorType(F32, 7),
        VectorType(F32, 16),
        F32,
        VectorType(I32, 8),
    ],
)
@pytest.mark.parametrize("engine", ["python", "native"])
def test_scaled_lowering_rejects_accumulator_types(
    scale16, concrete, role, bad_type, engine
):
    kernel, call = _scaled_call(scale16, concrete)
    value = call.operands[2] if role == "src2" else call.result
    value.type = bad_type
    if role == "src2":
        next(p for p in kernel.params if p.name == "c").type = bad_type
    _assert_scaled_lowering_rejects(
        kernel, "scaled WMMA requires src2 and dst to be vec<f32x8>", engine
    )


@pytest.mark.parametrize("scale16", [False, True])
@pytest.mark.parametrize("concrete", [False, True])
@pytest.mark.parametrize(
    "arity", ["missing_operand", "extra_operand", "missing_result", "extra_result"]
)
@pytest.mark.parametrize("engine", ["python", "native"])
def test_scaled_lowering_rejects_arity(scale16, concrete, arity, engine):
    kernel, call = _scaled_call(scale16, concrete)
    if arity == "missing_operand":
        call.operands.pop()
    elif arity == "extra_operand":
        call.operands.append(call.operands[-1])
    elif arity == "missing_result":
        call.results.clear()
    else:
        call.results.append(replace(call.result, name="%extra"))
    name = next(op for op in kernel.body.ops if op.name.startswith("tile.")).name
    if name == "tile.mma":
        name = f"tile.{call.attrs['op_id']}"
    _assert_scaled_lowering_rejects(
        kernel, f"{name} expects 5 operands and 1 result", engine
    )


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
                self.assertRaisesRegex(NotImplementedError, f"scale_src{index}"),
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
        # A native rejection is retried in Python unless strict C++ is requested.
        error_type = (
            RuntimeError
            if resolve_backend() == "cpp" and _cpp_strict()
            else NotImplementedError
        )
        for flavor in ("llvm20", "llvm22"):
            with (
                self.subTest(flavor=flavor),
                self.assertRaisesRegex(error_type, "requires llvm23"),
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
            ("wmma_scale", 32, "i32", "load <4 x i32>", 8),
            ("wmma_scale16", 16, "i64", "load <4 x i32>", 8),
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
            dtype_b="fp16",
            scale_dtype="e8m0",
            block_k=32,
            matrix_path="wmma_scale",
        )
        ok, why = is_valid_spec(bad_dtype)
        self.assertFalse(ok)
        self.assertIn("A/B must be", why)

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

    def test_python_unknown_scaled_atom_reports_supported_operations(self):
        # The supported-operation listing is a Python lowerer diagnostic.
        for scale16 in (False, True):
            with self.subTest(scale16=scale16):
                kernel = _build_scaled_atom(scale16=scale16)
                call = next(op for op in kernel.body.ops if op.name == "tile.mma")
                mode = "wmma_scale16" if scale16 else "wmma_scale"
                call.attrs["op_id"] = f"{mode}_f32_16x16x128_fp16_fp16"
                with self.assertRaisesRegex(
                    NotImplementedError, "not yet wired for gfx1250"
                ) as error:
                    _lower_kernel_to_llvm_python(
                        kernel, arch="gfx1250", llvm_flavor="llvm23"
                    )
                self.assertIn(_scaled_atom("bf8", scale16).op_id, str(error.exception))
                with self.assertRaisesRegex(NotImplementedError, "no HIP lowering"):
                    lower_kernel_to_hip(kernel, arch="gfx1250")


if __name__ == "__main__":
    unittest.main(verbosity=2)


@pytest.mark.parametrize("scale16", [False, True])
@pytest.mark.parametrize("concrete", [False, True])
@pytest.mark.parametrize("role", [0, 1, 3, 4])
@pytest.mark.parametrize("kind", ["scalar", "dtype", "short", "long"])
@pytest.mark.parametrize("engine", ["python", "native"])
@pytest.mark.parametrize("dtype", ["fp8", "bf8", "fp6", "bf6", "fp4"])
def test_scaled_lowering_rejects_sources(scale16, concrete, role, kind, engine, dtype):
    kernel, call = _scaled_call(scale16, concrete, dtype)
    if role < 2:
        bad = (
            I32
            if kind == "scalar"
            else VectorType(
                F32 if kind == "dtype" else I32,
                16 + (-1 if kind == "short" else 1 if kind == "long" else 0),
            )
        )
        message = f"scaled WMMA requires src{role} to be vec<i32x16>"
        param = ("a", "b")[role]
    else:
        expected = I64 if scale16 else I32
        bad = (
            F32
            if kind == "dtype"
            else (
                VectorType(I32, 1 if kind == "short" else 2)
                if kind != "scalar"
                else (I32 if scale16 else I64)
            )
        )
        # Split the shared scale value so each role is corrupted independently.
        call.operands[role] = replace(call.operands[role], name=f"%bad_scale{role}")

        kernel.params.append(Param(f"bad_scale{role}", bad))
        param = f"bad_scale{role}"
    call.operands[role].type = bad
    next(p for p in kernel.params if p.name == param).type = bad
    if role >= 3:
        op_id = call.attrs.get("op_id", call.name.removeprefix("tile."))
        message = f"tile.{op_id} expects {expected.name} scale operands, got {call.operands[3].type.name}/{call.operands[4].type.name}"
    _assert_scaled_lowering_rejects(kernel, message, engine)


@pytest.mark.parametrize("scale16", [False, True])
@pytest.mark.parametrize("role", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", [None, "", "unknown"])
def test_scaled_backend_rejects_matrix_dtypes(scale16, role, dtype):
    atom = _scaled_atom(scale16=scale16)
    if role < 3:
        srcs = list(atom.srcs)
        srcs[role] = replace(srcs[role], dtype=dtype)
        atom = replace(atom, srcs=tuple(srcs))
    else:
        atom = replace(atom, dst=replace(atom.dst, dtype=dtype))
    catalog = ArchTarget.from_gfx("gfx1250").mma
    with mock.patch.object(catalog, "by_op_id", return_value=atom):
        with pytest.raises(ValueError) as caught:
            gfx1250_scaled_wmma(atom.op_id)
        assert (
            str(caught.value)
            == f"unsupported scaled WMMA backend contract: {atom.op_id}"
        )


@pytest.mark.parametrize("scale16", [False, True])
@pytest.mark.parametrize("bad_width", [None, 16, 7, 9])
def test_scaled_python_preserves_independent_carrier_widths(scale16, bad_width):
    atom = _scaled_atom(scale16=scale16)
    atom = replace(
        atom,
        srcs=(
            atom.srcs[0],
            replace(atom.srcs[1], dtype="bf8e5m2", frag_len=8),
            atom.srcs[2],
        ),
    )
    b = IRBuilder("mixed_scaled_carriers")
    a = b.param("a", VectorType(I32, 16))
    bb = b.param("b", VectorType(I32, 8 if bad_width is None else bad_width))
    c = b.param("c", VectorType(F32, 8))
    scale = b.param("s", I64 if scale16 else I32)
    d = b.mma(atom, a, bb, c, scale, scale)
    dest = b.param("dest", PtrType(F32, "global"))
    b.global_store(dest, b.const_i32(0), b.vec_extract(d, 7))
    b.ret()
    catalog = ArchTarget.from_gfx("gfx1250").mma
    with mock.patch.object(catalog, "by_op_id", return_value=atom):
        for candidate in (b.kernel, parse(serialize(b.kernel))):
            if bad_width is None:
                ll = _lower_kernel_to_llvm_python(
                    candidate, arch="gfx1250", llvm_flavor="llvm23"
                )
                assert ".v16i32.v8i32" in ll
                assert "__builtin_amdgcn_wmma_scale" in lower_kernel_to_hip(
                    candidate, arch="gfx1250"
                )
            else:
                for lower in (
                    lambda: _lower_kernel_to_llvm_python(
                        candidate, arch="gfx1250", llvm_flavor="llvm23"
                    ),
                    lambda: lower_kernel_to_hip(candidate, arch="gfx1250"),
                ):
                    with pytest.raises(ValueError) as caught:
                        lower()
                    assert (
                        str(caught.value)
                        == "scaled WMMA requires src1 to be vec<i32x8>"
                    )


@pytest.mark.parametrize("scale16", [False, True])
@pytest.mark.parametrize("dtype", ["fp8", "bf8", "fp6", "bf6", "fp4"])
@pytest.mark.parametrize("source", [0, 1])
@pytest.mark.parametrize("width", [16.0, 16.5, 0, -1, True, False, "16", None])
def test_scaled_descriptor_rejects_invalid_source_widths(scale16, dtype, source, width):
    kernel, call = _scaled_call(scale16, False, dtype)
    atom = _scaled_atom(dtype, scale16)
    srcs = list(atom.srcs)
    srcs[source] = replace(srcs[source], frag_len=width)
    atom = replace(atom, srcs=tuple(srcs))
    # A matching malformed SSA must not bypass descriptor admission either.
    call.operands[source].type = VectorType(I32, width)
    kernel.params[source].type = call.operands[source].type
    catalog = ArchTarget.from_gfx("gfx1250").mma
    with mock.patch.object(catalog, "by_op_id", return_value=atom):
        for operation in (
            lambda: gfx1250_scaled_wmma(atom.op_id),
            lambda: _lower_kernel_to_llvm_python(
                kernel, arch="gfx1250", llvm_flavor="llvm23"
            ),
            lambda: lower_kernel_to_hip(kernel, arch="gfx1250"),
        ):
            with pytest.raises(ValueError) as caught:
                operation()
            assert (
                str(caught.value)
                == f"unsupported scaled WMMA backend contract: {atom.op_id}"
            )


@pytest.mark.parametrize("scale16", [False, True])
@pytest.mark.parametrize("dtype", ["fp8", "bf8", "fp6", "bf6", "fp4"])
@pytest.mark.parametrize("field", ["m", "n", "k"])
@pytest.mark.parametrize(
    "kind", ["float", "fraction", "bool", "string", "null", "zero", "negative"]
)
def test_scaled_descriptor_rejects_invalid_shape_scalars(scale16, dtype, field, kind):
    kernel, _ = _scaled_call(scale16, False, dtype)
    atom = _scaled_atom(dtype, scale16)
    value = getattr(atom, field)
    value = {
        "float": float(value),
        "fraction": value + 0.5,
        "bool": True,
        "string": str(value),
        "null": None,
        "zero": 0,
        "negative": -1,
    }[kind]
    atom = replace(atom, **{field: value})
    catalog = ArchTarget.from_gfx("gfx1250").mma
    with mock.patch.object(catalog, "by_op_id", return_value=atom):
        for operation in (
            lambda: gfx1250_scaled_wmma(atom.op_id),
            lambda: _lower_kernel_to_llvm_python(
                kernel, arch="gfx1250", llvm_flavor="llvm23"
            ),
            lambda: lower_kernel_to_hip(kernel, arch="gfx1250"),
        ):
            with pytest.raises(ValueError) as caught:
                operation()
            assert (
                str(caught.value)
                == f"unsupported scaled WMMA backend contract: {atom.op_id}"
            )
