# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MMA operands are passed as the register type the instruction takes.

An operand whose LLVM type already matches passes through unchanged. An operand in
another shape is reinterpreted with one ``bitcast`` when it holds the register's own
element type (a one-element vector) or the operand's element type in the target's
MMA catalog (the instruction's packed bytes). An operand holding any other number
format, or of another bit width, fails while lowering, naming the operand and the
instruction.

Every lowering here pins the Python engine: the C++ engine does not mirror this yet.
"""

import re

import pytest
from rocke.core.arch import ArchTarget
from rocke.core.ir import BF8E5M2, F16, F32, FP8E4M3, I8, I32, IRBuilder, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.tests.ir_validity import assert_target_accepts


def _mma_kernel(arch, op_id, a_operand, b_operand, accumulator, result_type) -> str:
    """D = A x B + C for one MMA; each operand is built by its callable."""
    op = ArchTarget.from_gfx(arch).mma.by_op_id(op_id)
    b = IRBuilder(f"mma_operand_{op_id}")
    out = b.param("D", PtrType(result_type, "global"))
    result = b.mma(op, a_operand(b), b_operand(b), accumulator(b))
    b.global_store(out, b.thread_id_x(), b.vec_extract(result, 0))
    return _lower_kernel_to_llvm_python(b.kernel, arch=arch)


def _mma_call(text: str) -> str:
    return next(
        line for line in text.splitlines() if "= call" in line and "amdgcn" in line
    )


def _f32_scalar(b):
    return b.const_f32(1.0)


def _f32_one_element_vector(b):
    return b.vec_pack([b.const_f32(2.0)], F32)


def _zeros(elem, count):
    return lambda b: b.zero_vec(elem, count)


_F32_MFMAS = [("mfma_f32_16x16x4_f32", 4), ("mfma_f32_32x32x2_f32", 16)]


@pytest.mark.parametrize(("op_id", "accumulator_count"), _F32_MFMAS)
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_an_f32_mfma_takes_scalars_unchanged(arch, op_id, accumulator_count):
    text = _mma_kernel(
        arch,
        op_id,
        _f32_scalar,
        _f32_scalar,
        _zeros(F32, accumulator_count),
        F32,
    )
    assert "bitcast" not in text
    assert_target_accepts(text, arch)


@pytest.mark.parametrize(("op_id", "accumulator_count"), _F32_MFMAS)
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_an_f32_mfma_reinterprets_only_the_one_element_vector(
    arch, op_id, accumulator_count
):
    text = _mma_kernel(
        arch,
        op_id,
        _f32_one_element_vector,
        _f32_scalar,
        _zeros(F32, accumulator_count),
        F32,
    )
    (reinterpreted,) = re.findall(
        r"(%reg\.\d+) = bitcast <1 x float> \S+ to float", text
    )
    assert re.search(
        rf"\(float {re.escape(reinterpreted)}, float 0x3FF0000000000000,",
        _mma_call(text),
    )
    assert_target_accepts(text, arch)


def test_integer_wmma_takes_packed_words_unchanged():
    text = _mma_kernel(
        "gfx1151",
        "wmma_i32_16x16x16_iu8",
        _zeros(I32, 4),
        _zeros(I32, 4),
        _zeros(I32, 8),
        I32,
    )
    assert "bitcast" not in text
    assert_target_accepts(text, "gfx1151")


def test_integer_wmma_reinterprets_only_the_packed_byte_operand():
    text = _mma_kernel(
        "gfx1151",
        "wmma_i32_16x16x16_iu8",
        _zeros(I8, 16),
        _zeros(I32, 4),
        _zeros(I32, 8),
        I32,
    )
    (reinterpreted,) = re.findall(
        r"(%reg\.\d+) = bitcast <16 x i8> \S+ to <4 x i32>", text
    )
    assert re.search(
        rf"\(i1 1, <4 x i32> {re.escape(reinterpreted)}, i1 1, <4 x i32> %cz\d+,",
        _mma_call(text),
    )
    assert_target_accepts(text, "gfx1151")


@pytest.mark.parametrize(
    ("op_id", "a_operand"),
    [
        ("wmma_i32_16x16x16_iu8", _zeros(I32, 4)),
        ("wmma_i32_16x16x16_iu8", _zeros(I8, 16)),
        ("wmma_i32_16x16x16_iu4", _zeros(I32, 2)),
    ],
)
def test_integer_wmma_marks_both_operands_signed_without_clamping(op_id, a_operand):
    words = 4 if op_id.endswith("iu8") else 2
    text = _mma_kernel(
        "gfx1151", op_id, a_operand, _zeros(I32, words), _zeros(I32, 8), I32
    )
    assert re.search(
        rf"\(i1 1, <{words} x i32> \S+, i1 1, <{words} x i32> \S+, "
        r"<8 x i32> \S+, i1 0\)",
        _mma_call(text),
    )


def test_gfx1250_fp8_wmma_reinterprets_each_operand_as_its_catalog_format():
    text = _mma_kernel(
        "gfx1250",
        "wmma_gfx1250_f32_16x16x64_fp8_bf8",
        _zeros(FP8E4M3, 32),
        _zeros(BF8E5M2, 32),
        _zeros(F32, 8),
        F32,
    )
    assert len(re.findall(r"bitcast <32 x i8> \S+ to <8 x i32>", text)) == 2
    assert_target_accepts(text, "gfx1250")


@pytest.mark.parametrize(
    ("arch", "op_id", "a_operand", "b_operand", "accumulator", "message"),
    [
        (
            "gfx942",
            "mfma_f32_16x16x4_f32",
            lambda b: b.const_i32(7),
            _f32_scalar,
            _zeros(F32, 4),
            (
                "MMA operand a of 'mfma_f32_16x16x4_f32' holds i32 values; the "
                "instruction takes f32 -- convert the values first"
            ),
        ),
        (
            "gfx942",
            "mfma_f32_16x16x4_f32",
            _f32_scalar,
            _zeros(F16, 2),
            _zeros(F32, 4),
            (
                "MMA operand b of 'mfma_f32_16x16x4_f32' holds f16 values; the "
                "instruction takes f32 -- convert the values first"
            ),
        ),
        (
            "gfx1151",
            "wmma_i32_16x16x16_iu8",
            _zeros(F16, 8),
            _zeros(I32, 4),
            _zeros(I32, 8),
            (
                "MMA operand a of 'wmma_i32_16x16x16_iu8' holds f16 values; the "
                "instruction takes i32 or i8 -- convert the values first"
            ),
        ),
        (
            "gfx1151",
            "wmma_i32_16x16x16_iu4",
            _zeros(I8, 8),
            _zeros(I32, 2),
            _zeros(I32, 8),
            (
                "MMA operand a of 'wmma_i32_16x16x16_iu4' holds i8 values; the "
                "instruction takes i32 or i4 -- convert the values first"
            ),
        ),
        (
            "gfx1250",
            "wmma_gfx1250_f32_16x16x64_fp8_bf8",
            _zeros(FP8E4M3, 32),
            _zeros(FP8E4M3, 32),
            _zeros(F32, 8),
            (
                "MMA operand b of 'wmma_gfx1250_f32_16x16x64_fp8_bf8' holds fp8e4m3 "
                "values; the instruction takes bf8e5m2 or i32 -- convert the values first"
            ),
        ),
    ],
    ids=["i32-into-f32", "f16-into-f32", "f16-into-iu8", "i8-into-iu4", "fp8-into-bf8"],
)
def test_an_operand_of_another_number_format_fails(
    arch, op_id, a_operand, b_operand, accumulator, message
):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _mma_kernel(arch, op_id, a_operand, b_operand, accumulator, F32)


@pytest.mark.parametrize("role", ["a", "b"])
def test_an_operand_of_another_width_fails_naming_its_role(role):
    narrow, words = _zeros(I8, 8), _zeros(I32, 4)
    message = (
        f"MMA operand {role} of 'wmma_i32_16x16x16_iu8' is vec<i8x8> (64 bits); the "
        "instruction takes <4 x i32> (128 bits) -- pass a 128-bit value"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _mma_kernel(
            "gfx1151",
            "wmma_i32_16x16x16_iu8",
            narrow if role == "a" else words,
            narrow if role == "b" else words,
            _zeros(I32, 8),
            I32,
        )
