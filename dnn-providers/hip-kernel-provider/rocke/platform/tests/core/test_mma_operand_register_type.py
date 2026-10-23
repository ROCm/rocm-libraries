# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MMA operands are passed as the register type the instruction takes.

An operand of the register type passes through unchanged. An operand in another
shape is reinterpreted once -- an LLVM ``bitcast``, a HIP ``__builtin_bit_cast`` --
when it holds the register's own element type (a one-element vector) or the
operand's element type in the target's MMA catalog (the instruction's packed
bytes). An operand holding any other number format, or of another bit width,
fails while lowering, naming the operand and the instruction, with the same text
in the LLVM and HIP lowerings.

Every LLVM case lowers with both engines and requires the same IR, or the same
error. The HIP cases do the same for the f32 MFMAs; the C++ HIP lowerer has no
integer or gfx1250 fp8 WMMA lowering, so those cases run the Python HIP lowering
and require the LLVM lowering's error text.
"""

import re
import shutil
import subprocess

import pytest
from rocke.core.arch import ArchTarget
from rocke.core.ir import (
    BF8E5M2,
    F16,
    F32,
    FP8E4M3,
    I8,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
)
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.tests.engines import (
    assert_both_engines_reject,
    assert_both_engines_reject_hip,
    lower_hip_with_both_engines,
    lower_with_both_engines,
)
from rocke.tests.ir_validity import assert_target_accepts


def _mma_kernel(
    arch, op_id, a_operand, b_operand, accumulator, result_type
) -> KernelDef:
    """D = A x B + C for one MMA; each operand is built by its callable."""
    op = ArchTarget.from_gfx(arch).mma.by_op_id(op_id)
    b = IRBuilder(f"mma_operand_{op_id}")
    out = b.param("D", PtrType(result_type, "global"))
    result = b.mma(op, a_operand(b), b_operand(b), accumulator(b))
    b.global_store(out, b.thread_id_x(), b.vec_extract(result, 0))
    return b.kernel


def _lower_mma(arch, op_id, a_operand, b_operand, accumulator, result_type) -> str:
    """The LLVM IR both engines emit for :func:`_mma_kernel`."""
    kernel = _mma_kernel(arch, op_id, a_operand, b_operand, accumulator, result_type)
    return lower_with_both_engines(kernel, arch)


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
    text = _lower_mma(
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
    text = _lower_mma(
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
    text = _lower_mma(
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
    text = _lower_mma(
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
    text = _lower_mma(
        "gfx1151", op_id, a_operand, _zeros(I32, words), _zeros(I32, 8), I32
    )
    assert re.search(
        rf"\(i1 1, <{words} x i32> \S+, i1 1, <{words} x i32> \S+, "
        r"<8 x i32> \S+, i1 0\)",
        _mma_call(text),
    )


def test_gfx1250_fp8_wmma_reinterprets_each_operand_as_its_catalog_format():
    text = _lower_mma(
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
                "instruction takes i32 -- convert the values first"
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
    kernel = _mma_kernel(arch, op_id, a_operand, b_operand, accumulator, F32)
    assert_both_engines_reject(kernel, arch, ValueError, message)
    _assert_hip_rejects(kernel, arch, op_id, message)


@pytest.mark.parametrize("role", ["a", "b"])
def test_an_operand_of_another_width_fails_naming_its_role(role):
    narrow, words = _zeros(I8, 8), _zeros(I32, 4)
    message = (
        f"MMA operand {role} of 'wmma_i32_16x16x16_iu8' is vec<i8x8> (64 bits); the "
        "instruction takes vec<i32x4> (128 bits) -- pass a 128-bit value"
    )
    kernel = _mma_kernel(
        "gfx1151",
        "wmma_i32_16x16x16_iu8",
        narrow if role == "a" else words,
        narrow if role == "b" else words,
        _zeros(I32, 8),
        I32,
    )
    assert_both_engines_reject(kernel, "gfx1151", ValueError, message)
    _assert_hip_rejects(kernel, "gfx1151", "wmma_i32_16x16x16_iu8", message)


def test_an_f32_operand_of_another_width_fails_in_every_lowering():
    message = (
        "MMA operand a of 'mfma_f32_16x16x4_f32' is vec<f32x2> (64 bits); the "
        "instruction takes f32 (32 bits) -- pass a 32-bit value"
    )
    kernel = _mma_kernel(
        "gfx942",
        "mfma_f32_16x16x4_f32",
        _zeros(F32, 2),
        _f32_scalar,
        _zeros(F32, 4),
        F32,
    )
    assert_both_engines_reject(kernel, "gfx942", ValueError, message)
    _assert_hip_rejects(kernel, "gfx942", "mfma_f32_16x16x4_f32", message)


# ------------------------------------------------------------------- HIP source

# The MMAs the C++ HIP lowerer lowers among those this file covers.
_NATIVE_HIP_MMAS = frozenset(op_id for op_id, _ in _F32_MFMAS)


def _assert_hip_rejects(kernel: KernelDef, arch: str, op_id: str, message: str) -> None:
    """The HIP lowering refuses ``kernel`` with the LLVM lowering's ``message``."""
    if op_id in _NATIVE_HIP_MMAS:
        assert_both_engines_reject_hip(kernel, arch, ValueError, message)
        return
    with pytest.raises(ValueError) as error:
        lower_kernel_to_hip(kernel, arch=arch)
    assert str(error.value) == message


def _loaded(elem, count=None):
    """An operand built from one value loaded from its own global buffer.

    A scalar when ``count`` is None, else that value in every one of ``count``
    lanes. The C++ HIP lowering spells float literals differently from the
    Python one, so the HIP parity kernels take every value from memory instead.
    """

    def build(b):
        source = b.param(f"S{len(b.kernel.params)}", PtrType(elem, "global"))
        value = b.global_load(source, b.thread_id_x(), elem)
        return value if count is None else b.vector_splat(value, count)

    return build


def _mma_builtin_call(source: str) -> str:
    return next(
        line
        for line in source.splitlines()
        if "__builtin_amdgcn_" in line and ("mfma" in line or "wmma" in line)
    )


def _assert_hipcc_accepts(source: str, arch: str, tmp_path) -> None:
    """Compile ``source`` for ``arch`` with hipcc, when hipcc is present."""
    if not shutil.which("hipcc"):
        pytest.skip("hipcc not in PATH")
    path = tmp_path / "kernel.hip"
    path.write_text(source)
    subprocess.run(
        [
            "hipcc",
            f"--offload-arch={arch}",
            "--cuda-device-only",
            "-fsyntax-only",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(("op_id", "accumulator_count"), _F32_MFMAS)
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_hip_f32_mfma_takes_scalars_unchanged(arch, op_id, accumulator_count, tmp_path):
    kernel = _mma_kernel(
        arch,
        op_id,
        _loaded(F32),
        _loaded(F32),
        _loaded(F32, accumulator_count),
        F32,
    )
    source = lower_hip_with_both_engines(kernel, arch)
    assert "__builtin_bit_cast" not in _mma_builtin_call(source)
    _assert_hipcc_accepts(source, arch, tmp_path)


@pytest.mark.parametrize(("op_id", "accumulator_count"), _F32_MFMAS)
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_hip_f32_mfma_reinterprets_only_the_one_element_vector(
    arch, op_id, accumulator_count, tmp_path
):
    kernel = _mma_kernel(
        arch,
        op_id,
        _loaded(F32),
        _loaded(F32, 1),
        _loaded(F32, accumulator_count),
        F32,
    )
    source = lower_hip_with_both_engines(kernel, arch)
    call = _mma_builtin_call(source)
    assert re.search(r"\(\w+, __builtin_bit_cast\(float, \w+\), \w+, 0, 0, 0\);", call)
    assert call.count("__builtin_bit_cast") == 1
    _assert_hipcc_accepts(source, arch, tmp_path)


def test_hip_integer_wmma_reinterprets_only_the_packed_byte_operand(tmp_path):
    kernel = _mma_kernel(
        "gfx1151",
        "wmma_i32_16x16x16_iu8",
        _zeros(I8, 16),
        _zeros(I32, 4),
        _zeros(I32, 8),
        I32,
    )
    source = lower_kernel_to_hip(kernel, arch="gfx1151")
    assert re.search(
        r"\(true, __builtin_bit_cast\(i32x4, \w+\), true, \w+, \w+, false\);",
        _mma_builtin_call(source),
    )
    _assert_hipcc_accepts(source, "gfx1151", tmp_path)


def test_hip_integer_wmma_takes_packed_words_unchanged():
    kernel = _mma_kernel(
        "gfx1151",
        "wmma_i32_16x16x16_iu4",
        _zeros(I32, 2),
        _zeros(I32, 2),
        _zeros(I32, 8),
        I32,
    )
    call = _mma_builtin_call(lower_kernel_to_hip(kernel, arch="gfx1151"))
    assert "__builtin_bit_cast" not in call


def test_hip_gfx1250_fp8_wmma_reinterprets_each_operand_as_its_catalog_format():
    kernel = _mma_kernel(
        "gfx1250",
        "wmma_gfx1250_f32_16x16x64_fp8_bf8",
        _zeros(FP8E4M3, 32),
        _zeros(BF8E5M2, 32),
        _zeros(F32, 8),
        F32,
    )
    call = _mma_builtin_call(lower_kernel_to_hip(kernel, arch="gfx1250"))
    assert len(re.findall(r"__builtin_bit_cast\(i32x8, \w+\)", call)) == 2
