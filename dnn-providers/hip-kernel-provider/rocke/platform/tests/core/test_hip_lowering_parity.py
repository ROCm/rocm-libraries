# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Both HIP lowerers agree op by op: the same source, or the same refusal.

The family-wide HIP source identity test compares whatever the kernel families
build. The cases here pin the ops those families may not reach on every
target: every native MFMA and WMMA atom in each target's MMA catalog, inline asm
and its refusals for malformed attributes, and the s_waitcnt refusal on a target
without the monolithic instruction. Each case lowers one small kernel through the Python lowerer and, through the
rocke_engine binding, the C++ one, and compares the text or the error.
"""

import pytest
from rocke.core.arch import ArchTarget
from rocke.core.ir import (
    BF8E5M2,
    BF16,
    F16,
    F32,
    FP8E4M3,
    I32,
    IRBuilder,
    PtrType,
    VectorType,
    dtype_to_ir_type,
    is_packed_only,
    type_bits,
)
from rocke.core.ir_serialize import serialize
from rocke.core.lower_hip import lower_kernel_to_hip

# rocke_status_t (cpp/include/rocke/ir.h): the C++ error code the binding reports
# for each Python exception type the lowerers raise.
_NATIVE_STATUS = {ValueError: 1, TypeError: 2, KeyError: 3, NotImplementedError: 5}


def _native_hip(builder: IRBuilder, gfx: str) -> str:
    rocke_engine = pytest.importorskip("rocke_engine")
    return rocke_engine.lower_serialized_ir_hip(serialize(builder.kernel), gfx)


def _assert_both_refuse(
    builder: IRBuilder, gfx: str, error: type[Exception], message: str
) -> None:
    """Python raises `error` with exactly `message`; C++ reports the matching
    status with the same text."""
    with pytest.raises(error) as python_error:
        lower_kernel_to_hip(builder.kernel, arch=gfx)
    assert python_error.value.args[0] == message
    with pytest.raises(RuntimeError) as native_error:
        _native_hip(builder, gfx)
    assert str(native_error.value).endswith(
        f"(status {_NATIVE_STATUS[error]}): {message}"
    )


def _assert_same_source(builder: IRBuilder, gfx: str) -> str:
    python_hip = lower_kernel_to_hip(builder.kernel, arch=gfx)
    assert _native_hip(builder, gfx) == python_hip
    return python_hip


def _waitcnt_kernel(op: str) -> IRBuilder:
    builder = IRBuilder(f"{op}_kernel")
    builder.param("out", PtrType(F32, "global"))
    if op == "s_waitcnt":
        builder.s_waitcnt(lgkmcnt=0)
    else:
        builder.sync_lds_only()
    builder.ret()
    return builder


@pytest.mark.parametrize("op", ["s_waitcnt", "sync_lds_only"])
def test_waitcnt_is_refused_where_the_target_has_split_wait_counters(op):
    _assert_both_refuse(
        _waitcnt_kernel(op),
        "gfx1250",
        NotImplementedError,
        "s_waitcnt is not available on gfx1250: the target replaced it with "
        "split wait counters, which the HIP lowering does not emit; lower "
        "through the LLVM path",
    )


@pytest.mark.parametrize("op", ["s_waitcnt", "sync_lds_only"])
@pytest.mark.parametrize("gfx", ["gfx942", "gfx950", "gfx1151", "gfx1201"])
def test_waitcnt_still_lowers_where_the_target_has_it(op, gfx):
    hip = _assert_same_source(_waitcnt_kernel(op), gfx)
    assert "__builtin_amdgcn_s_waitcnt(" in hip


# Every target with a catalog, so each MMA atom is lowered where it exists.
_TARGETS = (
    "gfx90a",
    "gfx942",
    "gfx950",
    "gfx1151",
    "gfx11-generic",
    "gfx1201",
    "gfx12-generic",
    "gfx1250",
)

# WMMA atoms take 8- and 4-bit A and B operands packed into 32-bit words, and
# the gfx11 and gfx1250 catalog rows count those words. The gfx12 rows count
# values instead and get a wider operand here; the HIP lowering has none of
# those atoms, so both engines refuse them alike.
_PACKED_WORD_DTYPES = frozenset({"i8", "u8", "i4", "u4", "fp8e4m3", "bf8e5m2"})


def _zero_operand(builder: IRBuilder, atom, dtype: str, frag_len: int):
    """A or B operand of ``atom``: zero bits read as ``frag_len`` values of
    ``dtype``, so that a format with no zero constant (the 6- and 4-bit floats,
    the unsigned integers) is built the same way as any other. Packed words, and
    the packed-only i4 / u4, stay ``i32`` words."""
    element = dtype_to_ir_type(dtype)
    words_counted = atom.family == "wmma" and dtype in _PACKED_WORD_DTYPES
    if words_counted or is_packed_only(element):
        return builder.zero_vec(I32, frag_len)
    values = VectorType(element, frag_len)
    return builder.bitcast(builder.zero_vec(I32, type_bits(values) // 32), values)


def _catalog_atoms():
    """(gfx, atom) for every native MFMA / WMMA atom a target lists. TF32
    operands have no zero constant to build them from; their atoms are covered
    by the TF32 MMA tests."""
    for gfx in _TARGETS:
        for atom in ArchTarget.from_gfx(gfx).mma.ops:
            lowered_here = atom.family in ("mma", "wmma") and atom.is_native
            if lowered_here and "tf32" not in (atom.srcs[0].dtype, atom.srcs[1].dtype):
                yield pytest.param(gfx, atom, id=f"{gfx}-{atom.op_id}")


def _mma_kernel(op, a_type, a_len: int, b_type, b_len: int, c_type, c_len: int):
    builder = IRBuilder("mma_atom")
    builder.param("out", PtrType(c_type, "global"))
    builder.mma(
        op,
        builder.zero_vec(a_type, a_len),
        builder.zero_vec(b_type, b_len),
        builder.zero_vec(c_type, c_len),
    )
    builder.ret()
    return builder


def _lower_both(builder: IRBuilder, gfx: str) -> tuple[str, str]:
    """Each engine's HIP source, or its error text when it refuses."""
    try:
        python_result = lower_kernel_to_hip(builder.kernel, arch=gfx)
    except (ValueError, NotImplementedError) as error:
        python_result = f"refused: {error}"
    try:
        native_result = _native_hip(builder, gfx)
    except RuntimeError as error:
        native_result = f"refused: {str(error).split('): ', 1)[1]}"
    return python_result, native_result


@pytest.mark.parametrize(("gfx", "atom"), list(_catalog_atoms()))
def test_each_mma_atom_lowers_to_the_same_source(gfx, atom):
    builder = IRBuilder("mma_atom")
    multiplicand_a, multiplicand_b, accumulator_input = atom.srcs
    accumulator = dtype_to_ir_type(accumulator_input.dtype)
    builder.param("out", PtrType(accumulator, "global"))
    builder.mma(
        atom,
        _zero_operand(builder, atom, multiplicand_a.dtype, multiplicand_a.frag_len),
        _zero_operand(builder, atom, multiplicand_b.dtype, multiplicand_b.frag_len),
        builder.zero_vec(accumulator, accumulator_input.frag_len),
    )
    builder.ret()
    python_result, native_result = _lower_both(builder, gfx)
    assert native_result == python_result


# The 4x4x4 MFMA atoms are in no target's catalog; kernels name them by op_id.
@pytest.mark.parametrize(
    ("op_id", "operand_type"),
    [("mfma_f32_4x4x4_f16", F16), ("mfma_f32_4x4x4_bf16", BF16)],
)
def test_4x4x4_mfma_atoms_lower_to_the_same_source(op_id, operand_type):
    builder = _mma_kernel(op_id, operand_type, 4, operand_type, 4, F32, 4)
    hip = _assert_same_source(builder, "gfx942")
    assert "f32x4 " in hip and "__builtin_amdgcn_mfma_f32_4x4x4" in hip


def _zero_values(builder: IRBuilder, element, count: int):
    """``count`` zero-bit values of ``element``, built from i32 words so that a
    format with no zero constant (fp8, bf8) is built like any other."""
    values = VectorType(element, count)
    return builder.bitcast(builder.zero_vec(I32, type_bits(values) // 32), values)


# Each ported atom checks the operands against its instruction's fixed signature
# (core/mfma.py validate_mfma, core/wmma.py validate_unscaled_wmma) before it
# emits anything, and both engines refuse a mismatch with the same text.
@pytest.mark.parametrize(
    ("gfx", "op_id", "operands", "message"),
    [
        (
            "gfx942",
            "mfma_f32_16x16x16_f16",
            ((F16, 4), (F16, 4), (F32, 16)),
            "MFMA requires src2 and dst to be vec<f32x4>",
        ),
        (
            "gfx942",
            "mfma_f32_16x16x32_fp8",
            ((BF8E5M2, 8), (FP8E4M3, 8), (F32, 4)),
            (
                "MFMA source format mismatch - src0=vec<bf8e5m2x8>, "
                "expected vec<fp8e4m3x8> or vec<i8x8> or vec<u8x8>"
            ),
        ),
        (
            "gfx1100",
            "wmma_f32_16x16x16_f16",
            ((F16, 8), (F16, 16), (F32, 8)),
            "unscaled WMMA requires src0 to be vec<f16x16>",
        ),
        (
            "gfx1100",
            "wmma_i32_16x16x16_iu8",
            ((I32, 4), (I32, 4), (I32, 4)),
            "unscaled WMMA requires src2 and dst to be vec<i32x8>",
        ),
    ],
)
def test_an_mma_atom_refuses_operands_its_instruction_does_not_take(
    gfx, op_id, operands, message
):
    builder = IRBuilder("mma_atom")
    builder.param("out", PtrType(operands[2][0], "global"))
    builder.mma(op_id, *(_zero_values(builder, *operand) for operand in operands))
    builder.ret()
    _assert_both_refuse(builder, gfx, ValueError, message)


def _inline_asm_kernel(**attr_changes) -> IRBuilder:
    """One tile.inline_asm op with a result; `attr_changes` overwrites its
    attrs (None deletes one) to build the malformed op a serialized kernel
    could carry."""
    builder = IRBuilder("inline_asm_kernel")
    builder.param("out", PtrType(F32, "global"))
    builder.inline_asm("v_mov_b32 $0, $1", "=v,v", [builder.const_i32(1)], I32)
    op = builder.kernel.body.ops[-1]
    assert op.name == "tile.inline_asm"
    for key, value in attr_changes.items():
        if value is None:
            del op.attrs[key]
        else:
            op.attrs[key] = value
    builder.ret()
    return builder


# test_optimization_barrier.py holds the other inline asm refusals (sideeffect,
# constraint values and types, clobbers); these are the template and
# constraints reads, which no verifier guards on the serialized-IR route.
@pytest.mark.parametrize(
    ("attr_changes", "error", "message"),
    [
        ({"template": None}, KeyError, "tile.inline_asm: missing 'template' attr"),
        (
            {"constraints": None},
            KeyError,
            "tile.inline_asm: missing 'constraints' attr",
        ),
        ({"template": 7}, TypeError, "HIP inline asm template must be a string"),
        (
            {"template": None, "constraints": 7},
            KeyError,
            "tile.inline_asm: missing 'template' attr",
        ),
    ],
)
def test_malformed_inline_asm_is_refused_the_same_way(attr_changes, error, message):
    _assert_both_refuse(_inline_asm_kernel(**attr_changes), "gfx942", error, message)
