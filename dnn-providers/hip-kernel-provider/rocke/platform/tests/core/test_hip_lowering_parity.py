# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Both HIP lowerers agree op by op: the same source, or the same refusal.

The family-wide HIP source identity test compares whatever the kernel families
build. The cases here pin the ops those families may not reach on every
target: every native MFMA and WMMA atom in each target's MMA catalog, the inline asm
template and constraints refusals, and the s_waitcnt refusal on a target
without the monolithic instruction. Each case lowers one small kernel through the
Python lowerer and, through the rocke_engine binding, the C++ one, and compares
the text or the error.
"""

from __future__ import annotations

import pytest
from rocke.core.arch import ArchTarget, MmaOp, arches_with_own_facts
from rocke.core.ir import (
    BF8E5M2,
    BF16,
    F16,
    F32,
    FP8E4M3,
    I32,
    IRBuilder,
    PtrType,
    Type,
    U8,
    Value,
    VectorType,
    dtype_to_ir_type,
    is_packed_only,
    type_bits,
)
from rocke.core.ir_serialize import serialize
from rocke.tests.engines import (
    HipRefusal,
    hip_outcomes_of_both_engines,
    ir_lower_cli_hip_outcomes,
    lower_hip_with_both_engines,
    native_hip_outcomes,
)


def _assert_both_refuse(
    builder: IRBuilder, gfx: str, error: type[Exception], message: str
) -> None:
    """Both engines refuse the kernel with ``error`` and exactly ``message``:
    Python raises it, and C++ reports the status code that stands for it."""
    refusal = HipRefusal(error.__name__, message)
    assert hip_outcomes_of_both_engines(serialize(builder.kernel), gfx) == (
        refusal,
        refusal,
    )


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


def test_ir_lower_cli_reports_what_the_binding_reports():
    """The installed test tree runs the C++ HIP lowering through
    rocke_ir_lower_cli, other lanes through the binding: for the same kernels,
    in one batch, both report the same source and the same refusals, in order,
    including the parser's refusals of text that is not a kernel. Runs where
    both are present."""
    pytest.importorskip("rocke_engine")
    waitcnt = serialize(_waitcnt_kernel("s_waitcnt").kernel)
    no_template = serialize(_inline_asm_kernel(template=None).kernel)
    kernels = [
        (waitcnt, "gfx942"),
        (waitcnt, "gfx1250"),
        (no_template, "gfx942"),
        ("", "gfx942"),
        ("not serialized IR", "gfx942"),
        (waitcnt, "gfx950"),
    ]
    via_program = ir_lower_cli_hip_outcomes(kernels)
    assert via_program == native_hip_outcomes(kernels)
    assert [isinstance(outcome, HipRefusal) for outcome in via_program] == [
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    assert via_program[3] == HipRefusal("ValueError", "empty IR input")
    assert via_program[4].error_type_name == "ValueError"


@pytest.mark.parametrize("op", ["s_waitcnt", "sync_lds_only"])
@pytest.mark.parametrize("gfx", ["gfx942", "gfx950", "gfx1151", "gfx1201"])
def test_waitcnt_still_lowers_where_the_target_has_it(op, gfx):
    hip = lower_hip_with_both_engines(_waitcnt_kernel(op).kernel, gfx)
    assert "__builtin_amdgcn_s_waitcnt(" in hip


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
    """(gfx, atom) for every native MFMA / WMMA atom of each distinct target
    catalog (a target that takes its facts from a generic has the generic's).
    TF32 atoms are covered by the TF32 MMA tests."""
    for gfx in arches_with_own_facts():
        for atom in ArchTarget.from_gfx(gfx).mma.ops:
            lowered_here = atom.family in ("mma", "wmma") and atom.is_native
            if lowered_here and "tf32" not in (atom.srcs[0].dtype, atom.srcs[1].dtype):
                yield pytest.param(gfx, atom, id=f"{gfx}-{atom.op_id}")


def _atom_kernel(atom) -> IRBuilder:
    """One tile.mma op of ``atom`` on zero operands, returned unterminated so a
    test can malform the op before ``ret``."""
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
    return builder


@pytest.mark.parametrize(("gfx", "atom"), list(_catalog_atoms()))
def test_each_mma_atom_lowers_to_the_same_source(gfx, atom):
    builder = _atom_kernel(atom)
    builder.ret()
    python_outcome, native_outcome = hip_outcomes_of_both_engines(
        serialize(builder.kernel), gfx
    )
    assert native_outcome == python_outcome


@pytest.mark.parametrize("missing", ["accumulator", "result"])
@pytest.mark.parametrize(("gfx", "atom"), list(_catalog_atoms()))
def test_mma_atom_without_accumulator_or_result_is_refused(gfx, atom, missing):
    """A serialized kernel is not verified, so an MMA op may lack its
    accumulator or its result. Both engines refuse it alike, and where the
    well-formed kernel lowers, with the instruction check's count error
    before reading either."""
    well_formed = _atom_kernel(atom)
    well_formed.ret()
    well_formed_outcome, _ = hip_outcomes_of_both_engines(
        serialize(well_formed.kernel), gfx
    )
    python_outcome, native_outcome = hip_outcomes_of_both_engines(
        serialize(_atom_kernel_without(atom, missing).kernel), gfx
    )
    assert native_outcome == python_outcome
    if not isinstance(well_formed_outcome, HipRefusal):
        instruction = "MFMA" if atom.family == "mma" else "unscaled WMMA"
        assert python_outcome == HipRefusal(
            "ValueError", f"{instruction} expects 3 operands and 1 result"
        )


def _atom_kernel_without(atom: MmaOp, missing: str) -> IRBuilder:
    """The kernel of :func:`_atom_kernel`, terminated, with its MMA op's
    ``missing`` part -- its "accumulator" operand or its "result" -- removed."""
    builder = _atom_kernel(atom)
    op = builder.kernel.body.ops[-1]
    assert op.name == "tile.mma"
    if missing == "accumulator":
        del op.operands[2]
    else:
        op.results.clear()
    builder.ret()
    return builder


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        (
            "accumulator",
            "XF32 MMA requires two vec<tf32x2> operands and its FP32 accumulator",
        ),
        ("result", "XF32 MMA result must match its FP32 accumulator"),
    ],
)
@pytest.mark.parametrize("op_id", ["mfma_f32_16x16x8_xf32", "mfma_f32_32x32x4_xf32"])
def test_xf32_mma_without_accumulator_or_result_is_refused(op_id, missing, message):
    """Both engines check an XF32 MMA op's A, B, C and result before any
    handler reads them, so one missing is refused by that check."""
    atom = ArchTarget.from_gfx("gfx942").mma.by_op_id(op_id)
    assert hip_outcomes_of_both_engines(
        serialize(_atom_kernel_without(atom, missing).kernel), "gfx942"
    ) == (HipRefusal("ValueError", message), HipRefusal("ValueError", message))


def _scaled_catalog_atoms():
    """(gfx, atom) for every scaled MFMA / WMMA atom of each distinct target
    catalog; their ops take A, B, C and the two scales."""
    for gfx in arches_with_own_facts():
        for atom in ArchTarget.from_gfx(gfx).mma.ops:
            if atom.family in ("mfma_scaled", "wmma_scaled"):
                yield pytest.param(gfx, atom, id=f"{gfx}-{atom.op_id}")


# An unscaled atom of the same target whose A, B and C both engines lower; its
# kernel, renamed to a scaled atom, is a scaled op without its scales.
_UNSCALED_TWIN = {
    "mfma_scaled": "mfma_f32_16x16x32_fp8",
    "wmma_scaled": "wmma_gfx1250_f32_16x16x64_fp8_fp8",
}


@pytest.mark.parametrize(("gfx", "atom"), list(_scaled_catalog_atoms()))
def test_scaled_mma_atom_without_its_scales_is_refused(gfx, atom):
    """The IR builder will not build a scaled op without its scales, but a
    serialized kernel can carry one; both engines refuse it alike, a scaled
    WMMA op with the operand count both check."""
    twin = ArchTarget.from_gfx(gfx).mma.by_op_id(_UNSCALED_TWIN[atom.family])
    builder = _atom_kernel(twin)
    builder.kernel.body.ops[-1].attrs["op_id"] = atom.op_id
    builder.ret()
    python_outcome, native_outcome = hip_outcomes_of_both_engines(
        serialize(builder.kernel), gfx
    )
    assert native_outcome == python_outcome
    assert isinstance(python_outcome, HipRefusal)
    if atom.family == "wmma_scaled":
        assert python_outcome == HipRefusal(
            "ValueError", f"tile.{atom.op_id} expects 5 operands and 1 result"
        )


# The 4x4x4 MFMA atoms are in no target's catalog; kernels name them by op_id.
@pytest.mark.parametrize(
    ("op_id", "operand_type"),
    [("mfma_f32_4x4x4_f16", F16), ("mfma_f32_4x4x4_bf16", BF16)],
)
def test_4x4x4_mfma_atoms_lower_to_the_same_source(op_id, operand_type):
    builder = IRBuilder("mma_atom")
    builder.param("out", PtrType(F32, "global"))
    builder.mma(
        op_id,
        builder.zero_vec(operand_type, 4),
        builder.zero_vec(operand_type, 4),
        builder.zero_vec(F32, 4),
    )
    builder.ret()
    hip = lower_hip_with_both_engines(builder.kernel, "gfx942")
    assert "f32x4 " in hip and "__builtin_amdgcn_mfma_f32_4x4x4" in hip


def _zero_values(builder: IRBuilder, element: Type, count: int) -> Value:
    """``count`` zero-bit values of ``element``, built from i32 words so that a
    format with no zero constant (fp8, bf8) is built like any other."""
    values = VectorType(element, count)
    return builder.bitcast(builder.zero_vec(I32, type_bits(values) // 32), values)


# Each ported atom checks the operands against its instruction's fixed signature
# (core/mfma.py validate_mfma, core/wmma.py validate_unscaled_wmma) before it
# emits anything, and both engines refuse a mismatch with the same text. A
# carrier the instruction takes but HIP has no type for (vec<u8x8>) is refused
# as an unsupported type by both.
@pytest.mark.parametrize(
    ("gfx", "op_id", "operands", "error", "message"),
    [
        (
            "gfx942",
            "mfma_f32_16x16x16_f16",
            ((F16, 4), (F16, 4), (F32, 16)),
            ValueError,
            "MFMA requires src2 and dst to be vec<f32x4>",
        ),
        (
            "gfx942",
            "mfma_f32_16x16x32_fp8",
            ((BF8E5M2, 8), (FP8E4M3, 8), (F32, 4)),
            ValueError,
            (
                "MFMA source format mismatch - src0=vec<bf8e5m2x8>, "
                "expected vec<fp8e4m3x8> or vec<i8x8> or vec<u8x8>"
            ),
        ),
        (
            "gfx942",
            "mfma_f32_16x16x32_fp8",
            ((U8, 8), (U8, 8), (F32, 4)),
            NotImplementedError,
            "no HIP mapping for type vec<u8x8>",
        ),
        (
            "gfx1100",
            "wmma_f32_16x16x16_f16",
            ((F16, 8), (F16, 16), (F32, 8)),
            ValueError,
            "unscaled WMMA requires src0 to be vec<f16x16>",
        ),
        (
            "gfx1100",
            "wmma_i32_16x16x16_iu8",
            ((I32, 4), (I32, 4), (I32, 4)),
            ValueError,
            "unscaled WMMA requires src2 and dst to be vec<i32x8>",
        ),
    ],
)
def test_an_mma_atom_refuses_operands_its_instruction_does_not_take(
    gfx: str,
    op_id: str,
    operands: tuple[tuple[Type, int], ...],
    error: type[Exception],
    message: str,
) -> None:
    builder = IRBuilder("mma_atom")
    builder.param("out", PtrType(operands[2][0], "global"))
    builder.mma(op_id, *(_zero_values(builder, *operand) for operand in operands))
    builder.ret()
    _assert_both_refuse(builder, gfx, error, message)


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
