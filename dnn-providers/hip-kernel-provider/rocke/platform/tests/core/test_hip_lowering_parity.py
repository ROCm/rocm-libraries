# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Both HIP lowerers agree op by op: the same source, or the same refusal.

The family-wide HIP source identity test compares whatever the kernel families
build. The cases here pin the ops those families may not reach on every
target: every native MFMA and WMMA atom in each target's MMA catalog, inline asm
and its refusals for malformed attributes, and the s_waitcnt refusal on a target
without the monolithic instruction. Each case lowers one small kernel through the
Python lowerer and, through the rocke_engine binding, the C++ one, and compares
the text or the error.
"""

import pytest
from rocke.core.arch import ArchTarget, arches_with_own_facts
from rocke.core.ir import (
    BF16,
    F16,
    F32,
    I32,
    IRBuilder,
    PtrType,
    VectorType,
    dtype_to_ir_type,
    is_packed_only,
    type_bits,
)
from rocke.core.ir_serialize import serialize
from rocke.tests.engines import (
    HipRefusal,
    hip_outcomes_of_both_engines,
    lower_hip_with_both_engines,
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
            if lowered_here and "tf32" not in (atom.a_dtype, atom.b_dtype):
                yield pytest.param(gfx, atom, id=f"{gfx}-{atom.op_id}")


@pytest.mark.parametrize(("gfx", "atom"), list(_catalog_atoms()))
def test_each_mma_atom_lowers_to_the_same_source(gfx, atom):
    builder = IRBuilder("mma_atom")
    accumulator = dtype_to_ir_type(atom.c_dtype)
    builder.param("out", PtrType(accumulator, "global"))
    builder.mma(
        atom,
        _zero_operand(builder, atom, atom.a_dtype, atom.a_frag_len),
        _zero_operand(builder, atom, atom.b_dtype, atom.b_frag_len),
        builder.zero_vec(accumulator, atom.c_frag_len),
    )
    builder.ret()
    python_outcome, native_outcome = hip_outcomes_of_both_engines(
        serialize(builder.kernel), gfx
    )
    assert native_outcome == python_outcome


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


def _inline_asm_kernel(*, with_result: bool, **attr_changes) -> IRBuilder:
    """One tile.inline_asm op; `attr_changes` overwrites its attrs (None
    deletes one) to build the malformed op a serialized kernel could carry."""
    builder = IRBuilder("inline_asm_kernel")
    builder.param("out", PtrType(F32, "global"))
    operand = builder.const_i32(1)
    if with_result:
        builder.inline_asm("v_mov_b32 $0, $1", "=v, v", [operand], I32)
    else:
        builder.inline_asm("s_nop $0", "n", [operand])
    op = builder.kernel.body.ops[-1]
    assert op.name == "tile.inline_asm"
    for key, value in attr_changes.items():
        if value is None:
            del op.attrs[key]
        else:
            op.attrs[key] = value
    builder.ret()
    return builder


@pytest.mark.parametrize("with_result", [True, False])
@pytest.mark.parametrize(
    "attr_changes", [{}, {"sideeffect": False}, {"clobber": "vcc"}]
)
def test_inline_asm_lowers_to_the_same_source(with_result, attr_changes):
    builder = _inline_asm_kernel(with_result=with_result, **attr_changes)
    hip = lower_hip_with_both_engines(builder.kernel, "gfx942")
    assert "asm " in hip


@pytest.mark.parametrize(
    ("attr_changes", "error", "message"),
    [
        ({"template": None}, KeyError, "tile.inline_asm: missing 'template' attr"),
        (
            {"constraints": None},
            KeyError,
            "tile.inline_asm: missing 'constraints' attr",
        ),
        ({"template": 7}, TypeError, "tile.inline_asm: 'template' must be a string"),
        (
            {"constraints": 7},
            TypeError,
            "tile.inline_asm: 'constraints' must be a string",
        ),
        ({"clobber": 1}, TypeError, "tile.inline_asm: 'clobber' must be a string"),
        ({"constraints": " , "}, ValueError, "tile.inline_asm: no output constraint"),
    ],
)
def test_malformed_inline_asm_is_refused_the_same_way(attr_changes, error, message):
    _assert_both_refuse(
        _inline_asm_kernel(with_result=True, **attr_changes), "gfx942", error, message
    )
