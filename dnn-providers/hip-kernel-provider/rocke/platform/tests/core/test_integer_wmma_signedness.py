# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Integer WMMA signedness comes from the operand dtypes, never from a literal.

Every integer WMMA (gfx11 i8 and i4, gfx12 i8 and i4 at K 16 and K 32) takes
one ``i1`` flag before A and one before B: 1 selects a signed operand, 0 an
unsigned one. Each instruction has one catalog row per signedness combination
-- the signed op_id, and the op_id with ``_u8_i8`` (A unsigned), ``_i8_u8``
(B unsigned) or ``_u8_u8`` appended (``_u4_i4`` ... for 4-bit operands) -- and
the lowering reads each flag from that row's operand dtype.

The cases below pin, for every integer WMMA and all four combinations, the two
flags both engines emit at every LLVM flavor, that the gfx11 unsigned rows
compile and link for gfx1100 and gfx11-generic and select the instruction
(compile-verified, not run), that the HIP lowering of every gfx11 row passes
its builtin the bools the row gives and hipcc turns them into the same flags,
that the gfx11 signed rows keep their declaration, and that neither engine's
source writes a flag as a literal.
"""

from __future__ import annotations

import re
import shutil

import pytest
from rocke.assets import platform_root
from rocke.core.arch import ArchTarget
from rocke.core.arch.target import integer_mma_op_ids
from rocke.core.ir import IRBuilder, PtrType, VectorType, dtype_to_ir_type
from rocke.core.isa.matrix_calls import MATRIX_CALLS, MatrixOperand
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import LLVM_FLAVORS
from rocke.tests.engines import lower_with_both_engines
from rocke.tests.ir_validity import assert_target_accepts, target_mnemonics

# (signed op_id, operand bits, the generic target whose row lists it)
_INSTRUCTIONS = (
    ("wmma_i32_16x16x16_iu8", 8, "gfx11-generic"),
    ("wmma_i32_16x16x16_iu4", 4, "gfx11-generic"),
    ("wmma_gfx12_i32_16x16x16_iu8", 8, "gfx12-generic"),
    ("wmma_gfx12_i32_16x16x16_iu4", 4, "gfx12-generic"),
    ("wmma_gfx12_i32_16x16x32_iu4", 4, "gfx12-generic"),
)
# The (A, B) flags of the signed row and of each unsigned combination, in
# integer_mma_op_ids order: signed x signed, unsigned x signed,
# signed x unsigned, unsigned x unsigned.
_FLAGS = (("i1 1", "i1 1"), ("i1 0", "i1 1"), ("i1 1", "i1 0"), ("i1 0", "i1 0"))

_ROWS = [
    (op_id, gfx, flags)
    for signed, bits, gfx in _INSTRUCTIONS
    for op_id, flags in zip(integer_mma_op_ids(signed, bits), _FLAGS)
]


def _kernel(gfx: str, op_id: str):
    """One MMA on zero operands of the call's register types, every result
    element stored, so that optimization keeps the instruction."""
    op = ArchTarget.from_gfx(gfx).mma.by_op_id(op_id)
    call = MATRIX_CALLS[op_id]
    b = IRBuilder(f"signedness_{op_id}")
    b.kernel.attrs["max_workgroup_size"] = op.wave_size

    def zero(form):
        register = form.register_type_at(LLVM_FLAVORS[0])
        if isinstance(register, VectorType):
            return b.zero_vec(register.elem, register.count)
        return b.const_i32(0)

    c_elem = dtype_to_ir_type(op.c_dtype)
    out = b.param("D", PtrType(c_elem, "global"))
    d = b.mma(op, zero(call.a), zero(call.b), b.zero_vec(c_elem, op.c_frag_len))
    base = b.mul(b.thread_id_x(), b.const_i32(op.c_frag_len))
    for slot in range(op.c_frag_len):
        b.global_store(out, b.add(base, b.const_i32(slot)), b.vec_extract(d, slot))
    b.ret()
    return b.kernel


def _flags(text: str, intrinsic: str) -> tuple[str, str]:
    """The flag arguments before A and B of the one call to ``intrinsic``."""
    (line,) = [
        line
        for line in text.splitlines()
        if f"@{intrinsic}(" in line and " call " in line
    ]
    arguments = line.split(f"@{intrinsic}(", 1)[1].split(", ")
    return arguments[0], arguments[2]


def test_every_integer_row_is_listed_with_its_operand_dtypes():
    """The four rows of each instruction differ only in the operand dtypes."""
    for op_id, gfx, (flag_a, flag_b) in _ROWS:
        op = ArchTarget.from_gfx(gfx).mma.by_op_id(op_id)
        assert op is not None, op_id
        assert op.a_dtype[0] == ("i" if flag_a == "i1 1" else "u"), op_id
        assert op.b_dtype[0] == ("i" if flag_b == "i1 1" else "u"), op_id


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("op_id, gfx, flags", _ROWS)
def test_both_engines_pass_the_flags_the_row_gives(op_id, gfx, flags, flavor):
    text = lower_with_both_engines(_kernel(gfx, op_id), gfx, flavor)
    assert _flags(text, MATRIX_CALLS[op_id].intrinsic) == flags


def test_every_integer_call_takes_its_flags_from_the_row():
    """Every table entry of an integer WMMA passes a Signedness argument, not an
    immediate, before A and before B."""
    integer = [op_id for op_id, _, _ in _ROWS]
    assert sorted(op_id for op_id in MATRIX_CALLS if "_iu" in op_id) == sorted(integer)
    for op_id in integer:
        args = MATRIX_CALLS[op_id].args
        a, b = args.index(MatrixOperand.A), args.index(MatrixOperand.B)
        assert args[a - 1].operand is MatrixOperand.A, op_id
        assert args[b - 1].operand is MatrixOperand.B, op_id


@pytest.mark.parametrize("target", ("gfx1100", "gfx11-generic"))
@pytest.mark.parametrize(
    "op_id, mnemonic",
    [
        (row, f"v_wmma_i32_16x16x16_iu{bits}")
        for signed, bits, gfx in _INSTRUCTIONS
        if gfx == "gfx11-generic"
        for row in integer_mma_op_ids(signed, bits)
    ],
)
def test_the_gfx11_rows_compile_link_and_select_the_instruction(
    op_id, mnemonic, target
):
    """Lowered for gfx11-generic at the host's LLVM flavor and compiled for
    gfx1100 and gfx11-generic; compile-verified, not run."""
    text = lower_with_both_engines(_kernel("gfx11-generic", op_id), "gfx11-generic")
    assert_target_accepts(text, target)
    mnemonics = target_mnemonics(text, target)
    assert [m for m in mnemonics if m.startswith("v_wmma")] == [mnemonic]


_GFX11_ROWS = [(op_id, flags) for op_id, gfx, flags in _ROWS if gfx == "gfx11-generic"]
# The HIP builtin's bool for each LLVM flag, and the flag as hipcc prints it.
_HIP_BOOL = {"i1 1": "true", "i1 0": "false"}
_HIPCC_FLAG = {"i1 1": "i1 true", "i1 0": "i1 false"}


def _hip_flags(source: str, op_id: str) -> tuple[str, str]:
    """The bools before A and B of the one gfx11 integer WMMA builtin call."""
    width = "iu8" if "_iu8" in op_id else "iu4"
    builtin = f"__builtin_amdgcn_wmma_i32_16x16x16_{width}_w32("
    (line,) = [line for line in source.splitlines() if builtin in line]
    arguments = line.split(builtin, 1)[1].split(", ")
    return arguments[0], arguments[2]


@pytest.mark.parametrize("op_id, flags", _GFX11_ROWS)
def test_the_hip_lowering_passes_the_bools_the_row_gives(op_id, flags):
    """Every gfx11 integer row, signed or not, lowers to its builtin with one
    bool per operand taken from the row."""
    source = lower_kernel_to_hip(_kernel("gfx11-generic", op_id), arch="gfx11-generic")
    assert _hip_flags(source, op_id) == tuple(_HIP_BOOL[flag] for flag in flags)


@pytest.mark.parametrize("op_id, flags", _GFX11_ROWS)
def test_hipcc_turns_the_hip_bools_into_the_rows_flags(op_id, flags):
    """hipcc compiles the HIP lowering for gfx1100 to the intrinsic call with
    the row's flags (compile-verified, not run)."""
    if not shutil.which("hipcc"):
        pytest.skip("hipcc not in PATH")
    from rocke.helpers.compile import emit_device_llvm_ir_via_hipcc

    llvm = emit_device_llvm_ir_via_hipcc(
        _kernel("gfx11-generic", op_id), arch="gfx1100"
    )
    intrinsic = MATRIX_CALLS[op_id].intrinsic
    assert _flags(llvm, intrinsic) == tuple(_HIPCC_FLAG[flag] for flag in flags)


@pytest.mark.parametrize(
    "op_id, declaration",
    [
        (
            "wmma_i32_16x16x16_iu8",
            (
                "declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32("
                "i1, <4 x i32>, i1, <4 x i32>, <8 x i32>, i1)"
            ),
        ),
        (
            "wmma_i32_16x16x16_iu4",
            (
                "declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32("
                "i1, <2 x i32>, i1, <2 x i32>, <8 x i32>, i1)"
            ),
        ),
    ],
)
def test_the_gfx11_signed_rows_keep_their_declaration(op_id, declaration):
    """Moving the gfx11 integer WMMAs into the table left their emitted
    declaration (no immarg on the flags) unchanged."""
    text = lower_with_both_engines(_kernel("gfx1151", op_id), "gfx1151")
    assert declaration in text.splitlines()


# A flag written as a literal: in an LLVM call (``i1 1, <``), a Python table
# entry (``Imm(I1, 1)``), a C++ table entry (``ARG_IMM, "i1", 1``) or a HIP
# builtin call (a format string starting ``true, {``).
_FLAG_LITERALS = re.compile(r'i1 1, <|Imm\(I1, 1\)|ARG_IMM, "i1", 1\b|"true, \{')


def test_neither_engine_writes_a_signedness_flag_literal():
    root = platform_root()
    sources = [
        *sorted((root / "python" / "rocke" / "core").rglob("*.py")),
        *sorted((root / "cpp" / "core").rglob("*.cpp")),
        *sorted((root / "cpp" / "include" / "rocke").rglob("*.h")),
    ]
    if not sources:
        pytest.skip(f"source-tree check: no engine sources under {root}")
    found = [
        f"{path.relative_to(root)}:{number}: {line.strip()}"
        for path in sources
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if _FLAG_LITERALS.search(line)
    ]
    assert found == []
