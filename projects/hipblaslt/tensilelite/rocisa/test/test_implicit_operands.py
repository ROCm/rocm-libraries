################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""
Per-opcode assertion tests for implicit-operand metadata exposed by rocisa.

Background (bead rocm-libraries-dzl):
    Most SALU compare/arithmetic opcodes implicitly read or write the SCC
    single-bit status register, and DirectToLds (DTL) buffer loads
    implicitly read m0. Neither operand is visible from dst/srcs, so the
    instruction classes carry boolean metadata (`reads_scc`, `writes_scc`,
    `is_dtl`) and the module exposes singleton RegisterContainer factories
    (`scc_resource()`, `m0_resource()`) that downstream validators attach
    to producer/consumer dataflow edges.

This test file is the per-opcode source of truth: each entry in the table
constructs one of the 43 SCC-touching SALU classes and asserts the
expected (reads_scc, writes_scc) pair. If a future rocisa change toggles
the wrong flag on the wrong opcode, the corresponding row fails.

To add coverage for a new SCC-touching opcode:
    1. Set `reads_scc` / `writes_scc` in the new class's constructor in
       the appropriate header (cmp.hpp / common.hpp / branch.hpp / etc.).
    2. Add a row to `SCC_OPCODE_TABLE` below: a tuple of (class_name,
       factory_lambda, expected_reads, expected_writes). The factory
       receives no arguments and returns a constructed instance.
    3. Run pytest against this file. The test will fail if the flags do
       not match the table.
"""

import pytest

from rocisa.container import sgpr, vgpr, MUBUFModifiers
from rocisa.instruction import (
    # SCmp* (cmp.hpp)
    SCmpEQI32,
    SCmpEQU32,
    SCmpEQU64,
    SCmpGeI32,
    SCmpGeU32,
    SCmpGtI32,
    SCmpGtU32,
    SCmpLeI32,
    SCmpLeU32,
    SCmpLgU32,
    SCmpLtI32,
    SCmpLtU32,
    # SBitcmp1* (cmp.hpp)
    SBitcmp1B32,
    # SCmpK* (cmp.hpp)
    SCmpKEQU32,
    SCmpKGeU32,
    SCmpKGtU32,
    SCmpKLGU32,
    # Common write-only ALUs (common.hpp)
    SAbsI32,
    SAddI32,
    SAddU32,
    SSubI32,
    SSubU32,
    SAndB32,
    SAndB64,
    SAndN2B32,
    SOrB32,
    SXorB32,
    SLShiftLeftB32,
    SLShiftRightB32,
    SLShiftLeftB64,
    SLShiftRightB64,
    SAShiftRightI32,
    SLShiftLeft2AddU32,
    # Carry-chain RW (common.hpp)
    SAddCU32,
    SSubBU32,
    # SaveExec writers (common.hpp)
    SAndSaveExecB32,
    SAndSaveExecB64,
    SOrSaveExecB32,
    SOrSaveExecB64,
    # SCC readers (common.hpp)
    SCSelectB32,
    SCMovB32,
    # Branch readers (branch.hpp)
    SCBranchSCC0,
    SCBranchSCC1,
    # Singleton factories (instruction.cpp)
    scc_resource,
    m0_resource,
    # MUBUF DTL reader
    BufferLoadB32,
    BufferLoadB64,
    # Counter-examples (no implicit SCC)
    SMovB32,
    SMovB64,
)


# ---------------------------------------------------------------------------
# SCC opcode coverage table: 43 entries.
#
# Each row is (label, factory, expected_reads_scc, expected_writes_scc).
# `label` is informative and used by pytest's id parametrization. `factory`
# constructs one instance with whatever arguments the class needs.
# ---------------------------------------------------------------------------

# 12 SCmp* (writes_scc only)
_SCMP_ROWS = [
    ("SCmpEQI32", lambda: SCmpEQI32(src0=sgpr(0), src1=sgpr(1)), False, True),
    ("SCmpEQU32", lambda: SCmpEQU32(src0=sgpr(0), src1=sgpr(1)), False, True),
    ("SCmpEQU64", lambda: SCmpEQU64(src0=sgpr(0, 2), src1=sgpr(2, 2)), False, True),
    ("SCmpGeI32", lambda: SCmpGeI32(src0=sgpr(0), src1=sgpr(1)), False, True),
    ("SCmpGeU32", lambda: SCmpGeU32(src0=sgpr(0), src1=sgpr(1)), False, True),
    ("SCmpGtI32", lambda: SCmpGtI32(src0=sgpr(0), src1=sgpr(1)), False, True),
    ("SCmpGtU32", lambda: SCmpGtU32(src0=sgpr(0), src1=sgpr(1)), False, True),
    ("SCmpLeI32", lambda: SCmpLeI32(src0=sgpr(0), src1=sgpr(1)), False, True),
    ("SCmpLeU32", lambda: SCmpLeU32(src0=sgpr(0), src1=sgpr(1)), False, True),
    ("SCmpLgU32", lambda: SCmpLgU32(src0=sgpr(0), src1=sgpr(1)), False, True),
    ("SCmpLtI32", lambda: SCmpLtI32(src0=sgpr(0), src1=sgpr(1)), False, True),
    ("SCmpLtU32", lambda: SCmpLtU32(src0=sgpr(0), src1=sgpr(1)), False, True),
]

# 4 SCmpK* (writes_scc only)
_SCMPK_ROWS = [
    ("SCmpKEQU32", lambda: SCmpKEQU32(src=sgpr(0), simm16=42), False, True),
    ("SCmpKGeU32", lambda: SCmpKGeU32(src=sgpr(0), simm16=42), False, True),
    ("SCmpKGtU32", lambda: SCmpKGtU32(src=sgpr(0), simm16=42), False, True),
    ("SCmpKLGU32", lambda: SCmpKLGU32(src=sgpr(0), simm16=42), False, True),
]

# 1 SBitcmp1B32 (writes_scc only)
_SBITCMP_ROWS = [
    ("SBitcmp1B32", lambda: SBitcmp1B32(src0=sgpr(0), src1=sgpr(1)), False, True),
]

# 16 dst+srcs writes-only ALUs (writes_scc only)
_SALU_WRITES_ROWS = [
    ("SAbsI32", lambda: SAbsI32(dst=sgpr(0), src=sgpr(1)), False, True),
    ("SAddI32", lambda: SAddI32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)), False, True),
    ("SAddU32", lambda: SAddU32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)), False, True),
    ("SSubI32", lambda: SSubI32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)), False, True),
    ("SSubU32", lambda: SSubU32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)), False, True),
    ("SAndB32", lambda: SAndB32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)), False, True),
    ("SAndB64", lambda: SAndB64(dst=sgpr(0, 2), src0=sgpr(2, 2), src1=sgpr(4, 2)), False, True),
    ("SAndN2B32", lambda: SAndN2B32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)), False, True),
    ("SOrB32", lambda: SOrB32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)), False, True),
    ("SXorB32", lambda: SXorB32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)), False, True),
    ("SLShiftLeftB32", lambda: SLShiftLeftB32(dst=sgpr(0), shiftHex=1, src=sgpr(1)), False, True),
    ("SLShiftRightB32", lambda: SLShiftRightB32(dst=sgpr(0), shiftHex=1, src=sgpr(1)), False, True),
    ("SLShiftLeftB64", lambda: SLShiftLeftB64(dst=sgpr(0, 2), shiftHex=1, src=sgpr(2, 2)), False, True),
    ("SLShiftRightB64", lambda: SLShiftRightB64(dst=sgpr(0, 2), shiftHex=1, src=sgpr(2, 2)), False, True),
    ("SAShiftRightI32", lambda: SAShiftRightI32(dst=sgpr(0), shiftHex=1, src=sgpr(1)), False, True),
    ("SLShiftLeft2AddU32", lambda: SLShiftLeft2AddU32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)), False, True),
]

# 2 carry-chain RW (reads_scc + writes_scc)
_CARRY_ROWS = [
    ("SAddCU32", lambda: SAddCU32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)), True, True),
    ("SSubBU32", lambda: SSubBU32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)), True, True),
]

# 4 SaveExec writers (writes_scc only)
_SAVEEXEC_ROWS = [
    ("SAndSaveExecB32", lambda: SAndSaveExecB32(dst=sgpr(0), src=sgpr(1)), False, True),
    ("SAndSaveExecB64", lambda: SAndSaveExecB64(dst=sgpr(0, 2), src=sgpr(2, 2)), False, True),
    ("SOrSaveExecB32", lambda: SOrSaveExecB32(dst=sgpr(0), src=sgpr(1)), False, True),
    ("SOrSaveExecB64", lambda: SOrSaveExecB64(dst=sgpr(0, 2), src=sgpr(2, 2)), False, True),
]

# 2 SCC readers (reads_scc only)
_SCC_READER_ROWS = [
    ("SCSelectB32", lambda: SCSelectB32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)), True, False),
    ("SCMovB32", lambda: SCMovB32(dst=sgpr(0), src=sgpr(1)), True, False),
]

# 2 SCondBranchSCC* (reads_scc only)
_SCC_BRANCH_ROWS = [
    ("SCBranchSCC0", lambda: SCBranchSCC0(labelName="lbl_scc0"), True, False),
    ("SCBranchSCC1", lambda: SCBranchSCC1(labelName="lbl_scc1"), True, False),
]

SCC_OPCODE_TABLE = (
    _SCMP_ROWS
    + _SCMPK_ROWS
    + _SBITCMP_ROWS
    + _SALU_WRITES_ROWS
    + _CARRY_ROWS
    + _SAVEEXEC_ROWS
    + _SCC_READER_ROWS
    + _SCC_BRANCH_ROWS
)


# Coverage sanity check: the original dzl bead enumerates 43 SCC-touching
# classes. If this number drifts, either the bead's accounting changed or
# someone forgot to add/remove a row. Either is worth a deliberate update.
def test_scc_opcode_table_covers_43_classes():
    assert len(SCC_OPCODE_TABLE) == 43, (
        f"Expected 43 SCC-touching classes per rocm-libraries-dzl, "
        f"got {len(SCC_OPCODE_TABLE)}. Update the table and this assertion "
        f"together when adding/removing SCC opcodes."
    )


@pytest.mark.parametrize(
    "label,factory,expected_reads,expected_writes",
    SCC_OPCODE_TABLE,
    ids=[row[0] for row in SCC_OPCODE_TABLE],
)
def test_scc_flag_metadata(label, factory, expected_reads, expected_writes):
    """Each SCC-touching class reports the correct (reads_scc, writes_scc)."""
    inst = factory()
    assert inst.reads_scc is expected_reads, (
        f"{label}.reads_scc: expected {expected_reads}, got {inst.reads_scc}"
    )
    assert inst.writes_scc is expected_writes, (
        f"{label}.writes_scc: expected {expected_writes}, got {inst.writes_scc}"
    )


# ---------------------------------------------------------------------------
# Counter-examples: opcodes that look SALU-shaped but do NOT touch SCC.
# Guards against accidental "set the flag on the base class" regressions.
# ---------------------------------------------------------------------------

_NON_SCC_ROWS = [
    ("SMovB32", lambda: SMovB32(dst=sgpr(0), src=sgpr(1))),
    ("SMovB64", lambda: SMovB64(dst=sgpr(0, 2), src=sgpr(2, 2))),
]


@pytest.mark.parametrize(
    "label,factory",
    _NON_SCC_ROWS,
    ids=[row[0] for row in _NON_SCC_ROWS],
)
def test_non_scc_opcode_flags_default_false(label, factory):
    """SALU opcodes that don't touch SCC must report both flags as False."""
    inst = factory()
    assert inst.reads_scc is False, f"{label}.reads_scc unexpectedly True"
    assert inst.writes_scc is False, f"{label}.writes_scc unexpectedly True"


# ---------------------------------------------------------------------------
# DTL (DirectToLds) is_dtl flag on MUBUFReadInstruction.
# ---------------------------------------------------------------------------

def test_buffer_load_is_dtl_when_lds_modifier_set():
    """is_dtl is True when the MUBUF modifier carries lds=True."""
    mod = MUBUFModifiers(offen=True, lds=True)
    inst = BufferLoadB32(dst=None, vaddr=vgpr(0), saddr=sgpr(0, 4),
                         soffset=0, mubuf=mod)
    assert inst.is_dtl is True


def test_buffer_load_is_dtl_false_when_lds_modifier_clear():
    """is_dtl is False when lds=False on the modifier."""
    mod = MUBUFModifiers(offen=True, lds=False)
    inst = BufferLoadB32(dst=vgpr(1), vaddr=vgpr(0), saddr=sgpr(0, 4),
                         soffset=0, mubuf=mod)
    assert inst.is_dtl is False


def test_buffer_load_is_dtl_false_when_no_modifier():
    """is_dtl is False when no MUBUF modifier is supplied at all."""
    inst = BufferLoadB32(dst=vgpr(1), vaddr=vgpr(0), saddr=sgpr(0, 4),
                         soffset=0)
    assert inst.is_dtl is False


def test_buffer_load_b64_is_dtl_picks_up_lds_flag():
    """Spot-check is_dtl is set on a different MUBUFReadInstruction subclass."""
    mod = MUBUFModifiers(offen=True, lds=True)
    inst = BufferLoadB64(dst=None, vaddr=vgpr(0), saddr=sgpr(0, 4),
                         soffset=0, mubuf=mod)
    assert inst.is_dtl is True


# ---------------------------------------------------------------------------
# Resource singletons.
#
# The validator builds dataflow edges by hashing/equality on
# RegisterContainer instances, so the producer-write side and the
# consumer-read side MUST hand back the same Python object on every call.
# These tests enforce that contract.
# ---------------------------------------------------------------------------

def test_scc_resource_returns_singleton_instance():
    """scc_resource() returns the same RegisterContainer across calls."""
    a = scc_resource()
    b = scc_resource()
    assert a is b, "scc_resource() must return the same Python object on every call"


def test_m0_resource_returns_singleton_instance():
    """m0_resource() returns the same RegisterContainer across calls."""
    a = m0_resource()
    b = m0_resource()
    assert a is b, "m0_resource() must return the same Python object on every call"


def test_scc_and_m0_resources_are_distinct():
    """SCC and m0 are different hardware resources; their singletons must differ."""
    assert scc_resource() is not m0_resource()


def test_scc_resource_renders_as_scc():
    """The SCC singleton stringifies as 'scc' (its register name)."""
    assert str(scc_resource()) == "scc0"


def test_m0_resource_renders_as_m0():
    """The m0 singleton stringifies as 'm0'."""
    assert str(m0_resource()) == "m0"


if __name__ == "__main__":
    test_scc_opcode_table_covers_43_classes()
    for row in SCC_OPCODE_TABLE:
        test_scc_flag_metadata(*row)
    for row in _NON_SCC_ROWS:
        test_non_scc_opcode_flags_default_false(*row)
    test_buffer_load_is_dtl_when_lds_modifier_set()
    test_buffer_load_is_dtl_false_when_lds_modifier_clear()
    test_buffer_load_is_dtl_false_when_no_modifier()
    test_buffer_load_b64_is_dtl_picks_up_lds_flag()
    test_scc_resource_returns_singleton_instance()
    test_m0_resource_returns_singleton_instance()
    test_scc_and_m0_resources_are_distinct()
    test_scc_resource_renders_as_scc()
    test_m0_resource_renders_as_m0()
    print("All tests passed.")
