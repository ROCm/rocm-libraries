# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Standalone tests for ``rocisa_stinkytofu_adaptor.container``.

Run from any working directory:

    python3 projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_container.py

Or with pytest:

    pytest projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_container.py

Treat any failure here as a regression that will silently corrupt
generated asm: KernelWriter compares ``RegisterContainer`` byte-for-byte
against ``toString`` output on every emitted line.
"""

from __future__ import annotations

import copy
import os
import pickle
import sys
import unittest

# ---------------------------------------------------------------------------
# Self-contained sys.path bootstrap (see test_register.py for rationale).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rocisa_stinkytofu_adaptor import rocIsa  # noqa: E402
from rocisa_stinkytofu_adaptor import getGlcBitName, getSlcBitName  # noqa: E402
from rocisa_stinkytofu_adaptor.container import (  # noqa: E402
    Container,
    ContinuousRegister,
    DSModifiers,
    DPPModifiers,
    EXEC,
    EXECLO,
    EXECHI,
    FLATModifiers,
    GLOBALModifiers,
    HWRegContainer,
    MemTokenData,
    MUBUFModifiers,
    SDWAModifiers,
    SMEMModifiers,
    True16Modifiers,
    VOP3PModifiers,
    Holder,
    HolderContainer,
    RegisterContainer,
    RegName,
    VCC,
    accvgpr,
    mgpr,
    replaceHolder,
    sgpr,
    vgpr,
)
from rocisa_stinkytofu_adaptor.enum import CacheScope, HighBitSel, NonVolatile, SelectBit, TemporalHint  # noqa: E402


# ===========================================================================
# RegName
# ===========================================================================


class TestRegNameConstruction(unittest.TestCase):
    def test_default_ctor_empty(self):
        rn = RegName()
        self.assertEqual(rn.name, "")
        self.assertEqual(rn.offsets, [])

    def test_name_only_ctor(self):
        rn = RegName("ValuA")
        self.assertEqual(rn.name, "ValuA")
        self.assertEqual(rn.offsets, [])

    def test_name_and_offsets_ctor(self):
        rn = RegName("ValuA", [2, 4])
        self.assertEqual(rn.name, "ValuA")
        self.assertEqual(rn.offsets, [2, 4])

    def test_offsets_are_copied(self):
        # Ctor copies the input list so post-construction mutation of
        # the source does not bleed into the RegName.
        src = [1, 2, 3]
        rn = RegName("X", src)
        src.append(99)
        self.assertEqual(rn.offsets, [1, 2, 3])


class TestRegNameOffsets(unittest.TestCase):
    def test_getOffsets_returns_list(self):
        rn = RegName("V", [1, 2, 3])
        self.assertEqual(rn.getOffsets(), [1, 2, 3])

    def test_setOffset_in_range(self):
        rn = RegName("V", [1, 2, 3])
        rn.setOffset(1, 99)
        self.assertEqual(rn.offsets, [1, 99, 3])

    def test_setOffset_out_of_range_raises(self):
        rn = RegName("V", [1, 2])
        with self.assertRaises(IndexError):
            rn.setOffset(2, 0)

    def test_addOffset_appends(self):
        rn = RegName("V", [1])
        rn.addOffset(5)
        rn.addOffset(7)
        self.assertEqual(rn.offsets, [1, 5, 7])

    def test_getTotalOffsets_sum(self):
        self.assertEqual(RegName("V", []).getTotalOffsets(), 0)
        self.assertEqual(RegName("V", [3, 4, 5]).getTotalOffsets(), 12)
        self.assertEqual(RegName("V", [-2, 7]).getTotalOffsets(), 5)


class TestRegNameTotalIdx(unittest.TestCase):
    """``getTotalIdx`` resolves ``name`` against ``rocIsa.getVgprIdx``."""

    def setUp(self):
        # Wipe the shared symbol table to keep tests order-independent.
        rocIsa.getInstance()._vgpr_idx.clear()

    def tearDown(self):
        rocIsa.getInstance()._vgpr_idx.clear()

    def test_unknown_name_resolves_to_zero(self):
        # Missing symbol defaults to 0, so total = sum(offsets).
        rn = RegName("MissingSym", [3])
        self.assertEqual(rn.getTotalIdx(), 3)

    def test_known_name_plus_offsets(self):
        rocIsa.getInstance().setVgprIdx("ValuA", 100)
        rn = RegName("ValuA", [4, 2])
        self.assertEqual(rn.getTotalIdx(), 106)  # 100 + 4 + 2
        self.assertEqual(rn.nameIdx, 100)

    def test_setNameIdx_re_resolves_after_remap(self):
        rocIsa.getInstance().setVgprIdx("ValuA", 100)
        rn = RegName("ValuA")
        rn.setNameIdx()
        self.assertEqual(rn.nameIdx, 100)
        # KernelWriter rebinds the symbol; getTotalIdx must reflect it.
        rocIsa.getInstance().setVgprIdx("ValuA", 200)
        self.assertEqual(rn.getTotalIdx(), 200)


class TestRegNameStringify(unittest.TestCase):
    """``__str__`` -> ``"name+off1+off2+..."``."""

    def test_name_only(self):
        self.assertEqual(str(RegName("ValuA")), "ValuA")

    def test_name_with_one_offset(self):
        self.assertEqual(str(RegName("ValuA", [3])), "ValuA+3")

    def test_name_with_many_offsets(self):
        self.assertEqual(str(RegName("ValuA", [1, 2, 3])), "ValuA+1+2+3")

    def test_negative_offset(self):
        self.assertEqual(str(RegName("X", [-5])), "X+-5")


class TestRegNameEquality(unittest.TestCase):
    def test_eq_same_name_and_offsets(self):
        self.assertEqual(RegName("A", [1, 2]), RegName("A", [1, 2]))

    def test_ne_different_name(self):
        self.assertNotEqual(RegName("A", [1]), RegName("B", [1]))

    def test_ne_different_offsets(self):
        self.assertNotEqual(RegName("A", [1, 2]), RegName("A", [1, 3]))

    def test_eq_with_non_regname_returns_NotImplemented(self):
        # NotImplemented bubbles up to Python's default ``False`` for !=.
        self.assertFalse(RegName("A") == "A")
        self.assertTrue(RegName("A") != "A")

    def test_hash_equal_for_equal_regnames(self):
        self.assertEqual(hash(RegName("A", [1, 2])), hash(RegName("A", [1, 2])))

    def test_usable_as_dict_key(self):
        d = {RegName("A", [1]): "first"}
        self.assertEqual(d[RegName("A", [1])], "first")


class TestRegNameCopy(unittest.TestCase):
    def test_deepcopy_independence(self):
        rn = RegName("A", [1, 2])
        clone = copy.deepcopy(rn)
        clone.addOffset(99)
        self.assertEqual(rn.offsets, [1, 2])
        self.assertEqual(clone.offsets, [1, 2, 99])

    def test_copy_independence(self):
        rn = RegName("A", [1, 2])
        clone = copy.copy(rn)
        clone.setOffset(0, 99)
        self.assertEqual(rn.offsets, [1, 2])
        self.assertEqual(clone.offsets, [99, 2])


class TestRegNamePickle(unittest.TestCase):
    def test_pickle_round_trip(self):
        rn = RegName("ValuA", [3, 4, 5])
        rt = pickle.loads(pickle.dumps(rn))
        self.assertEqual(rt, rn)
        self.assertEqual(str(rt), "ValuA+3+4+5")


# ===========================================================================
# RegisterContainer -- construction & basic attributes
# ===========================================================================


class TestRegisterContainerConstruction(unittest.TestCase):
    def test_unnamed_default(self):
        rc = RegisterContainer("v", None, 0, 1)
        self.assertEqual(rc.regType, "v")
        self.assertIsNone(rc.regName)
        self.assertEqual(rc.regIdx, 0)
        self.assertEqual(rc.regNum, 1)
        self.assertEqual(rc.msb, 0)
        self.assertFalse(rc.isInlineAsm)
        self.assertFalse(rc.isMinus)
        self.assertFalse(rc.isAbs)
        self.assertFalse(rc.isMacro)
        self.assertFalse(rc.isOff)

    def test_named(self):
        rn = RegName("ValuA")
        rc = RegisterContainer("v", rn, 0, 2)
        self.assertIs(rc.regName, rn)
        self.assertEqual(rc.regNum, 2)

    def test_regNum_ceil(self):
        # regNum is rounded up: 1.5 -> 2, 0.1 -> 1.
        self.assertEqual(RegisterContainer("v", None, 0, 1.5).regNum, 2)
        self.assertEqual(RegisterContainer("v", None, 0, 0.1).regNum, 1)
        self.assertEqual(RegisterContainer("v", None, 0, 3.0).regNum, 3)

    def test_seven_arg_kwargs(self):
        # Extended ctor: isAbs/isMacro/isOff as keyword-only overrides.
        rc = RegisterContainer("v", None, 0, 1, isAbs=True, isMacro=True, isOff=True)
        self.assertTrue(rc.isAbs)
        self.assertTrue(rc.isMacro)
        self.assertTrue(rc.isOff)


class TestRegisterContainerSetters(unittest.TestCase):
    def test_setInlineAsm(self):
        rc = RegisterContainer("v", None, 0, 1)
        rc.setInlineAsm(True)
        self.assertTrue(rc.isInlineAsm)

    def test_setMinus_and_setAbs(self):
        rc = RegisterContainer("v", None, 0, 1)
        rc.setMinus(True)
        rc.setAbs(True)
        self.assertTrue(rc.isMinus)
        self.assertTrue(rc.isAbs)

    def test_getMinus_returns_copy(self):
        rc = RegisterContainer("v", None, 5, 1)
        minus_rc = rc.getMinus()
        self.assertTrue(minus_rc.isMinus)
        self.assertFalse(rc.isMinus)  # original unchanged
        self.assertIsNot(minus_rc, rc)


# ===========================================================================
# replaceRegName
# ===========================================================================


class TestReplaceRegName(unittest.TestCase):
    def test_no_regName_is_noop(self):
        rc = RegisterContainer("v", None, 5, 1)
        rc.replaceRegName("X", 9)
        self.assertIsNone(rc.regName)
        self.assertEqual(rc.regIdx, 5)

    def test_exact_match_int_collapses_to_idx(self):
        # Exact int match collapses symbolic name into
        # ``regIdx = dst + sum(offsets)`` and clears regName.
        rc = RegisterContainer("v", RegName("ValuA", [3]), 0, 1)
        rc.replaceRegName("ValuA", 100)
        self.assertIsNone(rc.regName)
        self.assertEqual(rc.regIdx, 103)

    def test_exact_match_int_no_offsets(self):
        rc = RegisterContainer("v", RegName("ValuA"), 0, 1)
        rc.replaceRegName("ValuA", 7)
        self.assertIsNone(rc.regName)
        self.assertEqual(rc.regIdx, 7)

    def test_partial_match_int_substring_replaces(self):
        # ``ValuA+Sub`` with src="Sub", dst=42 -> name becomes "ValuA+42".
        rc = RegisterContainer("v", RegName("ValuA+Sub"), 0, 1)
        rc.replaceRegName("Sub", 42)
        self.assertIsNotNone(rc.regName)
        self.assertEqual(rc.regName.name, "ValuA+42")

    def test_str_dst_substring_replace(self):
        rc = RegisterContainer("v", RegName("Foo+Bar"), 0, 1)
        rc.replaceRegName("Bar", "Baz")
        self.assertEqual(rc.regName.name, "Foo+Baz")

    def test_str_dst_no_match_noop(self):
        rc = RegisterContainer("v", RegName("Foo"), 0, 1)
        rc.replaceRegName("Bar", "Baz")
        self.assertEqual(rc.regName.name, "Foo")

    def test_bool_rejected(self):
        # bool is an int subclass; rejected so callers don't drift into
        # the int overload by accident.
        rc = RegisterContainer("v", RegName("X"), 0, 1)
        with self.assertRaises(TypeError):
            rc.replaceRegName("X", True)


# ===========================================================================
# Composite name accessors / splitRegContainer / setMsb
# ===========================================================================


class TestRegNameAccessors(unittest.TestCase):
    def test_getRegNameWithType(self):
        rc = RegisterContainer("v", RegName("ValuA", [3]), 0, 1)
        # Bare ``name``, no offsets baked in.
        self.assertEqual(rc.getRegNameWithType(), "vgprValuA")

    def test_getCompleteRegNameWithType(self):
        rc = RegisterContainer("v", RegName("ValuA", [3]), 0, 1)
        # Includes the offset suffix.
        self.assertEqual(rc.getCompleteRegNameWithType(), "vgprValuA+3")

    def test_getCompleteRegName(self):
        rc = RegisterContainer("v", RegName("ValuA", [3]), 0, 1)
        self.assertEqual(rc.getCompleteRegName(), "ValuA+3")


class TestSplitRegContainer(unittest.TestCase):
    def test_unnamed_even_split(self):
        rc = RegisterContainer("v", None, 4, 2)
        r1, r2 = rc.splitRegContainer()
        self.assertEqual(r1.regIdx, 4)
        self.assertEqual(r1.regNum, 1)
        self.assertEqual(r2.regIdx, 5)
        self.assertEqual(r2.regNum, 1)

    def test_unnamed_odd_split(self):
        # ``new_reg_num = ceil(regNum/2)`` so r1 keeps the larger half
        # (3 -> 2,1).
        rc = RegisterContainer("v", None, 4, 3)
        r1, r2 = rc.splitRegContainer()
        self.assertEqual(r1.regNum, 2)
        self.assertEqual(r2.regNum, 1)
        self.assertEqual(r2.regIdx, 5)

    def test_named_appends_offset_one_to_right_half(self):
        rc = RegisterContainer("v", RegName("ValuA"), 0, 2)
        r1, r2 = rc.splitRegContainer()
        # Left half keeps the original RegName.
        self.assertEqual(r1.regName.offsets, [])
        # Right half gets ``+1`` appended to its offset chain.
        self.assertEqual(r2.regName.offsets, [1])

    def test_split_results_independent(self):
        # Mutating r2.regName.offsets must NOT bleed back into the source
        # rc.regName.offsets; split deep-clones the RegName field.
        rc = RegisterContainer("v", RegName("V", [3]), 0, 2)
        _r1, r2 = rc.splitRegContainer()
        r2.regName.addOffset(99)
        self.assertEqual(rc.regName.offsets, [3])

    def test_split_preserves_unrelated_state(self):
        rc = RegisterContainer("v", RegName("V"), 0, 2)
        rc.setMinus(True)
        rc.isMacro = True
        r1, r2 = rc.splitRegContainer()
        # Decorations propagate to both halves.
        self.assertTrue(r1.isMinus and r2.isMinus)
        self.assertTrue(r1.isMacro and r2.isMacro)


class TestSetMsb(unittest.TestCase):
    def setUp(self):
        rocIsa.getInstance()._vgpr_idx.clear()

    def tearDown(self):
        rocIsa.getInstance()._vgpr_idx.clear()

    def test_named_uses_totalIdx_div_256(self):
        rocIsa.getInstance().setVgprIdx("ValuA", 512)
        rc = RegisterContainer("v", RegName("ValuA", [1]), 0, 1)
        rc.setMsb()
        self.assertEqual(rc.msb, 2)  # (512+1)//256

    def test_unnamed_uses_regIdx_div_256(self):
        rc = RegisterContainer("v", None, 600, 1)
        rc.setMsb()
        self.assertEqual(rc.msb, 2)  # 600//256

    def test_msb_zero_for_low_index(self):
        rc = RegisterContainer("v", None, 5, 1)
        rc.setMsb()
        self.assertEqual(rc.msb, 0)


# ===========================================================================
# Hash / equality
# ===========================================================================


class TestRegisterContainerEquality(unittest.TestCase):
    def test_eq_same_fields(self):
        a = RegisterContainer("v", RegName("X"), 5, 2)
        b = RegisterContainer("v", RegName("X"), 5, 2)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_eq_ignores_decorations(self):
        # Equality only checks (regType, regIdx, regNum, regName); the
        # isMinus/isAbs/isInlineAsm decorations don't participate.
        a = RegisterContainer("v", None, 5, 1)
        b = RegisterContainer("v", None, 5, 1)
        b.setMinus(True)
        b.setAbs(True)
        self.assertEqual(a, b)

    def test_ne_regType(self):
        self.assertNotEqual(
            RegisterContainer("v", None, 0, 1),
            RegisterContainer("s", None, 0, 1),
        )

    def test_ne_regIdx(self):
        self.assertNotEqual(
            RegisterContainer("v", None, 0, 1),
            RegisterContainer("v", None, 1, 1),
        )

    def test_ne_regNum(self):
        self.assertNotEqual(
            RegisterContainer("v", None, 0, 1),
            RegisterContainer("v", None, 0, 2),
        )

    def test_ne_regName(self):
        self.assertNotEqual(
            RegisterContainer("v", RegName("X"), 0, 1),
            RegisterContainer("v", RegName("Y"), 0, 1),
        )

    def test_eq_with_non_container_returns_false(self):
        # ``__eq__`` returns False for non-RC objects (does not raise).
        self.assertFalse(RegisterContainer("v", None, 0, 1) == "v0")
        self.assertFalse(RegisterContainer("v", None, 0, 1) == None)  # noqa: E711

    def test_usable_as_dict_key(self):
        d = {RegisterContainer("v", None, 0, 1): "first"}
        self.assertEqual(d[RegisterContainer("v", None, 0, 1)], "first")


# ===========================================================================
# Aliasing (sameRegBaseAddr / __and__)
# ===========================================================================


class TestAliasing(unittest.TestCase):
    def test_sameRegBaseAddr_both_named_same(self):
        a = RegisterContainer("v", RegName("X"), 0, 1)
        b = RegisterContainer("v", RegName("X", [3]), 0, 1)
        # Compares ``regName.name`` only, ignoring offsets.
        self.assertTrue(a.sameRegBaseAddr(b))

    def test_sameRegBaseAddr_both_named_diff(self):
        a = RegisterContainer("v", RegName("X"), 0, 1)
        b = RegisterContainer("v", RegName("Y"), 0, 1)
        self.assertFalse(a.sameRegBaseAddr(b))

    def test_sameRegBaseAddr_both_unnamed_same_idx(self):
        a = RegisterContainer("v", None, 5, 1)
        b = RegisterContainer("v", None, 5, 3)
        self.assertTrue(a.sameRegBaseAddr(b))

    def test_sameRegBaseAddr_mixed_returns_false(self):
        a = RegisterContainer("v", RegName("X"), 0, 1)
        b = RegisterContainer("v", None, 0, 1)
        self.assertFalse(a.sameRegBaseAddr(b))

    def test_and_self_overlap(self):
        rc = RegisterContainer("v", None, 5, 4)
        # range [0, regNum) vs itself -> True.
        self.assertTrue(rc & rc)

    def test_and_disjoint_offsets(self):
        a = RegisterContainer("v", RegName("X", [0]), 0, 2)
        b = RegisterContainer("v", RegName("X", [4]), 0, 2)
        # ranges [0,2) and [4,6) -- no overlap.
        self.assertFalse(a & b)

    def test_and_adjacent_no_overlap(self):
        a = RegisterContainer("v", RegName("X", [0]), 0, 2)
        b = RegisterContainer("v", RegName("X", [2]), 0, 2)
        # ranges [0,2) and [2,4) -- touching, not overlapping.
        self.assertFalse(a & b)

    def test_and_overlapping_offsets(self):
        a = RegisterContainer("v", RegName("X", [0]), 0, 3)
        b = RegisterContainer("v", RegName("X", [2]), 0, 2)
        # ranges [0,3) and [2,4) -- overlap.
        self.assertTrue(a & b)

    def test_and_different_bases_disjoint(self):
        a = RegisterContainer("v", RegName("X"), 0, 1)
        b = RegisterContainer("v", RegName("Y"), 0, 1)
        self.assertFalse(a & b)


# ===========================================================================
# toString (byte-for-byte parity required for KernelWriter emit)
# ===========================================================================


class TestToString(unittest.TestCase):
    """The KernelWriter pipes ``str(rc)`` directly into the emitted asm.

    Byte-for-byte parity is a hard requirement -- any drift will break a
    downstream assembler / disassembler diff.
    """

    def setUp(self):
        # toString reads rocIsa.getAsmCaps() for the HasVgprMSB path.
        # Tests that don't care about MSB stay in the uninitialised-rocIsa
        # default ("no MSB"), so we do NOT init here.
        rocIsa._instance = None  # noqa: SLF001

    def test_isOff(self):
        rc = RegisterContainer("v", None, 0, 1, isOff=True)
        self.assertEqual(str(rc), "off")

    def test_inlineAsm_single(self):
        rc = RegisterContainer("v", None, 3, 1)
        rc.setInlineAsm(True)
        self.assertEqual(str(rc), "%3")

    def test_inlineAsm_minus(self):
        rc = RegisterContainer("v", None, 3, 1)
        rc.setInlineAsm(True)
        rc.setMinus(True)
        self.assertEqual(str(rc), "-%3")

    def test_inlineAsm_abs(self):
        rc = RegisterContainer("v", None, 3, 1)
        rc.setInlineAsm(True)
        rc.setAbs(True)
        self.assertEqual(str(rc), "abs(%3)")

    def test_unnamed_single_vgpr(self):
        self.assertEqual(str(RegisterContainer("v", None, 5, 1)), "v5")

    def test_unnamed_single_sgpr(self):
        self.assertEqual(str(RegisterContainer("s", None, 7, 1)), "s7")

    def test_unnamed_range(self):
        self.assertEqual(str(RegisterContainer("v", None, 0, 4)), "v[0:3]")
        self.assertEqual(str(RegisterContainer("s", None, 8, 2)), "s[8:9]")

    def test_unnamed_single_minus(self):
        rc = RegisterContainer("v", None, 5, 1)
        rc.setMinus(True)
        self.assertEqual(str(rc), "-v5")

    def test_unnamed_single_abs(self):
        rc = RegisterContainer("v", None, 5, 1)
        rc.setAbs(True)
        self.assertEqual(str(rc), "abs(v5)")

    def test_unnamed_single_minus_and_abs(self):
        # When both flags are set the abs prefix wraps the minus.
        rc = RegisterContainer("v", None, 5, 1)
        rc.setMinus(True)
        rc.setAbs(True)
        self.assertEqual(str(rc), "abs(-v5)")

    def test_named_single(self):
        rc = RegisterContainer("v", RegName("ValuA"), 0, 1)
        self.assertEqual(str(rc), "v[vgprValuA]")

    def test_named_range(self):
        rc = RegisterContainer("v", RegName("ValuA"), 0, 4)
        self.assertEqual(str(rc), "v[vgprValuA:vgprValuA+3]")

    def test_named_with_offset(self):
        rc = RegisterContainer("v", RegName("ValuA", [3]), 0, 1)
        self.assertEqual(str(rc), "v[vgprValuA+3]")

    def test_named_macro(self):
        rc = RegisterContainer("v", RegName("ValuA"), 0, 1, isMacro=True)
        # Macro path adds a ``\`` between ``[`` and ``vgpr``.
        self.assertEqual(str(rc), "v[\\vgprValuA]")

    def test_named_minus_abs(self):
        rc = RegisterContainer("v", RegName("X"), 0, 1)
        rc.setMinus(True)
        rc.setAbs(True)
        self.assertEqual(str(rc), "abs(-v[vgprX])")


# ===========================================================================
# Copy semantics
# ===========================================================================


class TestRegisterContainerCopy(unittest.TestCase):
    def test_deepcopy_independent_regName(self):
        rc = RegisterContainer("v", RegName("X", [1, 2]), 0, 1)
        clone = copy.deepcopy(rc)
        clone.regName.addOffset(99)
        # Mutation must not leak.
        self.assertEqual(rc.regName.offsets, [1, 2])
        self.assertEqual(clone.regName.offsets, [1, 2, 99])

    def test_copy_preserves_flags(self):
        rc = RegisterContainer("v", None, 5, 2)
        rc.setMinus(True)
        rc.setAbs(True)
        rc.setInlineAsm(True)
        clone = copy.copy(rc)
        self.assertTrue(clone.isMinus)
        self.assertTrue(clone.isAbs)
        self.assertTrue(clone.isInlineAsm)

    def test_getMinus_does_not_share_regName(self):
        # getMinus is the hot path KernelWriter uses for "-v0"; the clone
        # must not alias the source's RegName.
        rc = RegisterContainer("v", RegName("X", [1]), 0, 1)
        m = rc.getMinus()
        m.regName.addOffset(7)
        self.assertEqual(rc.regName.offsets, [1])

    def test_pickle_round_trip(self):
        rc = RegisterContainer("v", RegName("Y", [1, 2]), 5, 3)
        rc.setMinus(True)
        rc.isMacro = True
        rt = pickle.loads(pickle.dumps(rc))
        self.assertEqual(rt, rc)
        self.assertTrue(rt.isMinus)
        self.assertTrue(rt.isMacro)
        # toString quirk: the macro ``\`` is emitted only on the left
        # end of a range expression, never on the right.
        self.assertEqual(str(rt), "-v[\\vgprY+1+2:vgprY+1+2+2]")


# ===========================================================================
# Stinkytofu handoff (to_stinky)
# ===========================================================================
#
# These tests can be skipped if stinkytofu is not available in the test
# environment (the wrapper itself remains a usable rocisa-shape).


try:
    import stinkytofu as _stinky  # noqa: F401

    _STINKY_OK = True
except ImportError:
    _STINKY_OK = False


@unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built in this env")
class TestToStinky(unittest.TestCase):
    """``to_stinky`` builds a fresh stinky.Register from the wrapper state.

    Symbolic name carries the ``<regType>gpr`` prefix; physical idx is
    resolved through ``rocIsa.getVgprIdx()`` for named V registers.
    """

    def setUp(self):
        rocIsa.getInstance()._vgpr_idx.clear()

    def tearDown(self):
        rocIsa.getInstance()._vgpr_idx.clear()

    def test_basic_vgpr(self):
        rc = RegisterContainer("v", None, 5, 2)
        reg = rc.to_stinky()
        self.assertTrue(reg.is_register)
        self.assertEqual(reg.index, 5)
        self.assertEqual(reg.count, 2)

    def test_named_includes_vgpr_prefix(self):
        # When regIdx is unresolved (-1), symbolic name is attached so
        # stinkytofu can do its own scheduling/analysis.
        rc = RegisterContainer("v", RegName("ValuA", [1, 2]), -1, 1)
        reg = rc.to_stinky()
        self.assertTrue(reg.has_reg_name())
        name, offsets = reg.get_reg_name()
        self.assertEqual(name, "vgprValuA")
        self.assertEqual(offsets, [1, 2])

    def test_named_resolved_preserves_symbolic(self):
        # Resolved registers still carry symbolic names for readable asm.
        rc = RegisterContainer("v", RegName("ValuA", [1, 2]), 0, 1)
        reg = rc.to_stinky()
        self.assertTrue(reg.has_reg_name())
        name, offsets = reg.get_reg_name()
        self.assertEqual(name, "vgprValuA")
        self.assertEqual(offsets, [1, 2])

    def test_named_sgpr_prefix(self):
        rc = RegisterContainer("s", RegName("KArg"), -1, 1)
        reg = rc.to_stinky()
        name, _ = reg.get_reg_name()
        self.assertEqual(name, "sgprKArg")

    def test_macro_injects_backslash_prefix(self):
        # Macro context renders as ``v[\vgprAddr+0]``; the ``\`` is
        # encoded into the symbolic name for byte-parity.
        rc = RegisterContainer("v", RegName("Addr", [0]), -1, 1, isMacro=True)
        reg = rc.to_stinky()
        name, offsets = reg.get_reg_name()
        self.assertEqual(name, "\\vgprAddr")
        self.assertEqual(offsets, [0])

    def test_inlineAsm_raises(self):
        # ``%idx`` format is inline-asm-in-cpp only; stinky emit targets
        # .s. Module layer (T6) must catch this before to_stinky.
        rc = RegisterContainer("v", None, 3, 1)
        rc.setInlineAsm(True)
        with self.assertRaises(NotImplementedError):
            rc.to_stinky()

    def test_named_idx_resolves_through_rocIsa(self):
        # Physical idx for named V registers = regName.getTotalIdx().
        rocIsa.getInstance().setVgprIdx("ValuA", 64)
        rc = RegisterContainer("v", RegName("ValuA", [2]), 0, 1)
        reg = rc.to_stinky()
        self.assertEqual(reg.index, 66)  # 64 + 2

    def test_named_idx_unresolved_falls_back_to_zero(self):
        # Symbol missing from getVgprIdx() defaults to 0+sum(offsets).
        rc = RegisterContainer("v", RegName("UnknownSym", [3]), 0, 1)
        reg = rc.to_stinky()
        self.assertEqual(reg.index, 3)

    def test_minus_propagates(self):
        rc = RegisterContainer("v", None, 5, 1)
        rc.setMinus(True)
        reg = rc.to_stinky()
        self.assertTrue(reg.is_minus)

    def test_abs_propagates(self):
        rc = RegisterContainer("v", None, 5, 1)
        rc.setAbs(True)
        reg = rc.to_stinky()
        self.assertTrue(reg.is_abs)

    def test_isOff_returns_literal_off(self):
        rc = RegisterContainer("v", None, 0, 1, isOff=True)
        reg = rc.to_stinky()
        self.assertTrue(reg.is_literal_string)
        self.assertEqual(reg.literal_string, "off")

    def test_fresh_each_call(self):
        # to_stinky is documented as "build fresh each time" -- mutations
        # between calls must be reflected.
        rc = RegisterContainer("v", None, 5, 1)
        first = rc.to_stinky()
        rc.setMinus(True)
        second = rc.to_stinky()
        self.assertFalse(first.is_minus)
        self.assertTrue(second.is_minus)


# ===========================================================================
# KernelWriter scenario coverage
# ===========================================================================
#
# Walks through the exact call patterns KernelWriter uses (grepped from
# KernelWriterAssembly / Components).


class TestKernelWriterScenarios(unittest.TestCase):
    def test_vgpr_factory_pattern(self):
        # Real factory: vgpr("ValuA", 1) -> str-form RegisterContainer.
        rc = vgpr("ValuA", 1)
        self.assertEqual(str(rc), "v[vgprValuA]")
        # Used as a dict key in RegisterPool tracking. Two vgpr() calls
        # with matching args must hash/compare equal for this to work.
        d = {rc: "tracked"}
        self.assertEqual(d[vgpr("ValuA", 1)], "tracked")

    def test_split_then_emit(self):
        rc = RegisterContainer("v", RegName("Acc"), 0, 4)
        r1, r2 = rc.splitRegContainer()
        # KernelWriter emits both halves into different MFMAs.
        self.assertEqual(str(r1), "v[vgprAcc:vgprAcc+1]")
        self.assertEqual(str(r2), "v[vgprAcc+1:vgprAcc+1+1]")

    def test_replaceRegName_to_int_collapse(self):
        # Mirrors KernelWriter's post-allocation pass that collapses
        # symbolic refs to numeric indices.
        rc = RegisterContainer("v", RegName("ValuA"), 0, 2)
        rc.replaceRegName("ValuA", 32)
        self.assertEqual(str(rc), "v[32:33]")

    def test_replaceRegName_partial_int_substitution(self):
        # Mirrors prefix-stripping pattern (e.g. "vgprValuA+x" -> "vgprValuA+5").
        rc = RegisterContainer("v", RegName("ValuA+Tmp"), 0, 1)
        rc.replaceRegName("Tmp", 5)
        self.assertEqual(str(rc), "v[vgprValuA+5]")

    def test_minus_then_str(self):
        # KernelWriter uses ``getMinus()`` for ``v_sub_x dest, -src``.
        src = RegisterContainer("v", None, 7, 1)
        neg = src.getMinus()
        self.assertEqual(str(neg), "-v7")
        self.assertEqual(str(src), "v7")

    def test_overlap_check_for_aliasing_warning(self):
        # KernelWriter uses ``a && b`` to detect aliased register operands.
        dest = RegisterContainer("v", RegName("Acc", [0]), 0, 4)
        src = RegisterContainer("v", RegName("Acc", [2]), 0, 4)
        self.assertTrue(dest & src)  # ranges [0,4) and [2,6) overlap

    def test_inline_asm_emit(self):
        # ``%0``, ``%1`` style operands for inline asm blocks.
        rc = RegisterContainer("v", None, 0, 1)
        rc.setInlineAsm(True)
        self.assertEqual(str(rc), "%0")


# ===========================================================================
# Holder
# ===========================================================================


class TestHolderConstruction(unittest.TestCase):
    def test_int_ctor(self):
        # ``Holder(int idx)`` sets ``name = None``.
        h = Holder(5)
        self.assertEqual(h.idx, 5)
        self.assertIsNone(h.name)

    def test_string_ctor_no_offsets(self):
        # generateRegName("Foo") -> RegName("Foo", []).
        h = Holder("ValuC")
        self.assertEqual(h.idx, -1)
        self.assertIsNotNone(h.name)
        self.assertEqual(h.name.name, "ValuC")
        self.assertEqual(h.name.offsets, [])

    def test_string_ctor_single_offset(self):
        # "Foo+5" -> RegName("Foo", [5]).
        h = Holder("ValuC+5")
        self.assertEqual(h.idx, -1)
        self.assertEqual(h.name.name, "ValuC")
        self.assertEqual(h.name.offsets, [5])

    def test_string_ctor_multi_offset(self):
        # "Foo+1+2+3" -> RegName("Foo", [1, 2, 3]).
        h = Holder("ValuC+1+2+3")
        self.assertEqual(h.name.offsets, [1, 2, 3])

    def test_int_kwarg(self):
        h = Holder(idx=7)
        self.assertEqual(h.idx, 7)
        self.assertIsNone(h.name)

    def test_name_kwarg(self):
        h = Holder(name="Foo+2")
        self.assertEqual(h.idx, -1)
        self.assertEqual(h.name.offsets, [2])

    def test_bool_rejected(self):
        # bool is an int subclass but would silently become idx=0/1;
        # caller almost certainly meant something else.
        with self.assertRaises(TypeError):
            Holder(True)
        with self.assertRaises(TypeError):
            Holder(idx=False)

    def test_float_rejected(self):
        with self.assertRaises(TypeError):
            Holder(3.0)

    def test_both_kwargs_rejected(self):
        with self.assertRaises(TypeError):
            Holder(idx=1, name="Foo")

    def test_no_args_rejected(self):
        with self.assertRaises(TypeError):
            Holder()

    def test_multi_pos_rejected(self):
        with self.assertRaises(TypeError):
            Holder(1, 2)


class TestHolderSemantics(unittest.TestCase):
    def test_idx_mutable(self):
        # idx is a writable field.
        h = Holder(3)
        h.idx = 10
        self.assertEqual(h.idx, 10)

    def test_name_mutable(self):
        # name is a writable field.
        h = Holder(3)
        h.name = RegName("X", [1])
        self.assertEqual(h.name.name, "X")

    def test_equality(self):
        self.assertEqual(Holder(5), Holder(5))
        self.assertEqual(Holder("Foo+1"), Holder("Foo+1"))
        self.assertNotEqual(Holder(5), Holder(6))
        self.assertNotEqual(Holder(5), Holder("Foo"))
        self.assertNotEqual(Holder("Foo"), Holder("Bar"))

    def test_hashable(self):
        # Hash parity with equality lets KernelWriter use Holders as
        # dict keys / set members (e.g. dedup loops).
        s = {Holder(5), Holder(5), Holder("Foo")}
        self.assertEqual(len(s), 2)

    def test_repr_contains_state(self):
        # Debug-printability; exact format is not pinned.
        r = repr(Holder("Foo+2"))
        self.assertIn("Holder", r)
        self.assertIn("Foo", r)

    def test_eq_vs_other_type(self):
        # Cross-type comparison must not throw.
        self.assertFalse(Holder(5) == 5)
        self.assertTrue(Holder(5) != "Foo")

    def test_shallow_copy_independent(self):
        # Mutating the copy must not bleed into the original.
        h = Holder("ValuC+1")
        h2 = copy.copy(h)
        h2.name.offsets.append(99)
        self.assertEqual(h.name.offsets, [1])
        self.assertEqual(h2.name.offsets, [1, 99])

    def test_deep_copy_independent(self):
        h = Holder("ValuC+1+2")
        h2 = copy.deepcopy(h)
        h2.idx = 99
        h2.name.offsets[0] = 7
        self.assertEqual(h.idx, -1)
        self.assertEqual(h.name.offsets, [1, 2])

    def test_pickle_roundtrip_int(self):
        h = Holder(42)
        h2 = pickle.loads(pickle.dumps(h))
        self.assertEqual(h2.idx, 42)
        self.assertIsNone(h2.name)

    def test_pickle_roundtrip_name(self):
        h = Holder("Foo+3+4")
        h2 = pickle.loads(pickle.dumps(h))
        self.assertEqual(h2.idx, -1)
        self.assertEqual(h2.name.name, "Foo")
        self.assertEqual(h2.name.offsets, [3, 4])


# ===========================================================================
# HolderContainer
# ===========================================================================
#
# Subclass of RegisterContainer; resolution happens lazily via setRegNum().


class TestHolderContainerConstruction(unittest.TestCase):
    def test_string_ctor_named(self):
        # String ctor: type=1, holderName=str, parent regName =
        # RegName(holderName) with offsets=[].
        hc = HolderContainer("v", "ValuC", 2)
        self.assertEqual(hc.regType, "v")
        self.assertEqual(hc.regIdx, 0)
        self.assertEqual(hc.regNum, 2)
        self.assertIsNotNone(hc.regName)
        self.assertEqual(hc.regName.name, "ValuC")
        self.assertEqual(hc.regName.offsets, [])
        self.assertEqual(hc.holderName, "ValuC")
        self.assertEqual(hc.holderIdx, 0)
        self.assertEqual(hc.holderType, 1)
        self.assertEqual(hc.holderOffsets, [])

    def test_regname_ctor_named(self):
        # RegName ctor: holderOffsets seeded from RegName.offsets so
        # pre-existing offsets survive the resolution boundary.
        rn = RegName("ValuC", [3])
        hc = HolderContainer("v", rn, 2)
        self.assertEqual(hc.holderName, "ValuC")
        self.assertEqual(hc.holderType, 1)
        self.assertEqual(hc.holderOffsets, [3])
        # Parent regName carries source offsets verbatim.
        self.assertEqual(hc.regName.offsets, [3])

    def test_int_ctor_numeric(self):
        # Int ctor: type=0, regName=None, regIdx=holderIdx.
        hc = HolderContainer("v", 10, 4)
        self.assertEqual(hc.regType, "v")
        self.assertEqual(hc.regIdx, 10)
        self.assertEqual(hc.regNum, 4)
        self.assertIsNone(hc.regName)
        self.assertEqual(hc.holderName, "")
        self.assertEqual(hc.holderIdx, 10)
        self.assertEqual(hc.holderType, 0)
        self.assertEqual(hc.holderOffsets, [])

    def test_bool_rejected(self):
        # bool falls through to int otherwise; ensures intent isn't lost.
        with self.assertRaises(TypeError):
            HolderContainer("v", True, 1)

    def test_bad_type_rejected(self):
        with self.assertRaises(TypeError):
            HolderContainer("v", 3.0, 1)
        with self.assertRaises(TypeError):
            HolderContainer("v", None, 1)

    def test_inheritance(self):
        # KernelWriter checks `isinstance(p, RegisterContainer)` in
        # several places; HolderContainer must satisfy that.
        hc = HolderContainer("v", "Foo", 1)
        self.assertIsInstance(hc, RegisterContainer)
        self.assertIsInstance(hc, HolderContainer)


class TestHolderContainerSetRegNum(unittest.TestCase):
    def test_numeric_resolution(self):
        # type 0 sets ``regIdx = holderIdx + num``; holderIdx is
        # preserved so setRegNum is idempotent on rerun.
        hc = HolderContainer("v", 4, 2)
        hc.setRegNum(10)
        self.assertEqual(hc.regIdx, 14)
        self.assertEqual(hc.holderIdx, 4)

    def test_numeric_resolution_with_zero(self):
        # dst=0 is the canonical call site (replaceHolder(module, 0)).
        hc = HolderContainer("v", 8, 1)
        hc.setRegNum(0)
        self.assertEqual(hc.regIdx, 8)

    def test_named_resolution_simple(self):
        # type 1 rebuilds regName = RegName(holderName) and prepends num.
        hc = HolderContainer("v", "ValuC", 2)
        hc.setRegNum(5)
        self.assertEqual(hc.regName.name, "ValuC")
        self.assertEqual(hc.regName.offsets, [5])

    def test_named_resolution_with_holder_offsets(self):
        # If the RegName-form ctor was used with non-empty offsets,
        # setRegNum prepends num and re-appends the saved offsets.
        rn = RegName("ValuC", [7, 9])
        hc = HolderContainer("v", rn, 1)
        hc.setRegNum(3)
        self.assertEqual(hc.regName.name, "ValuC")
        self.assertEqual(hc.regName.offsets, [3, 7, 9])

    def test_named_resolution_idempotent_on_replay(self):
        # setRegNum unconditionally re-seeds regName from holderName, so
        # calling it twice yields the same result.
        hc = HolderContainer("v", "ValuC", 1)
        hc.setRegNum(5)
        hc.setRegNum(5)
        self.assertEqual(hc.regName.offsets, [5])

    def test_named_resolution_idempotent_with_offsets(self):
        rn = RegName("ValuC", [1])
        hc = HolderContainer("v", rn, 1)
        hc.setRegNum(2)
        hc.setRegNum(2)
        self.assertEqual(hc.regName.offsets, [2, 1])


class TestHolderContainerGetCopiedRC(unittest.TestCase):
    def test_numeric_snapshot(self):
        # type 0 -> RC(regType, None, regIdx, regNum).
        hc = HolderContainer("v", 4, 2)
        hc.setRegNum(10)
        rc = hc.getCopiedRC()
        self.assertNotIsInstance(rc, HolderContainer)
        self.assertIsInstance(rc, RegisterContainer)
        self.assertEqual(rc.regType, "v")
        self.assertIsNone(rc.regName)
        self.assertEqual(rc.regIdx, 14)
        self.assertEqual(rc.regNum, 2)

    def test_named_snapshot(self):
        # type 1 -> RC(regType, regName, regIdx, regNum). regIdx is 0
        # for the named flavour (the symbolic name carries the offset).
        hc = HolderContainer("v", "ValuC", 2)
        hc.setRegNum(5)
        rc = hc.getCopiedRC()
        self.assertNotIsInstance(rc, HolderContainer)
        self.assertEqual(rc.regName.name, "ValuC")
        self.assertEqual(rc.regName.offsets, [5])
        self.assertEqual(rc.regIdx, 0)
        self.assertEqual(rc.regNum, 2)

    def test_snapshot_is_independent(self):
        # Snapshot is a value, not a ref; mutating either side must not
        # bleed into the other.
        hc = HolderContainer("v", "ValuC", 2)
        hc.setRegNum(5)
        rc = hc.getCopiedRC()
        rc.regName.offsets.append(99)
        self.assertEqual(hc.regName.offsets, [5])

    def test_snapshot_renders_correctly(self):
        # End-to-end: the snapshot must emit the same string a directly-
        # constructed RC would.
        hc = HolderContainer("v", "ValuC", 2)
        hc.setRegNum(5)
        rc = hc.getCopiedRC()
        self.assertEqual(str(rc), "v[vgprValuC+5:vgprValuC+5+1]")


class TestHolderContainerSplit(unittest.TestCase):
    def test_split_numeric(self):
        # Numeric branch: r2.holderIdx bumps by 1; both halves stay as
        # HolderContainer.
        hc = HolderContainer("v", 4, 2)
        r1, r2 = hc.splitRegContainer()
        self.assertIsInstance(r1, HolderContainer)
        self.assertIsInstance(r2, HolderContainer)
        self.assertEqual(r1.holderIdx, 4)
        self.assertEqual(r2.holderIdx, 5)
        # regNum is halved (ceil) for r1, remainder for r2.
        self.assertEqual(r1.regNum, 1)
        self.assertEqual(r2.regNum, 1)

    def test_split_named(self):
        # Named branch: pushes 1 onto r2.regName.offsets. The RegName-
        # form ctor's saved offsets are preserved on the parent's
        # regName (used by addOffset).
        hc = HolderContainer("v", "ValuC", 2)
        r1, r2 = hc.splitRegContainer()
        self.assertIsInstance(r1, HolderContainer)
        self.assertIsInstance(r2, HolderContainer)
        self.assertEqual(r1.regName.offsets, [])
        self.assertEqual(r2.regName.offsets, [1])
        self.assertEqual(r1.holderName, "ValuC")
        self.assertEqual(r2.holderName, "ValuC")
        self.assertEqual(r1.regNum, 1)
        self.assertEqual(r2.regNum, 1)

    def test_split_independent(self):
        # Mutating halves must not bleed into the source.
        hc = HolderContainer("v", "ValuC", 2)
        r1, r2 = hc.splitRegContainer()
        r2.regName.offsets.append(99)
        self.assertEqual(hc.regName.offsets, [])


class TestHolderContainerCopy(unittest.TestCase):
    def test_shallow_copy_independent(self):
        hc = HolderContainer("v", "ValuC", 2)
        hc.setRegNum(5)
        c = copy.copy(hc)
        c.regName.offsets.append(99)
        c.holderOffsets.append(99)
        self.assertEqual(hc.regName.offsets, [5])
        self.assertEqual(hc.holderOffsets, [])

    def test_deep_copy_independent(self):
        hc = HolderContainer("v", "ValuC", 2)
        c = copy.deepcopy(hc)
        self.assertEqual(c.holderName, "ValuC")
        self.assertEqual(c.holderType, 1)
        c.holderName = "Other"
        self.assertEqual(hc.holderName, "ValuC")

    def test_copy_preserves_subclass(self):
        hc = HolderContainer("v", 5, 1)
        self.assertIsInstance(copy.copy(hc), HolderContainer)
        self.assertIsInstance(copy.deepcopy(hc), HolderContainer)


class TestHolderContainerPickle(unittest.TestCase):
    def test_pickle_roundtrip_named(self):
        hc = HolderContainer("v", "ValuC", 2)
        hc.setRegNum(5)
        c = pickle.loads(pickle.dumps(hc))
        self.assertIsInstance(c, HolderContainer)
        self.assertEqual(c.holderName, "ValuC")
        self.assertEqual(c.holderType, 1)
        self.assertEqual(c.regName.offsets, [5])
        self.assertEqual(c.regType, "v")
        self.assertEqual(c.regNum, 2)

    def test_pickle_roundtrip_numeric(self):
        hc = HolderContainer("v", 8, 1)
        hc.setRegNum(2)
        c = pickle.loads(pickle.dumps(hc))
        self.assertEqual(c.holderType, 0)
        self.assertEqual(c.holderIdx, 8)
        self.assertEqual(c.regIdx, 10)


# ===========================================================================
# replaceHolder
# ===========================================================================
#
# Adapter duck-types on .items() / .getParams() to detect Module /
# Instruction. TODO(T6): lights up once real Module/Instruction land.


class _MockInstr:
    """Minimal Instruction-like: exposes a mutable .getParams() list."""

    def __init__(self, params):
        self._params = list(params)

    def getParams(self):
        return self._params


class _MockModule:
    """Minimal Module-like: exposes .items() returning children."""

    def __init__(self, items):
        self._items = list(items)

    def items(self):
        return self._items


class _MockSWaitCnt:
    """Class-name-only marker for the SWaitCnt detection branch."""

    pass


_MockSWaitCnt.__name__ = "SWaitCnt"


class TestReplaceHolderLeaf(unittest.TestCase):
    def test_scalar_passthrough(self):
        # Unknown types are returned unchanged.
        self.assertEqual(replaceHolder(42, 5), 42)
        self.assertEqual(replaceHolder("hello", 5), "hello")
        self.assertIsNone(replaceHolder(None, 5))

    def test_holder_alone_unchanged(self):
        # A bare HolderContainer is neither Module nor Instruction so it
        # falls through unmodified.
        hc = HolderContainer("v", 4, 1)
        result = replaceHolder(hc, 7)
        self.assertIs(result, hc)
        self.assertEqual(hc.regIdx, 4)


class TestReplaceHolderInstruction(unittest.TestCase):
    def test_resolves_numeric_holder_in_params(self):
        # Instruction branch: walks getParams() and swaps each
        # HolderContainer for its resolved snapshot.
        hc = HolderContainer("v", 4, 1)
        inst = _MockInstr([hc, "literal"])
        replaceHolder(inst, 10)
        # Param 0 is now a plain RC, not a HolderContainer.
        self.assertNotIsInstance(inst.getParams()[0], HolderContainer)
        self.assertIsInstance(inst.getParams()[0], RegisterContainer)
        self.assertEqual(inst.getParams()[0].regIdx, 14)
        # Non-HolderContainer params are untouched.
        self.assertEqual(inst.getParams()[1], "literal")

    def test_resolves_named_holder_in_params(self):
        hc = HolderContainer("v", "ValuC", 2)
        inst = _MockInstr([hc])
        replaceHolder(inst, 5)
        rc = inst.getParams()[0]
        self.assertNotIsInstance(rc, HolderContainer)
        self.assertEqual(rc.regName.name, "ValuC")
        self.assertEqual(rc.regName.offsets, [5])
        self.assertEqual(rc.regNum, 2)

    def test_multiple_holders_resolved(self):
        # Same dst applied to every holder in the param list.
        h1 = HolderContainer("v", 4, 1)
        h2 = HolderContainer("v", "ValuC", 1)
        inst = _MockInstr([h1, h2])
        replaceHolder(inst, 3)
        self.assertEqual(inst.getParams()[0].regIdx, 7)
        self.assertEqual(inst.getParams()[1].regName.offsets, [3])

    def test_non_holder_register_left_intact(self):
        # Plain RegisterContainer params must pass through untouched
        # (only HolderContainer instances are resolved).
        rc = RegisterContainer("v", None, 5, 1)
        inst = _MockInstr([rc])
        replaceHolder(inst, 99)
        self.assertIs(inst.getParams()[0], rc)
        self.assertEqual(rc.regIdx, 5)


class TestReplaceHolderModule(unittest.TestCase):
    def test_recurses_into_module_items(self):
        # Module branch recurses into each child.
        hc = HolderContainer("v", 4, 1)
        inst = _MockInstr([hc])
        mod = _MockModule([inst])
        result = replaceHolder(mod, 10)
        self.assertIs(result, mod)
        self.assertEqual(inst.getParams()[0].regIdx, 14)

    def test_nested_modules(self):
        # Module-of-Modules: must recurse fully.
        hc = HolderContainer("v", 0, 1)
        inner_inst = _MockInstr([hc])
        inner_mod = _MockModule([inner_inst])
        outer_mod = _MockModule([inner_mod])
        replaceHolder(outer_mod, 7)
        self.assertEqual(inner_inst.getParams()[0].regIdx, 7)

    def test_mixed_children(self):
        # A Module containing instructions AND other modules.
        h1 = HolderContainer("v", 1, 1)
        h2 = HolderContainer("v", 2, 1)
        i1 = _MockInstr([h1])
        i2 = _MockInstr([h2])
        sub = _MockModule([i2])
        root = _MockModule([i1, sub])
        replaceHolder(root, 100)
        self.assertEqual(i1.getParams()[0].regIdx, 101)
        self.assertEqual(i2.getParams()[0].regIdx, 102)


class TestReplaceHolderSWaitCnt(unittest.TestCase):
    def test_raises(self):
        # SWaitCnt branch raises explicitly (intentional gap).
        with self.assertRaises(RuntimeError):
            replaceHolder(_MockSWaitCnt(), 0)


# ===========================================================================
# Factory functions: vgpr / sgpr / accvgpr / mgpr
# ===========================================================================
#
# Per rocisa::createGPR + the type-specific wrappers in container.cpp.
# Three dispatch arms each (Holder / int / str); plus type-tagging,
# regNum rounding (delegated to RegisterContainer), and modifier-kwarg
# acceptance matching the C++ signatures.


class TestVgprFactory(unittest.TestCase):
    def test_int_arg_makes_numeric_register(self):
        rc = vgpr(5)
        self.assertIsInstance(rc, RegisterContainer)
        self.assertNotIsInstance(rc, HolderContainer)
        self.assertEqual(rc.regType, "v")
        self.assertIsNone(rc.regName)
        self.assertEqual(rc.regIdx, 5)
        self.assertEqual(rc.regNum, 1)

    def test_int_arg_with_regnum(self):
        rc = vgpr(8, 4)
        self.assertEqual(rc.regIdx, 8)
        self.assertEqual(rc.regNum, 4)

    def test_str_arg_makes_symbolic_register(self):
        rc = vgpr("ValuA")
        self.assertEqual(rc.regType, "v")
        self.assertEqual(rc.regIdx, -1)
        self.assertIsNotNone(rc.regName)
        self.assertEqual(rc.regName.name, "ValuA")
        self.assertEqual(rc.regName.offsets, [])

    def test_str_arg_with_offsets_parsed(self):
        rc = vgpr("Foo+1+2", 1)
        self.assertEqual(rc.regName.name, "Foo")
        self.assertEqual(rc.regName.offsets, [1, 2])

    def test_str_arg_passes_modifier_kwargs(self):
        rc = vgpr("Foo", 1, isMacro=True, isAbs=True, isOff=True)
        self.assertTrue(rc.isMacro)
        self.assertTrue(rc.isAbs)
        self.assertTrue(rc.isOff)

    def test_holder_int_makes_typed_holdercontainer(self):
        h = Holder(idx=3)
        hc = vgpr(h, 2)
        self.assertIsInstance(hc, HolderContainer)
        self.assertEqual(hc.regType, "v")
        self.assertEqual(hc.holderIdx, 3)
        self.assertEqual(hc.holderType, 0)
        self.assertEqual(hc.regNum, 2)

    def test_holder_str_makes_named_holdercontainer(self):
        h = Holder(name="ValuC")
        hc = vgpr(h, 1)
        self.assertIsInstance(hc, HolderContainer)
        self.assertEqual(hc.regType, "v")
        self.assertEqual(hc.holderName, "ValuC")
        self.assertEqual(hc.holderType, 1)

    def test_bool_rejected(self):
        with self.assertRaises(TypeError):
            vgpr(True)
        with self.assertRaises(TypeError):
            vgpr(False, 1)

    def test_unsupported_type_rejected(self):
        with self.assertRaises(TypeError):
            vgpr(3.14)
        with self.assertRaises(TypeError):
            vgpr(None)

    def test_regnum_rounded_up_by_container(self):
        # vgpr delegates rounding semantics to RegisterContainer.__init__.
        rc = vgpr(0, 1.5)
        self.assertEqual(rc.regNum, 2)


class TestSgprFactory(unittest.TestCase):
    def test_int_arg(self):
        rc = sgpr(2)
        self.assertEqual(rc.regType, "s")
        self.assertIsNone(rc.regName)
        self.assertEqual(rc.regIdx, 2)

    def test_str_arg_default(self):
        rc = sgpr("AddressA")
        self.assertEqual(rc.regType, "s")
        self.assertEqual(rc.regName.name, "AddressA")
        self.assertFalse(rc.isMacro)
        self.assertFalse(rc.isAbs)
        self.assertFalse(rc.isOff)

    def test_str_arg_isMacro_only(self):
        rc = sgpr("Foo", 1, isMacro=True)
        self.assertTrue(rc.isMacro)
        self.assertFalse(rc.isAbs)
        self.assertFalse(rc.isOff)

    def test_str_arg_rejects_isAbs(self):
        # sgpr has no isAbs in the rocisa signature (only vgpr does).
        # isOff IS accepted by sgpr, matching native rocisa container.cpp.
        with self.assertRaises(TypeError):
            sgpr("Foo", 1, isAbs=True)

    def test_str_arg_accepts_isOff(self):
        # isOff mirrors the native rocisa sgpr binding (container.cpp).
        rc = sgpr("Foo", 1, isOff=True)
        self.assertTrue(rc.isOff)
        self.assertFalse(rc.isAbs)

    def test_holder_dispatch(self):
        hc = sgpr(Holder(idx=4), 1)
        self.assertIsInstance(hc, HolderContainer)
        self.assertEqual(hc.regType, "s")
        self.assertEqual(hc.holderIdx, 4)


class TestAccvgprFactory(unittest.TestCase):
    def test_int_arg(self):
        rc = accvgpr(7)
        self.assertEqual(rc.regType, "acc")
        self.assertEqual(rc.regIdx, 7)

    def test_str_arg(self):
        rc = accvgpr("Acc")
        self.assertEqual(rc.regType, "acc")
        self.assertEqual(rc.regName.name, "Acc")

    def test_holder_dispatch(self):
        hc = accvgpr(Holder(name="Acc"), 1)
        self.assertIsInstance(hc, HolderContainer)
        self.assertEqual(hc.regType, "acc")
        self.assertEqual(hc.holderName, "Acc")

    def test_rejects_modifier_kwargs(self):
        with self.assertRaises(TypeError):
            accvgpr("Acc", 1, isMacro=True)


class TestMgprFactory(unittest.TestCase):
    def test_int_arg(self):
        rc = mgpr(0)
        self.assertEqual(rc.regType, "m")
        self.assertEqual(rc.regIdx, 0)

    def test_str_arg(self):
        rc = mgpr("Mdesc")
        self.assertEqual(rc.regType, "m")
        self.assertEqual(rc.regName.name, "Mdesc")

    def test_holder_dispatch(self):
        hc = mgpr(Holder(idx=1), 4)
        self.assertIsInstance(hc, HolderContainer)
        self.assertEqual(hc.regType, "m")
        self.assertEqual(hc.holderIdx, 1)
        self.assertEqual(hc.regNum, 4)

    def test_rejects_modifier_kwargs(self):
        with self.assertRaises(TypeError):
            mgpr("Mdesc", 1, isMacro=True)


class TestFactoryEquivalentToDirectConstruction(unittest.TestCase):
    """Factories must produce containers indistinguishable from
    ``RegisterContainer(...)`` / ``HolderContainer(...)`` direct calls."""

    def test_vgpr_int_matches_manual_construction(self):
        from_factory = vgpr(5, 2)
        manual = RegisterContainer("v", None, 5, 2)
        self.assertEqual(from_factory.regType, manual.regType)
        self.assertEqual(from_factory.regIdx, manual.regIdx)
        self.assertEqual(from_factory.regNum, manual.regNum)
        self.assertEqual(from_factory.regName, manual.regName)

    def test_vgpr_holder_matches_manual_construction(self):
        h = Holder(idx=3)
        from_factory = vgpr(h, 1)
        manual = HolderContainer("v", 3, 1)
        self.assertEqual(from_factory.regType, manual.regType)
        self.assertEqual(from_factory.holderIdx, manual.holderIdx)
        self.assertEqual(from_factory.holderType, manual.holderType)


# ===========================================================================
# ContinuousRegister
# ===========================================================================
#
# rocisa::ContinuousRegister is a POD with read-only {idx, size}.
# The Python shim mirrors that surface; no equality or hashing in the
# C++ binding either.


class TestContinuousRegisterConstruction(unittest.TestCase):
    def test_positional_ctor(self):
        cr = ContinuousRegister(4, 2)
        self.assertEqual(cr.idx, 4)
        self.assertEqual(cr.size, 2)

    def test_keyword_ctor(self):
        cr = ContinuousRegister(idx=8, size=4)
        self.assertEqual(cr.idx, 8)
        self.assertEqual(cr.size, 4)

    def test_zero_ok(self):
        cr = ContinuousRegister(0, 0)
        self.assertEqual(cr.idx, 0)
        self.assertEqual(cr.size, 0)

    def test_negative_ok(self):
        # rocisa stores raw ints; -1 is a common sentinel.
        cr = ContinuousRegister(-1, 0)
        self.assertEqual(cr.idx, -1)

    def test_rejects_bool(self):
        with self.assertRaises(TypeError):
            ContinuousRegister(True, 1)
        with self.assertRaises(TypeError):
            ContinuousRegister(1, False)

    def test_rejects_non_int(self):
        with self.assertRaises(TypeError):
            ContinuousRegister(1.0, 2)
        with self.assertRaises(TypeError):
            ContinuousRegister("a", 2)


class TestContinuousRegisterReadOnly(unittest.TestCase):
    def test_idx_is_read_only(self):
        cr = ContinuousRegister(4, 2)
        with self.assertRaises(AttributeError):
            cr.idx = 5  # type: ignore[misc]

    def test_size_is_read_only(self):
        cr = ContinuousRegister(4, 2)
        with self.assertRaises(AttributeError):
            cr.size = 9  # type: ignore[misc]

    def test_extra_attribute_blocked(self):
        cr = ContinuousRegister(4, 2)
        with self.assertRaises(AttributeError):
            cr.something = 1  # type: ignore[attr-defined]


class TestContinuousRegisterRepr(unittest.TestCase):
    def test_repr_contains_fields(self):
        cr = ContinuousRegister(4, 2)
        r = repr(cr)
        self.assertIn("4", r)
        self.assertIn("2", r)
        self.assertIn("ContinuousRegister", r)


class TestContinuousRegisterCopyPickle(unittest.TestCase):
    def test_copy_returns_equal_state(self):
        cr = ContinuousRegister(4, 2)
        c = copy.copy(cr)
        self.assertIs(type(c), ContinuousRegister)
        self.assertEqual(c.idx, 4)
        self.assertEqual(c.size, 2)
        self.assertIsNot(c, cr)

    def test_deepcopy_returns_equal_state(self):
        cr = ContinuousRegister(4, 2)
        c = copy.deepcopy(cr)
        self.assertIs(type(c), ContinuousRegister)
        self.assertEqual(c.idx, 4)
        self.assertEqual(c.size, 2)

    def test_pickle_roundtrip(self):
        cr = ContinuousRegister(4, 2)
        c = pickle.loads(pickle.dumps(cr))
        self.assertEqual(c.idx, 4)
        self.assertEqual(c.size, 2)
        # Pickled copy is still read-only.
        with self.assertRaises(AttributeError):
            c.idx = 0  # type: ignore[misc]


class TestContinuousRegisterNoEqualityOrHash(unittest.TestCase):
    """Mirror rocisa C++: ContinuousRegister has no eq/hash binding."""

    def test_eq_is_identity(self):
        a = ContinuousRegister(4, 2)
        b = ContinuousRegister(4, 2)
        self.assertNotEqual(a, b)  # identity-only equality
        self.assertEqual(a, a)

    def test_hashable_via_identity(self):
        # Default object.__hash__ is identity-based; we only assert that
        # hashing does not raise (rocisa exposes no __hash__ override).
        a = ContinuousRegister(4, 2)
        hash(a)
        self.assertEqual(hash(a), hash(a))


# ===========================================================================
# Container ABC + hardware tokens (T4)
# ===========================================================================


class _WavefrontTestCase(unittest.TestCase):
    """Pin ``rocIsa`` kernel wavefront for EXEC/VCC ``toString`` branches."""

    def setUp(self):
        self._saved = rocIsa.getInstance().getKernel()
        rocIsa.getInstance().setKernel((12, 5, 0), 64)

    def tearDown(self):
        info = self._saved
        if info.isa is not None:
            rocIsa.getInstance().setKernel(info.isa, info.wavefrontSize)
        else:
            rocIsa.getInstance()._kernel_info = info


class TestContainerABC(unittest.TestCase):
    def test_not_instantiable(self):
        with self.assertRaises(TypeError):
            Container()

    def test_subclasses_are_container(self):
        self.assertIsInstance(EXEC(), Container)
        self.assertIsInstance(VCC(), Container)
        self.assertIsInstance(EXECLO(), Container)
        self.assertIsInstance(EXECHI(), Container)
        self.assertIsInstance(HWRegContainer("r", [1]), Container)

    def test_register_container_is_container(self):
        rc = RegisterContainer("v", None, 0, 1)
        self.assertIsInstance(rc, Container)
        hc = HolderContainer("v", 3, 1)
        self.assertIsInstance(hc, Container)
        self.assertIsInstance(hc, RegisterContainer)


class TestHWRegContainer(_WavefrontTestCase):
    def test_to_string(self):
        self.assertEqual(str(HWRegContainer("reg", [1, 1])), "hwreg(reg,1,1)")
        self.assertEqual(
            HWRegContainer("26", [4, 1]).toString(), "hwreg(26,4,1)"
        )

    def test_value_list_copied(self):
        raw = [1, 2]
        h = HWRegContainer("r", raw)
        raw.append(3)
        self.assertEqual(h.value, [1, 2])

    def test_clone_and_pickle(self):
        h = HWRegContainer("reg", [1, 1])
        self.assertEqual(str(h.clone()), str(h))
        self.assertEqual(str(copy.deepcopy(h)), str(h))
        self.assertEqual(str(pickle.loads(pickle.dumps(h))), str(h))


class TestEXECLOEXECHI(unittest.TestCase):
    def test_exec_lo_hi_tokens(self):
        self.assertEqual(str(EXECLO()), "exec_lo")
        self.assertEqual(str(EXECHI()), "exec_hi")
        self.assertIsInstance(EXECLO(), Container)

    def test_copy_pickle(self):
        self.assertEqual(str(copy.deepcopy(EXECLO())), "exec_lo")
        self.assertEqual(str(pickle.loads(pickle.dumps(EXECHI()))), "exec_hi")


class TestEXECWavefront(_WavefrontTestCase):
    def test_wavefront_64(self):
        self.assertEqual(str(EXEC()), "exec")
        self.assertEqual(str(EXEC(True)), "exec")

    def test_wavefront_32(self):
        rocIsa.getInstance().setKernel((12, 5, 0), 32)
        self.assertEqual(str(EXEC()), "exec_lo")
        self.assertEqual(str(EXEC(True)), "exec_lo")

    def test_clone_matches_str(self):
        e = EXEC(True)
        self.assertEqual(str(e.clone()), str(e))
        self.assertEqual(str(pickle.loads(pickle.dumps(e))), "exec")


class TestVCCWavefront(_WavefrontTestCase):
    def test_wavefront_64(self):
        self.assertEqual(str(VCC()), "vcc")
        self.assertEqual(str(VCC(True)), "vcc")

    def test_wavefront_32(self):
        rocIsa.getInstance().setKernel((12, 5, 0), 32)
        self.assertEqual(str(VCC()), "vcc_lo")
        self.assertEqual(str(VCC(True)), "vcc_hi")

    def test_to_string_alias(self):
        self.assertEqual(VCC().toString(), str(VCC()))


# ===========================================================================
# MemTokenData (T5)
# ===========================================================================


class _Gfx1250CapsTestCase(unittest.TestCase):
    """Exercise modifier ``toString`` under gfx1250 caps from ``getHardwareCaps``."""

    def setUp(self):
        self._saved = rocIsa.getInstance().getKernel()
        rocIsa.getInstance().setKernel((12, 5, 0), 64)

    def tearDown(self):
        info = self._saved
        if info.isa is not None:
            rocIsa.getInstance().setKernel(info.isa, info.wavefrontSize)
        else:
            rocIsa.getInstance()._kernel_info = info


class TestModifiersGfx1250(_Gfx1250CapsTestCase):
    """``toString`` parity with ``container.hpp`` under gfx1250 caps."""

    def test_ds_modifiers(self):
        self.assertEqual(str(DSModifiers(1, 2, gds=True)), " offset:2 gds")
        self.assertEqual(str(DSModifiers(2, offset0=3, offset1=4)), " offset0:3 offset1:4")
        self.assertEqual(str(DSModifiers(gds=True)), " offset:0 gds")

    def test_global_modifiers(self):
        self.assertEqual(str(GLOBALModifiers()), "")
        self.assertEqual(str(GLOBALModifiers(16)), " offset:16")
        gl2 = GLOBALModifiers(
            th=TemporalHint.TH_NT, scope=CacheScope.SCOPE_SE,
        )
        self.assertEqual(str(gl2), " th:TH_LOAD_NT scope:SCOPE_SE")
        self.assertEqual(
            str(GLOBALModifiers(16, TemporalHint.TH_NT, CacheScope.SCOPE_SE)),
            " offset:16 th:TH_LOAD_NT scope:SCOPE_SE",
        )

    def test_flat_modifiers_gfx1250(self):
        # gfx1250: HasGLCModifier=0, HasSC0Modifier=0 -> glc/slc bits are silent.
        flat = FLATModifiers(
            offset12=8, glc=True, slc=False, dlc=False,
            lds=True, isStore=False, scope=CacheScope.SCOPE_NONE,
        )
        self.assertEqual(str(flat), " offset:8  lds")

    def test_flat_modifiers_th_nv(self):
        flat = FLATModifiers(
            offset12=4, th=TemporalHint.TH_NT, nv=NonVolatile.NV,
        )
        self.assertEqual(str(flat), " offset:4 th:TH_LOAD_NT nv")

    def test_mubuf_modifiers_gfx1250(self):
        mubuf = MUBUFModifiers(
            offen=True, offset12=12, glc=True, slc=False,
            dlc=False, scope=CacheScope.SCOPE_NONE,
            nt=True, lds=False, isStore=True,
        )
        self.assertEqual(str(mubuf), " offen offset:12 ")

    def test_mubuf_modifiers_th_over_nt(self):
        mubuf = MUBUFModifiers(
            offen=True, offset12=12, nt=True,
            th=TemporalHint.TH_NT, isStore=False,
        )
        self.assertEqual(str(mubuf), " offen offset:12 th:TH_LOAD_NT")

    def test_mubuf_modifiers_store_th_lu(self):
        mubuf = MUBUFModifiers(
            offen=True, offset12=0, th=TemporalHint.TH_LU, isStore=True,
        )
        self.assertEqual(str(mubuf), " offen offset:0 th:TH_STORE_WB")

    def test_smem_modifiers_gfx1250(self):
        # HasSCOPEModifier=1 -> literal "glc" is suppressed (C++ SMEM path).
        smem = SMEMModifiers(
            glc=True, dlc=False, scope=CacheScope.SCOPE_NONE,
            nv=NonVolatile.NV_NONE, offset=8,
        )
        self.assertEqual(str(smem), " offset:8")

    def test_smem_modifiers_th_nv(self):
        smem = SMEMModifiers(
            offset=0, th=TemporalHint.TH_RT, nv=NonVolatile.NV,
        )
        self.assertEqual(str(smem), " th:TH_LOAD_RT nv")

    def test_smem_scope_when_set(self):
        smem = SMEMModifiers(
            glc=False, scope=CacheScope.SCOPE_DEV, offset=0,
        )
        self.assertEqual(str(smem), " scope:SCOPE_DEV")

    def test_sdwa_modifiers(self):
        sdwa = SDWAModifiers(
            dst_sel=SelectBit.WORD_0,
            src0_sel=SelectBit.WORD_0,
            src1_sel=SelectBit.WORD_1,
        )
        self.assertEqual(
            str(sdwa), " dst_sel:WORD_0 src0_sel:WORD_0 src1_sel:WORD_1",
        )

    def test_vop3p_modifiers(self):
        vop3p = VOP3PModifiers([0, 0], [0, 1], [0, 0])
        self.assertEqual(
            str(vop3p), " op_sel:[0,0] op_sel_hi:[0,1] byte_sel:[0,0]",
        )

    def test_dpp_modifiers(self):
        dpp = DPPModifiers(row_shr=1, quad_perm=[0, 1, 2, 3])
        self.assertEqual(str(dpp), " row_shr:1 quad_perm:[0,1,2,3]")

    def test_true16_modifiers(self):
        self.assertEqual(str(True16Modifiers()), "")
        self.assertEqual(str(True16Modifiers(HighBitSel.HIGH)), ".h")
        self.assertEqual(str(True16Modifiers(HighBitSel.LOW)), ".l")
        self.assertEqual(str(True16Modifiers(2)), ".h")

    def test_is_container(self):
        self.assertIsInstance(DSModifiers(), Container)
        self.assertIsInstance(FLATModifiers(), Container)

    def test_clone_and_pickle(self):
        for obj in (
            DSModifiers(gds=True),
            FLATModifiers(lds=True, th=TemporalHint.TH_NT, nv=NonVolatile.NV),
            GLOBALModifiers(th=TemporalHint.TH_NT, scope=CacheScope.SCOPE_SE),
            MUBUFModifiers(offen=True, offset12=12, glc=True, nt=True, th=TemporalHint.TH_NT),
            SMEMModifiers(offset=8, th=TemporalHint.TH_RT, nv=NonVolatile.NV),
            SDWAModifiers(dst_sel=SelectBit.WORD_0),
            VOP3PModifiers([1], [2], [3]),
            DPPModifiers(row_bcast=2),
            True16Modifiers(HighBitSel.LOW),
        ):
            self.assertEqual(str(obj.clone()), str(obj))
            self.assertEqual(str(copy.deepcopy(obj)), str(obj))
            self.assertEqual(str(pickle.loads(pickle.dumps(obj))), str(obj))


class TestGlcSlcBitNames(_Gfx1250CapsTestCase):
    def test_gfx1250_defaults_empty(self):
        self.assertEqual(getGlcBitName(), "")
        self.assertEqual(getSlcBitName(), "")

    def test_has_glc_modifier_branch(self):
        caps = dict(rocIsa.getInstance().getAsmCaps())
        caps["HasGLCModifier"] = 1
        caps["HasSC0Modifier"] = 0
        from rocisa_stinkytofu_adaptor.caps import (  # noqa: WPS433
            glc_bit_name_from_caps,
            slc_bit_name_from_caps,
        )

        self.assertEqual(glc_bit_name_from_caps(caps), "glc")
        self.assertEqual(slc_bit_name_from_caps(caps), "slc")


class TestMemTokenData(unittest.TestCase):
    def test_empty_tokens(self):
        self.assertEqual(str(MemTokenData()), "mem_token:")
        self.assertEqual(MemTokenData().toString(), "mem_token:")

    def test_single_and_multiple_tokens(self):
        self.assertEqual(str(MemTokenData([7])), "mem_token: 7")
        self.assertEqual(str(MemTokenData([1, 2, 3])), "mem_token: 1, 2, 3")

    def test_default_arg_none(self):
        m = MemTokenData(None)
        self.assertEqual(m.tokens, [])

    def test_tokens_list_copied(self):
        raw = [1, 2]
        m = MemTokenData(raw)
        raw.append(3)
        self.assertEqual(m.tokens, [1, 2])

    def test_tokens_mutable_like_cpp_binding(self):
        m = MemTokenData([1])
        m.tokens.append(2)
        self.assertEqual(str(m), "mem_token: 1, 2")

    def test_is_container(self):
        self.assertIsInstance(MemTokenData([0]), Container)

    def test_clone_and_pickle(self):
        m = MemTokenData([4, 5])
        self.assertEqual(str(m.clone()), str(m))
        self.assertEqual(str(copy.deepcopy(m)), str(m))
        self.assertEqual(str(pickle.loads(pickle.dumps(m))), str(m))

    def test_kernel_writer_style_single_meta(self):
        # KernelWriterAssembly passes a one-element list from memTokenLdsBufferMeta.
        m = MemTokenData([42])
        self.assertEqual(str(m), "mem_token: 42")


if __name__ == "__main__":
    unittest.main()
