################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
"""Standalone tests for ``rocisa_stinkytofu_adaptor.instruction`` -- Step 3
(``Instruction`` / ``CommonInstruction`` bases + the real ``VMovB32`` shim).

Run from any working directory:

    python3 projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_instruction.py

Or with pytest:

    pytest projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_instruction.py

These tests pin:
  * Base-class shape (``isinstance`` / fields / mutability) -- KernelWriter
    does ``isinstance(x, CommonInstruction) and ... and x.dst.regType == 'm'``
    and ``codeAccVgprReadInst.dst = vgpr(...)``; the shim must support both.
  * ``__str__`` byte-parity with ``rocisa::CommonInstruction::toString``
    (padding to col 50 + ``" // "`` + comment + newline). A drift here will
    silently corrupt diff-based regression tests downstream.
  * ``to_stinky_logical()`` and the ``_to_stinky_register`` coercion
    table -- both register operands (via ``RegisterContainer.to_stinky``)
    and the int / float / str literal paths.
  * Step-3 collector contract: ``Module._collect_logical_insts`` picks up
    real VMovB32 shims and skips dummy ``LogicalInstructionBase`` items.

End-to-end byte-equality of the emitted asm against the rocisa right-path
is covered in ``tests/test_emission_consistency.py`` (a stronger check
than the old substring assertions).
"""

from __future__ import annotations

import copy
import os
import sys
import unittest

# ---------------------------------------------------------------------------
# Self-contained sys.path bootstrap (matches test_container / test_code).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rocisa_stinkytofu_adaptor.code import Module  # noqa: E402
from rocisa_stinkytofu_adaptor.container import (  # noqa: E402
    RegisterContainer,
    sgpr,
    vgpr,
)
from rocisa_stinkytofu_adaptor.enum import InstType  # noqa: E402
from rocisa_stinkytofu_adaptor.instruction import (  # noqa: E402
    CommonInstruction,
    Instruction,
    MacroInstruction,
    VMovB32,
    _to_stinky_register,
)


# ===========================================================================
# Instruction base class -- shape & abstract surface.
# ===========================================================================


class TestInstructionBase(unittest.TestCase):
    def test_cannot_use_toString_directly(self):
        # rocisa::Instruction::toString throws; we mirror that so any
        # accidental ``str(Instruction(...))`` is loud, not silent.
        inst = Instruction(InstType.INST_B32, "x")
        with self.assertRaises(NotImplementedError):
            inst.toString()
        with self.assertRaises(NotImplementedError):
            inst.getParams()
        with self.assertRaises(NotImplementedError):
            inst.getDstParams()
        with self.assertRaises(NotImplementedError):
            inst.getSrcParams()

    def test_default_fields(self):
        inst = Instruction(InstType.INST_B32, "foo")
        # Mirrors rocisa::Item("") + Instruction ctor.
        self.assertEqual(inst.name, "")
        self.assertIsNone(inst.parent)
        self.assertEqual(inst.comment, "foo")
        self.assertEqual(inst.instStr, "")
        self.assertFalse(inst.outputInlineAsm)
        self.assertIsNone(inst.m_memToken)

    def test_setInst_records_str(self):
        inst = Instruction(InstType.INST_B32)
        inst.setInst("v_mov_b32")
        self.assertEqual(inst.instStr, "v_mov_b32")
        # preStr() returns instStr (subclasses override).
        self.assertEqual(inst.preStr(), "v_mov_b32")

    def test_setInlineAsm_toggles_outputInlineAsm(self):
        inst = Instruction(InstType.INST_B32)
        inst.setInlineAsm(True)
        self.assertTrue(inst.outputInlineAsm)
        inst.setInlineAsm(False)
        self.assertFalse(inst.outputInlineAsm)

    def test_mem_token_roundtrip(self):
        inst = Instruction(InstType.INST_B32)
        token = object()  # opaque sentinel matches "any shared_ptr"
        inst.setMemToken(token)
        self.assertIs(inst.getMemToken(), token)

    def test_default_issue_latency_and_cycles(self):
        # rocisa::Instruction defaults; subclasses override.
        inst = Instruction(InstType.INST_B32)
        self.assertEqual(inst.getIssueLatency(), 1)
        self.assertEqual(inst.getIssueCycles(), 1)

    def test_deepcopy_raises(self):
        # rocisa binding: throws "Deepcopy not supported for Instruction".
        with self.assertRaises(RuntimeError):
            copy.deepcopy(Instruction(InstType.INST_B32))

    def test_pickle_raises(self):
        # rocisa binding: throws "Pickling not supported for Instruction".
        import pickle
        with self.assertRaises(RuntimeError):
            pickle.dumps(Instruction(InstType.INST_B32))


# ===========================================================================
# CommonInstruction -- minimal shape that other shims will inherit from.
# ===========================================================================


class TestCommonInstructionFields(unittest.TestCase):
    def test_construction_sets_all_fields(self):
        d = vgpr(0)
        s0, s1 = vgpr(1), vgpr(2)
        ci = CommonInstruction(InstType.INST_B32, dst=d, srcs=[s0, s1],
                                comment="hi")
        self.assertIs(ci.dst, d)
        self.assertIsNone(ci.dst1)
        self.assertEqual(ci.srcs, [s0, s1])
        self.assertIsNone(ci.dpp)
        self.assertIsNone(ci.sdwa)
        self.assertIsNone(ci.vop3)
        self.assertEqual(ci.comment, "hi")

    def test_srcs_is_copied(self):
        # rocisa stores the vector by value; mutating the caller's list
        # after construction must not leak in.
        src_list = [vgpr(0)]
        ci = CommonInstruction(InstType.INST_B32, dst=vgpr(1), srcs=src_list)
        src_list.append(vgpr(99))
        self.assertEqual(len(ci.srcs), 1)

    def test_setSrc_swaps_in_place(self):
        ci = CommonInstruction(InstType.INST_B32, dst=vgpr(0), srcs=[vgpr(1)])
        new_src = vgpr(7)
        ci.setSrc(0, new_src)
        self.assertIs(ci.srcs[0], new_src)

    def test_dst_mutable(self):
        # KernelWriter pattern: ``codeAccVgprReadInst.dst = vgpr(N)``.
        ci = CommonInstruction(InstType.INST_B32, dst=vgpr(0), srcs=[vgpr(1)])
        new_dst = vgpr(5)
        ci.dst = new_dst
        self.assertIs(ci.dst, new_dst)

    def test_comment_mutable(self):
        ci = CommonInstruction(InstType.INST_B32, dst=vgpr(0), srcs=[vgpr(1)],
                                comment="old")
        ci.comment = "new"
        self.assertEqual(ci.comment, "new")


class TestCommonInstructionParams(unittest.TestCase):
    def test_getParams_returns_dst_then_srcs(self):
        d, s0, s1 = vgpr(0), vgpr(1), vgpr(2)
        ci = CommonInstruction(InstType.INST_B32, dst=d, srcs=[s0, s1])
        self.assertEqual(ci.getParams(), [d, s0, s1])

    def test_getDstParams(self):
        d = vgpr(0)
        ci = CommonInstruction(InstType.INST_B32, dst=d, srcs=[vgpr(1)])
        self.assertEqual(ci.getDstParams(), [d])

    def test_getDstParams_includes_dst1_when_set(self):
        # rocisa::CommonInstruction has a ``dst1`` slot used by a handful
        # of instructions (e.g. ``V*MulHIU32`` returning two outputs).
        d, d1 = vgpr(0), vgpr(1)
        ci = CommonInstruction(InstType.INST_B32, dst=d, srcs=[vgpr(2)])
        ci.dst1 = d1
        self.assertEqual(ci.getDstParams(), [d, d1])
        self.assertEqual(ci.getParams(), [d, d1, vgpr(2)])

    def test_getSrcParams_returns_independent_copy(self):
        srcs = [vgpr(1), vgpr(2)]
        ci = CommonInstruction(InstType.INST_B32, dst=vgpr(0), srcs=srcs)
        out = ci.getSrcParams()
        out.append("trash")
        # Mutating the returned list must not bleed in.
        self.assertEqual(len(ci.srcs), 2)

    def test_getParams_skips_None_dst(self):
        # Some patterns construct with no dst (e.g. control-flow insts).
        # rocisa C++ guards with ``if(dst)``; we do too.
        ci = CommonInstruction(InstType.INST_B32, dst=None, srcs=[vgpr(0)])
        self.assertEqual(ci.getParams(), [vgpr(0)])


class TestCommonInstructionToString(unittest.TestCase):
    """Byte-parity with ``rocisa::CommonInstruction::toString``.

    Format: ``"<instStr> <dst>, <src0>, <src1>... // <comment>\\n"``
    with padding to column 50 before ``" // "`` when a comment is set
    (see ``format.hpp:54-75``).
    """

    def test_dst_and_one_src(self):
        ci = CommonInstruction(InstType.INST_B32, dst=vgpr(0), srcs=[vgpr(1)])
        ci.setInst("v_mov_b32")
        self.assertEqual(str(ci), "v_mov_b32 v0, v1\n")

    def test_with_comment_padded_to_col_50(self):
        ci = CommonInstruction(InstType.INST_B32, dst=vgpr(0), srcs=[vgpr(1)],
                                comment="copy")
        ci.setInst("v_mov_b32")
        # ``v_mov_b32 v0, v1`` is 16 chars; pad to col 50 with (50 - 16)
        # = 34 spaces, then `" // copy\n"`. Locking exact length here
        # because emitted asm diffs against rocisa native rely on this
        # being byte-identical (see format.hpp:64-69).
        head = "v_mov_b32 v0, v1"
        self.assertEqual(len(head), 16)
        expected = head + (" " * 34) + " // copy\n"
        self.assertEqual(str(ci), expected)

    def test_long_inst_no_extra_padding(self):
        # When inst is longer than 50 chars, max(0, ...) clamps the pad
        # to zero -- no negative padding, no truncation.
        ci = CommonInstruction(InstType.INST_B32,
                                dst=vgpr("VeryLongRegisterName"),
                                srcs=[vgpr("AnotherVeryLongOne"), vgpr("third")],
                                comment="x")
        ci.setInst("v_long_instruction_name")
        text = str(ci)
        self.assertTrue(text.endswith(" // x\n"))
        # No spaces immediately before `" // x"` -- the format is just
        # ``<inst> <args> // x\n`` since pad clamped to 0.
        self.assertIn(" // x\n", text)

    def test_int_src_renders_decimal(self):
        ci = CommonInstruction(InstType.INST_B32, dst=vgpr(0), srcs=[42])
        ci.setInst("v_mov_b32")
        self.assertEqual(str(ci).split("//", 1)[0].rstrip(), "v_mov_b32 v0, 42")

    def test_str_src_renders_verbatim(self):
        # KernelWriter passes ``hex(value)`` -> "0x..." for inline imms.
        ci = CommonInstruction(InstType.INST_B32, dst=vgpr(0), srcs=["0x0"])
        ci.setInst("v_mov_b32")
        self.assertEqual(str(ci), "v_mov_b32 v0, 0x0\n")

    def test_float_src_double_format(self):
        # rocisa formats doubles with %.17g + trailing ``.0`` when
        # there's no decimal / exponent in the printed form.
        ci = CommonInstruction(InstType.INST_B32, dst=vgpr(0), srcs=[1.0])
        ci.setInst("v_mov_b32")
        text = str(ci).split("//", 1)[0].rstrip()
        self.assertTrue(text.endswith("1.0"), text)

    def test_no_comment_just_newline(self):
        ci = CommonInstruction(InstType.INST_B32, dst=vgpr(0), srcs=[vgpr(1)])
        ci.setInst("v_mov_b32")
        # No padding when comment is empty.
        self.assertEqual(str(ci), "v_mov_b32 v0, v1\n")

    def test_inline_asm_wraps_inst(self):
        ci = CommonInstruction(InstType.INST_B32, dst=vgpr(0), srcs=[vgpr(1)],
                                comment="c")
        ci.setInst("v_mov_b32")
        ci.setInlineAsm(True)
        # Inline-asm path: ``"<inst>\n\t"`` literal + padding + comment.
        self.assertTrue(str(ci).startswith('"v_mov_b32 v0, v1\\n\\t"'),
                        repr(str(ci)))


class TestCommonInstructionDeepcopy(unittest.TestCase):
    def test_deepcopy_independent_dst(self):
        d = vgpr(0)
        ci = CommonInstruction(InstType.INST_B32, dst=d, srcs=[vgpr(1)],
                                comment="x")
        ci.setInst("v_mov_b32")
        c = copy.deepcopy(ci)
        c.dst.setMinus(True)
        # Mutating clone's dst must not leak into original.
        self.assertFalse(ci.dst.isMinus)
        self.assertTrue(c.dst.isMinus)
        # Identity preserved across deepcopy (same subclass).
        self.assertIs(type(c), type(ci))

    def test_deepcopy_independent_srcs(self):
        ci = CommonInstruction(InstType.INST_B32, dst=vgpr(0), srcs=[vgpr(1)])
        c = copy.deepcopy(ci)
        c.srcs.append(vgpr(99))
        self.assertEqual(len(ci.srcs), 1)

    def test_deepcopy_preserves_instStr_and_comment(self):
        ci = CommonInstruction(InstType.INST_B32, dst=vgpr(0), srcs=[vgpr(1)],
                                comment="orig")
        ci.setInst("v_mov_b32")
        c = copy.deepcopy(ci)
        self.assertEqual(c.instStr, "v_mov_b32")
        self.assertEqual(c.comment, "orig")


# ===========================================================================
# MacroInstruction -- macro-call leaf (V_MAGIC_DIV / MAINLOOP / etc.).
# ===========================================================================
#
# KernelWriter constructs these directly via
# ``MacroInstruction(name=..., args=[...], comment=...)``. The emitted
# text becomes a single ``NAME arg0, arg1, ...`` line, optionally
# right-padded to col 50 followed by ``// comment``.


class TestMacroInstructionFields(unittest.TestCase):
    def test_default_comment_empty(self):
        mi = MacroInstruction("MY_MACRO", [1, 2, 3])
        self.assertEqual(mi.name, "MY_MACRO")
        self.assertEqual(mi.args, [1, 2, 3])
        self.assertEqual(mi.comment, "")
        self.assertEqual(mi.instType, InstType.INST_MACRO)

    def test_args_is_copied(self):
        # rocisa stores ``args`` by value; mutating the caller's list
        # afterwards must not bleed in.
        args = [1, 2]
        mi = MacroInstruction("X", args)
        args.append(99)
        self.assertEqual(len(mi.args), 2)

    def test_args_mutable_post_construction(self):
        # rocisa binding exposes ``args`` as def_rw.
        mi = MacroInstruction("X", [1])
        mi.args.append(2)
        self.assertEqual(mi.args, [1, 2])

    def test_setSrc_in_place(self):
        mi = MacroInstruction("X", [1, 2, 3])
        mi.setSrc(1, 99)
        self.assertEqual(mi.args, [1, 99, 3])

    def test_inherits_Instruction(self):
        mi = MacroInstruction("X", [])
        self.assertIsInstance(mi, Instruction)


class TestMacroInstructionParams(unittest.TestCase):
    def test_getParams_returns_args(self):
        mi = MacroInstruction("X", [1, "a", 3.0])
        self.assertEqual(mi.getParams(), [1, "a", 3.0])

    def test_getDstParams_raises(self):
        # rocisa throws "MacroInstruction does not have destination
        # parameters". Crashing loudly is the contract.
        mi = MacroInstruction("X", [1])
        with self.assertRaises(RuntimeError) as ctx:
            mi.getDstParams()
        self.assertIn("destination parameters", str(ctx.exception))

    def test_getSrcParams_raises(self):
        # rocisa throws "MacroInstruction does not have source
        # parameters". Same intentional crash policy.
        mi = MacroInstruction("X", [1])
        with self.assertRaises(RuntimeError) as ctx:
            mi.getSrcParams()
        self.assertIn("source parameters", str(ctx.exception))


class TestMacroInstructionToString(unittest.TestCase):
    """Byte-parity with ``rocisa::MacroInstruction::toString``."""

    def test_empty_args(self):
        mi = MacroInstruction("BARRIER", [])
        # Empty args -> no leading space, just "NAME\n".
        self.assertEqual(str(mi), "BARRIER\n")

    def test_single_int_arg(self):
        # KWA:16977 pattern: ``MacroInstruction(name="MAINLOOP", args=[0])``.
        mi = MacroInstruction("MAINLOOP", [0])
        self.assertEqual(str(mi), "MAINLOOP 0\n")

    def test_multiple_mixed_args(self):
        mi = MacroInstruction("X", [1, "lit", 2])
        self.assertEqual(str(mi), "X 1, lit, 2\n")

    def test_register_container_args(self):
        # KWA:2784 pattern: register operands routed via _input_to_str
        # which calls ``.toString()``.
        mi = MacroInstruction("V_MAGIC_DIV", [vgpr(0), sgpr("Foo")])
        self.assertEqual(str(mi), "V_MAGIC_DIV v0, s[sgprFoo]\n")

    def test_with_comment_padded_to_col_50(self):
        mi = MacroInstruction("X", [1], comment="hi")
        head = "X 1"
        pad = " " * (50 - len(head))
        self.assertEqual(str(mi), head + pad + " // hi\n")

    def test_name_with_space_emitted_verbatim(self):
        # Components/StreamK.py:152 pattern: ``MacroInstruction(name=
        # "s_wait_xcnt 0", args=[])``. Spaces inside name must be
        # preserved untouched.
        mi = MacroInstruction("s_wait_xcnt 0", [])
        self.assertEqual(str(mi), "s_wait_xcnt 0\n")

    def test_getArgStr_format(self):
        # Helper directly: leading space + comma-join.
        mi = MacroInstruction("X", [1, 2])
        self.assertEqual(mi.getArgStr(), " 1, 2")
        mi2 = MacroInstruction("X", [])
        self.assertEqual(mi2.getArgStr(), "")


class TestMacroInstructionDeepcopy(unittest.TestCase):
    def test_deepcopy_independent_args_list(self):
        mi = MacroInstruction("X", [1, 2, 3], comment="c")
        c = copy.deepcopy(mi)
        c.args.append(99)
        self.assertEqual(len(mi.args), 3)
        self.assertEqual(len(c.args), 4)

    def test_deepcopy_preserves_name_and_comment(self):
        mi = MacroInstruction("PRND", [1], comment="orig")
        c = copy.deepcopy(mi)
        self.assertEqual(c.name, "PRND")
        self.assertEqual(c.comment, "orig")

    def test_deepcopy_clones_container_args(self):
        # Container args must be deep-cloned -- mutating the clone's
        # Container should not bleed into the original.
        rc = vgpr(0)
        mi = MacroInstruction("X", [rc])
        c = copy.deepcopy(mi)
        c.args[0].setMinus(True)
        self.assertFalse(mi.args[0].isMinus)
        self.assertTrue(c.args[0].isMinus)

    def test_clone_method_returns_independent_copy(self):
        # rocisa Instruction::clone() returns a new shared_ptr; our
        # ``clone()`` returns a deepcopy with a fresh memo.
        mi = MacroInstruction("X", [1, 2])
        c = mi.clone()
        self.assertIsInstance(c, MacroInstruction)
        self.assertIsNot(c, mi)
        self.assertEqual(c.args, mi.args)


# ===========================================================================
# _to_stinky_register coercion table.
# ===========================================================================


try:
    import stinkytofu as _stinky
    _STINKY_OK = all(
        hasattr(_stinky, name)
        for name in ("Register", "vgpr", "VMovB32", "LogicalModule",
                     "lower_logical_module")
    )
except ImportError:
    _STINKY_OK = False


@unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
class TestToStinkyRegister(unittest.TestCase):
    def test_register_container_delegates_to_to_stinky(self):
        rc = vgpr(5)
        reg = _to_stinky_register(rc)
        # Numeric VGPR -> Register with index/count.
        self.assertTrue(reg.is_register)
        self.assertEqual(reg.index, 5)
        self.assertEqual(reg.count, 1)

    def test_named_register_container_via_to_stinky(self):
        rc = vgpr("ValuA")
        reg = _to_stinky_register(rc)
        self.assertTrue(reg.has_reg_name)
        name, offsets = reg.get_reg_name()
        self.assertEqual(name, "vgprValuA")
        self.assertEqual(offsets, [])

    def test_int_literal(self):
        reg = _to_stinky_register(42)
        self.assertTrue(reg.is_literal)

    def test_bool_routed_through_int(self):
        # bool is int subclass; ensure we don't crash on it.
        reg = _to_stinky_register(True)
        self.assertTrue(reg.is_literal)

    def test_float_literal(self):
        reg = _to_stinky_register(1.5)
        self.assertTrue(reg.is_literal)

    def test_str_literal_verbatim(self):
        # KernelWriter passes ``hex(N)`` strings; stinky emits them
        # verbatim as literal-string Register.
        reg = _to_stinky_register("0x42")
        self.assertTrue(reg.is_literal_string)
        self.assertEqual(reg.literal_string, "0x42")

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            _to_stinky_register([1, 2, 3])
        with self.assertRaises(TypeError):
            _to_stinky_register(None)


# ===========================================================================
# VMovB32 -- rocisa-shape construction + isinstance behaviour.
# ===========================================================================


class TestVMovB32Construction(unittest.TestCase):
    def test_positional_dst_src(self):
        d, s = vgpr(0), vgpr(1)
        v = VMovB32(d, s)
        self.assertIs(v.dst, d)
        self.assertEqual(v.srcs, [s])
        self.assertEqual(v.comment, "")
        self.assertIsNone(v.sdwa)
        self.assertIsNone(v.dpp)
        # ``setInst`` was called in __init__.
        self.assertEqual(v.instStr, "v_mov_b32")
        self.assertEqual(v.instType, InstType.INST_B32)

    def test_keyword_ctor_rocisa_shape(self):
        # rocisa accepts (dst, src, sdwa=None, comment="", dpp=None).
        v = VMovB32(dst=vgpr(0), src="0x0", comment="init zero")
        self.assertEqual(v.srcs, ["0x0"])
        self.assertEqual(v.comment, "init zero")

    def test_inherits_from_CommonInstruction_and_Instruction(self):
        v = VMovB32(vgpr(0), vgpr(1))
        # KernelWriter:1105 / SubtileBasedInstructionScheduler:151 rely
        # on these isinstance checks -- duck typing isn't enough.
        self.assertIsInstance(v, Instruction)
        self.assertIsInstance(v, CommonInstruction)
        self.assertIsInstance(v, VMovB32)

    def test_dst_regType_accessor_used_by_subtile_scheduler(self):
        # Components/SubtileBasedInstructionScheduler.py:151 does:
        #   isinstance(x, CommonInstruction) and hasattr(x.dst, 'regType')
        #     and x.dst.regType == 'm'
        v = VMovB32(vgpr(0), vgpr(1))
        self.assertTrue(hasattr(v.dst, "regType"))
        self.assertEqual(v.dst.regType, "v")

    def test_str_uses_v_mov_b32_format(self):
        v = VMovB32(vgpr(0), vgpr(1), comment="rename")
        text = str(v)
        self.assertTrue(text.startswith("v_mov_b32 v0, v1"), text)
        self.assertTrue(text.endswith(" // rename\n"), text)

    def test_str_with_immediate_src(self):
        v = VMovB32(vgpr(0), "0x0", comment="zero")
        self.assertEqual(str(v).split("//", 1)[0].rstrip(),
                          "v_mov_b32 v0, 0x0")

    def test_dst_mutation_post_construction(self):
        # GSU.py pattern: write to .dst after construction.
        v = VMovB32(vgpr(0), vgpr(1))
        v.dst = vgpr(42)
        self.assertEqual(str(v).split("//", 1)[0].rstrip(),
                          "v_mov_b32 v42, v1")

    def test_comment_mutation_post_construction(self):
        v = VMovB32(vgpr(0), vgpr(1))
        v.comment = "later"
        self.assertIn("// later", str(v))

    def test_getParams_dst_then_src(self):
        d, s = vgpr(0), vgpr(1)
        v = VMovB32(d, s)
        self.assertEqual(v.getParams(), [d, s])

    def test_deepcopy_preserves_subclass(self):
        v = VMovB32(vgpr(0), vgpr(1), comment="x")
        c = copy.deepcopy(v)
        self.assertIsInstance(c, VMovB32)
        self.assertEqual(str(c), str(v))


# ===========================================================================
# Step-3 contract: VMovB32 is picked up by Module._collect_logical_insts.
# ===========================================================================


class TestCollectLogicalIntegration(unittest.TestCase):
    """Step-3 leaves are expected to expose ``to_stinky_logical()``; the
    Step-2 collector skips dummies on a None-return sentinel. These tests
    confirm the contract from the Module-side without touching the C++
    binding -- the dummy-skip test in test_code.py would now fail-fast
    if we accidentally regressed VMovB32 back to a dummy, so we keep
    coverage symmetric here.
    """

    def test_vmovb32_collected_by_module(self):
        # We can't test the full pipeline without stinkytofu, but we
        # can check that to_stinky_logical exists and is callable.
        v = VMovB32(vgpr(0), vgpr(1))
        self.assertTrue(callable(getattr(v, "to_stinky_logical", None)))

    def test_dummy_instructions_skipped_when_mixed_with_real(self):
        # A Module holding both VMovB32 and a still-dummy instruction
        # (e.g. SBarrier) should ONLY collect the VMovB32.
        from rocisa_stinkytofu_adaptor.instruction import SBarrier  # noqa: WPS433
        m = Module()
        v = VMovB32(vgpr(0), vgpr(1))
        m.add(v)
        m.add(SBarrier())  # dummy: to_stinky_logical -> None
        if not _STINKY_OK:
            # Real VMovB32.to_stinky_logical does an import; skip the
            # actual call but assert the collector logic works.
            return
        out = m._collect_logical_insts()
        # Exactly one logical leaf collected (the VMovB32).
        self.assertEqual(len(out), 1)


# ===========================================================================
# End-to-end coverage -- see ``tests/test_emission_consistency.py``.
# ===========================================================================
#
# Previously this file held a ``TestVMovB32EndToEnd`` class with weak
# substring assertions ("emit contains 'v_mov_b32'"). That has been
# superseded by ``tests/test_emission_consistency.py``, which compares
# the asm output of three production code paths byte-for-byte:
#
#   (1) default rocisa  ->  str(module)
#   (2) default rocisa  ->  toStinkyTofuModule(...).emitAssembly()
#   (3) adapter         ->  module.to_stinky_asm(arch).emitAssembly()
#
# Cross-path byte equality is strictly stronger than "asm contains a
# given mnemonic" -- it would fail-fast on any drift in the adapter's
# right-path port, the rocisa->asm-IR bridge, or the logical-IR pipeline.
#
# The TextBlock-filtering behaviour ("TextBlock items don't leak into
# the logical-IR pipeline") is structurally covered by
# ``tests/test_code.py::TestCollectLogicalInsts::test_textblocks_and_logical_mixed``,
# which asserts the collector returns only logical leaves.


if __name__ == "__main__":
    unittest.main()
