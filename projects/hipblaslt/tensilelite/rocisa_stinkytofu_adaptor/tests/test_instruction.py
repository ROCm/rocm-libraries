# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Standalone tests for ``rocisa_stinkytofu_adaptor.instruction`` -- Step 3
(``Instruction`` / ``CommonInstruction`` bases + ``VMovB32`` / ``SMovB32`` /
``SMovB64`` / ``SMemLoadInstruction`` / ``SLoadB*`` shims).

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
  * Step-3 collector contract: ``VMovB32`` / ``SMovB32`` ``to_stinky_logical``
    is callable and, when stinkytofu is built, is picked up by
    ``Module._collect_logical_insts``.
    Dummy-only skip behaviour (e.g. ``SBarrier``) lives in
    ``test_code.TestCollectLogicalInsts``.

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
    SMEMModifiers,
    sgpr,
    vgpr,
)
from rocisa_stinkytofu_adaptor.enum import InstType  # noqa: E402
from rocisa_stinkytofu_adaptor.instruction import (  # noqa: E402
    CommonInstruction,
    Instruction,
    MacroInstruction,
    SAddI32,
    SAddU32,
    SAddCU32,
    SAndB32,
    SAndB64,
    SAndN2B32,
    SAndSaveExecB32,
    SAndSaveExecB64,
    SAShiftRightI32,
    SBarrier,
    SBitcmp1B32,
    SCmpEQI32,
    SCmpEQU32,
    SCmpEQU64,
    SCmpGeI32,
    SCmpGeU32,
    SCmpGtI32,
    SCmpGtU32,
    SCmpLeI32,
    SCmpLeU32,
    SCmpLgI32,
    SCmpLgU32,
    SCmpLgU64,
    SCmpLtI32,
    SCmpLtU32,
    SGetRegB32,
    SLShiftLeftB32,
    SLShiftRightB32,
    SLShiftLeftB64,
    SLShiftRightB64,
    SLShiftLeft1AddU32,
    SLShiftLeft2AddU32,
    SLShiftLeft3AddU32,
    SLShiftLeft4AddU32,
    SLoadB32,
    SLoadB64,
    SMemLoadInstruction,
    SMovB32,
    SMovB64,
    SMulHII32,
    SMulHIU32,
    SMulI32,
    SMulLOU32,
    SNop,
    SOrB32,
    SOrB64,
    SOrSaveExecB32,
    SOrSaveExecB64,
    SSubBU32,
    SSubI32,
    SSubU32,
    SXorB32,
    VAddF32,
    VAddU32,
    VAndB32,
    VAndOrB32,
    VCmpClassF32,
    VCmpEQF32,
    VCmpEQF64,
    VCmpEQI32,
    VCmpEQU32,
    VCmpGEF16,
    VCmpGEF32,
    VCmpGEF64,
    VCmpGEI32,
    VCmpGEU32,
    VCmpGTF16,
    VCmpGTF32,
    VCmpGTF64,
    VCmpGTI32,
    VCmpGtU32,
    VCmpLeI32,
    VCmpLeU32,
    VCmpLtI32,
    VCmpLtU32,
    VCmpNeI32,
    VCmpNeU32,
    VCmpNeU64,
    VCmpUF32,
    VCmpXClassF32,
    VCmpXEqU32,
    VCmpXGeU32,
    VCmpXGtU32,
    VCmpXLeI32,
    VCmpXLeU32,
    VCmpXLtF32,
    VCmpXLtI32,
    VCmpXLtU32,
    VCmpXLtU64,
    VCmpXNeU16,
    VCmpXNeU32,
    VCndMaskB32,
    VFmaF32,
    VFmaMixF32,
    VLShiftLeftB32,
    VLShiftLeftB64,
    VLShiftRightB32,
    VLShiftRightB64,
    VMovB32,
    VMulF32,
    VMulHII32,
    VMulHIU32,
    VMulI32I24,
    VMulLOU32,
    VMulU32U24,
    VOrB32,
    VReadfirstlaneB32,
    VSubF32,
    VSubI32,
    VSubU32,
    VXorB32,
    SAbsI32,
    SMaxI32,
    SMaxU32,
    SMinI32,
    SMinU32,
    VExpF16,
    VExpF32,
    VRcpF16,
    VRcpF32,
    VRcpIFlagF32,
    VRsqF16,
    VRsqF32,
    VRsqIFlagF32,
    VMaxF16,
    VMaxF32,
    VMaxF64,
    VMaxI32,
    VMaxPKF16,
    VMinF16,
    VMinF32,
    VMinF64,
    VMinI32,
    VMed3I32,
    VMed3F32,
    VNotB32,
    VPrngB32,
    VRndneF32,
    VAShiftRightI32,
    VPackF16toB32,
    VLShiftLeftOrB32,
    VCvtF16toF32,
    VCvtF32toF16,
    VCvtF32toU32,
    VCvtU32toF32,
    VCvtI32toF32,
    VCvtF32toI32,
    VCvtFP8toF32,
    VCvtBF8toF32,
    VCvtPkFP8toF32,
    VCvtPkBF8toF32,
    VCvtPkF32toBF8,
    VCvtSRF32toFP8,
    VCvtSRF32toBF8,
    VCvtPkF32toFP8,
    VCvtPkF32toBF16,
    VCvtScalePkFP8toF16,
    VCvtScalePkBF8toF16,
    VCvtScaleFP8toF16,
    VCvtScalePkF16toFP8,
    VCvtScalePkF16toBF8,
    VCvtScaleSRF16toFP8,
    VCvtScaleSRF16toBF8,
    BufferLoadU8,
    BufferLoadD16HIU8,
    BufferLoadD16U8,
    BufferLoadD16HIB16,
    BufferLoadD16B16,
    BufferLoadB32,
    BufferLoadB64,
    BufferLoadB96,
    BufferLoadB128,
    BufferStoreB8,
    BufferStoreD16HIU8,
    BufferStoreD16HIB16,
    BufferStoreB16,
    BufferStoreB32,
    BufferStoreB64,
    BufferStoreB128,
    BufferAtomicAddF32,
    BufferAtomicCmpswapB32,
    BufferAtomicCmpswapB64,
    FlatLoadD16HIU8,
    FlatLoadD16U8,
    FlatLoadD16HIB16,
    FlatLoadD16B16,
    FlatLoadB32,
    FlatLoadB64,
    FlatLoadB128,
    FlatStoreD16HIB16,
    FlatStoreB32,
    FlatStoreB64,
    FlatStoreB128,
    FlatAtomicCmpswapB32,
    DSLoadU8,
    DSLoadU16,
    DSLoadB32,
    DSLoadB64,
    DSLoadB128,
    DSLoad2B32,
    DSLoad2B64,
    DSStoreB8,
    DSStoreB16,
    DSStoreB32,
    DSStoreB64,
    DSStoreB96,
    DSStoreB128,
    DSStore2B32,
    DSStore2B64,
    DSBPermuteB32,
    TensorLoadToLds,
    BranchInstruction,
    SBranch,
    SCBranchSCC0,
    SCBranchSCC1,
    SCBranchVCCNZ,
    SCBranchVCCZ,
    SCBranchExecZ,
    SCBranchExecNZ,
    SEndpgm,
    _SWaitCnt,
    SWaitCnt,
    SWaitXCnt,
    SWaitTensorcnt,
    SWaitAlu,
    SSchedulingFence,
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
        for name in ("Register", "vgpr", "VMovB32", "SNop",
                     "SAddU32", "SAndB32", "SLShiftLeftB32",
                     "SBarrier", "SGetRegB32",
                     "LogicalModule", "lower_logical_module")
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
# SMovB32 / SMovB64 -- scalar moves (Phase A).
# ===========================================================================


class TestSMovB32Construction(unittest.TestCase):
    def test_positional_dst_src(self):
        d, s = sgpr(0), sgpr(1)
        m = SMovB32(d, s)
        self.assertIs(m.dst, d)
        self.assertEqual(m.srcs, [s])
        self.assertEqual(m.comment, "")
        self.assertEqual(m.instStr, "s_mov_b32")
        self.assertEqual(m.instType, InstType.INST_B32)

    def test_str_sgpr_to_sgpr(self):
        m = SMovB32(sgpr(0), sgpr(1), comment="probe")
        text = str(m)
        self.assertTrue(text.startswith("s_mov_b32 s0, s1"), text)
        self.assertTrue(text.endswith(" // probe\n"), text)

    def test_inherits_common_instruction(self):
        m = SMovB32(sgpr(0), sgpr(1))
        self.assertIsInstance(m, Instruction)
        self.assertIsInstance(m, CommonInstruction)
        self.assertIsInstance(m, SMovB32)

    def test_deepcopy(self):
        m = SMovB32(sgpr(0), sgpr(1), comment="x")
        c = copy.deepcopy(m)
        self.assertIsInstance(c, SMovB32)
        self.assertEqual(str(c), str(m))


class TestSMovB64Construction(unittest.TestCase):
    def test_inst_type_b64(self):
        m = SMovB64(sgpr(0, 2), sgpr(4, 2), comment="wide")
        self.assertEqual(m.instStr, "s_mov_b64")
        self.assertEqual(m.instType, InstType.INST_B64)

    def test_str_contains_mnemonic(self):
        m = SMovB64(sgpr(0, 2), sgpr(4, 2))
        self.assertIn("s_mov_b64", str(m))


# ===========================================================================
# SNop
# ===========================================================================


class TestSNopConstruction(unittest.TestCase):
    def test_inst_type_notype(self):
        m = SNop(waitState=0)
        self.assertEqual(m.instStr, "s_nop")
        self.assertEqual(m.instType, InstType.INST_NOTYPE)

    def test_str_wait_and_comment(self):
        m = SNop(waitState=3, comment="pad")
        text = str(m)
        self.assertIn("s_nop", text)
        self.assertIn("3", text)
        self.assertIn("pad", text)

    def test_get_params(self):
        m = SNop(waitState=7)
        self.assertEqual(m.getParams(), [7])
        self.assertEqual(m.getDstParams(), [])
        self.assertEqual(m.getSrcParams(), [7])

    def test_to_stinky_logical(self):
        m = SNop(waitState=2, comment="c")
        if hasattr(m, "to_stinky_logical"):
            name_fn = getattr(m, "getOpcodeName", None)
            if name_fn:
                self.assertEqual(name_fn(), "SNop")

    def test_deepcopy(self):
        m = SNop(waitState=1, comment="x")
        c = copy.deepcopy(m)
        self.assertIsInstance(c, SNop)
        self.assertIsNot(c, m)
        self.assertEqual(c.wait_state, 1)
        self.assertEqual(c.comment, "x")

    def test_positional_arg(self):
        """Supports SNop(3, 'comment') positional style used in Tensile."""
        m = SNop(3, "delay")
        self.assertEqual(m.wait_state, 3)
        self.assertEqual(m.comment, "delay")


# ===========================================================================
# SMemLoadInstruction / SLoadB32 / SLoadB64
# ===========================================================================


class TestSLoadB32Construction(unittest.TestCase):
    def test_is_smem_load_not_common(self):
        m = SLoadB32(sgpr(0), sgpr(2, 2), 0, comment="k")
        self.assertIsInstance(m, Instruction)
        self.assertIsInstance(m, SMemLoadInstruction)
        self.assertIsInstance(m, SLoadB32)
        self.assertNotIsInstance(m, CommonInstruction)

    def test_str_basic(self):
        m = SLoadB32(sgpr(0), sgpr(2, 2), 0, comment="probe")
        text = str(m)
        self.assertTrue(text.startswith("s_load_b32 s0, s[2:3], 0"), text)
        self.assertTrue(text.endswith(" // probe\n"), text)

    def test_get_params_order(self):
        d, b = sgpr(0), sgpr(2, 2)
        m = SLoadB32(d, b, 4)
        p = m.getParams()
        self.assertEqual(len(p), 3)
        self.assertIs(p[0], d)
        self.assertIs(p[1], b)
        self.assertEqual(p[2], 4)

    def test_smem_modifier_raises_on_logical_bridge(self):
        m = SLoadB32(sgpr(0), sgpr(2, 2), 0, smem=SMEMModifiers(offset=4))
        with self.assertRaises(NotImplementedError):
            m.to_stinky_logical()

    def test_deepcopy(self):
        m = SLoadB32(sgpr(0), sgpr(2, 2), 0, comment="x")
        c = copy.deepcopy(m)
        self.assertIsInstance(c, SLoadB32)
        self.assertEqual(str(c), str(m))


class TestSLoadB64Construction(unittest.TestCase):
    def test_prestr_b64(self):
        m = SLoadB64(sgpr(0, 2), sgpr(4, 2), 0)
        self.assertIn("s_load_b64", str(m))


# ===========================================================================
# Step-3 contract: VMovB32 is picked up by Module._collect_logical_insts.
# ===========================================================================


class TestCollectLogicalIntegration(unittest.TestCase):
    """``VMovB32`` / ``SMovB32`` instruction-side contract for
    ``_collect_logical_insts``.

    Collector walk semantics (fake leaves, TextBlock filter, dummy skip)
    are owned by ``test_code.TestCollectLogicalInsts`` -- this class only
    pins that the real shims expose ``to_stinky_logical`` and are actually
    collected when the stinkytofu binding is importable.
    """

    def test_vmovb32_exposes_to_stinky_logical(self):
        v = VMovB32(vgpr(0), vgpr(1))
        self.assertTrue(callable(getattr(v, "to_stinky_logical", None)))

    def test_smovb32_exposes_to_stinky_logical(self):
        s = SMovB32(sgpr(0), sgpr(1))
        self.assertTrue(callable(getattr(s, "to_stinky_logical", None)))

    def test_smovb64_exposes_to_stinky_logical(self):
        s = SMovB64(sgpr(0, 2), sgpr(4, 2))
        self.assertTrue(callable(getattr(s, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_vmovb32_collected_by_module(self):
        m = Module()
        m.add(VMovB32(vgpr(0), vgpr(1)))
        self.assertEqual(len(m._collect_logical_insts()), 1)

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_smovb32_collected_by_module(self):
        m = Module()
        m.add(SMovB32(sgpr(0), sgpr(1)))
        self.assertEqual(len(m._collect_logical_insts()), 1)

    def test_snop_exposes_to_stinky_logical(self):
        n = SNop(waitState=0)
        self.assertTrue(callable(getattr(n, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_sloadb32_collected_by_module(self):
        m = Module()
        m.add(SLoadB32(sgpr(0), sgpr(2, 2), 0))
        self.assertEqual(len(m._collect_logical_insts()), 1)

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_snop_collected_by_module(self):
        m = Module()
        m.add(SNop(waitState=0))
        self.assertEqual(len(m._collect_logical_insts()), 1)


# ===========================================================================
# Scalar ALU instructions (Phase 6 Step 1)
# ===========================================================================

_SCALAR_ALU_BINARY = [
    # (class, mnemonic, InstType)
    (SAddI32, "s_add_i32", InstType.INST_I32),
    (SAddU32, "s_add_u32", InstType.INST_U32),
    (SAddCU32, "s_addc_u32", InstType.INST_U32),
    (SMulI32, "s_mul_i32", InstType.INST_I32),
    (SMulHII32, "s_mul_hi_i32", InstType.INST_I32),
    (SMulHIU32, "s_mul_hi_u32", InstType.INST_U32),
    (SMulLOU32, "s_mul_lo_u32", InstType.INST_U32),
    (SSubI32, "s_sub_i32", InstType.INST_I32),
    (SSubU32, "s_sub_u32", InstType.INST_U32),
    (SSubBU32, "s_subb_u32", InstType.INST_U32),
    (SLShiftLeftB32, "s_lshl_b32", InstType.INST_B32),
    (SLShiftRightB32, "s_lshr_b32", InstType.INST_B32),
    (SLShiftLeftB64, "s_lshl_b64", InstType.INST_B64),
    (SLShiftRightB64, "s_lshr_b64", InstType.INST_B64),
    (SAShiftRightI32, "s_ashr_i32", InstType.INST_I32),
    (SLShiftLeft1AddU32, "s_lshl1_add_u32", InstType.INST_U32),
    (SLShiftLeft2AddU32, "s_lshl2_add_u32", InstType.INST_U32),
    (SLShiftLeft3AddU32, "s_lshl3_add_u32", InstType.INST_U32),
    (SLShiftLeft4AddU32, "s_lshl4_add_u32", InstType.INST_U32),
    (SAndB32, "s_and_b32", InstType.INST_B32),
    (SAndB64, "s_and_b64", InstType.INST_B64),
    (SAndN2B32, "s_andn2_b32", InstType.INST_B32),
    (SOrB32, "s_or_b32", InstType.INST_B32),
    (SOrB64, "s_or_b64", InstType.INST_B64),
    (SXorB32, "s_xor_b32", InstType.INST_B32),
]

_SCALAR_ALU_UNARY = [
    (SAndSaveExecB32, "s_and_saveexec_b32", InstType.INST_B32),
    (SAndSaveExecB64, "s_and_saveexec_b64", InstType.INST_B64),
    (SOrSaveExecB32, "s_or_saveexec_b32", InstType.INST_B32),
    (SOrSaveExecB64, "s_or_saveexec_b64", InstType.INST_B64),
]


# ===========================================================================
# SBarrier / SGetRegB32 / SSetRegB32 / SSetRegIMM32B32 (Phase 6 Step 2)
# ===========================================================================


class TestSBarrierConstruction(unittest.TestCase):
    """SBarrier is a zero-operand instruction with stinkytofu bridge."""

    def test_default_construction(self):
        b = SBarrier()
        self.assertIsInstance(b, Instruction)
        self.assertEqual(b.instStr, "s_barrier")
        self.assertEqual(b.instType, InstType.INST_NOTYPE)

    def test_with_comment(self):
        b = SBarrier(comment="sync")
        text = str(b)
        self.assertIn("s_barrier", text)
        self.assertIn("sync", text)

    def test_deepcopy(self):
        b = SBarrier(comment="x")
        c = copy.deepcopy(b)
        self.assertIsInstance(c, SBarrier)
        self.assertEqual(str(c), str(b))

    def test_has_to_stinky_logical(self):
        b = SBarrier()
        self.assertTrue(callable(getattr(b, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        m = Module()
        m.add(SBarrier())
        self.assertEqual(len(m._collect_logical_insts()), 1)


class TestSGetRegB32Construction(unittest.TestCase):
    """SGetRegB32 is a unary scalar control instruction."""

    def test_construction(self):
        inst = SGetRegB32(dst=sgpr(0), src=sgpr(1))
        self.assertIsInstance(inst, CommonInstruction)
        self.assertEqual(inst.instStr, "s_getreg_b32")

    def test_str(self):
        inst = SGetRegB32(dst=sgpr(0), src=sgpr(1), comment="hwid")
        text = str(inst)
        self.assertIn("s_getreg_b32", text)
        self.assertIn("s0", text)

    def test_has_to_stinky_logical(self):
        inst = SGetRegB32(dst=sgpr(0), src=sgpr(1))
        self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        m = Module()
        m.add(SGetRegB32(dst=sgpr(0), src=sgpr(1)))
        self.assertEqual(len(m._collect_logical_insts()), 1)


class TestScalarALUBinaryConstruction(unittest.TestCase):
    """All binary scalar ALU shims inherit CommonInstruction and emit correct asm."""

    def test_construction_and_str(self):
        for cls, mnemonic, itype in _SCALAR_ALU_BINARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2), comment="t")
                self.assertIsInstance(inst, CommonInstruction)
                self.assertEqual(inst.instStr, mnemonic)
                self.assertEqual(inst.instType, itype)
                text = str(inst)
                self.assertTrue(text.startswith(mnemonic), text)
                self.assertIn("s0", text)
                self.assertIn("s1", text)
                self.assertIn("s2", text)

    def test_deepcopy(self):
        for cls, _, _ in _SCALAR_ALU_BINARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=sgpr(4), src0=sgpr(5), src1=sgpr(6), comment="cp")
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _, _ in _SCALAR_ALU_BINARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _, _ in _SCALAR_ALU_BINARY:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)))
                self.assertEqual(len(m._collect_logical_insts()), 1)

    def test_immediate_src(self):
        inst = SAddU32(dst=sgpr(0), src0=sgpr(1), src1=42)
        text = str(inst)
        self.assertIn("42", text)


class TestScalarALUUnaryConstruction(unittest.TestCase):
    """Unary scalar ALU shims (dst, src) inherit CommonInstruction."""

    def test_construction_and_str(self):
        for cls, mnemonic, itype in _SCALAR_ALU_UNARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=sgpr(0), src=sgpr(1), comment="u")
                self.assertIsInstance(inst, CommonInstruction)
                self.assertEqual(inst.instStr, mnemonic)
                self.assertEqual(inst.instType, itype)
                text = str(inst)
                self.assertTrue(text.startswith(mnemonic), text)

    def test_deepcopy(self):
        for cls, _, _ in _SCALAR_ALU_UNARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=sgpr(0), src=sgpr(1), comment="x")
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _, _ in _SCALAR_ALU_UNARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=sgpr(0), src=sgpr(1))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _, _ in _SCALAR_ALU_UNARY:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=sgpr(0), src=sgpr(1)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestDevelopInstructionExports(unittest.TestCase):
    """Develop-added rocisa instructions must resolve (dummy phase).

    Tensile imports these by name during module load; missing exports
    fail before any kernel is generated. Real ``toString`` / logicalIR
    parity is deferred to the instruction vertical-slice batch.
    """

    _NAMES = (
        "SMemAtomicIncInstruction",
        "SMemAtomicDecInstruction",
        "SMemStoreInstruction",
    )

    def test_all_develop_exports_importable(self):
        import rocisa_stinkytofu_adaptor.instruction as inst_mod

        for name in self._NAMES:
            with self.subTest(name=name):
                cls = getattr(inst_mod, name)
                obj = cls()
                self.assertIn("DummyShim", repr(obj))


# ===========================================================================
# Vector ALU instructions (Phase 6 Step 4)
# ===========================================================================

_VECTOR_ALU_BINARY = [
    # (class, mnemonic, InstType)
    (VAddU32, "v_add_nc_u32", InstType.INST_U32),
    (VAddF32, "v_add_f32", InstType.INST_F32),
    (VSubF32, "v_sub_f32", InstType.INST_F32),
    (VSubI32, "v_sub_nc_i32", InstType.INST_I32),
    (VSubU32, "v_sub_nc_u32", InstType.INST_U32),
    (VMulF32, "v_mul_f32", InstType.INST_F32),
    (VMulLOU32, "v_mul_lo_u32", InstType.INST_LO_U32),
    (VMulHIU32, "v_mul_hi_u32", InstType.INST_HI_U32),
    (VMulHII32, "v_mul_hi_i32", InstType.INST_HI_I32),
    (VMulI32I24, "v_mul_i32_i24", InstType.INST_I32),
    (VMulU32U24, "v_mul_u32_u24", InstType.INST_U32),
    (VAndB32, "v_and_b32", InstType.INST_B32),
    (VOrB32, "v_or_b32", InstType.INST_B32),
    (VXorB32, "v_xor_b32", InstType.INST_B32),
]

_VECTOR_SHIFT = [
    (VLShiftLeftB32, "v_lshlrev_b32", InstType.INST_B32),
    (VLShiftRightB32, "v_lshrrev_b32", InstType.INST_B32),
    (VLShiftLeftB64, "v_lshlrev_b64", InstType.INST_B64),
    (VLShiftRightB64, "v_lshrrev_b64", InstType.INST_B64),
]

_VECTOR_TERNARY = [
    (VFmaF32, "v_fma_f32", InstType.INST_F32),
    (VFmaMixF32, "v_fma_mix_f32", InstType.INST_F32),
    (VAndOrB32, "v_and_or_b32", InstType.INST_B32),
]


class TestVectorALUBinaryConstruction(unittest.TestCase):
    """Binary vector ALU shims (dst, src0, src1) inherit CommonInstruction."""

    def test_construction_and_str(self):
        for cls, mnemonic, itype in _VECTOR_ALU_BINARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), comment="t")
                self.assertIsInstance(inst, CommonInstruction)
                self.assertEqual(inst.instStr, mnemonic)
                self.assertEqual(inst.instType, itype)
                text = str(inst)
                self.assertTrue(text.startswith(mnemonic), text)
                self.assertIn("v0", text)
                self.assertIn("v1", text)
                self.assertIn("v2", text)

    def test_deepcopy(self):
        for cls, _, _ in _VECTOR_ALU_BINARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(4), src0=vgpr(5), src1=vgpr(6), comment="cp")
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _, _ in _VECTOR_ALU_BINARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _, _ in _VECTOR_ALU_BINARY:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
                self.assertEqual(len(m._collect_logical_insts()), 1)

    def test_immediate_src(self):
        inst = VAddU32(dst=vgpr(0), src0=vgpr(1), src1=42)
        text = str(inst)
        self.assertIn("42", text)


class TestVectorShiftConstruction(unittest.TestCase):
    """Vector shift shims (dst, shiftHex, src) with rev mnemonics."""

    def test_construction_native_api(self):
        for cls, mnemonic, itype in _VECTOR_SHIFT:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), shiftHex=vgpr(1), src=vgpr(2), comment="sh")
                self.assertIsInstance(inst, CommonInstruction)
                self.assertEqual(inst.instStr, mnemonic)
                self.assertEqual(inst.instType, itype)
                text = str(inst)
                self.assertTrue(text.startswith(mnemonic), text)
                self.assertIn("v0", text)
                self.assertIn("v1", text)
                self.assertIn("v2", text)

    def test_construction_generic_api(self):
        for cls, mnemonic, _ in _VECTOR_SHIFT:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2))
                self.assertEqual(inst.instStr, mnemonic)
                text = str(inst)
                self.assertIn("v1", text)
                self.assertIn("v2", text)

    def test_deepcopy(self):
        for cls, _, _ in _VECTOR_SHIFT:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), shiftHex=vgpr(1), src=vgpr(2))
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _, _ in _VECTOR_SHIFT:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), shiftHex=vgpr(1), src=vgpr(2))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _, _ in _VECTOR_SHIFT:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=vgpr(0), shiftHex=vgpr(1), src=vgpr(2)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestVectorTernaryConstruction(unittest.TestCase):
    """Ternary vector ALU shims (dst, src0, src1, src2)."""

    def test_construction_and_str(self):
        for cls, mnemonic, itype in _VECTOR_TERNARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2),
                           src2=vgpr(3), comment="ter")
                self.assertIsInstance(inst, CommonInstruction)
                self.assertEqual(inst.instStr, mnemonic)
                self.assertEqual(inst.instType, itype)
                text = str(inst)
                self.assertTrue(text.startswith(mnemonic), text)
                self.assertIn("v0", text)
                self.assertIn("v1", text)
                self.assertIn("v2", text)
                self.assertIn("v3", text)

    def test_deepcopy(self):
        for cls, _, _ in _VECTOR_TERNARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), src2=vgpr(3))
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _, _ in _VECTOR_TERNARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), src2=vgpr(3))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _, _ in _VECTOR_TERNARY:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), src2=vgpr(3)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestVCndMaskB32Construction(unittest.TestCase):
    """VCndMaskB32 (native: dst, src0, src1, src2=VCC; logical IR: 2 srcs)."""

    def test_construction_with_src2(self):
        inst = VCndMaskB32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2),
                           src2=vgpr(3), comment="mask")
        self.assertIsInstance(inst, CommonInstruction)
        self.assertEqual(inst.instStr, "v_cndmask_b32")
        text = str(inst)
        self.assertIn("v0", text)
        self.assertIn("v1", text)
        self.assertIn("v2", text)
        self.assertIn("v3", text)

    def test_construction_without_src2(self):
        inst = VCndMaskB32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2))
        self.assertEqual(len(inst.srcs), 2)
        text = str(inst)
        self.assertIn("v_cndmask_b32", text)

    def test_deepcopy(self):
        inst = VCndMaskB32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), src2=vgpr(3))
        c = copy.deepcopy(inst)
        self.assertIsInstance(c, VCndMaskB32)
        self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        inst = VCndMaskB32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2))
        self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        m = Module()
        m.add(VCndMaskB32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
        self.assertEqual(len(m._collect_logical_insts()), 1)


class TestVReadfirstlaneB32Construction(unittest.TestCase):
    """VReadfirstlaneB32 (unary: dst, src)."""

    def test_construction_and_str(self):
        inst = VReadfirstlaneB32(dst=vgpr(0), src=vgpr(1), comment="lane0")
        self.assertIsInstance(inst, CommonInstruction)
        self.assertEqual(inst.instStr, "v_readfirstlane_b32")
        text = str(inst)
        self.assertIn("v_readfirstlane_b32", text)
        self.assertIn("v0", text)
        self.assertIn("v1", text)

    def test_deepcopy(self):
        inst = VReadfirstlaneB32(dst=vgpr(0), src=vgpr(1))
        c = copy.deepcopy(inst)
        self.assertIsInstance(c, VReadfirstlaneB32)
        self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        inst = VReadfirstlaneB32(dst=vgpr(0), src=vgpr(1))
        self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        m = Module()
        m.add(VReadfirstlaneB32(dst=vgpr(0), src=vgpr(1)))
        self.assertEqual(len(m._collect_logical_insts()), 1)


# ===========================================================================
# Compare instructions (Phase 6 Step 5)
# ===========================================================================

_SCALAR_CMP = [
    # (class, mnemonic, InstType)
    (SCmpEQI32, "s_cmp_eq_i32", InstType.INST_I32),
    (SCmpEQU32, "s_cmp_eq_u32", InstType.INST_U32),
    (SCmpEQU64, "s_cmp_eq_u64", InstType.INST_U64),
    (SCmpGeI32, "s_cmp_ge_i32", InstType.INST_I32),
    (SCmpGeU32, "s_cmp_ge_u32", InstType.INST_U32),
    (SCmpGtI32, "s_cmp_gt_i32", InstType.INST_I32),
    (SCmpGtU32, "s_cmp_gt_u32", InstType.INST_U32),
    (SCmpLeI32, "s_cmp_le_i32", InstType.INST_I32),
    (SCmpLeU32, "s_cmp_le_u32", InstType.INST_U32),
    (SCmpLgU32, "s_cmp_lg_u32", InstType.INST_U32),
    (SCmpLgI32, "s_cmp_lg_i32", InstType.INST_I32),
    (SCmpLgU64, "s_cmp_lg_u64", InstType.INST_U64),
    (SCmpLtI32, "s_cmp_lt_i32", InstType.INST_I32),
    (SCmpLtU32, "s_cmp_lt_u32", InstType.INST_U32),
    (SBitcmp1B32, "s_bitcmp1_b32", InstType.INST_B32),
]

_VECTOR_CMP = [
    (VCmpEQF32, "v_cmp_eq_f32", InstType.INST_F32),
    (VCmpEQF64, "v_cmp_eq_f64", InstType.INST_F64),
    (VCmpEQU32, "v_cmp_eq_u32", InstType.INST_U32),
    (VCmpEQI32, "v_cmp_eq_i32", InstType.INST_I32),
    (VCmpGEF16, "v_cmp_ge_f16", InstType.INST_F16),
    (VCmpGTF16, "v_cmp_gt_f16", InstType.INST_F16),
    (VCmpGEF32, "v_cmp_ge_f32", InstType.INST_F32),
    (VCmpGTF32, "v_cmp_gt_f32", InstType.INST_F32),
    (VCmpGEF64, "v_cmp_ge_f64", InstType.INST_F64),
    (VCmpGTF64, "v_cmp_gt_f64", InstType.INST_F64),
    (VCmpGEI32, "v_cmp_ge_i32", InstType.INST_I32),
    (VCmpGTI32, "v_cmp_gt_i32", InstType.INST_I32),
    (VCmpGEU32, "v_cmp_ge_u32", InstType.INST_U32),
    (VCmpGtU32, "v_cmp_gt_u32", InstType.INST_U32),
    (VCmpLeU32, "v_cmp_le_u32", InstType.INST_U32),
    (VCmpLeI32, "v_cmp_le_i32", InstType.INST_I32),
    (VCmpLtI32, "v_cmp_lt_i32", InstType.INST_I32),
    (VCmpLtU32, "v_cmp_lt_u32", InstType.INST_U32),
    (VCmpUF32, "v_cmp_u_f32", InstType.INST_F32),
    (VCmpNeI32, "v_cmp_ne_i32", InstType.INST_I32),
    (VCmpNeU32, "v_cmp_ne_u32", InstType.INST_U32),
    (VCmpNeU64, "v_cmp_ne_u64", InstType.INST_U64),
    (VCmpClassF32, "v_cmp_class_f32", InstType.INST_F32),
]

_VECTOR_CMPX = [
    (VCmpXClassF32, "v_cmpx_class_f32", InstType.INST_F32),
    (VCmpXEqU32, "v_cmpx_eq_u32", InstType.INST_U32),
    (VCmpXGeU32, "v_cmpx_ge_u32", InstType.INST_U32),
    (VCmpXGtU32, "v_cmpx_gt_u32", InstType.INST_U32),
    (VCmpXLeU32, "v_cmpx_le_u32", InstType.INST_U32),
    (VCmpXLeI32, "v_cmpx_le_i32", InstType.INST_I32),
    (VCmpXLtF32, "v_cmpx_lt_f32", InstType.INST_F32),
    (VCmpXLtI32, "v_cmpx_lt_i32", InstType.INST_I32),
    (VCmpXLtU32, "v_cmpx_lt_u32", InstType.INST_U32),
    (VCmpXLtU64, "v_cmpx_lt_u64", InstType.INST_U64),
    (VCmpXNeU16, "v_cmpx_ne_u16", InstType.INST_U16),
    (VCmpXNeU32, "v_cmpx_ne_u32", InstType.INST_U32),
]


class TestScalarCmpConstruction(unittest.TestCase):
    """Scalar compare shims (no dst, src0, src1)."""

    def test_construction_and_str(self):
        for cls, mnemonic, itype in _SCALAR_CMP:
            with self.subTest(cls=cls.__name__):
                inst = cls(src0=sgpr(0), src1=sgpr(1), comment="cmp")
                self.assertIsInstance(inst, CommonInstruction)
                self.assertEqual(inst.instStr, mnemonic)
                self.assertEqual(inst.instType, itype)
                self.assertIsNone(inst.dst)
                self.assertEqual(len(inst.srcs), 2)
                text = str(inst)
                self.assertIn(mnemonic, text)
                self.assertIn("s0", text)
                self.assertIn("s1", text)

    def test_deepcopy(self):
        for cls, _, _ in _SCALAR_CMP:
            with self.subTest(cls=cls.__name__):
                inst = cls(src0=sgpr(2), src1=sgpr(3), comment="cp")
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _, _ in _SCALAR_CMP:
            with self.subTest(cls=cls.__name__):
                inst = cls(src0=sgpr(0), src1=sgpr(1))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _, _ in _SCALAR_CMP:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(src0=sgpr(0), src1=sgpr(1)))
                self.assertEqual(len(m._collect_logical_insts()), 1)

    def test_immediate_operand(self):
        inst = SCmpEQI32(src0=sgpr(0), src1=42)
        text = str(inst)
        self.assertIn("42", text)


class TestVectorCmpConstruction(unittest.TestCase):
    """Vector compare shims (dst, src0, src1)."""

    def test_construction_and_str(self):
        for cls, mnemonic, itype in _VECTOR_CMP:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), comment="vcmp")
                self.assertIsInstance(inst, CommonInstruction)
                self.assertEqual(inst.instStr, mnemonic)
                self.assertEqual(inst.instType, itype)
                self.assertIsNotNone(inst.dst)
                self.assertEqual(len(inst.srcs), 2)
                text = str(inst)
                self.assertIn(mnemonic, text)
                self.assertIn("v0", text)
                self.assertIn("v1", text)
                self.assertIn("v2", text)

    def test_deepcopy(self):
        for cls, _, _ in _VECTOR_CMP:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(4), src0=vgpr(5), src1=vgpr(6), comment="cp")
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _, _ in _VECTOR_CMP:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _, _ in _VECTOR_CMP:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestVectorCmpXConstruction(unittest.TestCase):
    """Vector compareX shims (dst, src0, src1)."""

    def test_construction_and_str(self):
        for cls, mnemonic, itype in _VECTOR_CMPX:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), comment="cmpx")
                self.assertIsInstance(inst, CommonInstruction)
                self.assertEqual(inst.instStr, mnemonic)
                self.assertEqual(inst.instType, itype)
                text = str(inst)
                self.assertIn(mnemonic, text)
                self.assertIn("v0", text)
                self.assertIn("v1", text)
                self.assertIn("v2", text)

    def test_deepcopy(self):
        for cls, _, _ in _VECTOR_CMPX:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(4), src0=vgpr(5), src1=vgpr(6))
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _, _ in _VECTOR_CMPX:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _, _ in _VECTOR_CMPX:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


# ===========================================================================
# Step 6: Scalar Min/Max/Abs + Vector Unary/Misc tests
# ===========================================================================

_SCALAR_MINMAX = [
    (SAbsI32, "s_abs_i32", InstType.INST_I32),
    (SMaxI32, "s_max_i32", InstType.INST_I32),
    (SMaxU32, "s_max_u32", InstType.INST_U32),
    (SMinI32, "s_min_i32", InstType.INST_I32),
    (SMinU32, "s_min_u32", InstType.INST_U32),
]

_VECTOR_UNARY = [
    (VExpF16, "v_exp_f16", InstType.INST_F16),
    (VExpF32, "v_exp_f32", InstType.INST_F32),
    (VRcpF16, "v_rcp_f16", InstType.INST_F16),
    (VRcpF32, "v_rcp_f32", InstType.INST_F32),
    (VRcpIFlagF32, "v_rcp_iflag_f32", InstType.INST_F32),
    (VRsqF16, "v_rsq_f16", InstType.INST_F16),
    (VRsqF32, "v_rsq_f32", InstType.INST_F32),
    (VRsqIFlagF32, "v_rsq_iflag_f32", InstType.INST_F32),
    (VNotB32, "v_not_b32", InstType.INST_B32),
    (VPrngB32, "v_prng_b32", InstType.INST_B32),
    (VRndneF32, "v_rndne_f32", InstType.INST_F32),
]

_VECTOR_MINMAX = [
    (VMaxF16, "v_max_f16", InstType.INST_F16),
    (VMaxF32, "v_max_f32", InstType.INST_F32),
    (VMaxF64, "v_max_f64", InstType.INST_F64),
    (VMaxI32, "v_max_i32", InstType.INST_I32),
    (VMaxPKF16, "v_pk_max_f16", InstType.INST_F16),
    (VMinF16, "v_min_f16", InstType.INST_F16),
    (VMinF32, "v_min_f32", InstType.INST_F32),
    (VMinF64, "v_min_f64", InstType.INST_F64),
    (VMinI32, "v_min_i32", InstType.INST_I32),
    (VPackF16toB32, "v_pack_b32_f16", InstType.INST_B32),
]

_VECTOR_TERNARY_MISC = [
    (VMed3I32, "v_med3_i32", InstType.INST_I32),
    (VMed3F32, "v_med3_f32", InstType.INST_F32),
    (VLShiftLeftOrB32, "v_lshl_or_b32", InstType.INST_B32),
]


class TestScalarMinMaxAbsConstruction(unittest.TestCase):
    """SAbsI32 (unary), SMaxI32/SMaxU32/SMinI32/SMinU32 (binary)."""

    def test_sabsi32_construction(self):
        inst = SAbsI32(dst=sgpr(0), src=sgpr(1), comment="abs")
        self.assertIsInstance(inst, CommonInstruction)
        self.assertEqual(inst.instStr, "s_abs_i32")
        self.assertEqual(inst.instType, InstType.INST_I32)
        self.assertEqual(len(inst.srcs), 1)
        text = str(inst)
        self.assertIn("s_abs_i32", text)
        self.assertIn("s0", text)
        self.assertIn("s1", text)

    def test_binary_construction(self):
        for cls, mnemonic, itype in _SCALAR_MINMAX[1:]:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2), comment="mm")
                self.assertIsInstance(inst, CommonInstruction)
                self.assertEqual(inst.instStr, mnemonic)
                self.assertEqual(inst.instType, itype)
                self.assertEqual(len(inst.srcs), 2)
                text = str(inst)
                self.assertIn(mnemonic, text)
                self.assertIn("s0", text)
                self.assertIn("s1", text)
                self.assertIn("s2", text)

    def test_deepcopy(self):
        inst = SAbsI32(dst=sgpr(0), src=sgpr(1))
        c = copy.deepcopy(inst)
        self.assertIsInstance(c, type(inst))
        self.assertEqual(str(c), str(inst))
        for cls, _, _ in _SCALAR_MINMAX[1:]:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2))
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _, _ in _SCALAR_MINMAX:
            with self.subTest(cls=cls.__name__):
                if cls == SAbsI32:
                    inst = cls(dst=sgpr(0), src=sgpr(1))
                else:
                    inst = cls(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _, _ in _SCALAR_MINMAX:
            with self.subTest(cls=cls.__name__):
                m = Module()
                if cls == SAbsI32:
                    m.add(cls(dst=sgpr(0), src=sgpr(1)))
                else:
                    m.add(cls(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestVectorUnaryConstruction(unittest.TestCase):
    """Vector unary shims (dst, src)."""

    def test_construction_and_str(self):
        for cls, mnemonic, itype in _VECTOR_UNARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src=vgpr(1), comment="vu")
                self.assertIsInstance(inst, CommonInstruction)
                self.assertEqual(inst.instStr, mnemonic)
                self.assertEqual(inst.instType, itype)
                self.assertEqual(len(inst.srcs), 1)
                text = str(inst)
                self.assertIn(mnemonic, text)
                self.assertIn("v0", text)
                self.assertIn("v1", text)

    def test_deepcopy(self):
        for cls, _, _ in _VECTOR_UNARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(4), src=vgpr(5))
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _, _ in _VECTOR_UNARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src=vgpr(1))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _, _ in _VECTOR_UNARY:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=vgpr(0), src=vgpr(1)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestVectorMinMaxConstruction(unittest.TestCase):
    """Vector min/max binary shims (dst, src0, src1)."""

    def test_construction_and_str(self):
        for cls, mnemonic, itype in _VECTOR_MINMAX:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), comment="vmm")
                self.assertIsInstance(inst, CommonInstruction)
                self.assertEqual(inst.instStr, mnemonic)
                self.assertEqual(inst.instType, itype)
                self.assertEqual(len(inst.srcs), 2)
                text = str(inst)
                self.assertIn(mnemonic, text)
                self.assertIn("v0", text)
                self.assertIn("v1", text)
                self.assertIn("v2", text)

    def test_deepcopy(self):
        for cls, _, _ in _VECTOR_MINMAX:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2))
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _, _ in _VECTOR_MINMAX:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _, _ in _VECTOR_MINMAX:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestVectorTernaryMiscConstruction(unittest.TestCase):
    """VMed3I32, VMed3F32, VLShiftLeftOrB32 (dst, src0, src1, src2)."""

    def test_construction_and_str(self):
        for cls, mnemonic, itype in _VECTOR_TERNARY_MISC:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2),
                           src2=vgpr(3), comment="vtern")
                self.assertIsInstance(inst, CommonInstruction)
                self.assertEqual(inst.instStr, mnemonic)
                self.assertEqual(inst.instType, itype)
                self.assertEqual(len(inst.srcs), 3)
                text = str(inst)
                self.assertIn(mnemonic, text)
                self.assertIn("v0", text)
                self.assertIn("v1", text)
                self.assertIn("v2", text)
                self.assertIn("v3", text)

    def test_deepcopy(self):
        for cls, _, _ in _VECTOR_TERNARY_MISC:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), src2=vgpr(3))
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _, _ in _VECTOR_TERNARY_MISC:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), src2=vgpr(3))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _, _ in _VECTOR_TERNARY_MISC:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), src2=vgpr(3)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestVAShiftRightI32Construction(unittest.TestCase):
    """VAShiftRightI32 — vector shift (dst, shiftHex, src)."""

    def test_construction_and_str(self):
        inst = VAShiftRightI32(dst=vgpr(0), shiftHex=vgpr(1), src=vgpr(2),
                               comment="ashr")
        self.assertIsInstance(inst, CommonInstruction)
        self.assertEqual(inst.instStr, "v_ashrrev_i32")
        self.assertEqual(inst.instType, InstType.INST_I32)
        self.assertEqual(len(inst.srcs), 2)
        text = str(inst)
        self.assertIn("v_ashrrev_i32", text)
        self.assertIn("v0", text)

    def test_deepcopy(self):
        inst = VAShiftRightI32(dst=vgpr(0), shiftHex=vgpr(1), src=vgpr(2))
        c = copy.deepcopy(inst)
        self.assertIsInstance(c, type(inst))
        self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        inst = VAShiftRightI32(dst=vgpr(0), shiftHex=vgpr(1), src=vgpr(2))
        self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        m = Module()
        m.add(VAShiftRightI32(dst=vgpr(0), shiftHex=vgpr(1), src=vgpr(2)))
        self.assertEqual(len(m._collect_logical_insts()), 1)


# ===========================================================================
# Step 7 — Conversion (VCvt*) instructions
# ===========================================================================

_CVT_UNARY = [
    (VCvtF16toF32, "v_cvt_f32_f16", InstType.INST_NOTYPE),
    (VCvtF32toF16, "v_cvt_f16_f32", InstType.INST_NOTYPE),
    (VCvtF32toU32, "v_cvt_u32_f32", InstType.INST_NOTYPE),
    (VCvtU32toF32, "v_cvt_f32_u32", InstType.INST_NOTYPE),
    (VCvtI32toF32, "v_cvt_f32_i32", InstType.INST_NOTYPE),
    (VCvtF32toI32, "v_cvt_i32_f32", InstType.INST_NOTYPE),
    (VCvtFP8toF32, "v_cvt_f32_fp8", InstType.INST_NOTYPE),
    (VCvtBF8toF32, "v_cvt_f32_bf8", InstType.INST_NOTYPE),
    (VCvtPkFP8toF32, "v_cvt_pk_f32_fp8", InstType.INST_NOTYPE),
    (VCvtPkBF8toF32, "v_cvt_pk_f32_bf8", InstType.INST_NOTYPE),
]

_CVT_BINARY = [
    (VCvtPkF32toBF8, "v_cvt_pk_bf8_f32", InstType.INST_NOTYPE),
    (VCvtSRF32toFP8, "v_cvt_sr_fp8_f32", InstType.INST_NOTYPE),
    (VCvtSRF32toBF8, "v_cvt_sr_bf8_f32", InstType.INST_NOTYPE),
    (VCvtPkF32toFP8, "v_cvt_pk_fp8_f32", InstType.INST_NOTYPE),
    (VCvtPkF32toBF16, "v_cvt_pk_bf16_f32", InstType.INST_NOTYPE),
]

_CVT_SCALE = [
    (VCvtScalePkFP8toF16, "v_cvt_scalef32_pk_f16_fp8", InstType.INST_NOTYPE),
    (VCvtScalePkBF8toF16, "v_cvt_scalef32_pk_f16_bf8", InstType.INST_NOTYPE),
    (VCvtScaleFP8toF16, "v_cvt_scalef32_f16_fp8", InstType.INST_NOTYPE),
    (VCvtScalePkF16toFP8, "v_cvt_scalef32_pk_fp8_f16", InstType.INST_NOTYPE),
    (VCvtScalePkF16toBF8, "v_cvt_scalef32_pk_bf8_f16", InstType.INST_NOTYPE),
    (VCvtScaleSRF16toFP8, "v_cvt_scalef32_sr_fp8_f16", InstType.INST_NOTYPE),
    (VCvtScaleSRF16toBF8, "v_cvt_scalef32_sr_bf8_f16", InstType.INST_NOTYPE),
]


class TestCvtUnaryConstruction(unittest.TestCase):
    """Unary conversion shims (dst, src)."""

    def test_construction_and_str(self):
        for cls, mnemonic, itype in _CVT_UNARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src=vgpr(1), comment="cvt")
                self.assertIsInstance(inst, CommonInstruction)
                self.assertEqual(inst.instStr, mnemonic)
                self.assertEqual(inst.instType, itype)
                self.assertEqual(len(inst.srcs), 1)
                text = str(inst)
                self.assertIn(mnemonic, text)

    def test_deepcopy(self):
        for cls, _, _ in _CVT_UNARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(4), src=vgpr(5))
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _, _ in _CVT_UNARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src=vgpr(1))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _, _ in _CVT_UNARY:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=vgpr(0), src=vgpr(1)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestCvtBinaryConstruction(unittest.TestCase):
    """Binary conversion shims (dst, src0, src1)."""

    def test_construction_and_str(self):
        for cls, mnemonic, itype in _CVT_BINARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), comment="cvt2")
                self.assertIsInstance(inst, CommonInstruction)
                self.assertEqual(inst.instStr, mnemonic)
                self.assertEqual(inst.instType, itype)
                self.assertEqual(len(inst.srcs), 2)
                text = str(inst)
                self.assertIn(mnemonic, text)

    def test_deepcopy(self):
        for cls, _, _ in _CVT_BINARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(4), src0=vgpr(5), src1=vgpr(6))
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _, _ in _CVT_BINARY:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _, _ in _CVT_BINARY:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestCvtScaleConstruction(unittest.TestCase):
    """Scale conversion shims (dst, src, scale)."""

    def test_construction_and_str(self):
        for cls, mnemonic, itype in _CVT_SCALE:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src=vgpr(1), scale=vgpr(2), comment="sc")
                self.assertIsInstance(inst, CommonInstruction)
                self.assertEqual(inst.instStr, mnemonic)
                self.assertEqual(inst.instType, itype)
                self.assertEqual(len(inst.srcs), 2)
                text = str(inst)
                self.assertIn(mnemonic, text)

    def test_deepcopy(self):
        for cls, _, _ in _CVT_SCALE:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(4), src=vgpr(5), scale=vgpr(6))
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _, _ in _CVT_SCALE:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src=vgpr(1), scale=vgpr(2))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _, _ in _CVT_SCALE:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=vgpr(0), src=vgpr(1), scale=vgpr(2)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


# ===========================================================================
# Memory instructions -- Step 8 (Buffer/Flat/DS/Tensor).
# ===========================================================================


_BUFFER_LOAD = [
    (BufferLoadU8, "buffer_load_u8"),
    (BufferLoadD16HIU8, "buffer_load_d16_hi_u8"),
    (BufferLoadD16U8, "buffer_load_d16_u8"),
    (BufferLoadD16HIB16, "buffer_load_d16_hi_b16"),
    (BufferLoadD16B16, "buffer_load_d16_b16"),
    (BufferLoadB32, "buffer_load_b32"),
    (BufferLoadB64, "buffer_load_b64"),
    (BufferLoadB96, "buffer_load_b96"),
    (BufferLoadB128, "buffer_load_b128"),
]

_BUFFER_STORE = [
    (BufferStoreB8, "buffer_store_b8"),
    (BufferStoreD16HIU8, "buffer_store_d16_hi_b8"),
    (BufferStoreD16HIB16, "buffer_store_d16_hi_b16"),
    (BufferStoreB16, "buffer_store_b16"),
    (BufferStoreB32, "buffer_store_b32"),
    (BufferStoreB64, "buffer_store_b64"),
    (BufferStoreB128, "buffer_store_b128"),
    (BufferAtomicCmpswapB32, "buffer_atomic_cmpswap_b32"),
    (BufferAtomicCmpswapB64, "buffer_atomic_cmpswap_b64"),
]

_FLAT_LOAD = [
    (FlatLoadD16HIU8, "flat_load_d16_hi_u8"),
    (FlatLoadD16U8, "flat_load_d16_u8"),
    (FlatLoadD16HIB16, "flat_load_d16_hi_b16"),
    (FlatLoadD16B16, "flat_load_d16_b16"),
    (FlatLoadB32, "flat_load_b32"),
    (FlatLoadB64, "flat_load_b64"),
    (FlatLoadB128, "flat_load_b128"),
]

_FLAT_STORE = [
    (FlatStoreD16HIB16, "flat_store_d16_hi_b16"),
    (FlatStoreB32, "flat_store_b32"),
    (FlatStoreB64, "flat_store_b64"),
    (FlatStoreB128, "flat_store_b128"),
]

_DS_LOAD = [
    (DSLoadU8, "ds_load_u8"),
    (DSLoadU16, "ds_load_u16"),
    (DSLoadB32, "ds_load_b32"),
    (DSLoadB64, "ds_load_b64"),
    (DSLoadB128, "ds_load_b128"),
]

_DS_LOAD2 = [
    (DSLoad2B32, "ds_load2_b32"),
    (DSLoad2B64, "ds_load2_b64"),
]

_DS_STORE = [
    (DSStoreB8, "ds_store_b8"),
    (DSStoreB16, "ds_store_b16"),
    (DSStoreB32, "ds_store_b32"),
    (DSStoreB64, "ds_store_b64"),
    (DSStoreB96, "ds_store_b96"),
    (DSStoreB128, "ds_store_b128"),
]

_DS_STORE2 = [
    (DSStore2B32, "ds_store2_b32"),
    (DSStore2B64, "ds_store2_b64"),
]


class TestBufferLoadInstructions(unittest.TestCase):
    def test_construction_and_mnemonic(self):
        for cls, mnemonic in _BUFFER_LOAD:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), vaddr=vgpr(1), saddr=sgpr(4, 4),
                           soffset=0, comment="load")
                self.assertIn(mnemonic, str(inst))

    def test_deepcopy(self):
        for cls, _ in _BUFFER_LOAD:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), vaddr=vgpr(1), saddr=sgpr(4, 4), soffset=0)
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _ in _BUFFER_LOAD:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), vaddr=vgpr(1), saddr=sgpr(4, 4), soffset=0)
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _ in _BUFFER_LOAD:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=vgpr(0), vaddr=vgpr(1), saddr=sgpr(4, 4), soffset=0))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestBufferAtomicAddF32(unittest.TestCase):
    def test_construction(self):
        inst = BufferAtomicAddF32(dst=vgpr(2), vaddr=vgpr(1),
                                  saddr=sgpr(4, 4), soffset=0, comment="at")
        self.assertIn("buffer_atomic_add_f32", str(inst))

    def test_has_to_stinky_logical(self):
        inst = BufferAtomicAddF32(dst=vgpr(0), vaddr=vgpr(1),
                                  saddr=sgpr(4, 4), soffset=0)
        self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))


class TestBufferStoreInstructions(unittest.TestCase):
    def test_construction_and_mnemonic(self):
        for cls, mnemonic in _BUFFER_STORE:
            with self.subTest(cls=cls.__name__):
                inst = cls(src=vgpr(0), vaddr=vgpr(1), saddr=sgpr(4, 4),
                           soffset=0, comment="store")
                self.assertIn(mnemonic, str(inst))

    def test_deepcopy(self):
        for cls, _ in _BUFFER_STORE:
            with self.subTest(cls=cls.__name__):
                inst = cls(src=vgpr(0), vaddr=vgpr(1), saddr=sgpr(4, 4), soffset=0)
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _ in _BUFFER_STORE:
            with self.subTest(cls=cls.__name__):
                inst = cls(src=vgpr(0), vaddr=vgpr(1), saddr=sgpr(4, 4), soffset=0)
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _ in _BUFFER_STORE:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(src=vgpr(0), vaddr=vgpr(1), saddr=sgpr(4, 4), soffset=0))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestFlatLoadInstructions(unittest.TestCase):
    def test_construction_and_mnemonic(self):
        for cls, mnemonic in _FLAT_LOAD:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), vaddr=vgpr(2), comment="fload")
                self.assertIn(mnemonic, str(inst))

    def test_deepcopy(self):
        for cls, _ in _FLAT_LOAD:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), vaddr=vgpr(2))
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _ in _FLAT_LOAD:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), vaddr=vgpr(2))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _ in _FLAT_LOAD:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=vgpr(0), vaddr=vgpr(2)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestFlatStoreInstructions(unittest.TestCase):
    def test_construction_and_mnemonic(self):
        for cls, mnemonic in _FLAT_STORE:
            with self.subTest(cls=cls.__name__):
                inst = cls(src=vgpr(0), vaddr=vgpr(2), comment="fstore")
                self.assertIn(mnemonic, str(inst))

    def test_deepcopy(self):
        for cls, _ in _FLAT_STORE:
            with self.subTest(cls=cls.__name__):
                inst = cls(src=vgpr(0), vaddr=vgpr(2))
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _ in _FLAT_STORE:
            with self.subTest(cls=cls.__name__):
                inst = cls(src=vgpr(0), vaddr=vgpr(2))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _ in _FLAT_STORE:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(src=vgpr(0), vaddr=vgpr(2)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestFlatAtomicCmpswapB32(unittest.TestCase):
    def test_construction(self):
        inst = FlatAtomicCmpswapB32(vaddr=vgpr(0), tmp=vgpr(1),
                                     src=vgpr(2), comment="cas")
        self.assertIn("flat_atomic_cmpswap_b32", str(inst))

    def test_deepcopy(self):
        inst = FlatAtomicCmpswapB32(vaddr=vgpr(0), tmp=vgpr(1), src=vgpr(2))
        c = copy.deepcopy(inst)
        self.assertIsInstance(c, FlatAtomicCmpswapB32)
        self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        inst = FlatAtomicCmpswapB32(vaddr=vgpr(0), tmp=vgpr(1), src=vgpr(2))
        self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        m = Module()
        m.add(FlatAtomicCmpswapB32(vaddr=vgpr(0), tmp=vgpr(1), src=vgpr(2)))
        self.assertEqual(len(m._collect_logical_insts()), 1)


class TestDSLoadInstructions(unittest.TestCase):
    def test_construction_and_mnemonic(self):
        for cls, mnemonic in _DS_LOAD:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src=vgpr(1), comment="dsld")
                self.assertIn(mnemonic, str(inst))

    def test_deepcopy(self):
        for cls, _ in _DS_LOAD:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src=vgpr(1))
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _ in _DS_LOAD:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src=vgpr(1))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _ in _DS_LOAD:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=vgpr(0), src=vgpr(1)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestDSLoad2Instructions(unittest.TestCase):
    def test_construction_and_mnemonic(self):
        for cls, mnemonic in _DS_LOAD2:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src=vgpr(1), comment="dsld2")
                self.assertIn(mnemonic, str(inst))

    def test_deepcopy(self):
        for cls, _ in _DS_LOAD2:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src=vgpr(1))
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _ in _DS_LOAD2:
            with self.subTest(cls=cls.__name__):
                inst = cls(dst=vgpr(0), src=vgpr(1))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _ in _DS_LOAD2:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dst=vgpr(0), src=vgpr(1)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestDSStoreInstructions(unittest.TestCase):
    def test_construction_and_mnemonic(self):
        for cls, mnemonic in _DS_STORE:
            with self.subTest(cls=cls.__name__):
                inst = cls(dstAddr=vgpr(0), src=vgpr(1), comment="dsst")
                self.assertIn(mnemonic, str(inst))

    def test_deepcopy(self):
        for cls, _ in _DS_STORE:
            with self.subTest(cls=cls.__name__):
                inst = cls(dstAddr=vgpr(0), src=vgpr(1))
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _ in _DS_STORE:
            with self.subTest(cls=cls.__name__):
                inst = cls(dstAddr=vgpr(0), src=vgpr(1))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _ in _DS_STORE:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dstAddr=vgpr(0), src=vgpr(1)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestDSStore2Instructions(unittest.TestCase):
    def test_construction_and_mnemonic(self):
        for cls, mnemonic in _DS_STORE2:
            with self.subTest(cls=cls.__name__):
                inst = cls(dstAddr=vgpr(0), src0=vgpr(1), src1=vgpr(2),
                           comment="dsst2")
                self.assertIn(mnemonic, str(inst))

    def test_deepcopy(self):
        for cls, _ in _DS_STORE2:
            with self.subTest(cls=cls.__name__):
                inst = cls(dstAddr=vgpr(0), src0=vgpr(1), src1=vgpr(2))
                c = copy.deepcopy(inst)
                self.assertIsInstance(c, cls)
                self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        for cls, _ in _DS_STORE2:
            with self.subTest(cls=cls.__name__):
                inst = cls(dstAddr=vgpr(0), src0=vgpr(1), src1=vgpr(2))
                self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        for cls, _ in _DS_STORE2:
            with self.subTest(cls=cls.__name__):
                m = Module()
                m.add(cls(dstAddr=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
                self.assertEqual(len(m._collect_logical_insts()), 1)


class TestDSBPermuteB32(unittest.TestCase):
    def test_construction(self):
        inst = DSBPermuteB32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2),
                             comment="perm")
        self.assertIn("ds_bpermute_b32", str(inst))

    def test_deepcopy(self):
        inst = DSBPermuteB32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2))
        c = copy.deepcopy(inst)
        self.assertIsInstance(c, DSBPermuteB32)
        self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        inst = DSBPermuteB32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2))
        self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        m = Module()
        m.add(DSBPermuteB32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
        self.assertEqual(len(m._collect_logical_insts()), 1)


class TestTensorLoadToLds(unittest.TestCase):
    def test_construction(self):
        inst = TensorLoadToLds(group0=vgpr(0), group1=vgpr(1), comment="tld")
        self.assertIn("tensor_load_to_lds", str(inst))

    def test_deepcopy(self):
        inst = TensorLoadToLds(group0=vgpr(0), group1=vgpr(1))
        c = copy.deepcopy(inst)
        self.assertIsInstance(c, TensorLoadToLds)
        self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        inst = TensorLoadToLds(group0=vgpr(0), group1=vgpr(1))
        self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        m = Module()
        m.add(TensorLoadToLds(group0=vgpr(0), group1=vgpr(1)))
        self.assertEqual(len(m._collect_logical_insts()), 1)


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


# ===========================================================================
# Branch instructions
# ===========================================================================


class TestBranchInstructionConstruction(unittest.TestCase):
    """Branch instructions take (labelName, comment)."""

    def test_sbranch_default(self):
        inst = SBranch("loop_start")
        self.assertIsInstance(inst, BranchInstruction)
        self.assertIsInstance(inst, Instruction)
        self.assertEqual(inst.instStr, "s_branch")
        self.assertEqual(inst.labelName, "loop_start")

    def test_sbranch_str(self):
        inst = SBranch("loop_start", comment="jump back")
        text = str(inst)
        self.assertIn("s_branch", text)
        self.assertIn("loop_start", text)
        self.assertIn("jump back", text)

    def test_scbranch_scc0(self):
        inst = SCBranchSCC0("done")
        self.assertEqual(inst.instStr, "s_cbranch_scc0")
        self.assertIn("done", str(inst))

    def test_scbranch_scc1(self):
        inst = SCBranchSCC1("skip")
        self.assertEqual(inst.instStr, "s_cbranch_scc1")

    def test_scbranch_vccnz(self):
        inst = SCBranchVCCNZ("active_label")
        self.assertEqual(inst.instStr, "s_cbranch_vccnz")

    def test_scbranch_vccz(self):
        inst = SCBranchVCCZ("exit")
        self.assertEqual(inst.instStr, "s_cbranch_vccz")

    def test_scbranch_execz(self):
        inst = SCBranchExecZ("all_done")
        self.assertEqual(inst.instStr, "s_cbranch_execz")

    def test_scbranch_execnz(self):
        inst = SCBranchExecNZ("still_active")
        self.assertEqual(inst.instStr, "s_cbranch_execnz")

    def test_deepcopy(self):
        inst = SBranch("target", comment="c")
        dup = copy.deepcopy(inst)
        self.assertIsInstance(dup, SBranch)
        self.assertEqual(dup.labelName, "target")
        self.assertEqual(str(dup), str(inst))
        self.assertIsNot(dup, inst)

    def test_getParams(self):
        inst = SCBranchSCC0("lbl")
        self.assertEqual(inst.getParams(), ["lbl"])
        self.assertEqual(inst.getDstParams(), [])
        self.assertEqual(inst.getSrcParams(), ["lbl"])

    def test_has_to_stinky_logical(self):
        for cls in (SBranch, SCBranchSCC0, SCBranchSCC1,
                    SCBranchVCCNZ, SCBranchVCCZ,
                    SCBranchExecZ, SCBranchExecNZ):
            inst = cls("lbl")
            self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)),
                            f"{cls.__name__} missing to_stinky_logical")

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        m = Module()
        m.add(SBranch("lbl"))
        m.add(SCBranchSCC0("lbl"))
        self.assertEqual(len(m._collect_logical_insts()), 2)


# ===========================================================================
# SEndpgm
# ===========================================================================


class TestSEndpgmConstruction(unittest.TestCase):
    """SEndpgm is a zero-operand terminator instruction."""

    def test_default(self):
        inst = SEndpgm()
        self.assertIsInstance(inst, Instruction)
        self.assertEqual(inst.instStr, "s_endpgm")

    def test_str(self):
        inst = SEndpgm(comment="end")
        text = str(inst)
        self.assertIn("s_endpgm", text)
        self.assertIn("end", text)

    def test_deepcopy(self):
        inst = SEndpgm(comment="fin")
        dup = copy.deepcopy(inst)
        self.assertIsInstance(dup, SEndpgm)
        self.assertEqual(str(dup), str(inst))

    def test_params(self):
        inst = SEndpgm()
        self.assertEqual(inst.getParams(), [])
        self.assertEqual(inst.getDstParams(), [])
        self.assertEqual(inst.getSrcParams(), [])

    def test_has_to_stinky_logical(self):
        inst = SEndpgm()
        self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        m = Module()
        m.add(SEndpgm())
        self.assertEqual(len(m._collect_logical_insts()), 1)


# ===========================================================================
# Wait-count instructions
# ===========================================================================


class TestWaitCntInstructions(unittest.TestCase):
    """_SWaitCnt, SWaitCnt, SWaitXCnt, SWaitTensorcnt."""

    def test_swaitcnt_primitive_default(self):
        inst = _SWaitCnt()
        self.assertIsInstance(inst, Instruction)
        self.assertEqual(inst.lgkmcnt, -1)
        self.assertEqual(inst.vmcnt, -1)

    def test_swaitcnt_primitive_str_both_zero(self):
        inst = _SWaitCnt(lgkmcnt=0, vmcnt=0)
        self.assertIn("s_waitcnt 0", str(inst))

    def test_swaitcnt_primitive_str_lgkm_only(self):
        inst = _SWaitCnt(lgkmcnt=2)
        self.assertIn("lgkmcnt(2)", str(inst))

    def test_swaitcnt_primitive_str_both(self):
        inst = _SWaitCnt(lgkmcnt=1, vmcnt=3)
        text = str(inst)
        self.assertIn("lgkmcnt(1)", text)
        self.assertIn("vmcnt(3)", text)

    def test_swaitcnt_primitive_deepcopy(self):
        inst = _SWaitCnt(lgkmcnt=4, vmcnt=2, comment="wait")
        dup = copy.deepcopy(inst)
        self.assertEqual(dup.lgkmcnt, 4)
        self.assertEqual(dup.vmcnt, 2)
        self.assertEqual(str(dup), str(inst))

    def test_swaitcnt_composite(self):
        inst = SWaitCnt(vlcnt=0, vscnt=0, dscnt=0, kmcnt=0, waitAll=True)
        self.assertIsInstance(inst, Instruction)
        self.assertTrue(inst.waitAll)

    def test_swaitcnt_composite_deepcopy(self):
        inst = SWaitCnt(vlcnt=1, comment="sync")
        dup = copy.deepcopy(inst)
        self.assertEqual(dup.vlcnt, 1)
        self.assertEqual(dup.vscnt, -1)

    def test_swaitxcnt(self):
        inst = SWaitXCnt(cnt=5)
        self.assertIn("s_wait_xcnt", str(inst))
        self.assertIn("5", str(inst))

    def test_swaitxcnt_deepcopy(self):
        inst = SWaitXCnt(cnt=3, comment="x")
        dup = copy.deepcopy(inst)
        self.assertEqual(dup.cnt, 3)

    def test_swaittensorcnt(self):
        inst = SWaitTensorcnt(cnt=2)
        self.assertIn("s_wait_tensorcnt", str(inst))
        self.assertIn("2", str(inst))

    def test_swaittensorcnt_deepcopy(self):
        inst = SWaitTensorcnt(cnt=1, comment="tensor")
        dup = copy.deepcopy(inst)
        self.assertEqual(dup.cnt, 1)

    def test_all_have_to_stinky_logical(self):
        for inst in (_SWaitCnt(), SWaitCnt(), SWaitXCnt(), SWaitTensorcnt()):
            self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)),
                            f"{type(inst).__name__} missing to_stinky_logical")

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        m = Module()
        m.add(_SWaitCnt(lgkmcnt=0, vmcnt=0))
        m.add(SWaitTensorcnt(cnt=0))
        m.add(SWaitXCnt(cnt=0))
        # _SWaitCnt(lgkmcnt=0, vmcnt=0) decomposes into 2 typed waits
        # (dscnt from lgkmcnt + loadcnt from vmcnt) + tensorcnt + xcnt = 4
        self.assertEqual(len(m._collect_logical_insts()), 4)


class TestSWaitAlu(unittest.TestCase):
    """SWaitAlu — dependency counter wait instruction."""

    def test_default_construction(self):
        inst = SWaitAlu()
        self.assertIsInstance(inst, Instruction)
        self.assertEqual(inst.va_vdst, -1)
        self.assertEqual(inst.va_sdst, -1)
        self.assertEqual(inst.va_ssrc, -1)
        self.assertEqual(inst.hold_cnt, -1)
        self.assertEqual(inst.vm_vsrc, -1)
        self.assertEqual(inst.va_vcc, -1)
        self.assertEqual(inst.sa_sdst, -1)

    def test_keyword_construction(self):
        inst = SWaitAlu(vm_vsrc=0, va_vdst=3, comment="wait valu")
        self.assertEqual(inst.vm_vsrc, 0)
        self.assertEqual(inst.va_vdst, 3)
        self.assertEqual(inst.va_sdst, -1)
        self.assertEqual(inst.comment, "wait valu")

    def test_deepcopy(self):
        inst = SWaitAlu(va_vdst=2, hold_cnt=1, comment="dc")
        dup = copy.deepcopy(inst)
        self.assertIsNot(inst, dup)
        self.assertEqual(dup.va_vdst, 2)
        self.assertEqual(dup.hold_cnt, 1)
        self.assertEqual(dup.comment, "dc")
        self.assertEqual(dup.vm_vsrc, -1)

    def test_has_to_stinky_logical(self):
        inst = SWaitAlu(vm_vsrc=0)
        self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_to_stinky_logical(self):
        inst = SWaitAlu(va_vdst=1, vm_vsrc=0, comment="test")
        logical = inst.to_stinky_logical()
        self.assertIsNotNone(logical)

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        m = Module()
        m.add(SWaitAlu(vm_vsrc=0))
        m.add(SWaitAlu(va_vdst=3, va_sdst=1))
        self.assertEqual(len(m._collect_logical_insts()), 2)


class TestSSchedulingFence(unittest.TestCase):
    """SSchedulingFence — scheduling barrier pseudo-instruction."""

    def test_default_construction(self):
        inst = SSchedulingFence()
        self.assertIsInstance(inst, Instruction)
        self.assertEqual(inst.comment, "")

    def test_with_comment(self):
        inst = SSchedulingFence(comment="barrier")
        self.assertEqual(inst.comment, "barrier")

    def test_deepcopy(self):
        inst = SSchedulingFence(comment="fence")
        dup = copy.deepcopy(inst)
        self.assertIsNot(inst, dup)
        self.assertEqual(dup.comment, "fence")

    def test_has_to_stinky_logical(self):
        inst = SSchedulingFence()
        self.assertTrue(callable(getattr(inst, "to_stinky_logical", None)))

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_to_stinky_logical(self):
        inst = SSchedulingFence(comment="test fence")
        logical = inst.to_stinky_logical()
        self.assertIsNotNone(logical)

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        m = Module()
        m.add(SSchedulingFence())
        self.assertEqual(len(m._collect_logical_insts()), 1)


if __name__ == "__main__":
    unittest.main()
