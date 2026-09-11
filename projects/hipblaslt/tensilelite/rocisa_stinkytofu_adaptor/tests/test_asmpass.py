# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for ``rocisa_stinkytofu_adaptor.asmpass`` (``rocIsaPass`` Python port)."""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rocisa_stinkytofu_adaptor import asmpass  # noqa: E402
from rocisa_stinkytofu_adaptor.code import KernelBody, Macro, Module  # noqa: E402
from rocisa_stinkytofu_adaptor.container import sgpr, vgpr  # noqa: E402
from rocisa_stinkytofu_adaptor.instruction import MacroInstruction, VMovB32  # noqa: E402


class TestRocIsaPassOption(unittest.TestCase):
    def test_defaults_match_pass_hpp(self):
        o = asmpass.rocIsaPassOption()
        self.assertFalse(o.insertDelayAlu)
        self.assertTrue(o.removeDupFunc)
        self.assertTrue(o.removeDupAssign)
        self.assertTrue(o.getCycles)
        self.assertEqual(o.numWaves, 0)
        self.assertTrue(o.doOpt())


class TestRocIsaPassResult(unittest.TestCase):
    def test_default_cycles(self):
        r = asmpass.rocIsaPassResult()
        self.assertEqual(r.cycles, -1)


class TestGetActFuncNames(unittest.TestCase):
    def test_module_name(self):
        self.assertEqual(
            asmpass.getActFuncModuleName(8, 12, 3, 4),
            "ActFunc_VW8_Sgpr12_Tmp3_4",
        )

    def test_branch_name(self):
        self.assertEqual(
            asmpass.getActFuncBranchModuleName(),
            "InsertActFuncCallAddrCalc",
        )


class TestMacroToInstruction(unittest.TestCase):
    def test_expands_macro_call_and_drops_definition(self):
        body = Module("body")
        macro = Macro("FOO", ["vdst:req=v0", "ssrc:req=s0"])
        macro.add(
            VMovB32(
                dst=vgpr("vdst", isMacro=True),
                src=sgpr("ssrc", isMacro=True),
                comment="inside",
            )
        )
        body.add(macro)
        body.add(MacroInstruction(name="FOO", args=[vgpr(7), sgpr(2)]))

        kb = KernelBody("k")
        kb.addBody(body)
        kb.setGprs(256, 0, 256)

        opt = asmpass.rocIsaPassOption()
        opt.removeDupFunc = False
        opt.removeDupAssign = False
        opt.insertDelayAlu = False
        opt.getCycles = False

        res = asmpass.rocIsaPass(kb, opt)
        self.assertEqual(res.cycles, -1)

        items = body.items()
        self.assertEqual(len(items), 1)
        inst = items[0]
        self.assertIsInstance(inst, VMovB32)
        self.assertEqual(inst.dst.regIdx, 7)
        self.assertEqual(inst.srcs[0].regIdx, 2)
        self.assertEqual(inst.comment, "inside")

    def test_get_cycles_default_zero(self):
        body = Module("b")
        kb = KernelBody("k")
        kb.addBody(body)
        kb.setGprs(256, 0, 256)
        opt = asmpass.rocIsaPassOption()
        opt.removeDupFunc = False
        opt.removeDupAssign = False
        opt.insertDelayAlu = False
        res = asmpass.rocIsaPass(kb, opt)
        self.assertEqual(res.cycles, 0)


class TestModuleExports(unittest.TestCase):
    def test_public_symbols(self):
        for name in (
            "rocIsaPass",
            "rocIsaPassOption",
            "rocIsaPassResult",
            "getActFuncModuleName",
            "getActFuncBranchModuleName",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(asmpass, name))


if __name__ == "__main__":
    unittest.main()
