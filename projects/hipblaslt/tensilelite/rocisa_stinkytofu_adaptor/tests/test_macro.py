# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Standalone tests for ``rocisa_stinkytofu_adaptor.macro``.

Run from any working directory:

    python3 projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_macro.py

Or with pytest:

    pytest projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_macro.py
"""

from __future__ import annotations

import os
import sys
import unittest

# ---------------------------------------------------------------------------
# Self-contained sys.path bootstrap (mirrors test_functions.py).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rocisa_stinkytofu_adaptor import macro as _macro  # noqa: E402
from rocisa_stinkytofu_adaptor.code import Macro, Module  # noqa: E402


MACRO_PUBLIC_EXPORTS: tuple[str, ...] = (
    "MacroVMagicDiv",
    "PseudoRandomGenerator",
    "VMagicDiv",
    "PseudoRandomGeneratorModule",
)


class TestMacroModuleExports(unittest.TestCase):
    def test_all_symbols_exported(self):
        for name in MACRO_PUBLIC_EXPORTS:
            with self.subTest(name=name):
                self.assertTrue(hasattr(_macro, name),
                                f"macro.{name} missing")

    def test_all_are_callable(self):
        for name in MACRO_PUBLIC_EXPORTS:
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(_macro, name)))


class TestMacroVMagicDiv(unittest.TestCase):
    def test_algo1_returns_macro(self):
        result = _macro.MacroVMagicDiv(1)
        self.assertIsInstance(result, Macro)
        text = result.toString()
        self.assertIn(".macro V_MAGIC_DIV", text)
        self.assertIn("v_mul_hi_u32", text)
        self.assertIn("v_mul_lo_u32", text)
        self.assertIn("v_lshrrev_b64", text)
        self.assertIn(".endm", text)

    def test_algo2_returns_macro(self):
        result = _macro.MacroVMagicDiv(2)
        self.assertIsInstance(result, Macro)
        text = result.toString()
        self.assertIn("v_mul_hi_u32", text)
        self.assertIn("v_mul_lo_u32", text)
        self.assertIn("v_add_nc_u32", text)
        self.assertIn("v_lshrrev_b32", text)


class TestVMagicDiv(unittest.TestCase):
    def test_algo1_returns_module(self):
        from rocisa_stinkytofu_adaptor.container import vgpr, sgpr
        result = _macro.VMagicDiv(1, 10, vgpr(5), sgpr(2), sgpr(3), sgpr(4))
        self.assertIsInstance(result, Module)
        text = result.toString()
        self.assertIn("v_mul_hi_u32", text)
        self.assertIn("v_lshrrev_b64", text)

    def test_algo2_returns_module(self):
        from rocisa_stinkytofu_adaptor.container import vgpr, sgpr
        result = _macro.VMagicDiv(2, 10, vgpr(5), sgpr(2), sgpr(3), sgpr(4))
        self.assertIsInstance(result, Module)
        text = result.toString()
        self.assertIn("v_add_nc_u32", text)
        self.assertIn("v_lshrrev_b32", text)


class TestPseudoRandomGenerator(unittest.TestCase):
    def test_returns_macro(self):
        result = _macro.PseudoRandomGenerator()
        self.assertIsInstance(result, Macro)
        text = result.toString()
        self.assertIn(".macro PRND_GENERATOR", text)
        self.assertIn("v_xor_b32", text)
        self.assertIn("v_mul_u32_u24", text)

    def test_module_form(self):
        result = _macro.PseudoRandomGeneratorModule(0, 1, 2, 3)
        self.assertIsInstance(result, Module)
        text = result.toString()
        self.assertIn("v_xor_b32", text)
        self.assertIn("v_mul_u32_u24", text)
        self.assertIn("PRND_GENERATOR end", text)


if __name__ == "__main__":
    unittest.main()
