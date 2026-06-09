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
"""Standalone tests for ``rocisa_stinkytofu_adaptor.asmpass``.

Run from any working directory:

    python3 projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_asmpass.py

Or with pytest:

    pytest projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_asmpass.py

All exports in ``asmpass.py`` are dummy today. These tests pin the
import / callable / instantiable contract so Tensile can
``from rocisa.asmpass import …`` without AttributeError.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

# ---------------------------------------------------------------------------
# Self-contained sys.path bootstrap (mirrors test_functions.py).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rocisa_stinkytofu_adaptor import asmpass as _asmpass  # noqa: E402


ASMPASS_DUMMY_FUNCS: tuple[str, ...] = (
    "getActFuncModuleName",
    "getActFuncBranchModuleName",
    "rocIsaPass",
)

ASMPASS_DUMMY_CLASSES: tuple[str, ...] = (
    "rocIsaPassOption",
    "rocIsaPassResult",
)

ASMPASS_DUMMY_EXPORTS: tuple[str, ...] = ASMPASS_DUMMY_FUNCS + ASMPASS_DUMMY_CLASSES


class TestAsmpassModuleExports(unittest.TestCase):
    def test_all_dummy_symbols_exported(self):
        for name in ASMPASS_DUMMY_EXPORTS:
            with self.subTest(name=name):
                self.assertTrue(hasattr(_asmpass, name),
                                f"asmpass.{name} missing")

    def test_dummy_registry_matches_module(self):
        _skip = frozenset({"annotations", "make_dummy_func", "make_dummy_class"})
        module_exports = {
            name for name in dir(_asmpass)
            if not name.startswith("_") and name not in _skip
        }
        self.assertEqual(set(ASMPASS_DUMMY_EXPORTS), module_exports)


class TestAsmpassDummyCallables(unittest.TestCase):
    def test_each_dummy_func_callable_returns_none(self):
        for name in ASMPASS_DUMMY_FUNCS:
            with self.subTest(name=name):
                fn = getattr(_asmpass, name)
                self.assertTrue(callable(fn))
                with mock.patch("builtins.print"):
                    self.assertIsNone(fn())


class TestAsmpassDummyClasses(unittest.TestCase):
    def test_each_dummy_class_instantiable(self):
        for name in ASMPASS_DUMMY_CLASSES:
            with self.subTest(name=name):
                cls = getattr(_asmpass, name)
                self.assertTrue(callable(cls))
                with mock.patch("builtins.print"):
                    instance = cls()
                self.assertIsInstance(instance, cls)


if __name__ == "__main__":
    unittest.main()
