################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
"""Standalone tests for ``rocisa_stinkytofu_adaptor.functions``.

Run from any working directory:

    python3 projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_functions.py

Or with pytest if available:

    pytest projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_functions.py

Section A — ``ArgumentLoader`` (real): offset-bookkeeping contract that
Tensile's KernelWriterAssembly relies on (``self.argLoader.getOffset()``
is read directly to compute ``s_load_b*`` immediates). Instruction
emission is stubbed; only byte advancement is checked here.

Section B — dummy exports (structural): every ``make_dummy_func`` symbol
in ``functions.py`` must import, be callable, and return ``None``.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

# ---------------------------------------------------------------------------
# Self-contained sys.path bootstrap so the test runs without any install /
# editable-mode setup. The ``rocisa_stinkytofu_adaptor`` Python package
# lives at:
#     projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/rocisa_stinkytofu_adaptor/
# This file lives at:
#     projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_functions.py
# So the package's parent directory (where ``import
# rocisa_stinkytofu_adaptor`` resolves) is one level up.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rocisa_stinkytofu_adaptor import functions as _functions  # noqa: E402
from rocisa_stinkytofu_adaptor.functions import ArgumentLoader  # noqa: E402


# ---------------------------------------------------------------------------
# Registry of dummy function exports in ``functions.py``.
# ---------------------------------------------------------------------------

FUNCTIONS_DUMMY_EXPORTS: tuple[str, ...] = (
    # Branch helpers
    "BranchIfZero",
    "BranchIfNotZero",
    # Cast helper
    "VSaturateCastInt",
    # Vector math
    "vectorStaticDivideAndRemainder",
    "vectorStaticDivide",
    "vectorUInt32DivideAndRemainder",
    "vectorUInt32CeilDivideAndRemainder",
    "vectorStaticRemainder",
    "vectorStaticMultiply",
    "vectorStaticMultiplyAdd",
    "vectorAddMultiplyBpe",
    "vectorMultiplyBpe",
    "vectorMultiply64Bpe",
    # Scalar math
    "scalarStaticDivideAndRemainder",
    "scalarStaticCeilDivide",
    "scalarStaticRemainder",
    "scalarUInt24DivideAndRemainder",
    "scalarUInt32DivideAndRemainder",
    "scalarStaticMultiply64",
    "scalarMultiplyBpe",
    "scalarMultiply64Bpe",
    # Magic divide
    "sMagicDiv",
    "sMagicDiv2",
    # DS init
    "DSInit",
)


# ===========================================================================
# Section A — ArgumentLoader (real implementation)
# ===========================================================================


class TestArgumentLoaderConstruction(unittest.TestCase):
    def test_initial_offset_is_zero(self):
        # Mirrors ``ArgumentLoader() : kernArgOffset(0)`` in argument.hpp:34.
        loader = ArgumentLoader()
        self.assertEqual(loader.getOffset(), 0)

    def test_returns_int_not_dummy(self):
        # The whole point of this workaround: ``getOffset()`` must be a real
        # ``int`` because Tensile does ``getOffset() - numSgprPreload * 4``.
        loader = ArgumentLoader()
        self.assertIsInstance(loader.getOffset(), int)


class TestArgumentLoaderSetGetReset(unittest.TestCase):
    def test_setOffset_then_getOffset(self):
        loader = ArgumentLoader()
        loader.setOffset(64)
        self.assertEqual(loader.getOffset(), 64)

    def test_setOffset_overwrites(self):
        loader = ArgumentLoader()
        loader.setOffset(64)
        loader.setOffset(128)
        self.assertEqual(loader.getOffset(), 128)

    def test_setOffset_coerces_to_int(self):
        loader = ArgumentLoader()
        loader.setOffset(48)
        self.assertIsInstance(loader.getOffset(), int)

    def test_resetOffset_zeros(self):
        loader = ArgumentLoader()
        loader.setOffset(96)
        loader.resetOffset()
        self.assertEqual(loader.getOffset(), 0)

    def test_resetOffset_after_loads(self):
        loader = ArgumentLoader()
        loader.loadKernArg("AddressDbg", "KernArgAddress", dword=2)
        loader.resetOffset()
        self.assertEqual(loader.getOffset(), 0)


class TestArgumentLoaderLoadKernArg(unittest.TestCase):
    def test_default_dword_advances_4_bytes(self):
        loader = ArgumentLoader()
        loader.loadKernArg("AddressDbg", "KernArgAddress")
        self.assertEqual(loader.getOffset(), 4)

    def test_dword_2_advances_8_bytes(self):
        loader = ArgumentLoader()
        loader.loadKernArg("AddressDbg", "KernArgAddress", dword=2)
        self.assertEqual(loader.getOffset(), 8)

    def test_dword_4_advances_16_bytes(self):
        loader = ArgumentLoader()
        loader.loadKernArg("AddressDbg", "KernArgAddress", dword=4)
        self.assertEqual(loader.getOffset(), 16)

    def test_repeated_calls_accumulate(self):
        loader = ArgumentLoader()
        loader.loadKernArg("a", "KernArgAddress", dword=1)
        loader.loadKernArg("b", "KernArgAddress", dword=2)
        loader.loadKernArg("c", "KernArgAddress", dword=4)
        self.assertEqual(loader.getOffset(), 4 + 8 + 16)

    def test_with_sgprOffset_does_not_advance(self):
        # ``kernArgOffset += sgprOffset ? 0 : size`` — explicit sgprOffset
        # means the caller is providing its own offset, so the loader must
        # NOT bump kernArgOffset (argument.hpp:119).
        loader = ArgumentLoader()
        loader.loadKernArg("AddressDbg", "KernArgAddress",
                           sgprOffset=hex(64), dword=2)
        self.assertEqual(loader.getOffset(), 0)

    def test_with_sgprOffset_int_does_not_advance(self):
        # InstructionInput in C++ accepts both int and shared_ptr<RegisterContainer>;
        # in Python both surface as just "non-None" — covered identically.
        loader = ArgumentLoader()
        loader.loadKernArg("AddressDbg", "KernArgAddress",
                           sgprOffset=64, dword=2)
        self.assertEqual(loader.getOffset(), 0)

    def test_writeSgpr_false_still_advances(self):
        # The C++ advances outside the ``if(writeSgpr)`` block; this case
        # corresponds to "skip unused parm" (argument.hpp:57-58).
        loader = ArgumentLoader()
        loader.loadKernArg("UnusedParm", "KernArgAddress",
                           dword=2, writeSgpr=False)
        self.assertEqual(loader.getOffset(), 8)

    def test_returns_none(self):
        # Stubbed emission today; document the contract.
        loader = ArgumentLoader()
        self.assertIsNone(loader.loadKernArg("a", "KernArgAddress"))


class TestArgumentLoaderLoadAllKernArg(unittest.TestCase):
    def test_basic_total_advance(self):
        loader = ArgumentLoader()
        loader.loadAllKernArg(sgprStartIndex=0, srcAddr="KernArgAddress",
                              numSgprToLoad=8)
        self.assertEqual(loader.getOffset(), 8 * 4)

    def test_with_preload_total_advance_is_full(self):
        # numSgprPreload * 4 (initial) + actualLoad * 4 (chunked) = numSgprToLoad * 4
        loader = ArgumentLoader()
        loader.loadAllKernArg(sgprStartIndex=0, srcAddr="KernArgAddress",
                              numSgprToLoad=20, numSgprPreload=4)
        self.assertEqual(loader.getOffset(), 20 * 4)

    def test_preload_equals_total_still_advances_full(self):
        # Edge: every sgpr is preloaded (actualLoad == 0); only the initial
        # ``kernArgOffset += numSgprPreload * 4`` runs.
        loader = ArgumentLoader()
        loader.loadAllKernArg(sgprStartIndex=0, srcAddr="KernArgAddress",
                              numSgprToLoad=8, numSgprPreload=8)
        self.assertEqual(loader.getOffset(), 8 * 4)

    def test_zero_load_is_noop(self):
        loader = ArgumentLoader()
        loader.loadAllKernArg(sgprStartIndex=0, srcAddr="KernArgAddress",
                              numSgprToLoad=0)
        self.assertEqual(loader.getOffset(), 0)

    def test_chained_with_loadKernArg(self):
        loader = ArgumentLoader()
        loader.loadAllKernArg(sgprStartIndex=0, srcAddr="KernArgAddress",
                              numSgprToLoad=4)  # +16
        loader.loadKernArg("Foo", "KernArgAddress", dword=2)  # +8
        loader.loadKernArg("Bar", "KernArgAddress",
                           sgprOffset=0, dword=2)  # +0
        self.assertEqual(loader.getOffset(), 16 + 8)

    def test_returns_none(self):
        loader = ArgumentLoader()
        self.assertIsNone(loader.loadAllKernArg(0, "KernArgAddress", 4))


class TestArgumentLoaderTensileRegression(unittest.TestCase):
    def test_kernarg_wait_arithmetic_does_not_raise(self):
        # Reproduce the L1909-1916 + L2351 pattern: reset, loadAllKernArg,
        # then read getOffset() and subtract numSgprPreload*4.
        loader = ArgumentLoader()
        numSgprPreload = 4
        numsOfLoad = 24

        loader.resetOffset()
        loader.loadAllKernArg(sgprStartIndex=0, srcAddr="KernArgAddress",
                              numSgprToLoad=numsOfLoad,
                              numSgprPreload=numSgprPreload)

        # The expression that crashed pre-workaround:
        kernArgBytes = loader.getOffset() - numSgprPreload * 4
        self.assertEqual(kernArgBytes, (numsOfLoad - numSgprPreload) * 4)

    def test_two_loaders_independent(self):
        # KernelWriterAssembly creates ``argLoader`` and ``externalArgLoader``
        # (KernelWriterAssembly.py:2104-2105); state must not leak between
        # instances.
        a = ArgumentLoader()
        b = ArgumentLoader()
        a.loadKernArg("x", "KernArgAddress", dword=4)
        self.assertEqual(a.getOffset(), 16)
        self.assertEqual(b.getOffset(), 0)


# ===========================================================================
# Section B — dummy function exports (structural smoke)
# ===========================================================================


class TestFunctionsModuleExports(unittest.TestCase):
    def test_argument_loader_is_real_class(self):
        self.assertTrue(callable(ArgumentLoader))
        self.assertIsInstance(ArgumentLoader(), ArgumentLoader)

    def test_all_dummy_symbols_exported(self):
        for name in FUNCTIONS_DUMMY_EXPORTS:
            with self.subTest(name=name):
                self.assertTrue(hasattr(_functions, name),
                                f"functions.{name} missing")

    def test_dummy_registry_matches_module(self):
        # Guard against adding a dummy to functions.py without updating tests.
        _skip = frozenset({"annotations", "make_dummy_func"})
        module_dummies = {
            name for name in dir(_functions)
            if not name.startswith("_")
            and name not in _skip
            and name != "ArgumentLoader"
            and callable(getattr(_functions, name))
        }
        self.assertEqual(set(FUNCTIONS_DUMMY_EXPORTS), module_dummies)


class TestFunctionsDummyCallables(unittest.TestCase):
    def test_each_dummy_callable_returns_none(self):
        for name in FUNCTIONS_DUMMY_EXPORTS:
            with self.subTest(name=name):
                fn = getattr(_functions, name)
                self.assertTrue(callable(fn))
                with mock.patch("builtins.print"):
                    self.assertIsNone(fn())


if __name__ == "__main__":
    unittest.main()
