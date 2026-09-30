# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
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

    def test_returns_sload_instruction(self):
        from rocisa_stinkytofu_adaptor.instruction import SLoadB32, SLoadB64
        loader = ArgumentLoader()
        item = loader.loadKernArg("a", "KernArgAddress", dword=1)
        self.assertIsInstance(item, SLoadB32)
        item2 = loader.loadKernArg("b", "KernArgAddress", dword=2)
        self.assertIsInstance(item2, SLoadB64)

    def test_writeSgpr_false_returns_textblock(self):
        from rocisa_stinkytofu_adaptor.code import TextBlock
        loader = ArgumentLoader()
        item = loader.loadKernArg("x", "y", dword=2, writeSgpr=False)
        self.assertIsInstance(item, TextBlock)


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

    def test_returns_module(self):
        from rocisa_stinkytofu_adaptor.code import Module
        loader = ArgumentLoader()
        mod = loader.loadAllKernArg(0, "KernArgAddress", 4)
        self.assertIsInstance(mod, Module)

    def test_module_contains_sload_instructions(self):
        from rocisa_stinkytofu_adaptor.instruction import SLoadB128
        loader = ArgumentLoader()
        mod = loader.loadAllKernArg(0, "KernArgAddress", 4)
        self.assertEqual(len(mod.itemList), 1)
        self.assertIsInstance(mod.itemList[0], SLoadB128)

    def test_greedy_packing(self):
        from rocisa_stinkytofu_adaptor.instruction import (
            SLoadB32, SLoadB64, SLoadB128, SLoadB512,
        )
        loader = ArgumentLoader()
        mod = loader.loadAllKernArg(0, "KernArgAddress", 20)
        self.assertEqual(len(mod.itemList), 2)
        self.assertIsInstance(mod.itemList[0], SLoadB512)
        self.assertIsInstance(mod.itemList[1], SLoadB128)

    def test_unaligned_start(self):
        from rocisa_stinkytofu_adaptor.instruction import (
            SLoadB32, SLoadB64, SLoadB128,
        )
        loader = ArgumentLoader()
        mod = loader.loadAllKernArg(1, "KernArgAddress", 7)
        self.assertEqual(len(mod.itemList), 3)
        self.assertIsInstance(mod.itemList[0], SLoadB32)
        self.assertIsInstance(mod.itemList[1], SLoadB64)
        self.assertIsInstance(mod.itemList[2], SLoadB128)

    def test_countSMemLoad_on_output(self):
        from rocisa_stinkytofu_adaptor import countSMemLoad
        loader = ArgumentLoader()
        mod = loader.loadAllKernArg(0, "KernArgAddress", 20)
        self.assertEqual(countSMemLoad(mod), 2)


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
        _skip = frozenset({
            "annotations", "make_dummy_func", "math",
            "ArgumentLoader",
            "Module", "TextBlock", "sgpr", "vgpr", "Any",
            "ContinuousRegister", "EXEC", "VCC",
            "DataTypeEnum",
            # Real branch helpers (no longer dummies)
            "BranchIfZero", "BranchIfNotZero",
            "SLoadB32", "SLoadB64", "SLoadB128", "SLoadB256", "SLoadB512",
            "SAddCU32", "SAddU32", "SAndB32", "SAndB64",
            "SCBranchSCC0", "SCBranchSCC1", "SCBranchVCCNZ", "SCBranchVCCZ",
            "SCmpEQU32", "SCmpEQU64", "SCmpLgU32",
            "SLShiftLeftB32", "SLShiftLeftB64", "SLShiftRightB32",
            "SLShiftRightB64", "SMulHIU32", "SMulI32", "SMovB32", "SMovB64",
            "SNop", "SSubU32",
            "VAddCCOU32", "VAddLShiftLeftU32", "VAddU32", "VAndB32",
            "VCmpEQF32", "VCmpEQF64",
            "VCmpNeU32", "VCmpXEqU32", "VCmpXGeU32", "VCmpXGtU32",
            "VCvtF32toU32", "VCvtF64toU32", "VCvtU32toF32", "VCvtU32toF64",
            "VLShiftLeftAddU32", "VLShiftLeftB32", "VLShiftLeftB64",
            "VLShiftRightB32", "VLShiftRightB64", "VMadU32U24", "VMovB32",
            "VMulF32", "VMulF64", "VMulHIU32", "VMulLOU32", "VMulU32U24",
            "VRcpF64", "VRcpIFlagF32", "VReadfirstlaneB32", "VSubU32",
            # Real math functions (no longer dummies)
            "vectorStaticDivideAndRemainder", "vectorStaticDivide",
            "vectorUInt32DivideAndRemainder",
            "vectorUInt32CeilDivideAndRemainder",
            "vectorStaticRemainder", "vectorStaticMultiply",
            "vectorStaticMultiplyAdd", "vectorAddMultiplyBpe",
            "vectorMultiplyBpe", "vectorMultiply64Bpe",
            "scalarStaticDivideAndRemainder", "scalarStaticCeilDivide",
            "scalarStaticRemainder", "scalarUInt24DivideAndRemainder",
            "scalarUInt32DivideAndRemainder", "scalarStaticMultiply64",
            "scalarMultiplyBpe", "scalarMultiply64Bpe",
            "sMagicDiv", "sMagicDiv2",
            # Real cast helper (no longer dummy)
            "VSaturateCastInt",
            "SMovkI32", "VMed3I32", "VMinI32", "VMaxI32",
            "SaturateCastType",
        })
        module_dummies = {
            name for name in dir(_functions)
            if not name.startswith("_")
            and name not in _skip
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


# ==========================================================================
# Section C — Counting / analysis functions (real, in __init__.py)
# ==========================================================================

from rocisa_stinkytofu_adaptor import (  # noqa: E402
    countType, countInstruction, countGlobalRead, countSMemLoad,
    countLocalRead, countLocalWrite, countWeightedLocalRead,
    countWeightedLocalWrite, countDSStoreB128, countDSStoreB192,
    countDSStoreB256, countVMovB32, countMFMA, getMFMAs, findInstCount,
)
from rocisa_stinkytofu_adaptor.code import Module, TextBlock  # noqa: E402
from rocisa_stinkytofu_adaptor.instruction import (  # noqa: E402
    Instruction, BufferLoadB128, BufferLoadB32, FlatLoadB64,
    DSLoadB32, DSLoadB64, DSLoadB192, DSLoad2B32,
    DSStoreB32, DSStoreB64, DSStoreB128, DSStoreB192, DSStoreB256,
    DSStore2B32, VMovB32, SLoadB32, SLoadB128,
    GlobalLoadTR8B64, MFMAInstruction, SMFMAInstruction, MXMFMAInstruction,
    SAddU32, SNop,
)


class TestCountInstruction(unittest.TestCase):
    def test_empty_module(self):
        self.assertEqual(countInstruction(Module()), 0)

    def test_flat_module(self):
        m = Module()
        m.add(BufferLoadB128())
        m.add(DSLoadB32())
        m.add(DSStoreB32())
        self.assertEqual(countInstruction(m), 3)

    def test_nested_module(self):
        m = Module()
        m.add(BufferLoadB128())
        sub = Module()
        sub.add(DSLoadB32())
        sub.add(DSStoreB32())
        m.add(sub)
        self.assertEqual(countInstruction(m), 3)

    def test_skips_textblock(self):
        m = Module()
        m.add(TextBlock("// comment\n"))
        m.add(BufferLoadB128())
        self.assertEqual(countInstruction(m), 1)


class TestCountGlobalRead(unittest.TestCase):
    def test_buffer_loads(self):
        m = Module()
        m.add(BufferLoadB128())
        m.add(BufferLoadB32())
        m.add(DSLoadB32())
        self.assertEqual(countGlobalRead(m), 2)

    def test_flat_loads(self):
        m = Module()
        m.add(FlatLoadB64())
        self.assertEqual(countGlobalRead(m), 1)

    def test_global_load_tr(self):
        m = Module()
        m.add(GlobalLoadTR8B64())
        self.assertEqual(countGlobalRead(m), 1)

    def test_no_false_positives(self):
        m = Module()
        m.add(DSLoadB32())
        m.add(SLoadB32())
        self.assertEqual(countGlobalRead(m), 0)


class TestCountSMemLoad(unittest.TestCase):
    def test_sloads(self):
        m = Module()
        m.add(SLoadB32())
        m.add(SLoadB128())
        m.add(BufferLoadB128())
        self.assertEqual(countSMemLoad(m), 2)


class TestCountLocalRead(unittest.TestCase):
    def test_ds_loads(self):
        m = Module()
        m.add(DSLoadB32())
        m.add(DSLoadB64())
        m.add(DSLoad2B32())
        m.add(DSStoreB32())
        self.assertEqual(countLocalRead(m), 3)

    def test_nested(self):
        m = Module()
        sub = Module()
        sub.add(DSLoadB32())
        m.add(sub)
        m.add(DSLoadB64())
        self.assertEqual(countLocalRead(m), 2)


class TestCountLocalWrite(unittest.TestCase):
    def test_ds_stores(self):
        m = Module()
        m.add(DSStoreB32())
        m.add(DSStoreB64())
        m.add(DSStore2B32())
        m.add(DSLoadB32())
        self.assertEqual(countLocalWrite(m), 3)


class TestCountWeighted(unittest.TestCase):
    def test_weighted_local_read(self):
        m = Module()
        m.add(DSLoadB32())
        m.add(DSLoadB192())
        self.assertEqual(countWeightedLocalRead(m), 3)  # 1 + 2

    def test_weighted_local_write(self):
        m = Module()
        m.add(DSStoreB128())
        m.add(DSStoreB192())
        m.add(DSStoreB256())
        self.assertEqual(countWeightedLocalWrite(m), 5)  # 1 + 2 + 2


class TestCountExactType(unittest.TestCase):
    def test_ds_store_b128(self):
        m = Module()
        m.add(DSStoreB128())
        m.add(DSStoreB192())
        self.assertEqual(countDSStoreB128(m), 1)
        self.assertEqual(countDSStoreB192(m), 1)
        self.assertEqual(countDSStoreB256(m), 0)

    def test_vmov_b32(self):
        m = Module()
        m.add(VMovB32(dst="v0", src="v1"))
        m.add(VMovB32(dst="v2", src="v3"))
        self.assertEqual(countVMovB32(m), 2)


class TestCountMFMA(unittest.TestCase):
    def test_no_mfma(self):
        m = Module()
        m.add(BufferLoadB128())
        self.assertEqual(countMFMA(m), 0)
        self.assertEqual(getMFMAs(m), [])


class TestFindInstCount(unittest.TestCase):
    def test_found(self):
        target = DSLoadB32()
        m = Module()
        m.add(BufferLoadB128())
        m.add(target)
        cnt, found = findInstCount(m, target)
        self.assertTrue(found)
        self.assertEqual(cnt, 1)

    def test_not_found(self):
        target = DSLoadB32()
        m = Module()
        m.add(BufferLoadB128())
        cnt, found = findInstCount(m, target)
        self.assertFalse(found)

    def test_skips_textblock(self):
        target = DSStoreB32()
        m = Module()
        m.add(BufferLoadB128())
        m.add(TextBlock("// skip me\n"))
        m.add(DSLoadB32())
        m.add(target)
        cnt, found = findInstCount(m, target)
        self.assertTrue(found)
        self.assertEqual(cnt, 2)

    def test_nested_module(self):
        target = DSLoadB32()
        m = Module()
        sub = Module()
        sub.add(BufferLoadB128())
        sub.add(target)
        m.add(sub)
        cnt, found = findInstCount(m, target)
        self.assertTrue(found)
        self.assertEqual(cnt, 1)


class TestCountType(unittest.TestCase):
    def test_generic_isinstance(self):
        m = Module()
        m.add(BufferLoadB128())
        m.add(DSLoadB32())
        m.add(TextBlock("// x\n"))
        self.assertEqual(countType(m, Instruction), 2)


if __name__ == "__main__":
    unittest.main()
