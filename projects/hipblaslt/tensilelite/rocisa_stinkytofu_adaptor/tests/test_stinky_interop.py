# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for ``stinky_interop`` -- the signature wrapper around the lowered module.

The wrapper is the adapter's stand-in for C++
``StinkyAsmModuleWithSignature``, so it owes that class's behaviour on
``emitAssembly``: instruction-byte refresh, SGPR-count refresh, signature
banner, ``.set`` directives, instruction text -- in that order.
"""

import os
import sys
import unittest

_PKG_PARENT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rocisa_stinkytofu_adaptor.stinky_interop import (  # noqa: E402
    StinkyAsmModuleWithAdapterSignature,
    _apply_optimization_config,
)


class _FakeKernelDescriptor:
    def __init__(self, totalSgprs):
        self.totalVgprs = 256
        self.totalAgprs = 0
        self.totalSgprs = totalSgprs
        self.numSgprPreload = 26
        self.sgprWorkGroup = (1, 1, 1)


class _FakeSignature:
    """Duck-typed stand-in for ``code.SignatureBase`` (no arch caps needed)."""

    def __init__(self, totalSgprs=80):
        self.kernelDescriptor = _FakeKernelDescriptor(totalSgprs)
        self.setGprsCalls = []

    def setGprs(self, totalVgprs, totalAgprs, totalSgprs):
        self.setGprsCalls.append((totalVgprs, totalAgprs, totalSgprs))
        self.kernelDescriptor.totalVgprs = totalVgprs
        self.kernelDescriptor.totalAgprs = totalAgprs
        self.kernelDescriptor.totalSgprs = totalSgprs

    def toString(self):
        return f"sgprs={self.kernelDescriptor.totalSgprs}\n"


class _FakeModule:
    """Lowered module exposing only what the wrapper calls."""

    def __init__(self, required=None):
        self._required = required
        self.requiredCalls = []

    def emitAssembly(self):
        return "body\n"

    def getTotalInstructionBytes(self):
        return -1

    def getRequiredSgprCount(self, numSgprPreload, workgroupIds):
        self.requiredCalls.append((numSgprPreload, list(workgroupIds)))
        return self._required


class _FakeModuleWithoutCounter(_FakeModule):
    """Pre-dates the ``getRequiredSgprCount`` binding."""

    getRequiredSgprCount = None


class TestRefreshSgprCount(unittest.TestCase):
    def test_lowers_the_count_to_what_the_code_names(self):
        sig = _FakeSignature(totalSgprs=80)
        wrapper = StinkyAsmModuleWithAdapterSignature(_FakeModule(required=78), sig)
        self.assertEqual(wrapper.emitAssembly(), "sgprs=78\nbody\n")
        self.assertEqual(sig.setGprsCalls, [(256, 0, 78)])

    def test_never_raises_the_count(self):
        sig = _FakeSignature(totalSgprs=80)
        wrapper = StinkyAsmModuleWithAdapterSignature(_FakeModule(required=96), sig)
        wrapper.emitAssembly()
        self.assertEqual(sig.setGprsCalls, [])
        self.assertEqual(sig.kernelDescriptor.totalSgprs, 80)

    def test_zero_keeps_the_producers_number(self):
        # No function names an SGPR: nothing was measured, so nothing moves.
        sig = _FakeSignature(totalSgprs=80)
        wrapper = StinkyAsmModuleWithAdapterSignature(_FakeModule(required=0), sig)
        wrapper.emitAssembly()
        self.assertEqual(sig.setGprsCalls, [])

    def test_forwards_the_abi_floor_inputs(self):
        sig = _FakeSignature(totalSgprs=80)
        module = _FakeModule(required=78)
        StinkyAsmModuleWithAdapterSignature(module, sig).emitAssembly()
        self.assertEqual(module.requiredCalls, [(26, [1, 1, 1])])

    def test_older_binding_without_the_counter_is_a_no_op(self):
        sig = _FakeSignature(totalSgprs=80)
        wrapper = StinkyAsmModuleWithAdapterSignature(
            _FakeModuleWithoutCounter(), sig
        )
        self.assertEqual(wrapper.emitAssembly(), "sgprs=80\nbody\n")
        self.assertEqual(sig.setGprsCalls, [])

    def test_no_signature_emits_body_only(self):
        module = _FakeModule(required=78)
        wrapper = StinkyAsmModuleWithAdapterSignature(module, None)
        self.assertEqual(wrapper.emitAssembly(), "body\n")
        self.assertEqual(module.requiredCalls, [])


class TestApplyOptimizationConfig(unittest.TestCase):
    def test_forwards_module_options_and_drops_textblock_copy(self):
        class _Sig:
            def __init__(self):
                self.args = None
                self.descriptionTopic = "keep-me"
                self.cleared = False

            def setOptimizationConfig(self, *args):
                self.args = args

            def clearDescription(self):
                self.cleared = True

        sig = _Sig()
        _apply_optimization_config(
            sig,
            {
                "TileA0": 64,
                "TileB0": 8,
                "SubGroup0": 4,
                "SubGroup1": 32,
                "WaveGroup0": 2,
                "WaveGroup1": 1,
                "VectorWidthA": 1,
                "VectorWidthB": 1,
                "GlobalReadVectorWidthA": 2,
                "GlobalReadVectorWidthB": 2,
                "DirectToLdsA": False,
                "DirectToLdsB": False,
                "UseSgprForGRO": 0,
            },
        )
        self.assertEqual(
            sig.args,
            ((64, 8), (4, 32), (2, 1), 1, 1, 2, 2, False, False, 0),
        )
        self.assertEqual(sig.descriptionTopic.text, "")
        self.assertTrue(sig.cleared)

    def test_missing_setter_or_options_is_a_no_op(self):
        class _Bare:
            pass

        _apply_optimization_config(_Bare(), {"TileA0": 1})
        sig = _FakeSignature()
        _apply_optimization_config(sig, None)


if __name__ == "__main__":
    unittest.main()
