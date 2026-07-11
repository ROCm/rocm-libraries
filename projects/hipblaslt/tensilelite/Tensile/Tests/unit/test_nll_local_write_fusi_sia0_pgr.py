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

from unittest.mock import MagicMock, patch

import pytest
from rocisa.code import Module, TextBlock

from Tensile.KernelWriter import KernelWriter

pytestmark = pytest.mark.unit


def _marker_module(name, marker):
    module = Module(name)
    module.add(TextBlock("%s\n" % marker))
    return module


def _make_kernel(**overrides):
    k = {
        # all keys accessed anywhere in noLoadLoopBody
        "UnrollLoopSwapGlobalReadOrder": 0,
        "DirectToLdsA": False,
        "DirectToLdsB": False,
        "UseCustomMainLoopSchedule": False,
        "ExpandPointerSwap": False,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprMXSA": False,
        "DirectToVgprMXSB": False,
        "DirectToVgprSparseMetadata": False,
        "enableTDMA": False,
        "enableTDMB": False,
        "enableTDMMetadata": False,
        "EnableMatrixInstruction": False,
        "UseF32XEmulation": False,
        "AdaptiveGemmNTAB": 0,
        "NumWaves": 1,
        "LoopIters": 4,
        "numSubTiles": 1,
        "InnerUnroll": 1,
        # the four guards in the fix
        "NoLdsWriteCode": False,
        "ForceUnrollSubIter": True,
        "_ScheduleIterAlg": 0,
        "PrefetchGlobalRead": 2,
        # fields read in the per-iteration body
        "ProblemType": {
            "Sparse": False,
            "MXBlockA": 0,
            "MXBlockB": 0,
            "Gradient": False,
            "UseBias": False,
            "BiasSrc": "",
        },
    }
    k.update(overrides)
    return k


def _make_tps():
    tpA = {"is_sparse": False, "tpsMetadata": None, "tensorChar": "A"}
    tpB = {"is_sparse": False, "tpsMetadata": None, "tensorChar": "B"}
    return tpA, tpB


def _call_nll_body(kernel):
    """
    Build a MagicMock self and call KernelWriter.noLoadLoopBody directly
    with isNGLL=False (NLL path).  Only codes.localWriteA/B are explicitly
    configured; everything else auto-mocks so the loop body doesn't crash.
    """
    tpA, tpB = _make_tps()

    self = MagicMock()

    # states fields read before / during the loop
    self.states.numItersPLR = 1
    self.states.doPackPreSchedulingNextLoop = False
    self.states.lockLdsReadTokenSwap = False
    self.states.doFullPackCodePrefetch = False
    self.states.numVgprBuffer = 1
    self.states.numPackBuffer = 1
    # numIterPerCoalesced* must be numeric for the doRead* comparisons
    self.states.numIterPerCoalescedReadA = 1
    self.states.numIterPerCoalescedReadB = 1
    self.states.numIterPerCoalescedReadMXSA = 1
    self.states.numIterPerCoalescedReadMXSB = 1
    self.states.numIterPerCoalescedReadMetadata = 1
    self.states.numReadsIterCoalescedA = 1
    self.states.numReadsIterCoalescedB = 1
    self.states.numReadsIterCoalescedMXSA = 1
    self.states.numReadsIterCoalescedMXSB = 1
    self.states.numReadsIterCoalescedMetadata = 1

    # codes fields read after makeSchedule (unrollLoopHeader) and in the loop
    self.codes.unrollLoopHeader = Module("unrollLoopHeader")
    self.codes.perIterGlobalRead = [Module() for _ in range(kernel["LoopIters"])]

    # localWriteDo returns a real marker module keyed by tensorChar
    def _local_write_do(k, tP, swapAB=0):
        tag = "ds_write_%s" % tP["tensorChar"].lower()
        return _marker_module("localWrite", tag)

    self.localWriteDo.side_effect = _local_write_do

    # All methods whose return value is passed to Module.add() (a C++ binding)
    # must return a real Module, not a MagicMock.
    self.localReadDo.return_value = (Module(), Module(), Module())
    self.localReadInc.return_value = Module()
    self.localWriteSwapOffsets.return_value = Module()
    self.localReadSwapOffsets.return_value = Module()
    self.localReadInitPointers.return_value = Module()
    self._wait.return_value = Module()
    self.macIter.return_value = Module()
    self.mfmaIter.return_value = Module()
    self._makeSubIterSchedule.return_value = Module()
    self.closeLoop.return_value = Module()

    pack     = [Module() for _ in range(kernel["LoopIters"])]
    packPre  = [Module() for _ in range(kernel["LoopIters"])]

    KernelWriter.noLoadLoopBody(
        self, kernel, tpA, tpB,
        pack, packPre,
        isOptNLL=False, isNGLL=False,
        NLLfirst=True, NLLlast=True,
    )

    return self


def test_fusi_sia0_pgr_nll_emits_local_write():
    """FUSI+SIA0+PGR: noLoadLoopBody must call localWriteDo and populate codes.localWriteA/B."""
    kernel = _make_kernel()
    self = _call_nll_body(kernel)

    assert "ds_write_a" in str(self.codes.localWriteA)
    assert "ds_write_b" in str(self.codes.localWriteB)