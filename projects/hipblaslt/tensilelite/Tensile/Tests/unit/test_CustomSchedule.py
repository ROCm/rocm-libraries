################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import pytest
from unittest.mock import MagicMock

from Tensile.Components.CustomSchedule import hasCustomSchedule, ScheduleInfo
from Tensile.Components.CMSValidator import isValid, SchedulePosition, ValidationContext
from Tensile.Common import IsaVersion
from cms_test_utils import make_mock_id_map, make_mock_mfma_code


def _make_context(kernel, schedule_info):
    """Build a ValidationContext from mock idMap + mfmaCode.

    The per-shape positive-validation tests that used to live in this file
    (TestCustomScheduleBF16 / TestCustomScheduleTF32) were removed because
    they reproduced what `Tests/common/gemm/gfx950/custom_mainloop_scheduling*.yaml`
    already exercise via the full Tensile pipeline (which runs both the
    structural-rule isValid AND the dataflow-graph compare_graphs +
    validate_edge_wait_coverage as production gates). The remaining tests
    here exercise validator API surface and schedule-position ordering —
    neither of which needs the real-data kernel-writer path.
    """
    return ValidationContext(
        kernel=kernel,
        id_map=make_mock_id_map(schedule_info, kernel),
        mfma_code=make_mock_mfma_code(schedule_info.numMfma),
    )


# Helper to create a mock data type
def _mock_dtype(is_16bit=False, is_8bit=False, num_bytes=4):
    mock = MagicMock()
    mock.isHalf.return_value = is_16bit
    mock.isBFloat16.return_value = False # Assuming isHalf is enough for is16bit
    mock.isInt8.return_value = is_8bit
    mock.is8bitFloat.return_value = False # Assuming isInt8 is enough for is8bit
    mock.numBytes.return_value = num_bytes
    return mock

# Base kernel configuration factory
def create_base_kernel():
    kernel = {
        "UseCustomMainLoopSchedule": True,
        "EnableMatrixInstruction": True,
        "UnrollLoopSwapGlobalReadOrder": False,
        "ISA": IsaVersion(9,5,0),
        "WavefrontSize": 64,
        "ProblemType": {
            "DataType": _mock_dtype(),
            "DataTypeA": _mock_dtype(),
            "DataTypeB": _mock_dtype(),
            "TransposeA": False,
            "TransposeB": False,
            "TLUA": True,
            "TLUB": False,
        },
        "MacroTile0": 0, "MacroTile1": 0, "DepthU": 64,
        "PrefetchGlobalRead": 0, "PrefetchLocalRead": 0, "DirectToLds": 1,  "DtlPlusLdsBuf": False,
        "GlobalReadVectorWidthA": 0, "GlobalReadVectorWidthB": 0,
        "LocalReadVectorWidthA": 0, "LocalReadVectorWidthB": 0,
        "WaveSeparateGlobalReadA": 0,
        "WaveSeparateGlobalReadB": 0,
        "Use64bShadowLimit" : 1,
        "MatrixInstruction": [16,16,32,1],
        "MIWaveGroup": [],
        "LDSTrInst": False,
        "TransposeLDS": 0,
        "ForceUnrollSubIter": False,
        "SwapGlobalReadOrder": False, # For asserting it gets set
        "UsePLRPack": False, # For asserting it gets set
        "UseF32XEmulation": False,
        "VectorWidthA": 1,
        "VectorWidthB": 1,
        "MIWaveTileA": 2,
        "MIWaveTileB": 2,
    }
    return kernel

def update_kernel(kernel, updates):
    """Update kernel dict, auto-deriving TLUA/TLUB from TransposeA/TransposeB.

    Args:
        kernel: kernel dict to modify in-place
        updates: dict mirroring the kernel structure. "ProblemType" key (if present)
                 is applied via kernel["ProblemType"].update(); all other keys are
                 applied via kernel.update(). If TransposeA or TransposeB appear in
                 the ProblemType sub-dict, TLUA and TLUB are auto-derived.
    """
    if "ProblemType" in updates:
        pt = updates.pop("ProblemType")
        if "TransposeA" in pt or "TransposeB" in pt:
            transA = pt.get("TransposeA", kernel["ProblemType"]["TransposeA"])
            transB = pt.get("TransposeB", kernel["ProblemType"]["TransposeB"])
            pt["TLUA"] = not transA
            pt["TLUB"] = transB
        kernel["ProblemType"].update(pt)
    kernel.update(updates)


class TestCustomScheduleValidation:
    def test_no_custom_schedule(self):
        """A kernel that doesn't match any registered schedule returns False."""
        kernel = create_base_kernel()
        # An empty kernel should not have a custom schedule
        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert not has_schedule
        assert schedule_info is None

class TestSchedulePositionOrdering:
    """Test that SchedulePosition comparison correctly orders by
    `(loop_index, stream_index)`.

    After the rocm-libraries-5v4u collapse, the historical
    `(loop_index, vmfma_index, sub_index)` triple is folded into a single
    monotonic `stream_index` per body — the bridge picks values that
    preserve the original lex ordering. So the wrap-around "vmfma=-1" slot
    is simply assigned a smaller `stream_index` than the first-of-body slot,
    and these tests check the resulting `(loop_index, stream_index)`
    ordering directly.
    """

    def test_negative_stream_after_last_same_loop(self):
        """stream=-1 in loop 0 must be < stream=0 in loop 0."""
        a = SchedulePosition(loop_index=0, stream_index=-1)
        b = SchedulePosition(loop_index=0, stream_index=0)
        assert a < b
        assert b > a

    def test_loop1_negative_stream_after_loop0_positive(self):
        """stream=-1 in loop 1 must be > stream=2 in loop 0."""
        a = SchedulePosition(loop_index=1, stream_index=-1)
        b = SchedulePosition(loop_index=0, stream_index=2)
        assert a > b
        assert b < a

    def test_negative_stream_before_next_loop_stream0(self):
        """stream=-1 in loop 0 must be < stream=0 in loop 1."""
        a = SchedulePosition(loop_index=0, stream_index=-1)
        b = SchedulePosition(loop_index=1, stream_index=0)
        assert a < b
        assert b > a

    def test_equal_positions(self):
        """Identical positions must be equal."""
        a = SchedulePosition(loop_index=0, stream_index=4)
        b = SchedulePosition(loop_index=0, stream_index=4)
        assert a == b
        assert not (a != b)
        assert not (a < b)
        assert not (a > b)
        assert a <= b
        assert a >= b

    def test_not_equal_different_stream_index(self):
        """Same loop but different stream_index must not be equal."""
        a = SchedulePosition(loop_index=0, stream_index=3)
        b = SchedulePosition(loop_index=0, stream_index=4)
        assert a != b
        assert a < b

    def test_stream_index_ordering(self):
        """Within same loop, stream_index determines order."""
        a = SchedulePosition(loop_index=0, stream_index=2)
        b = SchedulePosition(loop_index=0, stream_index=5)
        c = SchedulePosition(loop_index=0, stream_index=10)
        assert a < b < c
        assert c > b > a
