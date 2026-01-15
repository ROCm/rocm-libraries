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
from Tensile.Components.CMSValidator import isValid
from Tensile.Common import IsaVersion

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
        },
        "MacroTile0": 0, "MacroTile1": 0, "DepthU": 64,
        "PrefetchGlobalRead": 0, "PrefetchLocalRead": 0, "DirectToLds": True,
        "GlobalReadVectorWidthA": 0, "GlobalReadVectorWidthB": 0,
        "LocalReadVectorWidth": 0,
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
        "MIWaveTileA": 2,
        "MIWaveTileB": 2,
    }
    return kernel

class TestCustomScheduleBF16:
    def test_no_custom_schedule(self):
        """Test that a kernel that doesn't match any condition returns False."""
        kernel = create_base_kernel()
        # An empty kernel should not have a custom schedule
        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert not has_schedule
        assert schedule_info is None

    @pytest.mark.parametrize("transA, transB", [(True, False), (False, True), (False, False), (True, True)])
    def test_schedule_256x256x64_16bit(self, transA, transB):
        """Tests the 256x256x64 16-bit schedule."""
        TN = transA and not transB
        NT = not transA and transB
        NN = not transA and not transB
        TT = transA and transB

        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "MacroTile0": 256, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2], "TransposeLDS": 0 if NT else 1, "MIWaveTileA": 8, "MIWaveTileB": 8,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)

        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 128
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message
        if TN:
            assert 'PackA0' not in schedule_info.optSchedule
            assert not kernel["UsePLRPack"]
        elif NT:
            assert 'PackA0' in schedule_info.optSchedule
            assert kernel["UsePLRPack"]
        elif NN:
            assert not kernel["SwapGlobalReadOrder"]
            assert 'PackA0' in schedule_info.optSchedule
            assert 'PackB0' not in schedule_info.optSchedule
            assert kernel["UsePLRPack"]
        elif TT:
            assert kernel["SwapGlobalReadOrder"]
            assert 'PackA0' not in schedule_info.optSchedule
            assert 'PackB0' in schedule_info.optSchedule
            assert kernel["UsePLRPack"]

    @pytest.mark.parametrize("force_unroll_sub_iter", [True, False])
    def test_schedule_256x256x128_8bit_TN(self, force_unroll_sub_iter: bool):
        """Tests the 256x256x128 8-bit TNschedule."""

        kernel = create_base_kernel()
        dtype_8bit = _mock_dtype(is_8bit=True, num_bytes=1)
        kernel["ProblemType"].update({
            "DataType": dtype_8bit, "DataTypeA": dtype_8bit, "DataTypeB": dtype_8bit,
            "TransposeA": True, "TransposeB": False
        })
        kernel.update({
            "MacroTile0": 256, "MacroTile1": 256, "DepthU": 128,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 0,
            "GlobalReadVectorWidthA": 16, "GlobalReadVectorWidthB": 16, "LocalReadVectorWidth": 16,
            "MatrixInstruction": [16,16,128,1], "MIWaveGroup": [2,2], "TransposeLDS": 1, "MIWaveTileA": 8, "MIWaveTileB": 8, "ForceUnrollSubIter": force_unroll_sub_iter,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)

        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 1
        assert schedule_info.numMfma == 64
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    def test_schedule_256x96x64_16bit_TN(self):
        """Tests the 256x96x64 16-bit TN schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": True, "TransposeB": False
        })
        kernel.update({
            "MacroTile0": 256, "MacroTile1": 96, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "TransposeLDS": 1, "MIWaveTileA": 8, "MIWaveTileB": 3,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 48
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize("transA, transB", [(False, False), (False, True), (True, False)])
    def test_schedule_192x256x64_16bit(self, transA, transB):
        """Tests the 192x256x64 16-bit NN schedule."""
        NN = not transA and not transB
        NT = not transA and transB
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "MacroTile0": 192, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": NN, "TransposeLDS": 0 if NT else 1, "MIWaveTileA": 6, "MIWaveTileB": 8,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 96
        if NN:
            assert kernel["SwapGlobalReadOrder"]
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst,  tr_lds", [
        (  True,  False,       False,       1),
        ( False,  False,        True,       1),
        ( False,   True,        True,       0)
        # fmt: on
        ])
    def test_schedule_256x192x64_16bit(self, transA, transB, lds_tr_inst, tr_lds):

        """Tests the 256x192x64 16-bit TN schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "MacroTile0": 256, "MacroTile1": 192, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": 8, "MIWaveTileB": 6,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 96
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize("transA, transB", [(True, False), (False, False), (False, True)])
    def test_schedule_160x256x64_16bit(self, transA, transB):
        """Tests the 160x256x64 16-bit schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "MacroTile0": 160, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": not transA, "TransposeLDS": not transB, "MIWaveTileA": 5, "MIWaveTileB": 8,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 80
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize("transA, transB", [(False, False), (False, True)])
    def test_schedule_256x160x64_16bit(self, transA, transB):
        """Tests the 256x160x64 16-bit schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "MacroTile0": 256, "MacroTile1": 160, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": True, "TransposeLDS": 0 if transB else 1, "MIWaveTileA": 8, "MIWaveTileB": 5,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 80
        assert kernel["SwapGlobalReadOrder"]
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize("transA, transB", [(True, False), (False, True), (False, False)])
    def test_schedule_256x240x64_16bit(self, transA, transB):
        """Tests the 256x240x64 16-bit schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "MacroTile0": 256, "MacroTile1": 240, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 2, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [4,1],
            "LDSTrInst": True, "TransposeLDS": 1 if transA or not (transA or transB) else 0, "MIWaveTileA": 4, "MIWaveTileB": 15,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 1
        assert schedule_info.numMfma == 120
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize("transA, transB", [(True, False), (False, False)])
    def test_schedule_256x208x64_16bit(self, transA, transB):
        """Tests the 256x208x64 16-bit schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "MacroTile0": 256, "MacroTile1": 208, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 2, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [4,1],
            "LDSTrInst": not transA , "TransposeLDS": 1, "MIWaveTileA": 4, "MIWaveTileB": 13,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 1
        assert schedule_info.numMfma == 104
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    def test_schedule_224x256x64_16bit_TN(self):
        """Tests the 224x256x64 16-bit TN schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": True, "TransposeB": False
        })
        kernel.update({
            "MacroTile0": 224, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "TransposeLDS": 1, "MIWaveTileA": 7, "MIWaveTileB": 8,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 112
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst,  tr_lds", [
        ( False,  False,        True,       1),
        (  True,  False,       False,       1),
        ( False,   True,        True,       0)
        # fmt: on
        ])
    def test_schedule_192x320x64_16bit(self, transA, transB, lds_tr_inst, tr_lds):
        """Tests the 192x320x64 16-bit NN schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "MacroTile0": 192, "MacroTile1": 320, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": 6, "MIWaveTileB": 10,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 1
        assert schedule_info.numMfma == 120
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst,  tr_lds", [
        (  True,  False,       False,       1),
        # fmt: on
        ])
    def test_schedule_224x128x64_16bit(self, transA, transB, lds_tr_inst, tr_lds):
        """Tests the 224x128x64 16-bit NN schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "MacroTile0": 224, "MacroTile1": 128, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": 7, "MIWaveTileB": 4,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 1
        assert schedule_info.numMfma == 56
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    def test_schedule_256x224x64_16bit_TN(self):
        """Tests the 256x224x64 16-bit TN schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": True, "TransposeB": False
        })
        kernel.update({
            "MacroTile0": 256, "MacroTile1": 224, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "TransposeLDS": 1, "MIWaveTileA": 8, "MIWaveTileB": 7,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 112
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize("transA, transB", [(False, False), (False, True)])
    def test_schedule_320x192x64_16bit(self, transA, transB):
        """Tests the 320x192x64 16-bit schedule."""
        NN = not transA and not transB
        NT = not transA and transB

        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "MacroTile0": 320, "MacroTile1": 192, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": True, "TransposeLDS": NN, "MIWaveTileA": 10, "MIWaveTileB": 6,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 120
        assert kernel["SwapGlobalReadOrder"]
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst,  tr_lds", [
        (  True,  False,       False,       1),
        ( False,   True,        True,       0),
        (  True,  False,        True,       1),
        ( False,   True,       False,       0),
        ( False,  False,        True,       1)
        # fmt: on
        ])
    def test_schedule_240x256x64_16bit(self, transA, transB, lds_tr_inst,  tr_lds):
        """Tests the 240x256x64 16-bit schedule."""
        NT = not transA and transB
        TN = transA and not transB
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "MacroTile0": 240, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 2, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [1,4],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": 15, "MIWaveTileB": 4,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == (1 if NT else 2)
        assert schedule_info.numMfma == 120
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst,  tr_lds", [
        (  True,  False,       False,       1),
        ( False,  False,        True,       1),
        ( False,   True,        True,       0)
        # fmt: on
        ])
    def test_schedule_208x256x64_16bit(self, transA, transB, lds_tr_inst,  tr_lds):
        """Tests the 208x256x64 16-bit schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "MacroTile0": 208, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 2, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [1,4],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": 13, "MIWaveTileB": 4,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 1
        assert schedule_info.numMfma == 104
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst,  tr_lds", [
        (  True,  False,       False,       1),
        ( False,  False,        True,       1),
        # fmt: on
        ])
    def test_schedule_128x224x64_16bit(self, transA, transB, lds_tr_inst,  tr_lds):
        """Tests the 208x256x64 16-bit schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "MacroTile0": 128, "MacroTile1": 224, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": 4, "MIWaveTileB": 7,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 56
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message


class TestCustomScheduleTF32:
    @pytest.mark.parametrize("transA, transB", [(True, False), (False, False)])
    def test_schedule_192x256x32_TF32(self, transA, transB):
        """Tests the 192x256x32 TF32 schedule."""        
        kernel = create_base_kernel()
        kernel["ProblemType"].update({
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "ForceUnrollSubIter": True,
            "MacroTile0": 192, "MacroTile1": 256, "DepthU": 32,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 0,
            "DirectToLds": True,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidth": 4,
            "MatrixInstruction": [16, 16, 32, 1], "MIWaveGroup": [2, 2],
            "LDSTrInst": False, "TransposeLDS": 1, "MIWaveTileA": 6, "MIWaveTileB": 8,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 144
        assert kernel["UsePLRPack"]
        valid, message = isValid(schedule_info, {"kernel": kernel})
        assert valid, message

    def test_schedule_128x192x32_TF32(self):
        """Tests the 128x192x32 TF32 TN schedule."""
        kernel = create_base_kernel()
        kernel["ProblemType"].update({
            "TransposeA": True, "TransposeB": False
        })
        kernel.update({
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "MacroTile0": 128, "MacroTile1": 192, "DepthU": 32,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 0,
            "DirectToLds": True,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidth": 4,
            "MatrixInstruction": [16, 16, 32, 1], "MIWaveGroup": [2, 2],
            "LDSTrInst": False, "TransposeLDS": 1, "MIWaveTileA": 4, "MIWaveTileB": 6,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 72
        assert kernel["UsePLRPack"]
        valid, message = isValid(schedule_info, {"kernel": kernel})
        assert valid, message

    def test_schedule_192x128x32_TF32(self):
        """Tests the 192x128x32 TF32 TN schedule."""
        kernel = create_base_kernel()
        kernel["ProblemType"].update({
            "TransposeA": True, "TransposeB": False
        })
        kernel.update({
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "MacroTile0": 192, "MacroTile1": 128, "DepthU": 32,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 0,
            "DirectToLds": True,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidth": 4,
            "MatrixInstruction": [16, 16, 32, 1], "MIWaveGroup": [2, 2],
            "LDSTrInst": True, "TransposeLDS": 1, "MIWaveTileA": 6, "MIWaveTileB": 4,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 72
        assert kernel["UsePLRPack"]
        valid, message = isValid(schedule_info, {"kernel": kernel})
        assert valid, message

    def test_schedule_256x256x32_TF32(self):
        """Tests the 256x256x32 TF32 TN schedule."""
        kernel = create_base_kernel()
        kernel["ProblemType"].update({
            "TransposeA": True, "TransposeB": False
        })
        kernel.update({
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "ForceUnrollSubIter": True,
            "MacroTile0": 256, "MacroTile1": 256, "DepthU": 32,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 0,
            "DirectToLds": True,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidth": 4,
            "MatrixInstruction": [16, 16, 32, 1], "MIWaveGroup": [2, 2],
            "LDSTrInst": False, "TransposeLDS": 1, "MIWaveTileA": 8, "MIWaveTileB": 8,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 192
        assert kernel["UsePLRPack"]
        valid, message = isValid(schedule_info, {"kernel": kernel})
        assert valid, message

    def test_schedule_256x192x32_TF32(self):
        """Tests the 256x192x32 TF32 TN schedule."""
        kernel = create_base_kernel()
        kernel["ProblemType"].update({
            "TransposeA": True, "TransposeB": False
        })
        kernel.update({
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "ForceUnrollSubIter": True,
            "MacroTile0": 256, "MacroTile1": 192, "DepthU": 32,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 0,
            "DirectToLds": True,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidth": 4,
            "MatrixInstruction": [16, 16, 32, 1], "MIWaveGroup": [2, 2],
            "LDSTrInst": False, "TransposeLDS": 1, "MIWaveTileA": 8, "MIWaveTileB": 6,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 144
        assert kernel["UsePLRPack"]
        valid, message = isValid(schedule_info, {"kernel": kernel})
        assert valid, message

    def test_schedule_128x256x32_TF32(self):
        """Tests the 128x256x32 TF32 TN schedule."""
        kernel = create_base_kernel()
        kernel["ProblemType"].update({
            "TransposeA": True, "TransposeB": False
        })
        kernel.update({
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "ForceUnrollSubIter": True,
            "MacroTile0": 128, "MacroTile1": 256, "DepthU": 32,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 0,
            "DirectToLds": True,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidth": 4,
            "MatrixInstruction": [16, 16, 32, 1], "MIWaveGroup": [2, 2],
            "LDSTrInst": False, "TransposeLDS": 1, "MIWaveTileA": 4, "MIWaveTileB": 8,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 96
        assert kernel["UsePLRPack"]
        valid, message = isValid(schedule_info, {"kernel": kernel})
        assert valid, message

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst,  tr_lds", [
        (  True,  False,       False,       1),
        # fmt: on
        ])
    def test_schedule_128x128x32(self, transA, transB, lds_tr_inst, tr_lds):
        """Tests the 129x128x32 TF32 TN schedule."""
        kernel = create_base_kernel()
        kernel["ProblemType"].update({
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "ForceUnrollSubIter": True,
            "MacroTile0": 128, "MacroTile1": 128, "DepthU": 32,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 0,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidth": 4,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": 4, "MIWaveTileB": 4,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 1
        numMfma = ((kernel["MacroTile0"] // kernel["MIWaveGroup"][0] // kernel["MatrixInstruction"][0]) *
                   (kernel["MacroTile1"] // kernel["MIWaveGroup"][1] // kernel["MatrixInstruction"][1]) *
                    3 # tf32 emulated with 3 bf16
        )
        assert schedule_info.numMfma == numMfma
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst,  tr_lds", [
        (  True,  False,       False,       1),
        # fmt: on
        ])
    def test_schedule_128x128x64(self, transA, transB, lds_tr_inst, tr_lds):
        """Tests the 129x128x64 TF32 TN schedule."""
        kernel = create_base_kernel()
        kernel["ProblemType"].update({
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "MacroTile0": 128, "MacroTile1": 128, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidth": 4,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": 4, "MIWaveTileB": 4,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        numMfma = ((kernel["MacroTile0"] // kernel["MIWaveGroup"][0] // kernel["MatrixInstruction"][0]) *
                   (kernel["MacroTile1"] // kernel["MIWaveGroup"][1] // kernel["MatrixInstruction"][1]) *
                    3 * # tf32 emulated with 3 bf16
                    2   # two sub-iterations due to DepthU=64
        )
        assert schedule_info.numMfma == numMfma
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message
    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst,  tr_lds", [
        (  True,  False,       False,       1),
        # fmt: on
        ])
    def test_schedule_128x160x64(self, transA, transB, lds_tr_inst, tr_lds):
        """Tests the 129x160x64 TF32 TN schedule."""
        kernel = create_base_kernel()
        kernel["ProblemType"].update({
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "MacroTile0": 128, "MacroTile1": 160, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidth": 4,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": 4, "MIWaveTileB": 5,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        numMfma = ((kernel["MacroTile0"] // kernel["MIWaveGroup"][0] // kernel["MatrixInstruction"][0]) *
                   (kernel["MacroTile1"] // kernel["MIWaveGroup"][1] // kernel["MatrixInstruction"][1]) *
                    3 * # tf32 emulated with 3 bf16
                    2   # two sub-iterations due to DepthU=64
        )
        assert schedule_info.numMfma == numMfma
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

class TestCustomScheduleValidation:
    def test_schedule_validation_disable(self):
        """
        Test of the flag that custom mainloop schedule (CMS) developers can use to override the
        validation checks.
        """
        kernel = create_base_kernel()
        invalid_schedule = {"P": [[3, 2, 1]]}

        # No verification message means that the schedule info is considered valid.
        scheduleInfo = ScheduleInfo(
            1, None, invalid_schedule, None, None, None, None
        )
        scheduleInfo.disableValidation()
        status, message = isValid(scheduleInfo, {"kernel" : {"DepthU": 42}})
        assert status == True
        assert message == "CMS validation explicitly disabled. Running on kernel with MT0xMT1xDepthU = ?x?x42 NN"

        # A non-empty verification message means that the schedule info is considered invalid.
        status, message = isValid(
            ScheduleInfo(1, None, invalid_schedule, None, None, None, None), {}
        )
        assert status == False

