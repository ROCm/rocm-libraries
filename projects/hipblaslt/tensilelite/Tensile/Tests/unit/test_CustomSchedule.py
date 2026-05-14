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
from Tensile.Components.CMSValidator import isValid, SchedulePosition, ValidatorPass
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

class TestCustomScheduleBF16:
    @staticmethod
    def get_num_mfma(kernel):
        numMfma = (kernel["MIWaveTileA"] * kernel["MIWaveTileB"] *
                   kernel["DepthU"] / kernel["MatrixInstruction"][2]   # two sub-iterations due to DepthU=64
        )
        return numMfma

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
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 256, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
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
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_8bit, "DataTypeA": dtype_8bit, "DataTypeB": dtype_8bit,
                "TransposeA": True, "TransposeB": False,
            },
            "MacroTile0": 256, "MacroTile1": 256, "DepthU": 128,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 0,
            "GlobalReadVectorWidthA": 16, "GlobalReadVectorWidthB": 16, "LocalReadVectorWidthA": 16, "LocalReadVectorWidthB": 16,
            "MatrixInstruction": [16,16,128,1], "MIWaveGroup": [2,2], "TransposeLDS": 1, "MIWaveTileA": 8, "MIWaveTileB": 8, "ForceUnrollSubIter": force_unroll_sub_iter,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)

        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 1
        assert schedule_info.numMfma == 64
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst, tr_lds, dtl_plus_lds_buf", [
        (  True,  False,        True,      1,             None),
        ( False,   True,        True,      0,                1),
        # fmt: on
        ])
    def test_schedule_256x96x64_16bit(self, transA, transB, lds_tr_inst, tr_lds, dtl_plus_lds_buf):
        """Tests the 256x96x64 16-bit schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 256, "MacroTile1": 96, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds,
            "MIWaveTileA": 8, "MIWaveTileB": 3,
        })

        if dtl_plus_lds_buf is not None:
            kernel.update({"DtlPlusLdsBuf": dtl_plus_lds_buf })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 48
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst,  tr_lds", [
        # TN case supports both LDSTrInst=True and LDSTrInst=False
        (  True,  False,       False,       1),
        (  True,  False,        True,       1),
        ( False,   True,        True,       0),
        ( False,  False,       False,       1)
        # fmt: on
        ])
    def test_schedule_96x256x64_16bit(self, transA, transB, lds_tr_inst, tr_lds):
        """Tests the 96x256x64 16-bit schedules."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 96, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds,
            "MIWaveTileA": 3, "MIWaveTileB": 8,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
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
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 192, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
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
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 256, "MacroTile1": 192, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
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
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 160, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
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

    @pytest.mark.parametrize("transA, transB", [(False, False), (False, True), (True, False)])
    def test_schedule_256x160x64_16bit(self, transA, transB):
        """Tests the 256x160x64 16-bit schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 256, "MacroTile1": 160, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            # Match CustomSchedule.py predicates:
            # - NN/NT useLDSTr=True, TN useLDSTr=False  -> useLDSTr == (not TransposeA)
            # - TLDS==1 for NN/TN, TLDS==0 for NT      -> TLDS == (not TransposeB)
            "LDSTrInst": not transA, "TransposeLDS": not transB, "MIWaveTileA": 8, "MIWaveTileB": 5,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 80
        # SwapGlobalReadOrder is set for NN/NT branches, not required for TN.
        if not (transA and (not transB)):
            assert kernel["SwapGlobalReadOrder"]
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize("transA, transB", [(True, False), (False, True), (False, False)])
    def test_schedule_256x240x64_16bit(self, transA, transB):
        """Tests the 256x240x64 16-bit schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 256, "MacroTile1": 240, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 2, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
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
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 256, "MacroTile1": 208, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 2, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
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

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst,  tr_lds", [
        (  True,  False,       False,       1),
        ( False,   True,        True,       0),
        ( False,  False,        True,       1)
        # fmt: on
        ])   
    def test_schedule_224x256x64_16bit(self, transA, transB, lds_tr_inst, tr_lds):
        """Tests the 224x256x64 16-bit schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 224, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": 7, "MIWaveTileB": 8,
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
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 192, "MacroTile1": 320, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
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
    "transA, transB, lds_tr_inst,  tr_lds, mt0, mt1", [
        ( True,  False,        False,       1, 224, 128),  # TN
        ( False,  True,        True,        0, 224, 128),  # NT
        ( False,  False,       True,        1, 224, 128),  # NN
        ( False,  True,        True,        0, 128, 224),  # NT
    ])
    def test_schedule_224x128x64_128x224x64_16bit(self, transA, transB, lds_tr_inst, tr_lds, mt0, mt1):
        """
        Tests the 224x128x64 16-bit schedules (TN/NT/NN).
        Tests the 128x224x64 16-bit schedule  (NT).
        """
        du = 64
        mi = [16,16,32,1]
        mi_wave_group = [2, 2]
        mi_wave_tile = (mt0 // (mi[0] * mi_wave_group[0]), mt1 // (mi[1] * mi_wave_group[1]))
        NT = (not transA and transB)
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": mt0, "MacroTile1": mt1, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
            "MatrixInstruction": mi, "MIWaveGroup": mi_wave_group,
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": mi_wave_tile[0], "MIWaveTileB": mi_wave_tile[1],
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == (2 if NT else 1)
        assert schedule_info.numMfma == 56
        assert bool(kernel.get("SwapGlobalReadOrder", False)) == NT
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize(
    # fmt: off
    "transA, transB, lds_tr_inst,  tr_lds", [
    (  True,  False,       False,       1),
    ( False,  False,        True,       1),
    ( False,   True,        True,       0),
    # fmt: on
    ])
    def test_schedule_256x224x64_16bit(self, transA, transB, lds_tr_inst, tr_lds):
        """Tests the 256x224x64 16-bit schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 256, "MacroTile1": 224, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": 8, "MIWaveTileB": 7,
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
    ( False,  False,        True,       1), #NN
    ( False,   True,        True,       0), #NT
    (  True,  False,       False,       1)  #TN
    # fmt: on
    ])
    def test_schedule_320x192x64_16bit(self, transA, transB, lds_tr_inst, tr_lds):
        """Tests the 320x192x64 16-bit schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 320, "MacroTile1": 192, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": 10, "MIWaveTileB": 6,
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
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 240, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 2, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
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
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 208, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 2, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
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
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 128, "MacroTile1": 224, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
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

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst,  tr_lds, mt0, mt1, code_paths", [
        (  True,  False,       False,       1, 128, 192,          1),
        (  True,  False,       False,       1, 192, 128,          2),
        # fmt: on
        ])
    def test_schedule_128x192x64_192x128x64_16bit(self, transA, transB, lds_tr_inst, tr_lds, mt0, mt1, code_paths):
        """Tests the 128x192x64 and 192x128x64 BF16 tiles."""

        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        du = 64
        mi = [16,16,32,1]
        mi_wave_group = [2, 2]
        mi_wave_tile = (mt0 // (mi[0] * mi_wave_group[0]), mt1 // (mi[1] * mi_wave_group[1]))

        kernel = create_base_kernel()
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": True, "TransposeB": False,
            },
            "MacroTile0": mt0, "MacroTile1": mt1, "DepthU": du,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
            "MatrixInstruction": mi, "MIWaveGroup": mi_wave_group,
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": mi_wave_tile[0], "MIWaveTileB": mi_wave_tile[1],
        })
        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == code_paths
        assert schedule_info.numMfma == TestCustomScheduleBF16.get_num_mfma(kernel)
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    def test_schedule_128x256x64_16bit(self):
        """Tests the 128x256x64  NN schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": False, "TransposeB": False,
            },
            "MacroTile0": 128, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "DirectToLds": True,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
            "MatrixInstruction": [16, 16, 32, 1], "MIWaveGroup": [2, 2],
            "LDSTrInst": True, "TransposeLDS": 1, "MIWaveTileA": 4, "MIWaveTileB": 8,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 64
        valid, message = isValid(schedule_info, {"kernel": kernel})
        assert valid, message

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst,  tr_lds", [
        (  True,  False,       True,       1),
        # fmt: on
        ])
    def test_schedule_352x192x64_16bit(self, transA, transB, lds_tr_inst, tr_lds):
        """Tests the 352x192x64 16-bit TN schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 352, "MacroTile1": 192, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": 11, "MIWaveTileB": 6,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == TestCustomScheduleBF16.get_num_mfma(kernel)
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst,  tr_lds", [
        (  True,  False,       True,       1),
        # fmt: on
        ])
    def test_schedule_224x320x64_16bit(self, transA, transB, lds_tr_inst, tr_lds):
        """Tests the 224x320x64 16-bit TN schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        update_kernel(kernel, {
            "ProblemType": {
                "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
                "TransposeA": transA, "TransposeB": transB,
            },
            "MacroTile0": 224, "MacroTile1": 320, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": 7, "MIWaveTileB": 10,
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 140
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

class TestCustomScheduleTF32:
    @staticmethod
    def get_num_mfma(kernel):
        numMfma = (kernel["MIWaveTileA"] * kernel["MIWaveTileB"] *
                    3 * # tf32 emulated with 3 bf16
                    kernel["DepthU"] / kernel["MatrixInstruction"][2]   # two sub-iterations due to DepthU=64
        )
        return numMfma
    
    @pytest.mark.parametrize("transA, transB, vwa", [
        (True, False, 1), 
        (False, False, 1),
        ])
    def test_schedule_192x256x32_TF32(self, transA, transB, vwa):
        """Tests the 192x256x32 TF32 schedule."""
        kernel = create_base_kernel()
        update_kernel(kernel, {
            "ProblemType": {
                "TransposeA": transA, "TransposeB": transB,
            },
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "ForceUnrollSubIter": True,
            "MacroTile0": 192, "MacroTile1": 256, "DepthU": 32,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "DirectToLds": True,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidthA": 4, "LocalReadVectorWidthB": 4,
            "MatrixInstruction": [16, 16, 32, 1], "MIWaveGroup": [2, 2],
            "LDSTrInst": False, "TransposeLDS": 1, "MIWaveTileA": 6, "MIWaveTileB": 8,
        })
        if vwa is not None:
            kernel.update({"VectorWidthA": vwa})

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
        update_kernel(kernel, {
            "ProblemType": {
                "TransposeA": True, "TransposeB": False,
            },
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "ForceUnrollSubIter": True,
            "MacroTile0": 128, "MacroTile1": 192, "DepthU": 32,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "DirectToLds": True,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidthA": 4, "LocalReadVectorWidthB": 4,
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
        update_kernel(kernel, {
            "ProblemType": {
                "TransposeA": True, "TransposeB": False,
            },
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "ForceUnrollSubIter": True,
            "MacroTile0": 192, "MacroTile1": 128, "DepthU": 32,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "DirectToLds": True,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidthA": 4, "LocalReadVectorWidthB": 4,
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

    @pytest.mark.parametrize(
    # fmt: off
    "transA, transB, lds_tr_inst,  tr_lds,  vwa, vwb", [
    (  True,  False,       False,       1,  None, None),
    ( False,   True,       False,       0,  4,    4),
    # fmt: on
    ])
    def test_schedule_256x256x32_TF32(self,transA, transB, lds_tr_inst, tr_lds, vwa, vwb):
        """Tests the 256x256x32 TF32 TN schedule."""
        kernel = create_base_kernel()
        update_kernel(kernel, {
            "ProblemType": {
                "TransposeA": transA, "TransposeB": transB,
            },
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "ForceUnrollSubIter": True,
            "MacroTile0": 256, "MacroTile1": 256, "DepthU": 32,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "DirectToLds": True,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidthA": 4, "LocalReadVectorWidthB": 4,
            "MatrixInstruction": [16, 16, 32, 1], "MIWaveGroup": [2, 2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": 8, "MIWaveTileB": 8,
        })

        if vwa is not None:
            kernel.update({"VectorWidthA": vwa})
        if vwb is not None:
            kernel.update({"VectorWidthB": vwb})

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 192
        assert kernel["UsePLRPack"]
        valid, message = isValid(schedule_info, {"kernel": kernel})
        assert valid, message

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, vwa", [
        (  True,  False, 1),
        ( False,  False, 1),
        # fmt: on
        ])
    def test_schedule_256x192x32_TF32(self, transA, transB, vwa):
        """Tests the 256x192x32 TF32 TN schedule."""
        kernel = create_base_kernel()
        update_kernel(kernel, {
            "ProblemType": {
                "TransposeA": transA, "TransposeB": transB,
            },
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "ForceUnrollSubIter": True,
            "MacroTile0": 256, "MacroTile1": 192, "DepthU": 32,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "DirectToLds": True,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidthA": 4, "LocalReadVectorWidthB": 4,
            "MatrixInstruction": [16, 16, 32, 1], "MIWaveGroup": [2, 2],
            "LDSTrInst": False, "TransposeLDS": 1, "MIWaveTileA": 8, "MIWaveTileB": 6,
        })
        if vwa is not None:
            kernel.update({"VectorWidthA": vwa})

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 144
        assert kernel["UsePLRPack"]
        if transA == False and transB == False:
            assert kernel["UseMFMAF32XEmulation"]
        valid, message = isValid(schedule_info, {"kernel": kernel})
        assert valid, message

    def test_schedule_128x256x32_TF32(self):
        """Tests the 128x256x32 TF32 TN schedule."""
        kernel = create_base_kernel()
        update_kernel(kernel, {
            "ProblemType": {
                "TransposeA": True, "TransposeB": False,
            },
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "ForceUnrollSubIter": True,
            "MacroTile0": 128, "MacroTile1": 256, "DepthU": 32,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "DirectToLds": True,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidthA": 4, "LocalReadVectorWidthB": 4,
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
        "transA, transB, lds_tr_inst,  tr_lds,  plr,  vwa,  vwb,      mi    , ncp", [
        (  True,  False,       False,       1,  1,   None, None, [16,16,32,1],   1),
        (  True,  False,       False,       1,  1,   None, None, [32,32,16,1],   1),
        (  False, False,       False,       1,  1,      2, None, [32,32,16,1],   2),
        (  False, False,        True,       1,  1,      2, None, [32,32,16,1],   2),
        (  False, True,         True,       0,  1,      2,    2, [32,32,16,1],   2),
        # fmt: on
        ])
    def test_schedule_128x128x32(self, transA, transB, lds_tr_inst, tr_lds, plr, vwa, vwb, mi, ncp):
        """Tests the 128x128x32 TF32 schedule."""
        kernel = create_base_kernel()
        macro_tile = (128,128,32)
        mi_wave_group = (2,2)
        mi_wave_tile = (macro_tile[0] // (mi[0] * mi_wave_group[0]), macro_tile[1] // (mi[1] * mi_wave_group[1]))

        update_kernel(kernel, {
            "ProblemType": {
                "TransposeA": transA, "TransposeB": transB,
            },
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "ForceUnrollSubIter": (macro_tile[2] == mi[2]), # production sets True when DU == matrixInstK (Solution.py:1442)
            "MacroTile0": macro_tile[0], "MacroTile1": macro_tile[1], "DepthU": macro_tile[2],
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": plr,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidthA": 4, "LocalReadVectorWidthB": 4,
            "MatrixInstruction": mi, "MIWaveGroup": [2,2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": mi_wave_tile[0], "MIWaveTileB": mi_wave_tile[1],
        })
        if vwa is not None:
            kernel.update({"VectorWidthA": vwa})
        if vwb is not None:
            kernel.update({"VectorWidthB": vwb})

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == ncp
        schedule_info.pretty_print()
        numMfma = (mi_wave_tile[0] * mi_wave_tile[1] *
                   3 *                      # tf32 emulated with 3 bf16
                   (1 if mi[0] == 16 else 2)   # two sub-iterations with mi32
        )
        assert schedule_info.numMfma == numMfma
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst,  tr_lds,  vwa", [
        (  True,  False,       False,       1, None),
        ( False,  False,        True,       1,    4), # NN doesn't depend on lds_tr_inst, so check for both values 
        ( False,  False,       False,       1,    4),
        # fmt: on
        ])
    def test_schedule_128x128x64(self, transA, transB, lds_tr_inst, tr_lds, vwa):
        """Tests the 128x128x64 TF32 TN schedule."""
        kernel = create_base_kernel()
        update_kernel(kernel, {
            "ProblemType": {
                "TransposeA": transA, "TransposeB": transB,
            },
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "MacroTile0": 128, "MacroTile1": 128, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidthA": 4, "LocalReadVectorWidthB": 4,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": 4, "MIWaveTileB": 4,
        })
        if vwa is not None:
            kernel.update({"VectorWidthA": vwa})

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
        "transA, transB, lds_tr_inst,  tr_lds, mt0, mt1", [
        (  True,  False,       False,       1, 128, 160),
        ( False,  False,        True,       1, 160, 128),
        (  True,  False,       False,       1, 160, 128),
        # fmt: on
        ])
    def test_schedule_128x160x64_160x128x64(self, transA, transB, lds_tr_inst, tr_lds, mt0, mt1):
        """Tests the 128x160x64, 160x128x64 TF32 TN schedule and 160x128x64 TF32 NN."""

        kernel = create_base_kernel()

        du = 64
        mi = [16,16,32,1]
        mi_wave_group = [2, 2]
        mi_wave_tile = (mt0 // (mi[0] * mi_wave_group[0]), mt1 // (mi[1] * mi_wave_group[1]))

        update_kernel(kernel, {
            "ProblemType": {
                "TransposeA": transA, "TransposeB": transB,
            },
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "MacroTile0": mt0, "MacroTile1": mt1, "DepthU": du,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidthA": 4, "LocalReadVectorWidthB": 4,
            "MatrixInstruction": mi, "MIWaveGroup": mi_wave_group,
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": mi_wave_tile[0], "MIWaveTileB": mi_wave_tile[1],
        })
        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == TestCustomScheduleTF32.get_num_mfma(kernel)
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

    @pytest.mark.parametrize(
        # fmt: off
        "transA, transB, lds_tr_inst,  tr_lds, mt0, mt1", [
        (  True,  False,       False,       1,  64, 128),
        (  True,  False,       False,       1, 128,  64),
        # fmt: on
        ])
    def test_schedule_64x128x64_128x64x64(self, transA, transB, lds_tr_inst, tr_lds, mt0, mt1):
        """Tests the 64x128x64 & 128x64x64 TF32 TN schedules."""
        kernel = create_base_kernel()
        du = 64
        mi = [16,16,32,1]
        mi_wave_group = [2, 2]
        mi_wave_tile = (mt0 // (mi[0] * mi_wave_group[0]), mt1 // (mi[1] * mi_wave_group[1]))

        update_kernel(kernel, {
            "ProblemType": {
                "TransposeA": transA, "TransposeB": transB,
            },
            "UseF32XEmulation": True, "UseDirect32XEmulation": True,
            "MacroTile0": mt0, "MacroTile1": mt1, "DepthU": du,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
            "GlobalReadVectorWidthA": 4, "GlobalReadVectorWidthB": 4, "LocalReadVectorWidthA": 4, "LocalReadVectorWidthB": 4,
            "MatrixInstruction": mi, "MIWaveGroup": mi_wave_group,
            "LDSTrInst": lds_tr_inst, "TransposeLDS": tr_lds, "MIWaveTileA": mi_wave_tile[0], "MIWaveTileB": mi_wave_tile[1],
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 1
        numMfma = (mi_wave_tile[0] * mi_wave_tile[1] *
                    3 * # tf32 emulated with 3 bf16
                    2   # two sub-iterations due to DepthU=64
        )
        assert schedule_info.numMfma == numMfma
        valid, message = isValid(schedule_info, {"kernel" : kernel})
        assert valid, message

class TestCustomScheduleValidation:
    def test_disable_single_pass(self):
        """Disabling a single pass allows an otherwise-invalid schedule to pass that check."""
        kernel = create_base_kernel()
        invalid_schedule = {"P": [[3, 2, 1]]}

        # Without disabling, the non-ascending schedule fails on ascending order.
        si = ScheduleInfo(1, None, invalid_schedule, None, None, None, None)
        status, message = isValid(si, {"kernel": kernel})
        assert status == False
        assert "Non-descending-order" in message

        # Disabling VERIFY_ASCENDING_ORDER skips that check.
        # Remaining structural and timeline passes are also disabled because this
        # minimal schedule lacks the keys/data they require.
        si = ScheduleInfo(1, 0, invalid_schedule, None, None, None, None)
        for p in ValidatorPass:
            if p != ValidatorPass.VERIFY_CORRECT_NUMBER_OF_INSTRUCTIONS:
                si.disableValidationPass(p, reason="Not relevant to this test")
        status, message = isValid(si, {"kernel": kernel})
        # The ascending-order error is gone; the schedule passes the remaining enabled check.
        assert "Non-descending-order" not in message
        assert status == True

    def test_disable_validation_pass_reason_required(self):
        """disableValidationPass requires a non-empty reason string and a valid ValidatorPass enum member."""
        si = ScheduleInfo(1, None, {}, None, None, None, None)

        with pytest.raises(ValueError):
            si.disableValidationPass(ValidatorPass.VERIFY_ASCENDING_ORDER, reason="")

        with pytest.raises(ValueError):
            si.disableValidationPass(ValidatorPass.VERIFY_ASCENDING_ORDER, reason="   ")

        with pytest.raises(TypeError):
            si.disableValidationPass("not_an_enum", reason="some reason")

        with pytest.raises(TypeError):
            si.disableValidationPass(42, reason="some reason")

    def test_disable_multiple_validation_passes(self):
        """Multiple validation passes can be disabled independently."""
        invalid_schedule = {"P": [[3, 2, 1]]}
        si = ScheduleInfo(1, None, invalid_schedule, None, None, None, None)
        si.disableValidationPass(ValidatorPass.VERIFY_ASCENDING_ORDER, reason="Reason A")
        si.disableValidationPass(ValidatorPass.VERIFY_CORRECT_NUMBER_OF_INSTRUCTIONS, reason="Reason B")

        assert si.reasonForDisablingValidationPass(ValidatorPass.VERIFY_ASCENDING_ORDER) == "Reason A"
        assert si.reasonForDisablingValidationPass(ValidatorPass.VERIFY_CORRECT_NUMBER_OF_INSTRUCTIONS) == "Reason B"
        # Passes not disabled return None.
        assert si.reasonForDisablingValidationPass(ValidatorPass.VERIFY_SCC_OVERLAP) is None

    def test_reason_for_disabling_validation_pass(self):
        """reasonForDisablingValidationPass returns the reason for disabled passes, None for enabled ones,
        and raises TypeError for non-enum arguments."""
        si = ScheduleInfo(1, None, {}, None, None, None, None)

        # Nothing disabled yet.
        assert si.reasonForDisablingValidationPass(ValidatorPass.VERIFY_ASCENDING_ORDER) is None

        si.disableValidationPass(ValidatorPass.VERIFY_ASCENDING_ORDER, reason="test reason")
        assert si.reasonForDisablingValidationPass(ValidatorPass.VERIFY_ASCENDING_ORDER) == "test reason"

        # Non-enum argument raises TypeError.
        with pytest.raises(TypeError):
            si.reasonForDisablingValidationPass("not_an_enum")

        with pytest.raises(TypeError):
            si.reasonForDisablingValidationPass(42)

    def test_disable_validation_all_passes(self):
        """disableValidation disables all passes with the given reason."""
        si = ScheduleInfo(1, None, {}, None, None, None, None)
        si.disableValidation("entire schedule unsupported")

        for p in ValidatorPass:
            assert si.reasonForDisablingValidationPass(p) == "entire schedule unsupported"

    def test_disable_validation_isvalid_early_return(self):
        """isValid returns (True, '') with a single warning when all passes are disabled."""
        kernel = create_base_kernel()
        si = ScheduleInfo(1, 0, {}, None, None, None, None)
        si.disableValidation("not supported yet")
        status, message = isValid(si, {"kernel": kernel})
        assert status == True
        assert message == ""

class TestSchedulePositionOrdering:
    """Test that SchedulePosition comparison handles vmfma_index=-1 correctly.

    vmfma_index=-1 is a wrap-around position: it represents instructions
    scheduled between iterations (after the last VMFMA of the previous
    iteration, before the first VMFMA of the current iteration). With
    explicit loop_index tracking, -1 naturally sorts before 0 within the
    same loop via integer comparison.
    """

    def test_neg1_after_last_vmfma_same_loop(self):
        """vmfma=-1 in loop 0 must be < vmfma=0 in loop 0."""
        a = SchedulePosition(loop_index=0, vmfma_index=-1, sub_index=0)
        b = SchedulePosition(loop_index=0, vmfma_index=0, sub_index=0)
        assert a < b
        assert b > a

    def test_neg1_loop1_after_last_vmfma_loop0(self):
        """vmfma=-1 in loop 1 must be > vmfma=num_vmfma-1 in loop 0."""
        a = SchedulePosition(loop_index=1, vmfma_index=-1, sub_index=0)
        b = SchedulePosition(loop_index=0, vmfma_index=2, sub_index=0)
        assert a > b
        assert b < a

    def test_neg1_before_next_loop_vmfma0(self):
        """vmfma=-1 in loop 0 must be < vmfma=0 in loop 1."""
        a = SchedulePosition(loop_index=0, vmfma_index=-1, sub_index=0)
        b = SchedulePosition(loop_index=1, vmfma_index=0, sub_index=0)
        assert a < b
        assert b > a

    def test_equal_positions(self):
        """Identical positions must be equal."""
        a = SchedulePosition(loop_index=0, vmfma_index=3, sub_index=1)
        b = SchedulePosition(loop_index=0, vmfma_index=3, sub_index=1)
        assert a == b
        assert not (a != b)
        assert not (a < b)
        assert not (a > b)
        assert a <= b
        assert a >= b

    def test_not_equal_different_sub_index(self):
        """Same (loop, vmfma) but different sub_index must not be equal."""
        a = SchedulePosition(loop_index=0, vmfma_index=3, sub_index=0)
        b = SchedulePosition(loop_index=0, vmfma_index=3, sub_index=1)
        assert a != b
        assert a < b

    def test_sub_index_ordering(self):
        """Within same (loop, vmfma), sub_index determines order."""
        a = SchedulePosition(loop_index=0, vmfma_index=2, sub_index=0)
        b = SchedulePosition(loop_index=0, vmfma_index=2, sub_index=5)
        c = SchedulePosition(loop_index=0, vmfma_index=2, sub_index=10)
        assert a < b < c
        assert c > b > a


# ----------------------------------------------------------------------------
# gfx1151 (RDNA 3.5) WMMA schedule tests
# ----------------------------------------------------------------------------
# These cover the schedules defined at the bottom of CustomSchedule.py.
# Coverage goals:
#   - dispatcher: hasCustomSchedule picks a gfx1151 schedule for gfx1151 kernels
#     and does NOT mis-dispatch CDNA 4 kernels to a gfx1151 schedule.
#   - predicate  : non-TN kernels and non-16bit dtypes are not routed to
#     TN-only fp16/bf16 gfx1151 schedules.
#   - validator  : the RDNA 3.5 profile hook is a no-op; every pass runs
#     under the dialect-aware validator (CMSValidatorDialect.py), so
#     `isValid` returns True with full coverage.

from Tensile.Components.CustomSchedule import _apply_rdna35_wmma_profile
import Tensile.Components.CMSValidator as cmsv
from Tensile.Components.CMSValidatorDialect import (
    RDNA35_WMMA_DIALECT,
    CDNA4_DIALECT,
    UnsupportedKernelError,
    resolve_dialect,
)


def _gfx1151_base_kernel():
    """Canonical gfx1151 kernel shell (MT / tile params filled in per case)."""
    dt = _mock_dtype(is_16bit=True, num_bytes=2)
    return {
        "UseCustomMainLoopSchedule": True,
        "EnableMatrixInstruction": True,
        "UnrollLoopSwapGlobalReadOrder": False,
        "ISA": IsaVersion(11, 5, 1),
        "WavefrontSize": 32,
        "ProblemType": {
            "DataType": dt, "DataTypeA": dt, "DataTypeB": dt,
            "TransposeA": True, "TransposeB": False,
            "TLUA": False, "TLUB": False,
        },
        "MacroTile0": 0, "MacroTile1": 0, "DepthU": 32,
        "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
        "DirectToLds": 0, "DtlPlusLdsBuf": False,
        "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8,
        "LocalReadVectorWidthA": 16, "LocalReadVectorWidthB": 16,
        "WaveSeparateGlobalReadA": 0, "WaveSeparateGlobalReadB": 0,
        "Use64bShadowLimit": 1,
        "MatrixInstruction": [16, 16, 16, 1],
        "MIWaveGroup": [2, 2],
        "LDSTrInst": False, "TransposeLDS": 1,
        "ForceUnrollSubIter": False,
        "SwapGlobalReadOrder": False,
        "UsePLRPack": False,
        "UseF32XEmulation": False,
        "MIWaveTileA": 1, "MIWaveTileB": 1,
        "1LDSBuffer": 0,
    }


# Tile parameters for a representative subset of gfx1151 schedules.
# Format: (MT0, MT1, DU, PGR, PLR, MIWG, WTA, WTB, MIK)
#   MIK=matrixInstK (16 for the bulk of fp16 tiles, 128 for the 16x16x128 case).
GFX1151_TILES = [
    # Production PGR=1 PLR=0 schedules (PLR>=1 is rejected by the
    # _reject_ldsb1_with_plr_prefetch correctness gate).
    (128, 96,  32, 1, 0, [2, 2], 4, 3, 16),
    (128, 128, 32, 1, 0, [2, 2], 4, 4, 16),
    (128, 64,  64, 1, 0, [2, 2], 4, 2, 16),
    (96,  96,  64, 1, 0, [2, 2], 3, 3, 16),
    (64,  128, 64, 1, 0, [1, 4], 4, 2, 16),
    (128, 112, 64, 1, 0, [4, 1], 2, 7, 16),
    (112, 128, 64, 1, 0, [1, 4], 7, 2, 16),
]


class TestCustomScheduleGfx1151:
    """Tests for gfx1151 (RDNA 3.5) WMMA schedules."""

    # ---- Dispatcher / predicate ----

    def test_dispatch_gfx1151_TN_16bit(self):
        """A TN fp16 gfx1151 kernel should pick up a gfx1151 schedule."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "MacroTile0": 128, "MacroTile1": 96, "DepthU": 32,
            "PrefetchGlobalRead": 1, "PrefetchLocalRead": 0,
            "MIWaveTileA": 4, "MIWaveTileB": 3,
            "1LDSBuffer": 1,
        })
        has, info = hasCustomSchedule(k)
        assert has
        assert isinstance(info, ScheduleInfo)

    def test_non_TN_does_not_dispatch(self):
        """gfx1151 schedules in this commit are TN-only; NN kernels must not match."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "ProblemType": {"TransposeA": False, "TransposeB": False},
            "MacroTile0": 96, "MacroTile1": 128, "DepthU": 32,
            "MIWaveTileA": 3, "MIWaveTileB": 4,
        })
        has, info = hasCustomSchedule(k)
        assert not has
        assert info is None

    def test_cdna4_isa_does_not_dispatch_to_gfx1151(self):
        """A CDNA 4 (gfx950) kernel must not match a gfx1151-registered tile."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "ISA": IsaVersion(9, 5, 0),
            "WavefrontSize": 64,
            "DirectToLds": 1,
            "MacroTile0": 96, "MacroTile1": 128, "DepthU": 32,
            "MIWaveTileA": 3, "MIWaveTileB": 4,
        })
        has, info = hasCustomSchedule(k)
        assert not has

    def test_fp32_does_not_dispatch_to_16bit_gfx1151(self):
        """The 16-bit gfx1151 schedules must not match a fp32 kernel."""
        k = _gfx1151_base_kernel()
        dt32 = _mock_dtype(is_16bit=False, is_8bit=False, num_bytes=4)
        update_kernel(k, {
            "ProblemType": {
                "DataType": dt32, "DataTypeA": dt32, "DataTypeB": dt32,
            },
            "MacroTile0": 96, "MacroTile1": 128, "DepthU": 32,
            "MIWaveTileA": 3, "MIWaveTileB": 4,
        })
        has, info = hasCustomSchedule(k)
        assert not has

    # ---- Schedule shape ----

    @pytest.mark.parametrize("MT0, MT1, DU, PGR, PLR, MIWG, WTA, WTB, MIK", GFX1151_TILES)
    def test_schedule_shape(self, MT0, MT1, DU, PGR, PLR, MIWG, WTA, WTB, MIK):
        """Each gfx1151 tile produces a well-formed ScheduleInfo."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "MacroTile0": MT0, "MacroTile1": MT1, "DepthU": DU,
            "PrefetchGlobalRead": PGR, "PrefetchLocalRead": PLR,
            "MatrixInstruction": [16, 16, MIK, 1],
            "MIWaveGroup": MIWG, "MIWaveTileA": WTA, "MIWaveTileB": WTB,
            "1LDSBuffer": 1,
        })
        has, info = hasCustomSchedule(k)
        assert has, f"no schedule dispatched for MT{MT0}x{MT1}x{DU}"
        assert isinstance(info, ScheduleInfo)
        assert info.numMfma > 0
        assert info.numCodePaths >= 1
        assert "SYNC" in info.optSchedule

    # ---- Validator coverage (this is the whole point of the helper) ----

    @pytest.mark.parametrize("MT0, MT1, DU, PGR, PLR, MIWG, WTA, WTB, MIK", GFX1151_TILES)
    def test_validator_passes_with_granular_disables(
            self, MT0, MT1, DU, PGR, PLR, MIWG, WTA, WTB, MIK):
        """Structural validator passes must still run and succeed on gfx1151."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "MacroTile0": MT0, "MacroTile1": MT1, "DepthU": DU,
            "PrefetchGlobalRead": PGR, "PrefetchLocalRead": PLR,
            "MatrixInstruction": [16, 16, MIK, 1],
            "MIWaveGroup": MIWG, "MIWaveTileA": WTA, "MIWaveTileB": WTB,
            "1LDSBuffer": 1,
        })
        has, info = hasCustomSchedule(k)
        assert has
        valid, msg = isValid(info, {"kernel": k})
        assert valid, f"MT{MT0}x{MT1}x{DU}: isValid said: {msg}"

    @staticmethod
    def _empty_schedule_info():
        return ScheduleInfo(numCodePaths=1, numMfma=1, optSchedule={}, syncCode=[],
                            nglshift=0, nllshift=0)

    def test_profile_hook_is_no_op(self):
        """_apply_rdna35_wmma_profile disables no validator passes.

        All hazard passes are dialect-aware via ``RDNA35_WMMA_DIALECT``
        and run on every gfx1151 schedule.
        ``VERIFY_CORRECT_NUMBER_OF_INSTRUCTIONS`` is dialect-aware
        (strict equality on CDNA 4, divisibility on RDNA 3.5), so no
        per-schedule opt-out is needed. The hook is retained as a
        uniform attach point across the registered
        ``_get_schedule_*_gfx1151`` functions for any future profile
        tweak, but currently performs no opt-outs.
        """
        info = self._empty_schedule_info()
        _apply_rdna35_wmma_profile(info)
        for pass_id in cmsv.ValidatorPass:
            reason = info.reasonForDisablingValidationPass(pass_id)
            assert reason is None, (
                f"{pass_id.name} was unexpectedly disabled by the RDNA 3.5 "
                f"profile hook (reason: {reason!r}). The hook is a no-op "
                f"now that every pass has hw-verified RDNA 3.5 semantics."
            )

    def test_dialect_is_rdna35_for_gfx1151_kernel(self):
        """resolve_dialect must return the RDNA 3.5 WMMA dialect for gfx1151."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "MacroTile0": 96, "MacroTile1": 128, "DepthU": 32,
            "MIWaveTileA": 3, "MIWaveTileB": 4,
        })
        assert resolve_dialect(k) is RDNA35_WMMA_DIALECT

    def test_dialect_is_cdna4_for_gfx950_kernel(self):
        """resolve_dialect must return the CDNA 4 dialect for a gfx950 kernel."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "ISA": IsaVersion(9, 5, 0),
            "WavefrontSize": 64,
            "DirectToLds": 1,
            "MacroTile0": 96, "MacroTile1": 128, "DepthU": 32,
        })
        assert resolve_dialect(k) is CDNA4_DIALECT

    # ---- TF32 pack-graph guard --------------------------------------------
    #
    # The pack_graph fields on RDNA35_WMMA_DIALECT are copied from CDNA 4
    # for forward compatibility only; they have never been hw-calibrated
    # on gfx1151. An RDNA 3.5 kernel that sets ``UseF32XEmulation`` or
    # ``UseMFMAF32XEmulation`` would consult those uncalibrated values in
    # ``add_pack_constraints``. The guard in ``resolve_dialect`` makes
    # this code path unreachable: it raises
    # ``UnsupportedKernelError`` so codegen aborts with a loud
    # diagnostic rather than emitting a silently-uncalibrated schedule.

    def test_rdna35_tf32_emulation_is_rejected(self):
        """resolve_dialect must reject RDNA 3.5 kernels with UseF32XEmulation."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "MacroTile0": 96, "MacroTile1": 128, "DepthU": 32,
            "MIWaveTileA": 3, "MIWaveTileB": 4,
            "UseF32XEmulation": True,
        })
        with pytest.raises(UnsupportedKernelError) as excinfo:
            resolve_dialect(k)
        msg = str(excinfo.value)
        assert "TF32" in msg or "F32XEmulation" in msg, (
            f"Expected the guard message to cite the uncalibrated pack-graph, got: {msg!r}"
        )

    def test_rdna35_mfma_tf32_emulation_is_rejected(self):
        """resolve_dialect must reject RDNA 3.5 kernels with UseMFMAF32XEmulation."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "MacroTile0": 96, "MacroTile1": 128, "DepthU": 32,
            "MIWaveTileA": 3, "MIWaveTileB": 4,
            "UseMFMAF32XEmulation": True,
        })
        with pytest.raises(UnsupportedKernelError):
            resolve_dialect(k)

    def test_cdna4_tf32_emulation_is_allowed(self):
        """The pack-graph guard only fires for RDNA 3.5; CDNA 4 kernels
        must continue to validate under TF32 emulation since the CDNA 4
        pack-graph is the calibrated source of those constants.
        """
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "ISA": IsaVersion(9, 5, 0),
            "WavefrontSize": 64,
            "DirectToLds": 1,
            "MacroTile0": 96, "MacroTile1": 128, "DepthU": 32,
            "UseF32XEmulation": True,
            "UseMFMAF32XEmulation": True,
        })
        assert resolve_dialect(k) is CDNA4_DIALECT

    def test_rdna35_without_tf32_emulation_resolves(self):
        """Default RDNA 3.5 kernels (no TF32 emulation) still resolve to
        RDNA35_WMMA_DIALECT - the guard must not fire for the production
        BF16/FP16 HHS path."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "MacroTile0": 96, "MacroTile1": 128, "DepthU": 32,
            "MIWaveTileA": 3, "MIWaveTileB": 4,
            "UseF32XEmulation": False,
        })
        assert resolve_dialect(k) is RDNA35_WMMA_DIALECT


# ----------------------------------------------------------------------------
# gfx1151 authored-stream-length checks
# ----------------------------------------------------------------------------
# `verify_correct_number_of_instructions` compares each authored stream's
# length against `len(idMap[name])` produced by KernelWriter at codegen
# time. We cannot run the full KernelWriter pipeline from a unit test, so
# this class instead checks ISA-agnostic invariants every well-formed
# gfx1151 schedule must satisfy:
#
#   1. Every mfmaIndex is in [-1, numMfma - 1].
#   2. For sub-iteration streams (`LRA<i>`, `LRB<i>`, `PackA<i>`,
#      `PackB<i>`) all sub-iterations of the same prefix have the same
#      length. This mirrors the per-sub-iter uniformity the KernelWriter
#      emits for gfx1151 wave32 WMMA.
#   3. len(GRA) == len(LWA) and len(GRB) == len(LWB) -- the VGPRs
#      populated by GR<X> drain through LW<X>, so the two streams must
#      have the same cardinality.
#
# Any deeper miscalibration (e.g. GRIncA cluster-size drift vs wave32
# idMap) surfaces at build time in `schedule_cms` through the live
# `isValid` call once `VERIFY_CORRECT_NUMBER_OF_INSTRUCTIONS` is
# re-enabled for the gfx1151 profile, and is exercised end-to-end by a
# rebuild + benchmark pass.


def _collect_gfx1151_schedule_metadata():
    """Return the CMSKernelInfo entries whose schedule function targets gfx1151."""
    from Tensile.Components.CustomSchedule import _SCHEDULE_METADATA
    return [m for m in _SCHEDULE_METADATA if "_gfx1151" in m.name]


def _gfx1151_kernel_for_metadata(meta):
    """Build a kernel dict that `hasCustomSchedule` will dispatch to `meta`."""
    k = _gfx1151_base_kernel()
    mi = list(meta.MatrixInstruction)
    miwg = list(meta.MIWaveGroup)
    update_kernel(k, {
        "ProblemType": {"TransposeA": meta.TransposeA, "TransposeB": meta.TransposeB},
        "MacroTile0": meta.MacroTile0,
        "MacroTile1": meta.MacroTile1,
        "DepthU": meta.DepthU,
        "PrefetchGlobalRead": meta.PrefetchGlobalRead,
        "PrefetchLocalRead": meta.PrefetchLocalRead,
        "DirectToLds": int(meta.DirectToLds),
        "DtlPlusLdsBuf": bool(meta.DtlPlusLdsBuf),
        "WaveSeparateGlobalReadA": meta.WaveSeparateGlobalReadA,
        "WaveSeparateGlobalReadB": meta.WaveSeparateGlobalReadB,
        "GlobalReadVectorWidthA": meta.GlobalReadVectorWidthA,
        "GlobalReadVectorWidthB": meta.GlobalReadVectorWidthB,
        "LocalReadVectorWidthA": meta.LocalReadVectorWidth,
        "LocalReadVectorWidthB": meta.LocalReadVectorWidth,
        "MatrixInstruction": mi,
        "MIWaveGroup": miwg,
        "MIWaveTileA": max(1, meta.MacroTile0 // (mi[0] * miwg[0])),
        "MIWaveTileB": max(1, meta.MacroTile1 // (mi[1] * miwg[1])),
        "LDSTrInst": meta.LDSTrInst,
        "TransposeLDS": meta.TransposeLDS,
    })
    return k


def _gfx1151_metadata_ids():
    """Build a stable id list for pytest parameter printing."""
    metas = _collect_gfx1151_schedule_metadata()
    return [m.name for m in metas]


class TestGfx1151StreamLengthAudit:
    """Audit wave32 idMap-shaped invariants for gfx1151 schedules."""

    @pytest.mark.parametrize(
        "meta",
        _collect_gfx1151_schedule_metadata(),
        ids=_gfx1151_metadata_ids(),
    )
    def test_indices_in_valid_mfma_range(self, meta):
        """Every authored mfmaIndex must be in [-1, numMfma - 1]."""
        k = _gfx1151_kernel_for_metadata(meta)
        has, info = hasCustomSchedule(k)
        assert has, f"{meta.name}: no schedule dispatched for {meta.MacroTile0}x{meta.MacroTile1}x{meta.DepthU}"
        numMfma = info.numMfma
        for key, paths in info.optSchedule.items():
            if key == "SYNC":
                continue
            for code_path_idx, path in enumerate(paths):
                for pos, idx in enumerate(path):
                    assert -1 <= idx <= numMfma - 1, (
                        f"{meta.name}: {key}[{code_path_idx}][{pos}]={idx} out of "
                        f"range [-1, {numMfma - 1}]"
                    )

    @pytest.mark.parametrize(
        "meta",
        _collect_gfx1151_schedule_metadata(),
        ids=_gfx1151_metadata_ids(),
    )
    def test_per_subiter_stream_lengths_uniform(self, meta):
        """LR<A,B><i> / Pack<A,B><i> must have the same length across all sub-iters."""
        k = _gfx1151_kernel_for_metadata(meta)
        has, info = hasCustomSchedule(k)
        assert has
        groups = {"LRA": [], "LRB": [], "PackA": [], "PackB": []}
        for key, paths in info.optSchedule.items():
            for prefix in groups:
                if key.startswith(prefix) and key[len(prefix):].isdigit():
                    for code_path_idx, path in enumerate(paths):
                        groups[prefix].append((key, code_path_idx, len(path)))
                    break
        for prefix, entries in groups.items():
            by_path = {}
            for key, cpi, length in entries:
                by_path.setdefault(cpi, []).append((key, length))
            for cpi, per_sub in by_path.items():
                lengths = {length for _, length in per_sub}
                if len(lengths) > 1:
                    detail = ", ".join(f"{key}={length}" for key, length in per_sub)
                    raise AssertionError(
                        f"{meta.name}: code-path {cpi} has non-uniform per-sub-iter "
                        f"{prefix} lengths: {detail}"
                    )

    @pytest.mark.parametrize(
        "meta",
        _collect_gfx1151_schedule_metadata(),
        ids=_gfx1151_metadata_ids(),
    )
    def test_gr_and_lw_same_cardinality(self, meta):
        """len(GR<X>) must equal len(LW<X>); both drain the same VGPR buffer."""
        k = _gfx1151_kernel_for_metadata(meta)
        has, info = hasCustomSchedule(k)
        assert has
        opt = info.optSchedule
        for suffix in ("A", "B"):
            gr_key, lw_key = f"GR{suffix}", f"LW{suffix}"
            if gr_key not in opt or lw_key not in opt:
                continue
            for cpi, (gr_path, lw_path) in enumerate(zip(opt[gr_key], opt[lw_key])):
                assert len(gr_path) == len(lw_path), (
                    f"{meta.name} code-path {cpi}: "
                    f"len({gr_key})={len(gr_path)} != len({lw_key})={len(lw_path)}"
                )


# ----------------------------------------------------------------------------
# Negative-hazard matrix for the gfx1151 validator passes
# ----------------------------------------------------------------------------
#
# Each test here has a positive counterpart (schedule passes, hazard absent)
# and a negative counterpart (schedule fails, hazard present). Coverage:
#
#   D1 VERIFY_SCC_OVERLAP                   direct call, RDNA35 dialect
#   D2 VERIFY_ASCENDING_ORDER               direct call, both dialects
#   D3 VERIFY_CORRECT_NUMBER_OF_INSTRUCTIONS direct call, divisibility +
#                                           strict-equality path (post-C1)
#   D4 ADD_GR_NOT_TOO_EARLY_CONSTRAINTS     dialect flag gate
#                                           (on CDNA 4, off RDNA 3.5)
#   D5 ADD_LOCAL_READ_CONSTRAINTS           dialect flag gate (on both)
#   D7 ADD_GR_FINISH_BEFORE_LR_CONSTRAINTS  dialect flag gate
#                                           (on CDNA 4, off RDNA 3.5)
#
# (D6 is the C2 TF32 emulation guard; covered by
# ``test_rdna35_tf32_emulation_is_rejected`` above.)
#
# Structural-check negatives (D1/D2/D3) are tested by calling the validator
# functions directly with synthetic ScheduleInfo objects. This avoids
# building the full Timeline pipeline, which is exercised end-to-end by
# ``TestCustomScheduleGfx1151.test_structural_checks_pass`` for every
# gfx1151 schedule.
#
# Timeline-based pass gate tests (D4/D5/D7) assert the
# ``ValidatorDialect`` flag that selects whether the pass runs on each
# architecture. The registered gfx1151 schedules exercise the passes
# end-to-end in ``test_structural_checks_pass``; the flag-level
# assertions below pin the gate so a future regression can't silently
# re-enable a pass that was deliberately flag-gated off on RDNA 3.5.

from Tensile.Components.CMSValidator import (
    verify_scc_overlap,
    verify_ascending_order,
    verify_correct_number_of_instructions,
)


def _rdna35_kernel_for_scc_test(shadow_limit: int = 1, miwave_tile: int = 4):
    """Build a gfx1151 kernel wired for a direct verify_scc_overlap call.

    ``verify_scc_overlap`` needs only ``kernel['DirectToLds']`` and
    ``kernel['Use64bShadowLimit']`` plus the dialect; the dialect
    supplies the (3,2,2,2) / (3,2,1) cluster shape for RDNA 3.5.
    """
    k = _gfx1151_base_kernel()
    update_kernel(k, {
        "MIWaveTileA": miwave_tile, "MIWaveTileB": miwave_tile,
        "MacroTile0": 64, "MacroTile1": 64, "DepthU": 32,
        "Use64bShadowLimit": shadow_limit,
    })
    return k


def _sched_info(optSchedule, *, num_mfma=32, num_code_paths=1):
    """Minimal ScheduleInfo for structural-check tests."""
    return ScheduleInfo(
        numCodePaths=num_code_paths,
        numMfma=num_mfma,
        optSchedule=optSchedule,
        syncCode=[],
        nglshift=0,
        nllshift=0,
    )


class TestGfx1151ValidatorNegativeHazards:
    """Per-pass negative+positive tests for the gfx1151 validator.

    Pairs of positive/negative tests that pin the behavioral contract of
    each re-enabled validator pass on the RDNA 3.5 dialect.
    """

    # ---- D1 -----------------------------------------------------------------

    def test_d1_scc_overlap_rdna35_clean_schedule_passes(self):
        """Clean GRIncA/GRIncB with LWSA/LWSB outside all SCC intervals: pass."""
        k = _rdna35_kernel_for_scc_test(shadow_limit=1, miwave_tile=4)
        opt = {
            "SYNC": [0],
            "GRIncA": [[0, 0, 1, 1, 2, 2, 3, 3, 4]],
            "GRIncB": [[5, 5, 6, 6, 7, 7, 8, 8, 9]],
            # On RDNA 3.5 GRA/GRB are skipped by verify_scc_overlap
            # (check_gr_m0_updates_when_dtl=False + DTL=0). Include them
            # anyway to assert the pass does ignore them.
            "GRA": [[2, 3]],
            "GRB": [[6, 7]],
            "LWSA": [[31]],
            "LWSB": [[31]],
        }
        sched = _sched_info(opt)
        ok, msg = verify_scc_overlap(
            sched,
            {"kernel": k, "dialect": RDNA35_WMMA_DIALECT},
            code_path=0,
        )
        assert ok, f"RDNA35 clean schedule failed unexpectedly: {msg}"

    def test_d1_scc_overlap_rdna35_lwsa_inside_grinca_interval_fails(self):
        """Landing LWSA inside a GRIncA SCC cluster must be rejected."""
        k = _rdna35_kernel_for_scc_test(shadow_limit=1, miwave_tile=4)
        # GRIncA cluster-0 interval spans [0, 1] (s_cmp_eq + 2x s_cselect).
        # Put LWSA at idx=1 (inside cluster-0) -- since LWSA is declared
        # AFTER GRIncA in optSchedule, inInterval uses lhsGt=False so
        # indices equal to the interval min (0) fail, and we can
        # additionally place LWSA exactly between the two GRIncA
        # instructions at indices 0 and 1.
        opt = {
            "SYNC": [0],
            "GRIncA": [[0, 0, 1, 2, 2, 3, 3, 4, 4]],
            "GRIncB": [[5, 5, 6, 7, 7, 8, 8, 9, 9]],
            "LWSA": [[0]],  # hits GRIncA cluster 0 (indices 0..1)
            "LWSB": [[31]],
        }
        sched = _sched_info(opt)
        ok, msg = verify_scc_overlap(
            sched,
            {"kernel": k, "dialect": RDNA35_WMMA_DIALECT},
            code_path=0,
        )
        assert not ok
        assert "LWSA" in msg and "GRIncA" in msg and "SCC" in msg, (
            f"Unexpected failure message: {msg}"
        )

    def test_d1_scc_overlap_rdna35_skips_gra_grb_when_dtl0(self):
        """On RDNA 3.5 (DTL=0, check_gr_m0_updates_when_dtl=False), GRA
        and GRB are not subject to the SCC-cluster check. A GRA index
        that would fail on CDNA 4 DTL=1 must pass on RDNA 3.5.
        """
        k = _rdna35_kernel_for_scc_test(shadow_limit=1, miwave_tile=4)
        opt = {
            "SYNC": [0],
            "GRIncA": [[0, 0, 1, 1, 2, 2, 3, 3, 4]],
            "GRIncB": [[5, 5, 6, 6, 7, 7, 8, 8, 9]],
            # GRA inside GRIncA cluster 0 -- would fail on CDNA 4 DTL=1
            # (see test_gr_simple in test_ValidateSCCoverlap.py), but
            # RDNA 3.5 DTL=0 skips this check.
            "GRA": [[0, 11]],
            "GRB": [[12, 13]],
            "LWSA": [[31]],
            "LWSB": [[31]],
        }
        sched = _sched_info(opt)
        ok, msg = verify_scc_overlap(
            sched,
            {"kernel": k, "dialect": RDNA35_WMMA_DIALECT},
            code_path=0,
        )
        assert ok, (
            f"RDNA 3.5 must skip GRA/GRB SCC overlap check (DTL=0, "
            f"check_gr_m0_updates_when_dtl=False); got: {msg}"
        )

    # ---- D2 -----------------------------------------------------------------

    def test_d2_ascending_order_monotone_passes(self):
        """Monotone-non-decreasing GRIncA sequence passes.

        Note: verify_ascending_order iterates optSchedule.keys() and calls
        ``schedule_get`` on each, so we must not include ``SYNC`` (whose
        value is an int, not a list of ints). The SCC-overlap pass has
        an internal exception for SYNC; verify_ascending_order does not.
        """
        k = _rdna35_kernel_for_scc_test(shadow_limit=1)
        opt = {
            "GRIncA": [[0, 0, 1, 1, 2, 2, 3, 3, 4]],
            "GRIncB": [[5, 5, 6, 6, 7, 7, 8, 8, 9]],
            "LWSA": [[31]], "LWSB": [[31]],
        }
        sched = _sched_info(opt)
        ok, msg = verify_ascending_order(
            sched,
            {"kernel": k, "dialect": RDNA35_WMMA_DIALECT},
            code_path=0,
        )
        assert ok, f"Monotone schedule failed: {msg}"

    def test_d2_ascending_order_out_of_order_fails(self):
        """Out-of-order GRIncA index must be rejected."""
        k = _rdna35_kernel_for_scc_test(shadow_limit=1)
        opt = {
            "GRIncA": [[0, 0, 2, 1, 2, 2, 3, 3, 4]],
            "GRIncB": [[5, 5, 6, 6, 7, 7, 8, 8, 9]],
            "LWSA": [[31]], "LWSB": [[31]],
        }
        sched = _sched_info(opt)
        ok, msg = verify_ascending_order(
            sched,
            {"kernel": k, "dialect": RDNA35_WMMA_DIALECT},
            code_path=0,
        )
        assert not ok
        assert "Non-descending-order rule failed" in msg and "GRIncA" in msg, (
            f"Unexpected failure message: {msg}"
        )

    def test_d2_ascending_order_is_dialect_agnostic(self):
        """The same hazard must fire under CDNA 4 (no ISA-specific logic)."""
        k = create_base_kernel()
        opt = {
            "GRIncA": [[0, 1, 0]],
        }
        sched = _sched_info(opt, num_mfma=8)
        ok, _ = verify_ascending_order(
            sched,
            {"kernel": k, "dialect": CDNA4_DIALECT},
            code_path=0,
        )
        assert not ok

    # ---- D3 -----------------------------------------------------------------

    def test_d3_count_rdna35_divisible_passes(self):
        """RDNA 3.5: idmap_len that's a multiple of authored_len passes
        (authored slots represent uniform packs of ops)."""
        opt = {"LRA0": [[0, 4]]}  # authored_len = 2
        sched = _sched_info(opt, num_mfma=8)
        id_map = {"LRA0": list(range(8))}  # idmap_len = 8; 8 % 2 == 0
        ok, msg = verify_correct_number_of_instructions(
            sched,
            {"kernel": {}, "dialect": RDNA35_WMMA_DIALECT, "idMap": id_map},
            code_path=0,
        )
        assert ok, f"Divisible stream lengths must pass on RDNA 3.5: {msg}"

    def test_d3_count_rdna35_non_divisible_fails(self):
        """RDNA 3.5: authored length that does not evenly divide idmap
        length must be rejected with a remainder diagnostic."""
        opt = {"LRA0": [[0, 2, 4]]}  # authored_len = 3
        sched = _sched_info(opt, num_mfma=8)
        id_map = {"LRA0": list(range(8))}  # 8 % 3 == 2
        ok, msg = verify_correct_number_of_instructions(
            sched,
            {"kernel": {}, "dialect": RDNA35_WMMA_DIALECT, "idMap": id_map},
            code_path=0,
        )
        assert not ok
        assert "does not evenly divide" in msg and "remainder 2" in msg, (
            f"Expected divisibility diagnostic, got: {msg}"
        )

    def test_d3_count_rdna35_zero_authored_fails(self):
        """RDNA 3.5: zero authored slots with non-empty idmap is a
        structural bug (would divide by zero otherwise)."""
        opt = {"LRA0": [[]]}
        sched = _sched_info(opt, num_mfma=8)
        id_map = {"LRA0": list(range(4))}
        ok, msg = verify_correct_number_of_instructions(
            sched,
            {"kernel": {}, "dialect": RDNA35_WMMA_DIALECT, "idMap": id_map},
            code_path=0,
        )
        assert not ok
        assert "0 authored slots" in msg

    def test_d3_count_cdna4_strict_equality_passes(self):
        """CDNA 4: strict equality required -- same-length schedule passes."""
        opt = {"LRA0": [[0, 2, 4, 6]]}
        sched = _sched_info(opt, num_mfma=8)
        id_map = {"LRA0": list(range(4))}  # equal
        ok, msg = verify_correct_number_of_instructions(
            sched,
            {"kernel": {}, "dialect": CDNA4_DIALECT, "idMap": id_map},
            code_path=0,
        )
        assert ok, f"Strict-equality passes on CDNA 4: {msg}"

    def test_d3_count_cdna4_strict_equality_fails_on_divisible_mismatch(self):
        """CDNA 4: a schedule that passes RDNA 3.5 divisibility (8 % 2 == 0)
        must still FAIL on CDNA 4 under strict equality (2 != 8). This
        pins the dialect split so neither side can silently adopt the
        other's invariant."""
        opt = {"LRA0": [[0, 4]]}  # authored_len = 2
        sched = _sched_info(opt, num_mfma=8)
        id_map = {"LRA0": list(range(8))}  # idmap_len = 8
        ok, msg = verify_correct_number_of_instructions(
            sched,
            {"kernel": {}, "dialect": CDNA4_DIALECT, "idMap": id_map},
            code_path=0,
        )
        assert not ok
        assert "2 instructions" in msg and "8 instructions" in msg

    # ---- D4 / D5 / D7 : dialect flag gates ---------------------------------

    def test_d4_gr_not_too_early_is_flag_gated_off_on_rdna35(self):
        """ADD_GR_NOT_TOO_EARLY_CONSTRAINTS has two flag-gated paths; the
        RDNA 3.5 dialect disables both (DTL=0 has no same-block LDS
        reuse, no same-loop GRInc->GR ordering requirement). CDNA 4
        keeps both on."""
        assert RDNA35_WMMA_DIALECT.gr_must_start_after_lr0s is False
        assert RDNA35_WMMA_DIALECT.gr_must_follow_grinc_in_same_loop is False
        assert CDNA4_DIALECT.gr_must_start_after_lr0s is True
        assert CDNA4_DIALECT.gr_must_follow_grinc_in_same_loop is True

    def test_d5_local_read_constraints_is_active_on_rdna35(self):
        """ADD_LOCAL_READ_CONSTRAINTS is not flag-gated; both dialects
        run the same pass but with dialect-specific LR consumer offsets.
        The (1, 2) half-offsets hold for all registered gfx1151
        schedules; this assertion pins the dialect values so a future
        edit can't silently drift them."""
        assert RDNA35_WMMA_DIALECT.lr0_consumer_half_offset == 1
        assert RDNA35_WMMA_DIALECT.lr1_consumer_half_offset == 2
        assert CDNA4_DIALECT.lr0_consumer_half_offset == 1
        assert CDNA4_DIALECT.lr1_consumer_half_offset == 2

    def test_d7_gr_finish_before_lr_is_flag_gated_off_on_rdna35(self):
        """ADD_GR_FINISH_BEFORE_LR_CONSTRAINTS encodes the CDNA 4 DTL=1
        "GR writes the LDS block next-iter LR1 reads" invariant. RDNA
        3.5 DTL=0 decouples GR (buffer_load -> VGPRs) from the LDS fill
        that drives LR1 (a separate LocalWrite stream with its own
        handshake), so the constraint is gated off there."""
        assert RDNA35_WMMA_DIALECT.gr_finish_before_lr is False
        assert CDNA4_DIALECT.gr_finish_before_lr is True

    def test_flag_matrix_is_consistent_with_timing_dialect(self):
        """Sanity: the RDNA 3.5 dialect must match the C1/C2/C3
        calibration decisions: stream_length_strict_equality=False (C1),
        plus the three flag-gates above. A single-dialect regression
        that accidentally flips any of these is a validator soundness
        bug that every gfx1151 schedule would be affected by."""
        assert RDNA35_WMMA_DIALECT.stream_length_strict_equality is False
        assert CDNA4_DIALECT.stream_length_strict_equality is True

