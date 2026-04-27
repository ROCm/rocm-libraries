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
from Tensile.Components.CMSValidator import isValid, SchedulePosition, ValidationContext, ValidationConcern, active_concerns
from Tensile.Common import IsaVersion
from cms_test_utils import (
    make_mock_id_map, make_mock_mfma_code,
    kernel_to_solution_config, generate_real_idmap, subset_id_map,
)


def _make_context(kernel, schedule_info, asm=None, isaInfoMap=None):
    """Build a ValidationContext.

    Default path uses mock idMap (preserves existing behavior for ~30 callers).
    When asm + isaInfoMap are passed, uses the real kernel-writer-produced
    idMap. Required for the 128x128x64 vwa=4 parametrizations whose mock
    idMap can't model VW>1 swap-pack interleaving.
    """
    if asm is not None and isaInfoMap is not None:
        config = kernel_to_solution_config(kernel)
        real_id_map, real_mfma_code, _ = generate_real_idmap(config, asm, isaInfoMap)
        mock_fb = make_mock_id_map(schedule_info, kernel)
        id_map = subset_id_map(real_id_map, schedule_info.optSchedule,
                               syncCode=schedule_info.syncCode,
                               snopCode=schedule_info.snopCode,
                               fallback_id_map=mock_fb)
        mfma_code = real_mfma_code[:int(schedule_info.numMfma)]
        return ValidationContext(kernel=kernel, id_map=id_map, mfma_code=mfma_code)
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
        assert valid, message

class TestCustomScheduleTF32:
    @pytest.fixture(autouse=True)
    def _inject_isa(self, isa_infrastructure):
        """Stash assembler + isaInfoMap so test_schedule_128x128x64 vwa=4
        cases can opt into real idMap. Other tests in this class are
        unaffected — _make_context defaults to the mock path."""
        self._isaInfoMap = isa_infrastructure[1]
        self._asm = isa_infrastructure[2]

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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        numMfma = (mi_wave_tile[0] * mi_wave_tile[1] *
                   3 *                      # tf32 emulated with 3 bf16
                   (1 if mi[0] == 16 else 2)   # two sub-iterations with mi32
        )
        assert schedule_info.numMfma == numMfma
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        # vwa=4 cases need real idMap because mock register layout doesn't
        # model VW>1 swap-pack interleaving (see CMSValidator_TODO.md).
        ctx_args = (self._asm, self._isaInfoMap) if vwa == 4 else (None, None)
        valid, message = isValid(schedule_info,
                                 _make_context(kernel, schedule_info, *ctx_args))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
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
        valid, message = isValid(schedule_info, _make_context(kernel, schedule_info))
        assert valid, message

class TestCustomScheduleValidation:
    def test_ascending_order_catches_invalid_schedule(self):
        """A non-ascending schedule is caught by the ordering concern."""
        kernel = create_base_kernel()
        invalid_schedule = {"P": [[3, 2, 1]]}

        si = ScheduleInfo(1, None, invalid_schedule, None, None, None, None)
        status, message = isValid(si, _make_context(kernel, si))
        assert status == False
        assert "Non-descending-order" in message

    def test_active_concerns_gfx950_with_packs(self):
        """gfx950 kernel with packs includes PACK_DATA_READY."""
        kernel = create_base_kernel()
        idmap = {"LRA0": [], "PackA0": [], "GRA": [], "GRIncA": [], "SYNC": []}
        concerns = active_concerns(kernel, idmap)
        assert ValidationConcern.PACK_DATA_READY in concerns
        assert ValidationConcern.LR_DATA_READY in concerns
        assert ValidationConcern.INSTRUCTION_ORDERING in concerns
        assert ValidationConcern.SCHEDULE_COMPLETENESS in concerns
        assert ValidationConcern.SCALAR_REGISTER_SAFETY in concerns

    def test_active_concerns_gfx950_tf32(self):
        """gfx950 kernel with TF32 includes QUAD_CYCLE_TIMING."""
        kernel = create_base_kernel()
        kernel["UseF32XEmulation"] = True
        idmap = {"LRA0": [], "PackA0": [], "GRA": []}
        concerns = active_concerns(kernel, idmap)
        assert ValidationConcern.QUAD_CYCLE_TIMING in concerns

    def test_active_concerns_gfx1151(self):
        """gfx1151 currently only has INSTRUCTION_ORDERING in its catalog.

        Other concerns (LW_ORDERING, GR_VGPR_READY, LDS coherence, etc.) are
        declared as future work in the catalog comments but not yet active —
        they will be added in stage 12 when the corresponding rules are built.
        """
        kernel = _gfx1151_base_kernel()
        kernel["MacroTile0"] = 96
        kernel["MacroTile1"] = 128
        idmap = {"LRA0": [], "GRA": [], "SYNC": []}
        concerns = active_concerns(kernel, idmap)
        assert ValidationConcern.INSTRUCTION_ORDERING in concerns
        # No other concerns are active yet for gfx1151
        assert ValidationConcern.PACK_DATA_READY not in concerns
        assert ValidationConcern.QUAD_CYCLE_TIMING not in concerns
        assert ValidationConcern.SCALAR_REGISTER_SAFETY not in concerns
        assert ValidationConcern.SCHEDULE_COMPLETENESS not in concerns

    def test_active_concerns_unknown_isa(self):
        """Unknown ISA returns empty set (graceful fallback)."""
        kernel = create_base_kernel()
        kernel["ISA"] = (99, 99, 99)
        concerns = active_concerns(kernel, {})
        assert concerns == set()

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
#   - validator  : active_concerns() auto-skips rules that don't apply to
#     gfx1151's ISA catalog, so isValid returns True with the correct
#     subset of validation (ordering checks, not CDNA-4 timing).


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
    (96,  128, 32,  2, 1, [2, 2], 3, 4, 16),
    (128, 96,  32,  2, 1, [2, 2], 4, 3, 16),
    (192, 64,  32,  2, 1, [4, 1], 3, 4, 16),
    (64,  192, 32,  2, 1, [1, 4], 4, 3, 16),
    (64,  128, 32,  2, 1, [2, 2], 2, 4, 16),
    (128, 64,  64,  2, 1, [2, 2], 4, 2, 16),
    (32,  128, 64,  2, 1, [1, 4], 2, 2, 16),
    (16,  16,  128, 2, 1, [1, 1], 1, 1, 16),
]


class TestCustomScheduleGfx1151:
    """Tests for gfx1151 (RDNA 3.5) WMMA schedules."""

    # ---- Dispatcher / predicate ----

    def test_dispatch_gfx1151_TN_16bit(self):
        """A TN fp16 gfx1151 kernel should pick up a gfx1151 schedule."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "MacroTile0": 96, "MacroTile1": 128, "DepthU": 32,
            "MIWaveTileA": 3, "MIWaveTileB": 4,
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
        })
        has, info = hasCustomSchedule(k)
        assert has, f"no schedule dispatched for MT{MT0}x{MT1}x{DU}"
        assert isinstance(info, ScheduleInfo)
        assert info.numMfma > 0
        assert info.numCodePaths >= 1
        assert "SYNC" in info.optSchedule

    # ---- Validator coverage ----

    @pytest.mark.parametrize("MT0, MT1, DU, PGR, PLR, MIWG, WTA, WTB, MIK", GFX1151_TILES)
    def test_validator_passes_gfx1151(
            self, MT0, MT1, DU, PGR, PLR, MIWG, WTA, WTB, MIK):
        """Applicable validator rules must run and succeed on gfx1151 schedules."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "MacroTile0": MT0, "MacroTile1": MT1, "DepthU": DU,
            "PrefetchGlobalRead": PGR, "PrefetchLocalRead": PLR,
            "MatrixInstruction": [16, 16, MIK, 1],
            "MIWaveGroup": MIWG, "MIWaveTileA": WTA, "MIWaveTileB": WTB,
        })
        has, info = hasCustomSchedule(k)
        assert has
        valid, msg = isValid(info, _make_context(k, info))
        assert valid, f"MT{MT0}x{MT1}x{DU}: isValid said: {msg}"

    def test_gfx1151_concerns_skip_timeline_rules(self):
        """gfx1151 catalog only has INSTRUCTION_ORDERING, so all timeline rules auto-skip."""
        k = _gfx1151_base_kernel()
        k["MacroTile0"] = 96
        k["MacroTile1"] = 128
        idmap = {"LRA0": [], "GRA": [], "SYNC": []}
        concerns = active_concerns(k, idmap)
        # No timeline rule concerns match, so timeline construction is skipped entirely.
        assert ValidationConcern.PACK_DATA_READY not in concerns
        assert ValidationConcern.LR_DATA_READY not in concerns
        assert ValidationConcern.LDS_WRITE_AFTER_READ not in concerns
        assert ValidationConcern.LDS_READ_AFTER_WRITE not in concerns
        # Ordering is the only active concern
        assert concerns == {ValidationConcern.INSTRUCTION_ORDERING}
