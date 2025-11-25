import pytest
from unittest.mock import MagicMock

from Tensile.Components.CustomSchedule import hasCustomSchedule, ScheduleInfo, verifyAscendingOrder
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
        "ISA": IsaVersion(9,5,0),
        "ProblemType": {
            "DataType": _mock_dtype(),
            "DataTypeA": _mock_dtype(),
            "DataTypeB": _mock_dtype(),
            "TransposeA": False,
            "TransposeB": False,
        },
        "MacroTile0": 0, "MacroTile1": 0, "DepthU": 0,
        "PrefetchGlobalRead": 0, "PrefetchLocalRead": 0, "DirectToLds": False,
        "GlobalReadVectorWidthA": 0, "GlobalReadVectorWidthB": 0,
        "LocalReadVectorWidth": 0,
        "MatrixInstruction": [],
        "MIWaveGroup": [],
        "LDSTrInst": False,
        "TransposeLDS": 0,
        "ForceUnrollSubIter": False,
        "SwapGlobalReadOrder": False, # For asserting it gets set
        "UsePLRPack": False, # For asserting it gets set
    }
    return kernel

class TestCustomSchedule:
    def test_no_custom_schedule(self):
        """Test that a kernel that doesn't match any condition returns False."""
        kernel = create_base_kernel()
        # An empty kernel should not have a custom schedule
        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert not has_schedule
        assert schedule_info is None


    def test_schedule_256x256x64_16bit_TN(self):
        """Tests the 256x256x64 16-bit TN schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": True, "TransposeB": False
        })
        kernel.update({
            "MacroTile0": 256, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "DirectToLds": True,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2], "TransposeLDS": 1
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)

        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 128
        assert schedule_info.isValid({"kernel" : kernel})[0]
        assert 'PackA0' not in schedule_info.optSchedule
        assert not kernel["UsePLRPack"]

    def test_schedule_256x256x64_16bit_NT(self):
        """Tests the 256x256x64 16-bit NT schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": False, "TransposeB": True
        })
        kernel.update({
            "MacroTile0": 256, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "DirectToLds": True,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": False, "TransposeLDS": 0
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)

        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 128
        assert 'PackA0' in schedule_info.optSchedule
        assert kernel["UsePLRPack"]
        assert schedule_info.isValid({"kernel" : kernel})[0]

    @pytest.mark.parametrize("transA, transB", [(False, False), (True, True)])
    def test_schedule_256x256x64_16bit_NN_TT(self, transA, transB):
        """Tests the 256x256x64 16-bit NN and TT schedules."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": transA, "TransposeB": transB
        })
        kernel.update({
            "MacroTile0": 256, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "DirectToLds": True,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": False, "TransposeLDS": 1
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)

        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 128
        assert kernel["UsePLRPack"]
        assert schedule_info.isValid({"kernel" : kernel})[0]
        if transA and transB: # isTT
            assert kernel["SwapGlobalReadOrder"]
            assert 'PackB0' in schedule_info.optSchedule
            assert 'PackA0' not in schedule_info.optSchedule
        else: # isNN
            assert not kernel["SwapGlobalReadOrder"]
            assert 'PackA0' in schedule_info.optSchedule
            assert 'PackB0' not in schedule_info.optSchedule

    def test_schedule_256x256x128_8bit_TN(self):
        """Tests the 256x256x128 8-bit TN schedule."""
        kernel = create_base_kernel()
        dtype_8bit = _mock_dtype(is_8bit=True, num_bytes=1)
        kernel["ProblemType"].update({
            "DataType": dtype_8bit, "DataTypeA": dtype_8bit, "DataTypeB": dtype_8bit,
            "TransposeA": True, "TransposeB": False
        })
        kernel.update({
            "MacroTile0": 256, "MacroTile1": 256, "DepthU": 128,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 0, "DirectToLds": True,
            "GlobalReadVectorWidthA": 16, "GlobalReadVectorWidthB": 16, "LocalReadVectorWidth": 16,
            "MatrixInstruction": [16,16,128,1], "MIWaveGroup": [2,2], "TransposeLDS": 1
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)

        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 1
        assert schedule_info.numMfma == 64
        assert len(schedule_info.mfmaReorder) > 0
        assert schedule_info.isValid({"kernel" : kernel})[0]

    def test_schedule_192x256x64_16bit_NN(self):
        """Tests the 192x256x64 16-bit NN schedule."""
        kernel = create_base_kernel()
        dtype_16bit = _mock_dtype(is_16bit=True, num_bytes=2)
        kernel["ProblemType"].update({
            "DataType": dtype_16bit, "DataTypeA": dtype_16bit, "DataTypeB": dtype_16bit,
            "TransposeA": False, "TransposeB": False
        })
        kernel.update({
            "MacroTile0": 192, "MacroTile1": 256, "DepthU": 64,
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "DirectToLds": True,
            "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 8,
            "MatrixInstruction": [16,16,32,1], "MIWaveGroup": [2,2],
            "LDSTrInst": True, "TransposeLDS": 1
        })

        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert has_schedule
        assert isinstance(schedule_info, ScheduleInfo)
        assert schedule_info.numCodePaths == 2
        assert schedule_info.numMfma == 96
        assert kernel["SwapGlobalReadOrder"]
        assert schedule_info.isValid({"kernel" : kernel})[0]


class TestCustomScheduleValidation:
    def test_schedule_validation_non_descending_order(self):
        """
        Test of the rule that instructions in each category
        appear in non-descending order
        """

        sched = ScheduleInfo(
            None, None, {"P": [[3, 2, 1]]}, None, None, None, None
        )
        status, message = verifyAscendingOrder(sched)

        expected = "Non-descending-order rule failed, schedule key 'P', sequence [3, 2, 1]: value 2 at index 1 is less than 3 at index 0."
        assert status == False
        assert message == expected

        sched = ScheduleInfo(
            None, None, {"P": [[1, 1, 2]]}, None, None, None, None
        )
        status, message = verifyAscendingOrder(sched)
        assert status == True

    def test_schedule_validation_disable(self):
        """
        Test of the flag that custom mainloop schedule (CMS) developers can use to override the
        validation checks.
        """
        kernel = create_base_kernel()
        invalid_schedule = {"P": [[3, 2, 1]]}

        # No verification message means that the schedule info is considered valid.
        scheduleInfo = ScheduleInfo(
            None, None, invalid_schedule, None, None, None, None
        )
        scheduleInfo.disableValidation()
        status, message = scheduleInfo.isValid({"kernel" : {"DepthU": 42}})
        assert status == True
        assert message == "CMS validation explicitly disabled. Running on kernel with MT0xMT1xDepthU = ?x?x42"

        # A non-empty verification message means that the schedule info is considered invalid.
        status, message = ScheduleInfo(
            None, None, invalid_schedule, None, None, None, None
        ).isValid({})
        assert status == False


