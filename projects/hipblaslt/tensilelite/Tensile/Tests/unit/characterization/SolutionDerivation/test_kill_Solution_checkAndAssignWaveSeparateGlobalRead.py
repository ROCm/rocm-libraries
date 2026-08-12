import importlib

import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")
Solution = S.Solution

pytestmark = pytest.mark.unit

DEPTHU_MSG = (
    "didn't support WaveSeparateGlobalRead when DepthU is not multiple "
    "of wave %u in TLU%s"
)
MACROTILE_MSG = (
    "didn't support WaveSeparateGlobalRead when MacroTile is not multiple "
    "of wave %u in TLU%s"
)

def _state(**over):
    state = {
        "NumThreads": 128,
        "WavefrontSize": 64,
        "Valid": True,
        "WaveSeparateGlobalReadA": 1,
        "_DepthUA": 8,
        "MacroTileA": 128,
        "ProblemType": {"TLUA": True},
    }
    state.update(over)
    return state

def test_num_of_waves_uses_floor_division():

    state = _state(NumThreads=160, WavefrontSize=64, _DepthUA=5)
    Solution.checkAndAssignWaveSeparateGlobalRead(state, "A", False)
    assert state["Valid"] is False

def test_depthu_strictly_greater_than_zero_short_circuits():

    state = _state(NumThreads=32, WavefrontSize=64, _DepthUA=0)
    Solution.checkAndAssignWaveSeparateGlobalRead(state, "A", False)
    assert state["Valid"] is True

def test_depthu_guard_boundary_one():

    state = _state(NumThreads=128, WavefrontSize=64, _DepthUA=1)
    Solution.checkAndAssignWaveSeparateGlobalRead(state, "A", False)
    assert state["Valid"] is False

def test_depthu_reject_prints_exact_message(capsys):

    state = _state(NumThreads=128, WavefrontSize=64, _DepthUA=5)
    Solution.checkAndAssignWaveSeparateGlobalRead(state, "A", True)
    captured = capsys.readouterr()
    expected = (DEPTHU_MSG % (5, "A")) + "\n"
    assert expected in captured.out
    assert state["Valid"] is False

def test_macrotile_reject_prints_exact_message(capsys):

    state = _state(
        NumThreads=128,
        WavefrontSize=64,
        ProblemType={"TLUA": False},
        MacroTileA=5,
    )
    Solution.checkAndAssignWaveSeparateGlobalRead(state, "A", True)
    captured = capsys.readouterr()
    expected = (MACROTILE_MSG % (5, "A")) + "\n"
    assert expected in captured.out
    assert state["Valid"] is False
