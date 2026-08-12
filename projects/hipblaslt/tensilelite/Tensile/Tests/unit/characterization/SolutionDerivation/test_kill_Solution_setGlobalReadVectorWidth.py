import importlib

import pytest

from Tensile.Common.DataType import DataType

S = importlib.import_module("Tensile.SolutionStructs.Solution")
Solution = S.Solution

pytestmark = pytest.mark.unit

def test_grvw_16_is_valid_member():

    state = {"UseSubtileImpl": False, "NumThreads": 256}
    rv = Solution.setGlobalReadVectorWidth(state, "A", 512, 16, printRejectionReason=False)
    assert rv is True
    assert state["GlobalReadVectorWidthA"] == 16

def test_grvw_32_is_valid_member():

    state = {"UseSubtileImpl": False, "NumThreads": 256}
    rv = Solution.setGlobalReadVectorWidth(state, "A", 512, 32, printRejectionReason=False)
    assert rv is True
    assert state["GlobalReadVectorWidthA"] == 32

def test_tdm_A_float4_full_state():
    state = {
        "UseSubtileImpl": False,
        "TDMInst": 3,
        "NumThreads": 256,
        "ProblemType": {"DataTypeA": DataType("Float4"), "Sparse": 0},
    }
    rv = Solution.setGlobalReadVectorWidth(state, "A", 512, 4, printRejectionReason=False)
    assert rv is True
    assert state["GlobalReadVectorWidthA"] == 2
    assert state["NumLoadsA"] == 1
    assert state["NumLoadsCoalescedA"] == 1
    assert state["NumLoadsPerpendicularA"] == 1
    assert set(state) == {
        "UseSubtileImpl", "TDMInst", "NumThreads", "ProblemType",
        "GlobalReadVectorWidthA", "NumLoadsA",
        "NumLoadsCoalescedA", "NumLoadsPerpendicularA",
    }

def test_tdm_B_float4_full_state():

    state = {
        "UseSubtileImpl": False,
        "TDMInst": 3,
        "NumThreads": 256,
        "ProblemType": {"DataTypeB": DataType("Float4"), "Sparse": 0},
    }
    rv = Solution.setGlobalReadVectorWidth(state, "B", 512, 4, printRejectionReason=False)
    assert rv is True
    assert state["GlobalReadVectorWidthB"] == 2
    assert state["NumLoadsB"] == 1
    assert state["NumLoadsCoalescedB"] == 1
    assert state["NumLoadsPerpendicularB"] == 1
    assert set(state) == {
        "UseSubtileImpl", "TDMInst", "NumThreads", "ProblemType",
        "GlobalReadVectorWidthB", "NumLoadsB",
        "NumLoadsCoalescedB", "NumLoadsPerpendicularB",
    }

def test_tdm_MXSA_default_grvw_one():
    state = {"UseSubtileImpl": False, "TDMInst": 3, "NumThreads": 256}
    rv = Solution.setGlobalReadVectorWidth(state, "MXSA", 512, 4, printRejectionReason=False)
    assert rv is True
    assert state["GlobalReadVectorWidthMXSA"] == 1
    assert state["NumLoadsMXSA"] == 1
    assert state["NumLoadsCoalescedMXSA"] == 1
    assert state["NumLoadsPerpendicularMXSA"] == 1
    assert set(state) == {
        "UseSubtileImpl", "TDMInst", "NumThreads",
        "GlobalReadVectorWidthMXSA", "NumLoadsMXSA",
        "NumLoadsCoalescedMXSA", "NumLoadsPerpendicularMXSA",
    }

def test_tdm_MXSB_default_grvw_one():
    state = {"UseSubtileImpl": False, "TDMInst": 3, "NumThreads": 256}
    rv = Solution.setGlobalReadVectorWidth(state, "MXSB", 512, 4, printRejectionReason=False)
    assert rv is True
    assert state["GlobalReadVectorWidthMXSB"] == 1
    assert state["NumLoadsMXSB"] == 1
    assert state["NumLoadsCoalescedMXSB"] == 1
    assert state["NumLoadsPerpendicularMXSB"] == 1
    assert set(state) == {
        "UseSubtileImpl", "TDMInst", "NumThreads",
        "GlobalReadVectorWidthMXSB", "NumLoadsMXSB",
        "NumLoadsCoalescedMXSB", "NumLoadsPerpendicularMXSB",
    }

def test_tdm_A_sixbit_float_grvw_four():
    state = {
        "UseSubtileImpl": False,
        "TDMInst": 3,
        "NumThreads": 256,
        "ProblemType": {"DataTypeA": DataType("Float6"), "Sparse": 0},
    }
    Solution.setGlobalReadVectorWidth(state, "A", 512, 4, printRejectionReason=False)
    assert state["GlobalReadVectorWidthA"] == 4

def test_sparse_A_s1_grvw_four():

    state = {
        "UseSubtileImpl": False,
        "TDMInst": 3,
        "NumThreads": 256,
        "ProblemType": {"DataTypeA": DataType("Half"), "Sparse": 1},
    }
    Solution.setGlobalReadVectorWidth(state, "A", 512, 4, printRejectionReason=False)
    assert state["GlobalReadVectorWidthA"] == 4

def test_sparse_A_s0_grvw_stays_one():

    state = {
        "UseSubtileImpl": False,
        "TDMInst": 3,
        "NumThreads": 256,
        "ProblemType": {"DataTypeA": DataType("Half"), "Sparse": 0},
    }
    Solution.setGlobalReadVectorWidth(state, "A", 512, 4, printRejectionReason=False)
    assert state["GlobalReadVectorWidthA"] == 1

def test_sparse_B_s2_grvw_four():

    state = {
        "UseSubtileImpl": False,
        "TDMInst": 3,
        "NumThreads": 256,
        "ProblemType": {"DataTypeB": DataType("Half"), "Sparse": 2},
    }
    Solution.setGlobalReadVectorWidth(state, "B", 512, 4, printRejectionReason=False)
    assert state["GlobalReadVectorWidthB"] == 4

def test_sparse_B_s1_grvw_stays_one():

    state = {
        "UseSubtileImpl": False,
        "TDMInst": 3,
        "NumThreads": 256,
        "ProblemType": {"DataTypeB": DataType("Half"), "Sparse": 1},
    }
    Solution.setGlobalReadVectorWidth(state, "B", 512, 4, printRejectionReason=False)
    assert state["GlobalReadVectorWidthB"] == 1

def test_reject_sets_valid_false_via_state_arg():
    state = {"UseSubtileImpl": False, "NumThreads": 256}
    rv = Solution.setGlobalReadVectorWidth(state, "A", 500, 4, printRejectionReason=False)
    assert rv is False
    assert "Valid" in state
    assert state["Valid"] is False

def test_reject_message_exact_text(capsys):
    state = {"UseSubtileImpl": False, "NumThreads": 256}
    rv = Solution.setGlobalReadVectorWidth(state, "A", 500, 4, printRejectionReason=True)
    out = capsys.readouterr().out
    assert "\nreject: totalVectorsA 500 % NumThreads 256 != 0\n" in out
    assert rv is False
