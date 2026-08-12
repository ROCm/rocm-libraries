import importlib

import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")

pytestmark = pytest.mark.unit

class _FakeIsaInfo:
    def __init__(self, has_ecc_half):
        self.archCaps = {"HasEccHalf": has_ecc_half}

class _FakeDataType:
    def __init__(self, num_registers):
        self._num_registers = num_registers

    def numRegisters(self):
        return self._num_registers

def _base_state(dtva, dtvb, plr=0, num_registers=0.5):
    isa = (9, 0, 10)
    return {
        "ISA": isa,
        "EnableMatrixInstruction": True,
        "PrefetchLocalRead": plr,
        "DirectToVgprA": dtva,
        "DirectToVgprB": dtvb,
        "ProblemType": {"DataType": _FakeDataType(num_registers)},
    }

def _isa_map():
    return {(9, 0, 10): _FakeIsaInfo(True)}

def test_and_or_mutant_A_true_B_false():

    state = _base_state(dtva=True, dtvb=False)
    result = S.Solution.isVgprForLocalReadPackingDoable(state, _isa_map())
    assert result is False

def test_and_or_mutant_A_false_B_true():

    state = _base_state(dtva=False, dtvb=True)
    result = S.Solution.isVgprForLocalReadPackingDoable(state, _isa_map())
    assert result is False

def test_dtva_and_dtvb_both_true_exception_keeps_doable():

    state = _base_state(dtva=True, dtvb=True)
    result = S.Solution.isVgprForLocalReadPackingDoable(state, _isa_map())
    assert result is True

def test_all_conditions_pass_returns_true():

    state = _base_state(dtva=False, dtvb=False, plr=1)
    result = S.Solution.isVgprForLocalReadPackingDoable(state, _isa_map())
    assert result is True
