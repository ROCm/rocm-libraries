import importlib

import pytest
from types import SimpleNamespace as NS

S = importlib.import_module("Tensile.SolutionStructs.Solution")

pytestmark = pytest.mark.unit

def _make_inputs(has_mfma=False, has_wmma_v1=False, has_wmma_v2=False, has_wmma_v3=False):
    isa = (9, 0, 0)
    state = {"ISA": [9, 0, 0]}
    isaInfoMap = {
        isa: NS(
            asmCaps={
                "HasMFMA": has_mfma,
                "HasWMMA_V1": has_wmma_v1,
                "HasWMMA_V2": has_wmma_v2,
                "HasWMMA_V3": has_wmma_v3,
            }
        )
    }
    return state, isaInfoMap

def test_else_branch_returns_initial_values_and_warns(capsys):
    """No MFMA and no WMMA caps: falls to the else branch.

    Kills the initial-value mutants (outputVectorWidth=4->5, RegsPerOut=1->None,
    RegsPerOut=1->2) because the else branch never overwrites them, and kills all
    four print(...) string/None mutants via exact stdout comparison.
    """
    state, isaInfoMap = _make_inputs()
    outputVectorWidth, RegsPerOut = S.Solution.getMIOutputInfo(state, isaInfoMap)

    assert (outputVectorWidth, RegsPerOut) == (4, 1)
    assert RegsPerOut == 1
    assert isinstance(RegsPerOut, int) and RegsPerOut is not None

    captured = capsys.readouterr()
    assert captured.out == "WARNING: unexpect code flow\n"

def test_wmma_v2_only_takes_or_branch():
    """HasWMMA_V2 True, HasWMMA_V3 False: the 'or' condition is True.

    Kills the 'or'->'and' mutant (which would fall through to the else branch and
    return (4,1)) and the (8,1)->(9,1)/(8,2) tuple mutants.
    """
    state, isaInfoMap = _make_inputs(has_wmma_v2=True, has_wmma_v3=False)
    result = S.Solution.getMIOutputInfo(state, isaInfoMap)
    assert result == (8, 1)

def test_wmma_v3_only_takes_or_branch():
    """HasWMMA_V3 True, HasWMMA_V2 False: the 'or' condition is still True.

    Second direction of the 'or'->'and' mutant; the mutant would return (4,1).
    """
    state, isaInfoMap = _make_inputs(has_wmma_v2=False, has_wmma_v3=True)
    result = S.Solution.getMIOutputInfo(state, isaInfoMap)
    assert result == (8, 1)

def test_wmma_v2_and_v3_still_eight_one():
    """Both WMMA_V2 and WMMA_V3 True: reinforces the (8,1) tuple value mutants."""
    state, isaInfoMap = _make_inputs(has_wmma_v2=True, has_wmma_v3=True)
    result = S.Solution.getMIOutputInfo(state, isaInfoMap)
    assert result == (8, 1)
