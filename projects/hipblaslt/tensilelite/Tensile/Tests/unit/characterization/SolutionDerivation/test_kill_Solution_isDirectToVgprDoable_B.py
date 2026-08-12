import copy
import importlib

import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")
pytestmark = pytest.mark.unit

def _mk(real_state, **over):
    st = copy.deepcopy(real_state)
    st.pop("SolutionIndex", None)
    st.pop("SolutionNameMin", None)
    st["Valid"] = True
    st["enableGLTrB"] = True
    st["GlobalReadVectorWidthB"] = 8
    st["LocalReadVectorWidthB"] = 8
    for k, v in over.items():
        st[k] = v
    return st

def _run(st, isa_info_map):
    return S.Solution.isDirectToVgprDoable(st, "B", True, isa_info_map)

def test_b_passthrough_returns_true(real_state, isa_info_map, capsys):
    st = _mk(real_state)
    assert _run(st, isa_info_map) is True
    assert st["Valid"] is True
    assert capsys.readouterr().out == ""

def test_b_reject_matrixinstbm_not_one(real_state, isa_info_map, capsys):
    st = _mk(real_state, MatrixInstBM=2)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "MatrixInstBM should be 1 for DirectToVgprB" in capsys.readouterr().out

def test_b_reject_gltr_grvw_not_8(real_state, isa_info_map, capsys):
    st = _mk(real_state, GlobalReadVectorWidthB=4)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "enableGLTrB and GlobalReadVectorWidth != 8" in capsys.readouterr().out

def test_b_reject_grvw_ne_lrvw(real_state, isa_info_map, capsys):
    st = _mk(real_state, LocalReadVectorWidthB=4)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "GlobalReadVectorWidthB(8) != LocalReadVectorWidth(4)" in capsys.readouterr().out

def test_b_reject_prefetchglobalread_zero(real_state, isa_info_map, capsys):
    st = _mk(real_state, PrefetchGlobalRead=0)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "DirectToVgprB does not supports PrefetchGlobalRead == 0" in capsys.readouterr().out

def test_b_nn_tail_sets_assert_summation_element_multiple(real_state, isa_info_map, capsys):
    st = _mk(real_state)
    st["ProblemType"] = copy.deepcopy(st["ProblemType"])
    st["ProblemType"]["TransposeA"] = False
    st["ProblemType"]["TransposeB"] = False
    st["AssertSummationElementMultiple"] = 1
    st["DepthU"] = 16
    assert _run(st, isa_info_map) is True
    assert st["AssertSummationElementMultiple"] == 16
