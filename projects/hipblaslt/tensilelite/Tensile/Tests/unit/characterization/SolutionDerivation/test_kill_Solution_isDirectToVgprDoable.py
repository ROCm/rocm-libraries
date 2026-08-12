import copy
import importlib

import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")
pytestmark = pytest.mark.unit

_PASS_A = {"GlobalReadVectorWidthA": 4, "NumLoadsCoalescedA": 2}

def _mk(real_state, **over):
    st = copy.deepcopy(real_state)
    st.update(_PASS_A)
    st.pop("SolutionIndex", None)
    st.pop("SolutionNameMin", None)
    st["Valid"] = True
    for k, v in over.items():
        st[k] = v
    return st

def _run(st, isa_info_map, tc="A"):
    return S.Solution.isDirectToVgprDoable(st, tc, True, isa_info_map)

def test_passthrough_returns_true(real_state, isa_info_map, capsys):
    st = _mk(real_state)
    assert _run(st, isa_info_map) is True
    assert st["Valid"] is True
    assert capsys.readouterr().out == ""

def test_reject_matrix_instruction_only(real_state, isa_info_map, capsys):
    st = _mk(real_state, EnableMatrixInstruction=False)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "DirectToVgpr is for MatrixInstruction only" in capsys.readouterr().out

def test_reject_lrvw_lt_miinputperthread(real_state, isa_info_map):
    st = _mk(real_state, LocalReadVectorWidthA=1)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False

def test_reject_lsu_tlu_nonswizzle(real_state, isa_info_map, capsys):
    st = _mk(real_state, LocalSplitU=2)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "Non-Swizzled DirectToVgpr + LSU + TLU=False has not been enabled yet" in capsys.readouterr().out

def test_reject_dtva_and_dtvb(real_state, isa_info_map, capsys):
    st = _mk(real_state, DirectToVgprA=True, DirectToVgprB=True)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "DirectToVgprA + DirectToVgprB disabled" in capsys.readouterr().out

def test_dtvab_sets_side_effect_params_before_reject(real_state, isa_info_map):
    st = _mk(real_state, DirectToVgprA=True, DirectToVgprB=True)
    _run(st, isa_info_map)
    assert st["PrefetchGlobalRead"] == 1
    assert st["ExpandPointerSwap"] is False
    assert st["1LDSBuffer"] == 0
    assert st["PrefetchLocalRead"] == 0

def test_reject_plr0_tlu_false(real_state, isa_info_map, capsys):
    st = _mk(real_state, PrefetchLocalRead=0)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "does not supports TLUA = False and PrefetchLocalRead = 0" in capsys.readouterr().out

def test_reject_grvw_times_numbytes_lt_4(real_state, isa_info_map, capsys):
    st = _mk(real_state, GlobalReadVectorWidthA=1)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "does not support TLUA + numByte * GlobalReadVectorWidthA < 4" in capsys.readouterr().out

def test_reject_matrixinstbn_not_one(real_state, isa_info_map, capsys):
    st = _mk(real_state, MatrixInstBN=2)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "MatrixInstBN should be 1 for DirectToVgprA. Current value is 2" in capsys.readouterr().out

def test_reject_waveseparateglobalread(real_state, isa_info_map, capsys):
    st = _mk(real_state, WaveSeparateGlobalReadA=1)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "does not supports WaveSeparateGlobalReadA" in capsys.readouterr().out

def test_reject_numloadscoalesced_mismatch(real_state, isa_info_map, capsys):
    st = _mk(real_state, NumLoadsCoalescedA=99)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "does not supports TLU=False and NumLoadsCoalescedA !=" in capsys.readouterr().out

def test_reject_gltr_grvw_not_8(real_state, isa_info_map, capsys):
    st = _mk(real_state, enableGLTrA=True)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "does not supports enableGLTrA and GlobalReadVectorWidth != 8" in capsys.readouterr().out

def test_reject_grvw_ne_lrvw(real_state, isa_info_map, capsys):
    st = _mk(real_state, GlobalReadVectorWidthA=2, NumLoadsCoalescedA=4)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "GlobalReadVectorWidthA(2) != LocalReadVectorWidth(4)" in capsys.readouterr().out

def test_reject_scheduleiteralg_lt_3(real_state, isa_info_map, capsys):
    st = _mk(real_state, _ScheduleIterAlg=2)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "does not supports ScheduleIterAlg < 3" in capsys.readouterr().out

def test_reject_innerunroll_gt_1(real_state, isa_info_map, capsys):
    st = _mk(real_state, InnerUnroll=2)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "does not supports InnerUnroll>1" in capsys.readouterr().out

def test_reject_tlu_eq_unrollmajorlds(real_state, isa_info_map, capsys):
    st = _mk(real_state, UnrollMajorLDSA=False)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "does not supports TLUA = UnrollMajorLDSA" in capsys.readouterr().out

def test_reject_unrollloopswapglobalreadorder(real_state, isa_info_map, capsys):
    st = _mk(real_state, UnrollLoopSwapGlobalReadOrder=1)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "does not supports UnrollLoopSwapGlobalReadOrder" in capsys.readouterr().out

def test_reject_sparse(real_state, isa_info_map, capsys):
    st = _mk(real_state)
    st["ProblemType"]["Sparse"] = 1
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "does not supports Sparse" in capsys.readouterr().out

def test_reject_prefetchglobalread_zero(real_state, isa_info_map, capsys):
    st = _mk(real_state, PrefetchGlobalRead=0)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "does not supports PrefetchGlobalRead == 0." in capsys.readouterr().out

def test_reject_nn_transposelds_zero(real_state, isa_info_map, capsys):
    st = _mk(real_state, TransposeLDS=0)
    st["ProblemType"]["TransposeA"] = False
    st["ProblemType"]["TransposeB"] = False
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "does not supports NN case with TransposeLDS == 0." in capsys.readouterr().out
