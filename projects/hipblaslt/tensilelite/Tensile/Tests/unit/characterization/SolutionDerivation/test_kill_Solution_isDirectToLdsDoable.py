import copy
import importlib

import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")
pytestmark = pytest.mark.unit

_PASS_A = {"GlobalReadVectorWidthA": 2, "UseGeneralizedNLCOneA": True}

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
    return S.Solution.isDirectToLdsDoable(st, tc, isa_info_map, True)

def test_passthrough_returns_true(real_state, isa_info_map, capsys):
    st = _mk(real_state)
    assert _run(st, isa_info_map) is True
    assert st["Valid"] is True
    assert capsys.readouterr().out == ""

def test_subtile_impl_early_return_true(real_state, isa_info_map, capsys):
    st = _mk(real_state, UseSubtileImpl=True)
    assert _run(st, isa_info_map) is True
    assert capsys.readouterr().out == ""

def test_b64_x2_returns_false_with_warning(real_state, isa_info_map, capsys):
    st = _mk(real_state, GlobalReadVectorWidthA=4)
    assert _run(st, isa_info_map) is False
    assert "can't use DirectToLds with b64 buffer load" in capsys.readouterr().out

def test_reject_b128_not_supported(real_state, isa_info_map, capsys):
    st = _mk(real_state, GlobalReadVectorWidthA=8)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "b128 DirectToLds not supported" in capsys.readouterr().out

def test_reject_load_less_than_32bits(real_state, isa_info_map, capsys):
    st = _mk(real_state, GlobalReadVectorWidthA=1)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "DirectToLds not supported for loads less than 32bits" in capsys.readouterr().out

def test_reject_matrix_instruction_only(real_state, isa_info_map, capsys):
    st = _mk(real_state, EnableMatrixInstruction=False)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "DirectToLds is for MatrixInstruction only for now (tentative)" in capsys.readouterr().out

def test_reject_lrvw_gt_miinputperthread(real_state, isa_info_map, capsys):
    st = _mk(real_state, LocalReadVectorWidthA=8)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "DirectToLds does not work with LocalReadVectorWidth > MIInputPerThread" in capsys.readouterr().out

def test_reject_numthreads_not_multiple_of_wavefront(real_state, isa_info_map, capsys):
    st = _mk(real_state, NumThreads=250)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "can't use DirectToLds for NumThreads % WavefrontSize != 0" in capsys.readouterr().out

def test_tlu_eq_unrollmajorlds_returns_false_with_warning(real_state, isa_info_map, capsys):
    st = _mk(real_state, UnrollMajorLDSA=False)
    assert _run(st, isa_info_map) is False
    assert "can't use DirectToLds for TLUA == UnrollMajorLDSA" in capsys.readouterr().out

def test_reject_wsgr_lsc_lsp_mismatch(real_state, isa_info_map, capsys):
    st = _mk(real_state, WaveSeparateGlobalReadA=1)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    out = capsys.readouterr().out
    assert "can't use DirectToLds for LSCA and LSPA" in out
    assert "WavefrontSize * GlobalReadVectorWidthA" in out

def test_reject_wsgr_equals_two(real_state, isa_info_map, capsys):
    st = _mk(real_state, WaveSeparateGlobalReadA=2, LSCA=16, LSPA=8)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "can't use DirectToLds for WSGRA = 2" in capsys.readouterr().out

def test_reject_nlcone_elif_lsc_lsp_numthreads_mismatch(real_state, isa_info_map, capsys):
    st = _mk(real_state, UseGeneralizedNLCOneA=False)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    out = capsys.readouterr().out
    assert "can't use DirectToLds for LSCA and LSPA" in out
    assert "NumThreads * GlobalReadVectorWidthA" in out

def test_reject_localreadvectorwidth_equals_two(real_state, isa_info_map, capsys):
    st = _mk(real_state, LocalReadVectorWidthA=2)
    assert _run(st, isa_info_map) is False
    assert st["Valid"] is False
    assert "can't use DirectToLds for LocalReadVectorWidth == 2" in capsys.readouterr().out

def test_lds_b_passthrough_uses_macrotile1(real_state, isa_info_map, capsys):
    st = copy.deepcopy(real_state)
    st.pop("SolutionIndex", None)
    st.pop("SolutionNameMin", None)
    st["Valid"] = True
    st["GlobalReadVectorWidthB"] = 2
    st["UseGeneralizedNLCOneB"] = True
    assert _run(st, isa_info_map, tc="B") is True
    assert st["Valid"] is True
