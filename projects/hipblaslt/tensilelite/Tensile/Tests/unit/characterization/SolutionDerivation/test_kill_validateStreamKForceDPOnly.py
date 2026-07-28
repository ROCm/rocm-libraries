import importlib

import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")

pytestmark = pytest.mark.unit

MSG_STREAMK = "StreamKForceDPOnly requires DP-first two-tile Stream-K"
MSG_ATOMIC = "StreamKForceDPOnly does not support atomic Stream-K"

def _base_state(streamk, atomic, force=True):
    return {
        "StreamKForceDPOnly": force,
        "StreamK": streamk,
        "StreamKAtomic": atomic,
    }

def test_wrong_streamk_rejects_and_prints_exact_message(capsys):
    state = _base_state(streamk=0, atomic=0)
    result = S._validateStreamKForceDPOnly(state, True)
    out = capsys.readouterr().out
    assert result is False
    assert state["Valid"] is False
    assert MSG_STREAMK in out

def test_wrong_streamk_message_is_case_exact(capsys):
    state = _base_state(streamk=0, atomic=0)
    S._validateStreamKForceDPOnly(state, True)
    out = capsys.readouterr().out
    assert MSG_STREAMK in out
    assert MSG_STREAMK.lower() not in out.replace(MSG_STREAMK, "")
    assert MSG_STREAMK.upper() not in out
    assert "XX" not in out
    assert "None" not in out

def test_streamk_equal_three_and_no_atomic_is_valid(capsys):
    state = _base_state(streamk=3, atomic=0)
    result = S._validateStreamKForceDPOnly(state, True)
    out = capsys.readouterr().out
    assert result is True
    assert "Valid" not in state
    assert out == ""

def test_atomic_rejects_and_prints_exact_message(capsys):
    state = _base_state(streamk=3, atomic=1)
    result = S._validateStreamKForceDPOnly(state, True)
    out = capsys.readouterr().out
    assert result is False
    assert state["Valid"] is False
    assert MSG_ATOMIC in out

def test_atomic_message_is_case_exact(capsys):
    state = _base_state(streamk=3, atomic=1)
    S._validateStreamKForceDPOnly(state, True)
    out = capsys.readouterr().out
    assert MSG_ATOMIC in out
    assert MSG_ATOMIC.lower() not in out.replace(MSG_ATOMIC, "")
    assert MSG_ATOMIC.upper() not in out
    assert "XX" not in out
    assert "None" not in out

def test_force_disabled_is_valid_without_reading_streamk():
    state = {"StreamKForceDPOnly": False}
    result = S._validateStreamKForceDPOnly(state, True)
    assert result is True
    assert "Valid" not in state

def test_streamk_four_rejects():
    state = _base_state(streamk=4, atomic=0)
    result = S._validateStreamKForceDPOnly(state, True)
    assert result is False
    assert state["Valid"] is False

def test_atomic_zero_passes_through_to_final_true(capsys):
    state = _base_state(streamk=3, atomic=0)
    result = S._validateStreamKForceDPOnly(state, True)
    out = capsys.readouterr().out
    assert result is True
    assert "Valid" not in state
    assert out == ""
