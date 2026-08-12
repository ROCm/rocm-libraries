import importlib

import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")

pytestmark = pytest.mark.unit

def _make_solution(name, state):
    """Build a lightweight Solution whose str() is pinned via the _name cache,
    bypassing full derivation so __eq__ can be exercised in isolation."""
    obj = S.Solution.__new__(S.Solution)
    obj._state = state
    obj._name = name
    return obj

def test_eq_non_solution_returns_false():
    sol = _make_solution("K", {"DeviceNames": ["gfx942"]})
    assert (sol == 42) is False
    assert (sol == "not a solution") is False
    assert (sol == None) is False

def test_eq_different_str_returns_false():
    a = _make_solution("NameA", {"DeviceNames": ["gfx942"]})
    b = _make_solution("NameB", {"DeviceNames": ["gfx942"]})
    assert (a == b) is False

def test_eq_same_str_same_device_names_true():
    a = _make_solution("SameName", {"DeviceNames": ["gfx942"]})
    b = _make_solution("SameName", {"DeviceNames": ["gfx942"]})
    assert (a == b) is True

def test_eq_same_str_different_device_names_false():
    a = _make_solution("SameName", {"DeviceNames": ["gfx942"]})
    b = _make_solution("SameName", {"DeviceNames": ["gfx90a"]})
    assert (a == b) is False

def test_eq_both_missing_device_names_true():
    a = _make_solution("SameName", {})
    b = _make_solution("SameName", {})
    assert (a == b) is True

def test_eq_one_missing_device_names_false():
    a = _make_solution("SameName", {"DeviceNames": ["gfx942"]})
    b = _make_solution("SameName", {})
    assert (a == b) is False
    assert (b == a) is False

def test_eq_missing_vs_explicit_none_are_equal():
    a = _make_solution("SameName", {})
    b = _make_solution("SameName", {"DeviceNames": None})
    assert (a == b) is True

def test_eq_on_real_state_reflexive_and_symmetric(real_state):
    import copy

    a = _make_solution("RealKernel", real_state)
    b = _make_solution("RealKernel", copy.deepcopy(real_state))
    assert (a == a) is True
    assert (a == b) is True
    assert (b == a) is True
