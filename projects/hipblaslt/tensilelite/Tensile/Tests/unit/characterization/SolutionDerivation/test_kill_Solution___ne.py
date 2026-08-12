import importlib
import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")
pytestmark = pytest.mark.unit

def _make(name, device_names):
    obj = S.Solution.__new__(S.Solution)
    obj._name = name
    obj._state = {"DeviceNames": device_names}
    return obj

def test_ne_false_for_equal_solutions_kills_none_and_none_arg_mutants():
    a = _make("kernelX", ["gfx942"])
    b = _make("kernelX", ["gfx942"])
    assert a.__eq__(b) is True
    assert (a != b) is False

def test_ne_true_for_unequal_by_name():
    a = _make("kernelX", ["gfx942"])
    b = _make("kernelY", ["gfx942"])
    assert (a != b) is True

def test_ne_true_for_unequal_by_device_names():
    a = _make("kernelX", ["gfx942"])
    b = _make("kernelX", ["gfx90a"])
    assert (a != b) is True

def test_ne_is_negation_of_eq_identity_case():
    a = _make("kernelX", ["gfx942"])
    assert (a != a) is False
    assert a.__ne__(a) == (not a.__eq__(a))
