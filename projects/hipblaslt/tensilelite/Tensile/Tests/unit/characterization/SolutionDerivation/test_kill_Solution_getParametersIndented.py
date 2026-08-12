import importlib

import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")

pytestmark = pytest.mark.unit

def test_get_parameters_indented_exact_full_output():
    state = {"ProblemType": "PT_SENTINEL", "Alpha": 1, "Beta": 2}
    out = S.Solution.getParametersIndented(state, ">>")
    expected = (
        ">>ProblemType: PT_SENTINEL\n"
        ">>Alpha: 1\n"
        ">>Beta: 2\n"
        ">>ProblemType: PT_SENTINEL\n"
    )
    assert out == expected

def test_get_parameters_indented_returns_str_not_none():
    state = {"ProblemType": "PT_SENTINEL", "Alpha": 1}
    out = S.Solution.getParametersIndented(state, "  ")
    assert isinstance(out, str)
    assert out != ""

def test_get_parameters_indented_header_uses_problemtype_value():
    state = {"ProblemType": "REALVALUE"}
    out = S.Solution.getParametersIndented(state, "|")
    assert out == "|ProblemType: REALVALUE\n|ProblemType: REALVALUE\n"
    assert "None" not in out

def test_get_parameters_indented_iterates_all_keys_sorted():
    state = {"ProblemType": "PT", "Z": 26, "A": 1, "M": 13}
    out = S.Solution.getParametersIndented(state, "")
    expected = (
        "ProblemType: PT\n"
        "A: 1\n"
        "M: 13\n"
        "ProblemType: PT\n"
        "Z: 26\n"
    )
    assert out == expected

def test_get_parameters_indented_values_and_keys_distinct():
    state = {"ProblemType": "PT", "Foo": "FOOVAL", "Bar": "BARVAL"}
    out = S.Solution.getParametersIndented(state, "->")
    assert "->Foo: FOOVAL\n" in out
    assert "->Bar: BARVAL\n" in out
    assert "None" not in out
