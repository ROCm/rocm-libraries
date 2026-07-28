import importlib

import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")

pytestmark = pytest.mark.unit

@pytest.fixture
def controlled_types(monkeypatch):
    """Install a deterministic expected-type map and skip set.

    The real ``validParameters``-derived ``_expectedParamTypes`` happens to give
    every parameter a single-element type set, so the ``" or ".join`` separator
    is never exercised by production data. We inject a two-type expectation to
    force that branch, plus a single-type entry and a skip entry to drive the
    loop's continue path. ``validateParameterTypes`` reads these as module
    globals, so patching them on the Solution module changes its behavior.
    """
    expected = {
        "IntParam": {int},
        "MultiParam": {bool, int},
    }
    skip = {"SkipParam"}
    monkeypatch.setattr(S, "_expectedParamTypes", expected)
    monkeypatch.setattr(S, "_skipTypeCheck", skip)
    return expected, skip

def test_srcfile_default_is_empty_string(controlled_types):
    """Kills mutant_1: default ``srcFile=""`` mutated to ``srcFile="XXXX"``.

    Call without passing srcFile and assert the third element of the emitted
    record is exactly the empty string, not the mutated sentinel.
    """
    state = {"IntParam": "not-an-int"}

    records = S.validateParameterTypes(state)

    assert len(records) == 1
    collectorKey, valueRepr, srcFile = records[0]
    assert srcFile == ""
    assert collectorKey == ("IntParam", "str", "int")
    assert valueRepr == repr("not-an-int")

def test_srcfile_passthrough_when_provided(controlled_types):
    """Extra guard on the record's srcFile slot (also independent of mutant_1)."""
    state = {"IntParam": "bad"}

    records = S.validateParameterTypes(state, srcFile="a.yaml")

    assert records == [(("IntParam", "str", "int"), repr("bad"), "a.yaml")]

def test_skip_key_uses_continue_not_break(controlled_types):
    """Kills mutant_6: ``continue`` mutated to ``break``.

    A skip-listed key appears BEFORE a genuine mismatch. With ``continue`` the
    loop proceeds and records the later mismatch; with ``break`` it exits early
    and the mismatch is lost.
    """
    state = {}
    state["SkipParam"] = "whatever"
    state["IntParam"] = "wrong"

    records = S.validateParameterTypes(state)

    assert len(records) == 1
    assert records[0][0] == ("IntParam", "str", "int")

def test_unknown_key_uses_continue_not_break(controlled_types):
    """Also kills mutant_6 via the ``key not in _expectedParamTypes`` branch.

    An unknown key precedes the mismatch; ``break`` would drop the mismatch.
    """
    state = {}
    state["UnknownKey"] = 123
    state["IntParam"] = "wrong"

    records = S.validateParameterTypes(state)

    assert len(records) == 1
    assert records[0][0] == ("IntParam", "str", "int")

def test_expected_type_separator_is_space_or_space(controlled_types):
    """Kills mutant_13 (\"XX or XX\") and mutant_14 (\" OR \").

    Force a multi-type expectation and feed a value whose type is in neither,
    so the sorted type names are joined with the separator. Assert the exact
    joined string in the collector key.
    """
    state = {"MultiParam": "string-value"}

    records = S.validateParameterTypes(state)

    assert len(records) == 1
    collectorKey, valueRepr, srcFile = records[0]
    param, actual, expectedStr = collectorKey
    assert param == "MultiParam"
    assert actual == "str"
    assert expectedStr == "bool or int"
    assert valueRepr == repr("string-value")

def test_clean_state_returns_empty(controlled_types):
    """Well-typed values produce no records (bounds the emit condition)."""
    state = {
        "IntParam": 5,
        "MultiParam": True,
        "SkipParam": object(),
        "UnknownKey": 1.5,
    }

    assert S.validateParameterTypes(state) == []
