import importlib

import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")

pytestmark = pytest.mark.unit

def test_extractable_index_A_branch_uses_dropped_last_element_slice():

    state = {"PackedC0IndicesX": [10, 11, 12], "PackedC1IndicesX": []}
    assert S.isExtractableIndex(state, 11, "A") is True

def test_extractable_index_A_branch_last_element_excluded():

    state = {"PackedC0IndicesX": [10, 11, 12], "PackedC1IndicesX": []}
    assert S.isExtractableIndex(state, 12, "A") is False

def test_extractable_index_B_branch_uses_dropped_last_element_slice():

    state = {"PackedC0IndicesX": [], "PackedC1IndicesX": [20, 21, 22]}
    assert S.isExtractableIndex(state, 21, "B") is True

def test_extractable_index_B_branch_last_element_excluded():
    state = {"PackedC0IndicesX": [], "PackedC1IndicesX": [20, 21, 22]}
    assert S.isExtractableIndex(state, 22, "B") is False

def test_extractable_index_default_and_else_branch_or_semantics():

    state = {"PackedC0IndicesX": [10, 11, 12], "PackedC1IndicesX": [20, 21, 22]}

    assert S.isExtractableIndex(state, 11) is True

    assert S.isExtractableIndex(state, 21) is True

    assert S.isExtractableIndex(state, 12) is False
