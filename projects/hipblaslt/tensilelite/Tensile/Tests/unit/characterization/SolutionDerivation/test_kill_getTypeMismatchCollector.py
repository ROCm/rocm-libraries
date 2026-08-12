import importlib

import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")

pytestmark = pytest.mark.unit

@pytest.fixture(autouse=True)
def _clean_collector():
    S.resetTypeMismatchCollector()
    yield
    S.resetTypeMismatchCollector()

def test_get_returns_exact_inner_key_set_and_values():
    S._typeMismatchCollector[("P", "int", "float")] = {
        "count": 7,
        "values": {"1", "2"},
        "files": {"a.yaml"},
    }

    out = S.getTypeMismatchCollector()

    assert set(out) == {("P", "int", "float")}
    inner = out[("P", "int", "float")]
    assert set(inner) == {"count", "values", "files"}
    assert inner["count"] == 7
    assert inner["values"] == {"1", "2"}
    assert inner["files"] == {"a.yaml"}

def test_get_count_key_is_lowercase_literal():
    S._typeMismatchCollector[("Q", "str", "int")] = {
        "count": 3,
        "values": {"x"},
        "files": {"b.yaml"},
    }

    inner = S.getTypeMismatchCollector()[("Q", "str", "int")]

    assert "count" in inner
    assert "COUNT" not in inner
    assert "XXcountXX" not in inner
    assert inner["count"] == 3

def test_get_returns_deep_copies():
    orig_values = {"v1"}
    orig_files = {"f1"}
    S._typeMismatchCollector[("R", "a", "b")] = {
        "count": 1,
        "values": orig_values,
        "files": orig_files,
    }

    out = S.getTypeMismatchCollector()[("R", "a", "b")]
    out["values"].add("mutated")
    out["files"].add("mutated")

    assert orig_values == {"v1"}
    assert orig_files == {"f1"}
