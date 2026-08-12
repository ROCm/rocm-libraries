import importlib
import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")
from Tensile.Common.TypeValidationErrors import ConfigTypeError

pytestmark = pytest.mark.unit

def _seed_collector():
    """Build a collector whose sort order under the ORIGINAL key
    (lambda kv: kv[0][0], i.e. paramName) differs from:
      - sorting by the whole item / full key tuple (key=None or no key)
      - sorting by kv[0][1] (actualType)

    Entries (insertion order matters for stable sorting):
      1: (BBB, int,  e1)
      2: (AAA, str,  e2)
      3: (BBB, bool, e3)

    Order of the emitted parameter lines:
      original  (by paramName, stable):   AAA/str, BBB/int, BBB/bool
      full-tuple(by whole key):           AAA/str, BBB/bool, BBB/int
      by actualType (kv[0][1]):           BBB/bool, BBB/int, AAA/str
    All three differ, so any of the three mutants changes the message.
    """
    S._typeMismatchCollector.clear()
    S._typeMismatchCollector[("BBB", "int", "e1")] = {
        "count": 1, "values": {"1"}, "files": {"fA"}}
    S._typeMismatchCollector[("AAA", "str", "e2")] = {
        "count": 1, "values": {"2"}, "files": {"fB"}}
    S._typeMismatchCollector[("BBB", "bool", "e3")] = {
        "count": 1, "values": {"3"}, "files": {"fC"}}

def _param_lines(message):
    """Extract the ordered per-parameter report lines from the message."""
    out = []
    for line in message.splitlines():
        stripped = line.strip()
        if stripped.startswith(("AAA:", "BBB:")):
            out.append(stripped)
    return out

@pytest.fixture(autouse=True)
def _restore_collector():
    saved = dict(S._typeMismatchCollector)
    try:
        yield
    finally:
        S._typeMismatchCollector.clear()
        S._typeMismatchCollector.update(saved)

def test_empty_collector_does_not_raise():
    S._typeMismatchCollector.clear()

    assert S.raiseIfTypeMismatches() is None

def test_lines_sorted_by_param_name_exact_order():
    """Pins the exact ordered output of the report.

    Kills all three survivors, which each reorder these lines:
      - key=None / removed key  -> sorts by full tuple -> BBB/bool before BBB/int
      - key=lambda kv: kv[0][1] -> sorts by actualType -> BBB lines lead, AAA last
    The original sorts by paramName (stable), giving the exact order below.
    """
    _seed_collector()
    with pytest.raises(ConfigTypeError) as excinfo:
        S.raiseIfTypeMismatches()
    lines = _param_lines(str(excinfo.value))
    assert lines == [
        "AAA: found str in 1 solutions (values: 2) - expected e2",
        "BBB: found int in 1 solutions (values: 1) - expected e1",
        "BBB: found bool in 1 solutions (values: 3) - expected e3",
    ]

def test_first_reported_param_is_lowest_name():
    """Focused assertion that the FIRST line is AAA.

    Under mutant 23 (sort by actualType) the first line would be a BBB
    entry (bool < int < str), so this pins the paramName-primary ordering.
    """
    _seed_collector()
    with pytest.raises(ConfigTypeError) as excinfo:
        S.raiseIfTypeMismatches()
    lines = _param_lines(str(excinfo.value))
    assert lines[0].startswith("AAA:")

def test_bbb_int_precedes_bbb_bool_under_stable_name_sort():
    """Within equal paramName BBB, insertion order (int then bool) is
    preserved by the stable paramName-only sort.

    Under mutants 18/20 (full-tuple sort) the tie is broken by actualType,
    putting bool before int and flipping this order.
    """
    _seed_collector()
    with pytest.raises(ConfigTypeError) as excinfo:
        S.raiseIfTypeMismatches()
    lines = _param_lines(str(excinfo.value))
    bbb = [ln for ln in lines if ln.startswith("BBB:")]
    assert bbb == [
        "BBB: found int in 1 solutions (values: 1) - expected e1",
        "BBB: found bool in 1 solutions (values: 3) - expected e3",
    ]
