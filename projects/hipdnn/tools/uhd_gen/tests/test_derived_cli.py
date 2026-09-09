"""--derived: parsing, and that it reaches both the descriptor and the fingerprint."""
from __future__ import annotations
import pytest
from uhd_gen.__main__ import _parse_derived

GOOD = 'intensity={"/":["$q.flops","$q.bytes"]}'


def test_pairs_are_parsed_in_declaration_order():
    got = _parse_derived([GOOD, 'twice={"*":["$derived.intensity",2]}'])
    assert [n for n, _ in got] == ["intensity", "twice"]


def test_a_malformed_expression_fails_here_rather_than_at_load():
    """At load it would degrade to declared order, which is a legal state and so silent."""
    with pytest.raises(ValueError, match="not valid JsonLogic"):
        _parse_derived(["intensity=$q.flops / $q.bytes"])


def test_a_missing_expression_is_rejected():
    with pytest.raises(ValueError, match="NAME=EXPRESSION"):
        _parse_derived(["intensity"])


def test_duplicate_names_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        _parse_derived([GOOD, GOOD])


def test_none_yields_no_derived():
    assert _parse_derived(None) == []
