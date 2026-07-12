################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.Properties``: the ``Property`` base
(state / equality / hash / FromOriginalState) and the ``Predicate`` subclass
(``And`` / ``Or`` factories and the matching-order ``__lt__``)."""

import pytest

from Tensile.Properties import Property, Predicate

pytestmark = pytest.mark.unit


def test_property_state_minimal(snapshot):
    assert Property(tag="Foo").state() == snapshot


def test_property_state_full(snapshot):
    assert Property(tag="EqualSize", index=2, value=128).state() == snapshot


def test_property_from_original_state(snapshot):
    p = Property.FromOriginalState({"type": "EqualSize", "index": 1, "value": 64})
    assert {"tag": p.tag, "index": p.index, "value": p.value, "state": p.state()} == snapshot


def test_property_equality_and_hash():
    a = Property("T", 1, 5)
    b = Property("T", 1, 5)
    c = Property("T", 2, 5)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    # Different class -> not equal.
    assert a != Predicate("T", 1, 5)


def test_predicate_and(snapshot):
    p = Predicate("EqualSize", 0, 1)
    q = Predicate("RangeMatching", 1, 2)
    assert {
        "empty": Predicate.And([]).state(),
        "single": Predicate.And([p]).state(),
        "multi_tag": Predicate.And([p, q]).tag,
    } == snapshot


def test_predicate_or(snapshot):
    p = Predicate("EqualSize", 0, 1)
    q = Predicate("RangeMatching", 1, 2)
    assert {
        "empty": Predicate.Or([]).state(),
        "single": Predicate.Or([p]).state(),
        "multi_tag": Predicate.Or([p, q]).tag,
    } == snapshot


def test_predicate_lt_matching_order():
    eq = Predicate("EqualityMatching")
    rng = Predicate("RangeMatching")
    tp = Predicate("TruePred")
    assert eq < rng
    assert rng < tp
    assert not (tp < eq)


def test_predicate_lt_default_and_dict_value():
    # Non-matching-order tags fall to the (tag, index, value) tuple compare;
    # dict values are reduced to a set first.
    a = Predicate("And", index=0, value={0: 1, 1: 2})
    b = Predicate("And", index=1, value={0: 1})
    # Same tag, index 0 < 1 -> a < b.
    assert a < b
