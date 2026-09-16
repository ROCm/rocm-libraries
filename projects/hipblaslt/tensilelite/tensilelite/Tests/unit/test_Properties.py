################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for Tensile.Properties (Property and Predicate).

These are pure data/value classes used for library solution selection; the
tests assert concrete construction, serialization, equality/hashing, and the
predicate-ordering rules directly, with no mocking.
"""

import pytest

from Tensile.Properties import Predicate, Property

pytestmark = pytest.mark.unit


class TestProperty:
    def test_from_original_state_full(self):
        p = Property.FromOriginalState({"type": "Foo", "index": 3, "value": 5})
        assert (p.tag, p.index, p.value) == ("Foo", 3, 5)

    def test_from_original_state_defaults_to_none(self):
        p = Property.FromOriginalState({})
        assert (p.tag, p.index, p.value) == (None, None, None)

    def test_state_omits_none_index_and_value(self):
        assert Property("Foo").state() == {"type": "Foo"}

    def test_state_keeps_zero_index_and_value(self):
        # 0 is falsy but not None, so it must be retained.
        assert Property("Foo", index=0, value=0).state() == {
            "type": "Foo", "index": 0, "value": 0,
        }

    def test_state_recurses_into_value(self):
        assert Property("Foo", index=2, value=[1, 2]).state() == {
            "type": "Foo", "index": 2, "value": [1, 2],
        }

    def test_state_round_trips_through_from_original_state(self):
        original = Property("Foo", 3, 5)
        assert Property.FromOriginalState(original.state()) == original

    def test_equality_requires_all_fields(self):
        base = Property("A", 1, 2)
        assert base == Property("A", 1, 2)
        assert base != Property("B", 1, 2)
        assert base != Property("A", 9, 2)
        assert base != Property("A", 1, 9)

    def test_not_equal_across_classes(self):
        assert Property("A") != Predicate("A")

    def test_not_equal_to_non_property(self):
        assert Property("A") != "A"

    def test_equal_objects_hash_equal_and_dedupe(self):
        a = Property("A", 1, 2)
        b = Property("A", 1, 2)
        assert hash(a) == hash(b)
        assert len({a, b}) == 1


class TestPredicateAndOr:
    def test_and_of_empty_is_truepred(self):
        result = Predicate.And([])
        assert isinstance(result, Predicate)
        assert result.tag == "TruePred"

    def test_or_of_empty_is_truepred(self):
        assert Predicate.Or([]).tag == "TruePred"

    def test_and_of_single_returns_that_predicate(self):
        only = Predicate("EqualityMatching")
        assert Predicate.And([only]) is only

    def test_or_of_single_returns_that_predicate(self):
        only = Predicate("RangeMatching")
        assert Predicate.Or([only]) is only

    def test_and_of_many_wraps_in_and(self):
        p1 = Predicate("EqualityMatching")
        p2 = Predicate("RangeMatching")
        combined = Predicate.And([p1, p2])
        assert combined.tag == "And"
        assert combined.value == (p1, p2)

    def test_or_of_many_wraps_in_or_and_accepts_any_iterable(self):
        p1 = Predicate("EqualityMatching")
        p2 = Predicate("RangeMatching")
        combined = Predicate.Or(p for p in (p1, p2))  # generator, not a list
        assert combined.tag == "Or"
        assert combined.value == (p1, p2)


class TestPredicateOrdering:
    def test_matching_order_sorts_known_tags(self):
        unordered = [
            Predicate("TruePred"),
            Predicate("FreeSizeMatching"),
            Predicate("EqualityMatching"),
            Predicate("GridBasedMatching"),
            Predicate("PredictionMatching"),
            Predicate("RangeMatching"),
        ]
        assert [p.tag for p in sorted(unordered)] == [
            "EqualityMatching",
            "RangeMatching",
            "PredictionMatching",
            "GridBasedMatching",
            "FreeSizeMatching",
            "TruePred",
        ]

    def test_known_tag_versus_unknown_tag_falls_back_to_field_comparison(self):
        # Both tags must be in the matching order to use it; here only one is,
        # so it falls back to tuple comparison: "EqualityMatching" < "Foo".
        assert Predicate("EqualityMatching") < Predicate("Foo")

    def test_unknown_tags_compare_by_index(self):
        assert Predicate("Foo", index=1) < Predicate("Foo", index=2)

    def test_dict_values_are_reduced_to_value_sets(self):
        # dict keys should not affect predicate ordering; only the set of
        # values is compared.
        exact = Predicate("Foo", value={"chip": 0x75A0})
        multi = Predicate("Foo", value={"mi350": 0x75A0, "mi355": 0x75A3})
        assert exact < multi


class TestPredicateState:
    def test_nested_and_serializes_recursively(self):
        p1 = Predicate("EqualityMatching", index=0, value=5)
        p2 = Predicate("RangeMatching", index=1, value=10)
        combined = Predicate.And([p1, p2])
        assert combined.state() == {
            "type": "And",
            "value": [
                {"type": "EqualityMatching", "index": 0, "value": 5},
                {"type": "RangeMatching", "index": 1, "value": 10},
            ],
        }
