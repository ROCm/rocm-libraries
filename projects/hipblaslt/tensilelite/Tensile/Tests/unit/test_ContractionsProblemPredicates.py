# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.Contractions import ProblemPredicate


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "key, predicate",
    [
        ("AssertStrideAEqual", "StrideAEqual"),
        ("AssertStrideBEqual", "StrideBEqual"),
        ("AssertStrideCEqual", "StrideCEqual"),
        ("AssertStrideDEqual", "StrideDEqual"),
        ("AssertSizeEqual", "SizeEqual"),
        ("AssertSizeGreaterThan", "SizeGreaterThan"),
        ("AssertSizeLessThan", "SizeLessThan"),
        ("AssertSizeMultiple", "SizeMultiple"),
    ],
)
def test_dimension_assertions_become_runtime_predicates(key, predicate):
    result = ProblemPredicate.FromOriginalKeyPair((key, {0: 4, 1: -1}))

    assert result.state() == {"type": predicate, "index": 0, "value": 4}


def test_multiple_dimension_assertions_are_combined():
    result = ProblemPredicate.FromOriginalKeyPair(("AssertSizeEqual", {0: 4, 2: 8}))

    assert result.tag == "And"
    assert [predicate.state() for predicate in result.value] == [
        {"type": "SizeEqual", "index": 0, "value": 4},
        {"type": "SizeEqual", "index": 2, "value": 8},
    ]


@pytest.mark.parametrize(
    "key, value, expected",
    [
        ("AssertAlphaValue", 1, {"type": "AlphaValue", "value": "1"}),
        ("AssertBetaValue", 0, {"type": "BetaValue", "value": "0"}),
        ("AssertCEqualsD", True, {"type": "CEqualsD"}),
    ],
)
def test_value_assertions_become_runtime_predicates(key, value, expected):
    assert ProblemPredicate.FromOriginalKeyPair((key, value)).state() == expected


@pytest.mark.parametrize("key", ["AssertAlphaValue", "AssertBetaValue", "AssertCEqualsD"])
def test_disabled_value_assertions_do_not_add_predicates(key):
    assert ProblemPredicate.FromOriginalKeyPair((key, False)) is None


def test_unknown_assertions_remain_strict():
    with pytest.raises(RuntimeError, match="Unknown assertion key"):
        ProblemPredicate.FromOriginalKeyPair(("AssertFutureConstraint", 4))
