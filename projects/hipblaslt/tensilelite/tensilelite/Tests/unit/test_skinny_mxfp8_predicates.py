# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pins the predicate evaluator the skinny MXFP8 tests rely on to the C++ semantics."""

import pytest

from skinny_mxfp8_predicates import evaluate, make_problem
from tensilelite.Contractions import ProblemPredicate

pytestmark = pytest.mark.unit


def predicate(tag, index=None, value=None):
    return ProblemPredicate(tag, index=index, value=value)


def problem(m=64, n=2, k=256, lda=256, ldb=256):
    return make_problem(m, n, 1, k, m, m, lda, ldb)


def test_unknown_tag_raises():
    with pytest.raises(AssertionError, match="no evaluator"):
        evaluate(predicate("NoSuchPredicate", value=1), problem())


def test_size_less_than_is_strict_at_the_boundary():
    cap = 1048577
    assert evaluate(predicate("SizeLessThan", 0, cap), problem(m=cap - 1))
    assert not evaluate(predicate("SizeLessThan", 0, cap), problem(m=cap))
    assert evaluate(predicate("SizeLessThan", 1, cap), problem(n=cap - 1))
    assert not evaluate(predicate("SizeLessThan", 1, cap), problem(n=cap))


@pytest.mark.parametrize("tag, lda, ldb", [("StrideAMultiple", 0, 256), ("StrideBMultiple", 256, 0)])
def test_stride_multiple_rejects_a_zero_stride(tag, lda, ldb):
    assert not evaluate(predicate(tag, 1, 4), problem(lda=lda, ldb=ldb))


@pytest.mark.parametrize("tag", ["StrideAMultiple", "StrideBMultiple"])
def test_stride_multiple_rejects_a_zero_multiple_and_an_out_of_range_index(tag):
    assert evaluate(predicate(tag, 1, 4), problem())
    assert not evaluate(predicate(tag, 1, 0), problem())
    assert not evaluate(predicate(tag, 7, 4), problem())
