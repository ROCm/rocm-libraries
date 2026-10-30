# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-solution MXScaleFormatEqual predicate emitted for an MX block."""

import pytest

import tensilelite.Contractions as C

pytestmark = pytest.mark.unit


class _SkipSizePredicates:
    """isInt8x4 skips the free-size predicates, so the MX predicate stands alone."""

    def isInt8x4(self):
        return True


def _problem_type(mx_block_a, mx_block_b, mx_scale_format):
    problem_type = C.ProblemType()
    problem_type.aType = _SkipSizePredicates()
    problem_type.mxBlockA = mx_block_a
    problem_type.mxBlockB = mx_block_b
    problem_type.mxScaleFormat = mx_scale_format
    return problem_type


def _state():
    # Keys CompoundPredicates reads on the way to the MX predicate.
    return {
        "InternalSupportParams": {"KernArgsVersion": 0},
        "PackedC0IndicesX": [],
        "ProblemType": {
            "SwizzleTensorA": False,
            "SwizzleTensorB": False,
            "FusedGemmA2A": False,
        },
    }


def _emitted(mx_block_a, mx_block_b, mx_scale_format):
    return C.ProblemPredicate.CompoundPredicates(
        _state(), _problem_type(mx_block_a, mx_block_b, mx_scale_format)
    )


@pytest.mark.parametrize("mx_scale_format", [1, 0, 2])
def test_mx_block_emits_one_format_equal(mx_scale_format):
    preds = _emitted(32, 32, mx_scale_format)
    assert [(p.tag, p.value) for p in preds] == [
        ("MXScaleFormatEqual", mx_scale_format)
    ]


def test_no_mx_block_emits_no_format_equal():
    preds = _emitted(0, 0, 1)
    assert [p for p in preds if p.tag == "MXScaleFormatEqual"] == []
