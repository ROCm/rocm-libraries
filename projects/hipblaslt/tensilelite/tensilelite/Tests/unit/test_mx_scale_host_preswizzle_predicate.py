# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The MXScaleHostPreSwizzle problem predicate must be emitted per solution.

A kernel built for MXScaleFormat ``HostPreSwizzle`` reads the MX scale tensor
through the gfx950 e8m0_shuffle permutation; one built for ``NoSwizzle`` or
``InMemorySwizzle`` reads a compact row-major E8M0 array. The two buffers have
identical extents and element type, so no other predicate separates them: a
mismatch is silently wrong answers rather than a failed launch.

This is the regression guard for exactly that. Registering compact-scale custom
kernels into problem shards that also serve pre-swizzled callers made every one
of those kernels return wrong answers, because no predicate carried the
distinction.
"""

import pytest

from tensilelite.Contractions import (
    MX_SCALE_FORMAT_CODES,
    ProblemPredicate,
    mxScaleFormatCode,
    mxScaleHostPreSwizzle,
)


PREDICATE_TAG = "MXScaleHostPreSwizzle"


class _DataTypeStub:
    def isInt8x4(self):
        return False


class _ProblemTypeStub:
    """Just the fields CompoundPredicates reads on the way to this predicate."""

    def __init__(self, mxBlockA=32, mxBlockB=32):
        self.mxBlockA = mxBlockA
        self.mxBlockB = mxBlockB
        self.aType = _DataTypeStub()


def _state(mxScaleFormat=None, **overrides):
    """The smallest solution dict CompoundPredicates will walk."""
    state = {
        "ProblemType": {
            "SwizzleTensorA": False,
            "SwizzleTensorB": False,
            "FusedGemmA2A": False,
            "TLUA": False,
            "TLUB": False,
        },
        "InternalSupportParams": {"KernArgsVersion": 0},
        "MacroTile0": 128,
        "MacroTile1": 128,
        "GlobalReadVectorWidthA": 1,
        "GlobalReadVectorWidthB": 1,
        "PackedC0IndicesX": [],
        "WorkGroupMappingXCC": 1,
        "WorkGroupMappingXCCGroup": -1,
    }
    if mxScaleFormat is not None:
        state["MXScaleFormat"] = mxScaleFormat
    state.update(overrides)
    return state


def _emitted(state, problemType):
    """The MXScaleHostPreSwizzle predicates CompoundPredicates produces."""
    preds = ProblemPredicate.CompoundPredicates(state, problemType)
    return [p for p in preds if p.tag == PREDICATE_TAG]


# -- the format -> host-layout mapping -------------------------------------


def test_only_host_pre_swizzle_needs_a_shuffled_buffer():
    # InMemorySwizzle is False on purpose: that kernel permutes the scales
    # itself and still reads the compact buffer. DataInitialization applies the
    # host swizzle only for mxScaleFormat == 1.
    assert mxScaleHostPreSwizzle({"MXScaleFormat": "HostPreSwizzle"}) is True
    assert mxScaleHostPreSwizzle({"MXScaleFormat": "NoSwizzle"}) is False
    assert mxScaleHostPreSwizzle({"MXScaleFormat": "InMemorySwizzle"}) is False


@pytest.mark.parametrize("absent", [{}, {"MXScaleFormat": "Auto"}])
def test_unresolved_format_is_compact(absent):
    # The 75 gfx950 custom-kernel logic files carry MXScaleFormat only inside
    # DefaultSolution, and as "Auto". Both must land on the compact layout,
    # which is the layout those kernels actually read.
    assert mxScaleFormatCode(absent) == MX_SCALE_FORMAT_CODES["NoSwizzle"]
    assert mxScaleHostPreSwizzle(absent) is False


# -- emission ---------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt,expected",
    [
        ("HostPreSwizzle", True),
        ("NoSwizzle", False),
        ("InMemorySwizzle", False),
        (None, False),
    ],
)
def test_mx_solution_emits_the_predicate(fmt, expected):
    preds = _emitted(_state(fmt), _ProblemTypeStub())
    assert len(preds) == 1, "an MX solution must carry exactly one layout predicate"
    assert preds[0].value is expected


def test_a_compact_kernel_rejects_a_pre_swizzled_caller():
    """The defect, stated as the predicate values that must not agree."""
    compact_kernel = _emitted(_state("NoSwizzle"), _ProblemTypeStub())[0]
    pre_swizzled_kernel = _emitted(_state("HostPreSwizzle"), _ProblemTypeStub())[0]

    # The C++ predicate is `problem.mxScaleHostPreSwizzle() == value`, so a
    # pre-swizzled problem (True) matches only the pre-swizzled kernel.
    assert compact_kernel.value is not True
    assert pre_swizzled_kernel.value is True
    assert compact_kernel.value != pre_swizzled_kernel.value


def test_non_mx_solutions_are_untouched():
    # Emitting this for every GEMM would add a predicate to tens of thousands of
    # shipped solutions for no benefit.
    assert _emitted(_state("NoSwizzle"), _ProblemTypeStub(mxBlockA=0, mxBlockB=0)) == []


@pytest.mark.parametrize("mxBlockA,mxBlockB", [(32, 0), (0, 32)])
def test_single_operand_mx_still_emits(mxBlockA, mxBlockB):
    preds = _emitted(_state("HostPreSwizzle"), _ProblemTypeStub(mxBlockA, mxBlockB))
    assert len(preds) == 1
    assert preds[0].value is True
