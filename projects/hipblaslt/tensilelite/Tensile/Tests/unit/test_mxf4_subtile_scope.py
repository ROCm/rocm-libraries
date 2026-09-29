################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""The scope gate for the MXF4 subtile optimizations.

``isMxf4SubtilePath`` and the tile predicates layered on it decide, for every
kernel in the library, whether the fused post-loop store and the partitioned
block schedule are built. Getting the gate wrong in the permissive direction
silently reshapes the epilogue of kernels that were never measured with it, and
that failure mode is invisible in a test that only drives MXF4 configs.

These are pure-predicate tests: no emit, no GPU, no toolchain.
"""

import pytest

from Tensile.Common.DataType import DataType
from Tensile.Common.Utilities import (
    isMxf4SubtilePath,
    plsinBlockSchedTile,
    plsinEarlyStoreTile,
    plsinStagingEligible,
    plsinSubtileTypes,
)

pytestmark = pytest.mark.unit

FP4 = "float4"
BF16 = "bfloat16"
FP32 = "float"
FP16 = "half"


def kern(dtA=FP4, dtB=FP4, dtD=BF16, subtile=True, mt0=256, mt1=256):
    return {
        "UseSubtileImpl": subtile,
        "MacroTile0": mt0,
        "MacroTile1": mt1,
        "ProblemType": {
            "DataTypeA": DataType(dtA),
            "DataTypeB": DataType(dtB),
            "DestDataType": DataType(dtD),
        },
    }


# (description, kernel, in scope?)
_SCOPE_CASES = [
    ("mxf4 in, bf16 out, subtile", kern(), True),
    ("fp32 out", kern(dtD=FP32), False),
    ("fp16 out", kern(dtD=FP16), False),
    ("bf16 in", kern(dtA=BF16, dtB=BF16), False),
    ("only A is fp4", kern(dtB=BF16), False),
    ("only B is fp4", kern(dtA=BF16), False),
    ("UseSubtileImpl off", kern(subtile=False), False),
]


@pytest.mark.parametrize(
    "kernel,expected",
    [pytest.param(k, e, id=d) for d, k, e in _SCOPE_CASES],
)
def test_is_mxf4_subtile_path(kernel, expected):
    assert isMxf4SubtilePath(kernel) is expected


def test_use_subtile_impl_alone_does_not_open_the_gate():
    """The distinction the gate exists to draw.

    ``UseSubtileImpl`` is set on gfx1250 as well as gfx950, and gfx950 MX merely
    requires it rather than being the only thing that sets it, so gating on it
    would pull in every subtile kernel in the library.
    """
    bf16Subtile = kern(dtA=BF16, dtB=BF16, dtD=BF16)
    assert bf16Subtile["UseSubtileImpl"] is True
    assert not isMxf4SubtilePath(bf16Subtile)


@pytest.mark.parametrize(
    "kernel,expected",
    [pytest.param(k, e, id=d) for d, k, e in _SCOPE_CASES],
)
def test_plsin_subtile_types_ignores_use_subtile_impl(kernel, expected):
    """``plsinSubtileTypes`` is the type half of the gate on its own.

    It must answer for the operand types alone, so that the only case where it
    disagrees with the full gate is the one where UseSubtileImpl is off.
    """
    typesOnly = plsinSubtileTypes(kernel)
    if kernel["UseSubtileImpl"]:
        assert typesOnly is expected
    else:
        assert typesOnly is True


@pytest.mark.parametrize("mt0,mt1,early,blockSched", [
    (256, 256, True, True),      # the one geometry block scheduling is measured on
    (128, 128, True, False),     # PLSIN-eligible, deliberately not block-scheduled
    (192, 256, True, False),     # satisfies the <=256 bound; the equality excludes it
    (256, 192, True, False),
    (320, 256, False, False),    # >256: lends its K=0 operand registers to the store
    (256, 320, False, False),
])
def test_tile_scope(mt0, mt1, early, blockSched):
    """The tile predicates, held apart from the type predicate.

    ``plsinBlockSchedTile`` is an equality rather than a bound on purpose:
    MT192x256 also passes ``plsinEarlyStoreTile`` and would otherwise be dragged
    in untested. MT>256x256 is excluded even from early-store work because it
    lends its K=0 operand registers to the store.
    """
    kernel = kern(mt0=mt0, mt1=mt1)
    assert plsinEarlyStoreTile(kernel) is early
    assert plsinBlockSchedTile(kernel) is blockSched
    assert plsinStagingEligible(kernel) is blockSched


def test_tile_scope_requires_the_type_scope():
    """No tile is in scope once the operand types are out of it."""
    outOfScope = kern(dtD=FP32)
    assert plsinEarlyStoreTile(outOfScope) is False
    assert plsinBlockSchedTile(outOfScope) is False
    assert plsinStagingEligible(outOfScope) is False


@pytest.mark.parametrize("kernel", [
    pytest.param({}, id="empty"),
    pytest.param({"UseSubtileImpl": True}, id="no ProblemType"),
    pytest.param({"UseSubtileImpl": True, "ProblemType": {}}, id="empty ProblemType"),
    pytest.param(
        {"UseSubtileImpl": True,
         "ProblemType": {"DataTypeA": 21, "DataTypeB": 21, "DestDataType": 7}},
        id="raw type values",
    ),
])
def test_partially_built_state_is_not_mxf4(kernel):
    """Naming runs before the data types are DataType objects.

    ``isMxf4SubtilePath`` has to answer there too, and "not MXF4" is the
    conservative answer: it is what keeps ``PostLoopStoreInNll`` out of the name
    hash of kernels that can never set it.
    """
    assert isMxf4SubtilePath(kernel) is False
