# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Divisions that truncate to zero when an operand is narrower than one unit.

Two places compute "how many of X fit in one Y" with integer division and then
divide by the result. Both are fine while X is at least as wide as Y, and both
raise ``ZeroDivisionError`` the moment it is narrower -- the ratio floors to 0
and becomes a divisor.

  ``swizzleGeometry``      bytes per element, 0.5 for F4, ``int()``-ed to 0.
  ``numMXSA`` / ``numMXSB`` MX scale local reads narrower than one MX unit.

Neither produces a wrong number first: the value is correct wherever the old
form returned at all, so this changes no generated kernel. It only replaces a
crash with the answer.

CPU-only.
"""

import pytest

from Tensile.Common.DataType import DataType
from Tensile.Common.Utilities import SWIZZLE_LOAD_BYTES, swizzleGeometry

pytestmark = pytest.mark.unit


def solution(dataType, miInput=32, matrixInstK=128, waveSize=64):
    return {
        "ProblemType": {"DataTypeA": DataType(dataType), "DataTypeB": DataType(dataType)},
        "MIInputPerThreadA": miInput,
        "MIInputPerThreadB": miInput,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": matrixInstK,
        "WavefrontSize": waveSize,
    }


@pytest.mark.parametrize("tc", ["A", "B"])
def test_swizzle_geometry_handles_a_sub_byte_type(tc):
    """F4 is half a byte per element, which used to floor to zero bytes.

    ``swizzleTensor`` on an F4 operand reached this and brought down the whole
    generator rather than emitting anything, so the failure was a missing
    library, not a wrong kernel.
    """
    assert DataType("F4").numBytes() == 0.5, "the premise of this test changed"
    geometry = swizzleGeometry(solution("F4"), tc)
    assert geometry["packK"] >= 1
    assert geometry["laneSize"] >= 1
    assert geometry["swizzleK"] >= 1


@pytest.mark.parametrize("dataType", ["F4", "F8", "B", "S"])
@pytest.mark.parametrize("tc", ["A", "B"])
def test_swizzle_geometry_lane_size_tracks_bytes_per_element(dataType, tc):
    """A lane holds as many elements as fit in one load, and never more.

    Pinning the relation rather than per-type constants keeps the test honest
    about *why* the sub-byte case is different: the narrower the element, the
    more of them a fixed-size load carries.
    """
    bpe = DataType(dataType).numBytes()
    geometry = swizzleGeometry(solution(dataType), tc)
    assert geometry["laneSize"] <= SWIZZLE_LOAD_BYTES / bpe, (
        f"{dataType}: a lane cannot hold {geometry['laneSize']} elements of "
        f"{bpe} bytes in a {SWIZZLE_LOAD_BYTES}-byte load"
    )


def tilesPerRead(blockWidth, mxUnit):
    """The ratio as ``numMXSA``/``numMXSB`` now compute it.

    Restated here rather than called: the production expression sits partway
    through ``KernelWriter`` setup and reaching it needs a fully built
    ``tensorParameters`` with a resolved local-read instruction. So the two
    tests below characterize the arithmetic change -- that the new form agrees
    with the old everywhere the old returned at all, and is defined where the
    old was not -- which is exactly the claim that lets this land on its own.
    They do not cover the call site; the ``swizzleGeometry`` tests above do run
    production code.
    """
    return blockWidth * 4 / mxUnit


@pytest.mark.parametrize("blockWidth, mxUnit", [(0.25, 2), (0.25, 4), (0.5, 4)])
def test_narrow_mx_scale_read_needs_more_reads_not_an_error(blockWidth, mxUnit):
    """A read covering part of an MX unit needs proportionally more reads.

    The old form was ``int(blockWidth * 4 // mxUnit)``, which is 0 for every
    case here and then divides. These are the ratios below 1.
    """
    ratio = tilesPerRead(blockWidth, mxUnit)
    assert ratio < 1, "this case is supposed to be a sub-unit read"
    assert int(blockWidth * 4 // mxUnit) == 0, "the old form would not have divided by zero"
    assert int(8 / ratio) > 8, "a narrower read should need more reads, not fewer"


@pytest.mark.parametrize("blockWidth, mxUnit", [(0.25, 1), (0.5, 1), (0.5, 2),
                                                (1, 1), (1, 2), (1, 4),
                                                (2, 2), (4, 4)])
def test_whole_unit_reads_are_unchanged(blockWidth, mxUnit):
    """Wherever the old form returned a value, the new one returns the same.

    This is what makes the fix safe to land on its own: no kernel that
    generates today generates differently afterwards.
    """
    count = 8
    old = count // int(blockWidth * 4 // mxUnit)
    new = int(count / tilesPerRead(blockWidth, mxUnit))
    assert old == new
