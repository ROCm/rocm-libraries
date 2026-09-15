# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the MXBlock{A,B} scaling-tile spelling.

A config may write the tile as ``[k]`` or ``[free, k]``; ProblemType splits it
into the ``MXBlock{A,B}`` / ``MXBlockFree{A,B}`` int pair everything downstream
reads, and the kernel name spells a 2D tile ``B<free>x<k>``.
"""

import pytest

from Tensile.Common.TypeValidationErrors import ConfigTypeError
from Tensile.SolutionStructs.Problem import ProblemType, _defaultProblemType


def _problemType(**overrides):
    cfg = dict(_defaultProblemType)
    cfg["DataType"] = "F8"
    cfg["DestDataType"] = "s"
    cfg["ComputeDataType"] = "s"
    cfg["DataTypeMXSA"] = "e8"
    cfg["DataTypeMXSB"] = "e8"
    cfg.update(overrides)
    return ProblemType(cfg, printIndexAssignmentInfo=False)


@pytest.mark.parametrize("spelling", [32, [32]])
def test_k_only_forms_agree(spelling):
    """A bare int and a one-element list are the same 1x32 tile."""
    pt = _problemType(MXBlockA=spelling, MXBlockB=spelling)
    assert (pt["MXBlockA"], pt["MXBlockFreeA"]) == (32, 1)
    assert (pt["MXBlockB"], pt["MXBlockFreeB"]) == (32, 1)
    assert "MXAE8B32" in str(pt)


def test_two_element_form_sets_free_extent():
    pt = _problemType(MXBlockA=[128, 128], MXBlockB=[128])
    assert (pt["MXBlockA"], pt["MXBlockFreeA"]) == (128, 128)
    assert (pt["MXBlockB"], pt["MXBlockFreeB"]) == (128, 1)
    name = str(pt)
    assert "MXAE8B128x128" in name  # 2D tile
    assert "MXBE8B128" in name and "MXBE8B128x" not in name  # A and B are independent


def test_rejects_wrong_length():
    with pytest.raises(ConfigTypeError, match="MXBlockA"):
        _problemType(MXBlockA=[1, 128, 128])
