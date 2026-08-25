# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Mutation-targeted characterization tests for ValidParameters space-fill validators."""

import pytest

import Tensile.Common.ValidParameters as VP

pytestmark = pytest.mark.unit


def test_space_fill_algo_accepts_max_order_id_five():
    """OrderID 5 is accepted as the current inclusive upper boundary."""
    assert VP.checkSpaceFillAlgoIsValid("SpaceFillingAlgo", [5]) is None


def test_space_fill_algo_wgm_accepts_grid_dim_255():
    """GridDim 255 is accepted as the current inclusive upper boundary."""
    assert VP.checkSpaceFillAlgoWGMIsValid("SFCWGM", [[0, 255]]) is None
