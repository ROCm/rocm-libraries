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

"""Characterization tests for the validators in
``Tensile.Common.ValidParameters``: ``checkParametersAreValid`` (the central
parameter validator) and the two space-filling sub-validators
``checkSpaceFillAlgoIsValid`` / ``checkSpaceFillAlgoWGMIsValid``.

All Tier A: pure functions over plain dicts/lists. Accept paths return ``None``
(asserted) and reject paths raise ``Exception`` whose message is pinned via
``pytest.raises``. Targeted synthetic ``validParams`` dicts drive each branch
of ``checkParametersAreValid`` deterministically (the real ``validParameters``
is used only for the realistic accept case).
"""

import pytest

import Tensile.Common.ValidParameters as VP

pytestmark = pytest.mark.unit


# ===========================================================================
# checkSpaceFillAlgoIsValid — accept + every reject branch
# ===========================================================================

def test_space_fill_algo_valid_returns_none():
    assert VP.checkSpaceFillAlgoIsValid("SpaceFillingAlgo", [0, 1, 2]) is None
    assert VP.checkSpaceFillAlgoIsValid("SpaceFillingAlgo", []) is None


@pytest.mark.parametrize(
    "value",
    [5, "abc", (0, 1), [0, 1, 2, 3], [0, 6], [-1, 0]],
    ids=["int", "str", "tuple", "too_long", "order_too_big", "order_negative"],
)
def test_space_fill_algo_invalid_raises(value, snapshot):
    with pytest.raises(Exception) as excinfo:
        VP.checkSpaceFillAlgoIsValid("SpaceFillingAlgo", value)
    assert str(excinfo.value) == snapshot


# ===========================================================================
# checkSpaceFillAlgoWGMIsValid — accept + every reject branch
# ===========================================================================

def test_space_fill_algo_wgm_valid_returns_none():
    assert VP.checkSpaceFillAlgoWGMIsValid("SFCWGM", [[0, 1], [2, 3]]) is None
    assert VP.checkSpaceFillAlgoWGMIsValid("SFCWGM", []) is None


@pytest.mark.parametrize(
    "value",
    [
        7,                       # not a list
        "xy",                    # str, not a list
        [[0, 1], [2, 3], [4, 5]],  # too many levels (>2)
        [[0, 1, 2]],             # pair length != 2
        [[0]],                   # pair length != 2 (short)
        [[0, 256]],              # gridDim out of range [0,256)
        [[0, -1]],               # gridDim out of range (negative)
    ],
    ids=["int", "str", "too_many_levels", "triple_pair", "short_pair", "dim_too_big", "dim_negative"],
)
def test_space_fill_algo_wgm_invalid_raises(value, snapshot):
    with pytest.raises(Exception) as excinfo:
        VP.checkSpaceFillAlgoWGMIsValid("SFCWGM", value)
    assert str(excinfo.value) == snapshot


# ===========================================================================
# checkParametersAreValid — early returns
# ===========================================================================

def test_check_params_early_return():
    # ProblemSizes is the only name accepted unconditionally (no validParams lookup).
    assert VP.checkParametersAreValid(("ProblemSizes", ["anything", 1, 2]), {}) is None


# ===========================================================================
# checkParametersAreValid — accept paths (return None, no raise)
# ===========================================================================

def test_check_params_accept_value_in_list():
    assert VP.checkParametersAreValid(("P", [2]), {"P": [1, 2, 3]}) is None


def test_check_params_accept_any_value_sentinel():
    # validParams[name] == -1 -> any value accepted (the `!= -1` short-circuit).
    assert VP.checkParametersAreValid(("P", [99999]), {"P": -1}) is None


def test_check_params_accept_multiple_values():
    assert VP.checkParametersAreValid(("P", [1, 2, 3]), {"P": [1, 2, 3, 4]}) is None


def test_check_params_accept_against_real_table():
    # A realistic accept using the module's own validParameters: pick the first
    # real key with a non-empty declared value list and feed back one of its
    # own values (insertion order is deterministic, so this is stable).
    name = next(k for k, v in VP.validParameters.items() if isinstance(v, list) and v)
    value = VP.validParameters[name][0]
    assert VP.checkParametersAreValid((name, [value]), VP.validParameters) is None


# ===========================================================================
# checkParametersAreValid — reject paths
# ===========================================================================

def test_check_params_unknown_name_raises(snapshot):
    with pytest.raises(Exception) as excinfo:
        VP.checkParametersAreValid(("NotARealParameter", [1]), {"P": [1, 2, 3]})
    # The message embeds sorted(validParameters.keys()) (138 names); pin only
    # the first line to stay reviewable and table-roster-independent.
    assert str(excinfo.value).split("\n")[0] == snapshot


def test_check_params_value_not_in_short_list_raises(snapshot):
    # List <= 32 entries -> no "(only first 32 combos printed)" suffix.
    with pytest.raises(Exception) as excinfo:
        VP.checkParametersAreValid(("P", [99]), {"P": [1, 2, 3]})
    assert str(excinfo.value) == snapshot


def test_check_params_value_not_in_long_list_raises(snapshot):
    # List > 32 entries -> the truncated-message (msgExt) variant fires.
    with pytest.raises(Exception) as excinfo:
        VP.checkParametersAreValid(("P", [999]), {"P": list(range(40))})
    assert str(excinfo.value) == snapshot


# ===========================================================================
# checkParametersAreValid — sub-validator dispatch (the two elif arms)
# ===========================================================================

def test_check_params_dispatch_space_filling_algo_valid():
    # name == "SpaceFillingAlgo" with the -1 sentinel bypasses the value-list
    # check and dispatches to checkSpaceFillAlgoIsValid (valid -> None).
    assert VP.checkParametersAreValid(("SpaceFillingAlgo", [[0, 1, 2]]), {"SpaceFillingAlgo": -1}) is None


def test_check_params_dispatch_space_filling_algo_invalid_propagates(snapshot):
    with pytest.raises(Exception) as excinfo:
        VP.checkParametersAreValid(("SpaceFillingAlgo", [[0, 9]]), {"SpaceFillingAlgo": -1})
    assert str(excinfo.value) == snapshot


def test_check_params_dispatch_sfcwgm_valid():
    assert VP.checkParametersAreValid(("SFCWGM", [[[0, 1], [2, 3]]]), {"SFCWGM": -1}) is None


def test_check_params_dispatch_sfcwgm_invalid_propagates(snapshot):
    with pytest.raises(Exception) as excinfo:
        VP.checkParametersAreValid(("SFCWGM", [[[0, 999]]]), {"SFCWGM": -1})
    assert str(excinfo.value) == snapshot
