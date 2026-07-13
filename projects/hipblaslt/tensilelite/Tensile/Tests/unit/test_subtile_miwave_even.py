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
################################################################################

# TODO: TEMPORARY FIX. These tests cover a temporary rejection that guards
# UseSubtileImpl solutions with an odd MIWaveTile * MIWaveGroup product, which
# currently produces a numerical mismatch we have not yet root-caused. Remove
# together with _validateSubtileMIWaveEven once the mismatch is fixed.

import pytest

from Tensile.SolutionStructs.Solution import _validateSubtileMIWaveEven


def _state(useSubtile, miWaveTile, miWaveGroup):
    return {
        "UseSubtileImpl": useSubtile,
        "MIWaveTile": miWaveTile,
        "MIWaveGroup": miWaveGroup,
        "Valid": True,
    }


def test_subtile_odd_product_is_rejected():
    # Every factor odd -> odd product -> rejected.
    state = _state(True, [1, 1], [1, 1])

    valid = _validateSubtileMIWaveEven(state, False)

    assert valid is False
    assert state["Valid"] is False


def test_subtile_odd_miwavetile_odd_miwavegroup_is_rejected():
    state = _state(True, [3, 1], [3, 1])

    valid = _validateSubtileMIWaveEven(state, False)

    assert valid is False
    assert state["Valid"] is False


def test_subtile_even_miwavegroup_2x2_is_accepted():
    # MIWaveGroup=[2, 2] makes the product even regardless of MIWaveTile.
    state = _state(True, [1, 1], [2, 2])

    valid = _validateSubtileMIWaveEven(state, False)

    assert valid is True
    assert state["Valid"] is True


def test_subtile_even_miwavetile_is_accepted():
    # A single even MIWaveTile factor makes the product even.
    state = _state(True, [2, 1], [1, 1])

    valid = _validateSubtileMIWaveEven(state, False)

    assert valid is True
    assert state["Valid"] is True


@pytest.mark.parametrize("mi_wave_group", [[2, 2], [4, 1], [1, 4]])
def test_subtile_multi_wave_group_is_accepted(mi_wave_group):
    state = _state(True, [1, 1], mi_wave_group)

    valid = _validateSubtileMIWaveEven(state, False)

    assert valid is True
    assert state["Valid"] is True


def test_non_subtile_odd_product_is_not_rejected():
    # Rejection only applies to the subtile (UseSubtileImpl) path.
    state = _state(False, [1, 1], [1, 1])

    valid = _validateSubtileMIWaveEven(state, False)

    assert valid is True
    assert state["Valid"] is True
