# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only tests for the RMS epilogue unit visit order."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

from Tensile.Common.DataType import DataType
from Tensile.Components.Subtile.RMSEpilogueEmit import RMSEpilogueEmitter
from Tensile.Components.Subtile.RMSEpilogueLayout import CmsRmsLayout, SubtileRmsLayout


def _makeLayout(mt0, mt1, wgM, wgN):
    writer = SimpleNamespace(states=SimpleNamespace(laneSGPRCount=2))
    kernel = {
        "WavefrontSize": 64,
        "MIWaveGroup": [wgM, wgN],
        "MacroTile0": mt0,
        "MacroTile1": mt1,
        "RMSEpilogue": True,
        "UseSubtileImpl": True,
        "UseCustomMainLoopSchedule": 0,
        "ProblemType": {"DestDataType": DataType("B")},
    }
    emitter = RMSEpilogueEmitter(writer, kernel)
    return emitter, SubtileRmsLayout(emitter, None)


# (MT0, MT1, wgM, wgN) covering T_M in {2, 4, 6, 8, 12}.
_configs = [
    (64, 64, 2, 2),
    (256, 256, 4, 1),
    (192, 192, 2, 2),
    (256, 256, 2, 2),
    (384, 256, 2, 2),
]


@pytest.mark.parametrize("mt0, mt1, wgM, wgN", _configs)
def test_unitOrderIsPermutation(mt0, mt1, wgM, wgN):
    emitter, layout = _makeLayout(mt0, mt1, wgM, wgN)
    order = [layout.unitCoords(u) for u in range(emitter.numUnits)]
    expected = {(mp, n) for mp in range(emitter.T_M // 2) for n in range(emitter.T_N)}
    assert len(order) == len(expected)
    assert set(order) == expected


@pytest.mark.parametrize("mt0, mt1, wgM, wgN", _configs)
def test_unitOrderKeepsAscendingMpPerColumn(mt0, mt1, wgM, wgN):
    emitter, layout = _makeLayout(mt0, mt1, wgM, wgN)
    order = [layout.unitCoords(u) for u in range(emitter.numUnits)]
    for n in range(emitter.T_N):
        mps = [mp for mp, col in order if col == n]
        assert mps == sorted(mps)


@pytest.mark.parametrize("mt0, mt1, wgM, wgN", _configs)
def test_unitOrderInterleavesOrFallsBack(mt0, mt1, wgM, wgN):
    emitter, layout = _makeLayout(mt0, mt1, wgM, wgN)
    order = [layout.unitCoords(u) for u in range(emitter.numUnits)]
    if (emitter.T_M // 2) % 2 != 0:
        assert not layout.interleavePairs
        assert order == [(u // emitter.T_N, u % emitter.T_N) for u in range(emitter.numUnits)]
        return
    assert layout.interleavePairs
    for i in range(0, len(order), 2):
        mp, n = order[i]
        assert mp % 2 == 0
        assert order[i + 1] == (mp + 1, n)


@pytest.mark.parametrize("mt0, mt1, wgM, wgN", _configs)
def test_slotAndKeyFollowUnitCoords(mt0, mt1, wgM, wgN):
    emitter, layout = _makeLayout(mt0, mt1, wgM, wgN)
    for u in range(emitter.numUnits):
        mp, n = layout.unitCoords(u)
        assert layout.unitSlot(u) == n
        assert layout.gammaKey(u) == mp


@pytest.mark.parametrize("mt0, mt1, wgM, wgN", _configs)
def test_gammaBankCount(mt0, mt1, wgM, wgN):
    _, layout = _makeLayout(mt0, mt1, wgM, wgN)
    assert layout.numGammaBanks == (2 if layout.interleavePairs else 1)


def test_cmsLayoutUsesSingleGammaBank():
    assert CmsRmsLayout.numGammaBanks == 1
