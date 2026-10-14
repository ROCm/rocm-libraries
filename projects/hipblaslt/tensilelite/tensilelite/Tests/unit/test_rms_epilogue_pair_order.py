# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only tests for the RMS epilogue pair visit order."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

from Tensile.Common.DataType import DataType
from Tensile.Components.Subtile.RMSEpilogueEmit import RMSEpilogueEmitter


def _makeEmitter(mt0, mt1, wgM, wgN):
    writer = SimpleNamespace(states=SimpleNamespace(laneSGPRCount=2))
    kernel = {
        "WavefrontSize": 64,
        "MIWaveGroup": [wgM, wgN],
        "MacroTile0": mt0,
        "MacroTile1": mt1,
        "RMSEpilogue": True,
        "ProblemType": {"DestDataType": DataType("B")},
    }
    return RMSEpilogueEmitter(writer, kernel)


# (MT0, MT1, wgM, wgN) covering T_M in {2, 4, 6, 8, 12}.
_configs = [
    (64, 64, 2, 2),
    (256, 256, 4, 1),
    (192, 192, 2, 2),
    (256, 256, 2, 2),
    (384, 256, 2, 2),
]


@pytest.mark.parametrize("mt0, mt1, wgM, wgN", _configs)
def test_pairOrderIsPermutation(mt0, mt1, wgM, wgN):
    emitter = _makeEmitter(mt0, mt1, wgM, wgN)
    order = emitter._pairOrder()
    expected = {(mp, n) for mp in range(emitter.T_M // 2) for n in range(emitter.T_N)}
    assert len(order) == emitter.numPairs
    assert set(order) == expected


@pytest.mark.parametrize("mt0, mt1, wgM, wgN", _configs)
def test_pairOrderKeepsAscendingMpPerColumn(mt0, mt1, wgM, wgN):
    emitter = _makeEmitter(mt0, mt1, wgM, wgN)
    order = emitter._pairOrder()
    for n in range(emitter.T_N):
        mps = [mp for mp, col in order if col == n]
        assert mps == sorted(mps)


@pytest.mark.parametrize("mt0, mt1, wgM, wgN", _configs)
def test_pairOrderInterleavesOrFallsBack(mt0, mt1, wgM, wgN):
    emitter = _makeEmitter(mt0, mt1, wgM, wgN)
    order = emitter._pairOrder()
    if (emitter.T_M // 2) % 2 != 0:
        assert not emitter.interleavePairs
        assert order == [(t // emitter.T_N, t % emitter.T_N) for t in range(emitter.numPairs)]
        return
    assert emitter.interleavePairs
    for i in range(0, len(order), 2):
        mp, n = order[i]
        assert mp % 2 == 0
        assert order[i + 1] == (mp + 1, n)
