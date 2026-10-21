# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""2D MX (MXBlockFree > 1) solution checks for unaligned MacroTiles.

_validateMXFreePhase keeps unaligned tiles to what mxPhaseLraFixup handles.
"""

from Tensile.SolutionStructs.Solution import _validateMXFreePhase


def _state(mtA=256, mtB=224, free=128, depthU=256, numWaves=4, vwB=1, **overrides):
    state = {
        "ProblemType": {"MXBlockA": 128, "MXBlockB": 128,
                        "MXBlockFreeA": free, "MXBlockFreeB": free},
        "MacroTileMXSA": mtA,
        "MacroTileMXSB": mtB,
        "MatrixInstK": 128,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstBM": 1,
        "MatrixInstBN": 1,
        "VectorWidthA": 1,
        "VectorWidthB": vwB,
        "DepthU": depthU,
        "NumWaves": numWaves,
        "MXScaleFormat": "InMemorySwizzle",
        "UseSubtileImpl": False,
        "PrefetchAcrossPersistent": 0,
        "DirectToVgprMXSA": False,
        "DirectToVgprMXSB": False,
        "enableTDMA": True,
        "enableTDMB": True,
        "TDMFuse": 0,
        "Valid": True,
    }
    state.update(overrides)
    return state


def test_free_phase_accepts_the_mt224_config():
    assert _validateMXFreePhase(_state(), False)


def test_free_phase_ignores_aligned_and_1d_tiles():
    assert _validateMXFreePhase(_state(mtB=256, UseSubtileImpl=True), False)
    assert _validateMXFreePhase(_state(free=1, UseSubtileImpl=True), False)


def test_free_phase_rejects_what_the_fixup_cannot_index():
    assert not _validateMXFreePhase(_state(UseSubtileImpl=True), False)
    assert not _validateMXFreePhase(_state(PrefetchAcrossPersistent=1), False)
    assert not _validateMXFreePhase(_state(MXScaleFormat="NoSwizzle"), False)
    assert not _validateMXFreePhase(_state(MatrixInstK=256), False)
    # 16 * VW 4 = 64 columns per vector, but gcd(224, 128) = 32: at phase 32
    # a vector straddles a scale row.
    assert not _validateMXFreePhase(_state(vwB=4), False)

