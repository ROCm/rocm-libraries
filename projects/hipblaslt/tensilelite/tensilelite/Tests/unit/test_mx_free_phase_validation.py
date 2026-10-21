# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""2D MX (MXBlockFree > 1) solution checks for unaligned MacroTiles.

_validateMXFreePhase keeps unaligned tiles to what mxPhaseLraFixup handles.
_validateMXTdmMSplit rejects a wave-separated TDM M-split that would drop or
misplace scale rows.
"""

from Tensile.SolutionStructs.Solution import _validateMXFreePhase, _validateMXTdmMSplit


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


def test_tdm_k_split_is_untouched():
    # DepthU 256 / MXBlock 128 = 2 K-groups >= 2 components (4 waves).
    assert _validateMXTdmMSplit(_state(mtB=384), False)


def test_tdm_m_split_needs_even_aligned_rows():
    # DepthU 128: 1 K-group < 2 components, so the components split rows.
    assert _validateMXTdmMSplit(_state(depthU=128, mtB=256), False)
    assert not _validateMXTdmMSplit(_state(depthU=128, mtB=384), False)  # 3 rows / 2
    assert not _validateMXTdmMSplit(_state(depthU=128, mtB=448), False)  # unaligned
    # Fewer rows than components: every component loads the whole tile.
    assert _validateMXTdmMSplit(_state(depthU=128, mtA=128, mtB=128), False)
