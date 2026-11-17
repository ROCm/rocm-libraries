# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""2D MX (MXBlockFree > 1) solution checks for unaligned MacroTiles.

_validateMXFreePhase keeps unaligned tiles to what mxPhaseLraFixup handles;
_validateMXFreePhaseReadOffset keeps their scale read offsets below 64K.
"""

from tensilelite.SolutionStructs.Solution import _validateMXFreePhase, \
    _validateMXFreePhaseReadOffset


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
        "ExpandPointerSwap": False,
        "StoreSwapAddr": False,
        "LdsOffsetA_Blk": 0,
        "LdsPadMXSA": 0,
        "LdsPadMXSB": 0,
        "LdsBlockSizePerPadMXSA": 0,
        "LdsBlockSizePerPadMXSB": 0,
        "UnrollMajorLDSMXSA": True,
        "UnrollMajorLDSMXSB": True,
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



def test_free_phase_read_offset_rejects_a_swap_offset_past_64k():
    # MT 224 at the worst phase spans 3 scale rows; DepthU 256 / MXBlock 128 = 2
    # bytes per row, so one MXS buffer is 6 bytes.
    eps = dict(ExpandPointerSwap=True)
    assert _validateMXFreePhaseReadOffset(_state(LdsOffsetA_Blk=65530, **eps), False)
    assert not _validateMXFreePhaseReadOffset(_state(LdsOffsetA_Blk=65536, **eps), False)


def test_free_phase_read_offset_ignores_address_swaps_and_aligned_tiles():
    big = dict(ExpandPointerSwap=True, LdsOffsetA_Blk=1 << 17)
    assert _validateMXFreePhaseReadOffset(_state(LdsOffsetA_Blk=1 << 17), False)
    assert _validateMXFreePhaseReadOffset(_state(StoreSwapAddr=True, **big), False)
    assert _validateMXFreePhaseReadOffset(_state(mtA=256, mtB=256, **big), False)


def test_aligned_2d_rejects_a_vector_wider_than_a_scale_row():
    # MT 256 is a multiple of free 128, but VW 16 makes one B vector 256
    # columns: the one scale it loads would cover 2 scale rows.
    assert not _validateMXFreePhase(_state(mtB=256, vwB=16), False)


def test_aligned_2d_accepts_a_vector_within_a_scale_row():
    assert _validateMXFreePhase(_state(mtB=256, vwB=4), False)
    assert _validateMXFreePhase(_state(mtB=256, vwB=8), False)
