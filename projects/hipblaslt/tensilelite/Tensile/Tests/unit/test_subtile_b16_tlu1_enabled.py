# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""bf16/fp16 TLU=1 resolves to a geometry instead of being rejected.

Solution.py's TLU=1 dtype branch used to end at::

    if dtype.isBFloat16() or dtype.isHalf():
        reject(state, printRejectionReason,
               f"UseSubtileImpl=1 TLU=1 is not implemented for dtype {dtype}")
        return

That reject was correct while no bf16 TLU=1 geometry existed -- without it these
solutions cleared validation and then asserted inside kernelBodySubtile rather
than failing cleanly.  It is no longer correct: `subtileStackForB16TLU1` picks
a stack and `_ABTilePair{tc}` names the geometry that lays it out.

The reject and the implementation are mutually exclusive and occupy the same
lines, so there is no partially-correct outcome here -- either bf16 NN subtile
kernels are generated or they are not.  The imports below are themselves part
of the guard: the selection helpers only exist on the implemented side, so
losing it turns this file into a collection error rather than a silent gap in
coverage.
"""

import pytest

from Tensile.Common.DataType import DataType
from Tensile.Components.Subtile.Kernel import AB_GEOMETRY_MAP
from Tensile.SolutionStructs.Validators.Subtile import (
    SUBTILE_STACK_SIZES_B16,
    subtileStackForB16TLU1,
    subtileTLU1StackReason,
)

MI_M = 16
MI_K = 32

# The mapping Solution.py applies to the chosen stack. Duplicated here on
# purpose: if the branch is rewritten, this is the statement of what it owes.
STACK_TO_GEOMETRY = {
    2: "AB_B16_TLU1_2x1",
    4: "AB_B16_TLU1_4x1",
}


def _state(mtTiles, waveGroup, dtype="bfloat16"):
    dt = DataType(dtype)
    return {
        "ProblemType": {"DataTypeA": dt, "DataTypeB": dt},
        "ISA": (9, 5, 0),
        "MIWaveGroup": list(waveGroup),
        "MIWaveTile": [mtTiles // waveGroup[0], mtTiles // waveGroup[1]],
        "MacroTile0": mtTiles * MI_M,
        "MacroTile1": mtTiles * MI_M,
        "MatrixInstM": MI_M,
        "MatrixInstK": MI_K,
        "WavefrontSize": 64,
        "DepthU": 256,
    }


# (mtTiles, MIWaveGroup, expected stack). MT = mtTiles * 16.
ACCEPTED = [
    (4, (1, 1), 4),     # MT64
    (4, (4, 1), 4),     # MT64,  waves share one strip
    (4, (2, 2), 4),     # MT64
    (8, (2, 2), 4),     # MT128
    (16, (4, 1), 4),    # MT256, a full 128 B line at stack 4
    (20, (2, 2), 2),    # MT320, the bf16 NN shape; see the dedicated test below
]


@pytest.mark.parametrize("mtTiles,waveGroup,stack", ACCEPTED)
def test_bf16_tlu1_selects_a_stack(mtTiles, waveGroup, stack):
    """The branch must produce a height, not a rejection."""
    assert subtileStackForB16TLU1(_state(mtTiles, waveGroup), "A", mtTiles) == stack


@pytest.mark.parametrize("mtTiles,waveGroup,stack", ACCEPTED)
def test_fp16_takes_the_same_path_as_bf16(mtTiles, waveGroup, stack):
    """The branch is `isBFloat16() or isHalf()`; both are 2 bytes and identical
    to every rule here, so fp16 must not be left behind if the branch is
    rewritten."""
    state = _state(mtTiles, waveGroup, dtype="half")
    assert subtileStackForB16TLU1(state, "A", mtTiles) == stack


@pytest.mark.parametrize("mtTiles,waveGroup,stack", ACCEPTED)
def test_every_selected_stack_has_a_geometry(mtTiles, waveGroup, stack):
    """A returned height indexes the _ABTilePair map, so it must name a real
    geometry -- a height with no entry is a KeyError during solution setup."""
    assert stack in STACK_TO_GEOMETRY
    geometry = AB_GEOMETRY_MAP[STACK_TO_GEOMETRY[stack]]
    assert tuple(geometry.gr.subtileShape) == (stack, 1)
    assert geometry.gr.tlu and geometry.lr.tlu


def test_mt320x320_resolves_to_the_two_tile_stack():
    """The shape bf16 TLU=1 subtile support was built for, pinned end to end.

    MT320x320 NN bf16 at MIWaveGroup [2,2] is 20 MMA-M tiles per operand. The
    4-stack cannot lay that out, so the ladder falls to 2 and the geometry is
    AB_B16_TLU1_2x1 -- which is what Solution.py records as _ABTilePairA for
    the generated kernel.
    """
    state = _state(20, (2, 2))
    assert subtileTLU1StackReason(state, "A", 20, 4) is not None
    assert subtileStackForB16TLU1(state, "A", 20) == 2
    assert STACK_TO_GEOMETRY[2] == "AB_B16_TLU1_2x1"


def test_ladder_is_four_then_two():
    """bf16 reaches a full 128 B cache line at a stack of 4, so taller buys no
    coverage and only costs LDS. 2 is reachable because it lays out tile counts
    a 4-stack refuses."""
    assert SUBTILE_STACK_SIZES_B16 == (4, 2)
    assert set(SUBTILE_STACK_SIZES_B16) == set(STACK_TO_GEOMETRY)
    for stack in SUBTILE_STACK_SIZES_B16:
        assert stack * MI_M * 2 <= 128


def test_preferred_height_is_tried_first():
    """The ladder must not reorder: a shape that 4 can lay out has to get 4."""
    for mtTiles, waveGroup, stack in ACCEPTED:
        state = _state(mtTiles, waveGroup)
        if subtileTLU1StackReason(state, "A", mtTiles, 4) is None:
            assert stack == 4


def test_unlayoutable_shape_still_returns_none():
    """The guard is against a blanket reject, not against all rejection.

    10 tiles at MIWaveGroup [2,2] gives a wave 5 MMA-M tiles, which neither
    stack tiles, so there is genuinely no geometry and None is correct. A
    version of subtileStackForB16TLU1 that never rejects would be as wrong as
    one that always does.
    """
    assert subtileStackForB16TLU1(_state(10, (2, 2)), "A", 10) is None


def test_rejection_reason_exists_for_the_preferred_height():
    """On total failure the caller reports the preferred height's reason, so
    that reason has to be non-None and describe the shape asked for."""
    state = _state(10, (2, 2))
    assert subtileStackForB16TLU1(state, "A", 10) is None
    reason = subtileTLU1StackReason(state, "A", 10, SUBTILE_STACK_SIZES_B16[0])
    assert reason and "tensor A" in reason
