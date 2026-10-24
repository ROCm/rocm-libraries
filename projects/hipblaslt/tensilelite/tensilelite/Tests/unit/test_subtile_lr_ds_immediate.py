# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The 16-bit ds-immediate bound on the swizzled non-TLU=1 subtile LR path.

emitSingleDsRead addresses every local read off one per-lane base register plus
a ds immediate, with no staging register to hold a 64 KB-aligned top (the TLU=1
bf16 path has one -- sharedVgprLRBigOffset).  Past the 16-bit field the operand
cannot be addressed at all and the assembler rejects the kernel with "expected a
16-bit unsigned offset", so _subtileLRDsImmediateReason has to reject the
solution first.

The expectations below are the offsets actually emitted for the B operand of an
NN bf16 subtile kernel at DepthU 128, MIWaveGroup [2,2], captured from
Tensile/Components/Subtile/SubtileLREmit.py.
"""

import pytest

from Tensile.SolutionStructs.Validators.Subtile import (
    _SUBTILE_DS_IMM_LIMIT,
    _subtileMaxLRDsOffset,
)


# (macroTile, subtileSize, localSubtileGrid, globalSubtileGrid, expected offset)
# B operand, NN bf16, DepthU 128, MIWaveGroup [2,2].  MIWaveTile[1] = MT/32.
MEASURED = [
    (256, 2048, (8, 2), (16, 2), 47104),
    (288, 2048, (9, 2), (18, 2), 53248),
    (320, 2048, (10, 2), (20, 2), 59392),
    (352, 2048, (11, 2), (22, 2), 65536),
    (384, 2048, (12, 2), (24, 2), 71680),
]


@pytest.mark.parametrize("macroTile, subtileSize, localGrid, globalGrid, expected",
                         MEASURED)
def test_max_lr_ds_offset_matches_emit(macroTile, subtileSize, localGrid,
                                       globalGrid, expected):
  assert _subtileMaxLRDsOffset(subtileSize, localGrid, globalGrid) == expected


@pytest.mark.parametrize("macroTile, subtileSize, localGrid, globalGrid, expected",
                         MEASURED)
def test_ds_immediate_bound_splits_at_mt352(macroTile, subtileSize, localGrid,
                                            globalGrid, expected):
  """MT 256/288/320 fit; 352 is the first that does not (65536 == the limit)."""
  fits = _subtileMaxLRDsOffset(subtileSize, localGrid, globalGrid) < _SUBTILE_DS_IMM_LIMIT
  assert fits == (macroTile <= 320)


def test_globalGrid_may_be_float():
  """globalSubtileGrid[1] arrives as a float (DepthU / MatrixInstK)."""
  assert _subtileMaxLRDsOffset(2048, (11, 2), (22, 2.0)) == 65536


def test_single_k_window_drops_the_window_term():
  """localSubtileGrid[1] == 1 leaves only the subtile-row walk."""
  assert _subtileMaxLRDsOffset(2048, (11, 1), (22, 1)) == 10 * 2048
