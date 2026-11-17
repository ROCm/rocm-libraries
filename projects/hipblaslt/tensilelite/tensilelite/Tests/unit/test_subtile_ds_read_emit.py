# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emit tests for the swizzled subtile ds_read path.

``emitSubtileDsRead`` walks a subtile's K slices and issues one ds_read_b128
per 4 destination VGPRs, consuming ``sharedVgprLROffset`` entries in order;
``localReadDoSubtile`` drives it across the local subtile grid for one tensor.
Both are otherwise only reached from the gfx950 hardware suites, so the tile
geometry here is a stub -- the point is the read count, the destination and
address registers each read picks, and the LDS offset per subtile.
"""

import re

import pytest
from types import SimpleNamespace

from tensilelite.Components.Subtile.SubtileLREmit import (
    emitSubtileDsRead, localReadDoSubtile,
)

pytestmark = pytest.mark.unit


def _tile(firstVgpr, numRegs):
    return SimpleNamespace(regList=SimpleNamespace(
        indices=list(range(firstVgpr, firstVgpr + numRegs))))


def _tileInfo(tc='A', subtileShapeK=2, numRegs=4, localGrid=(1, 1),
              subtileSize=64, gridX=2, numTiles=8):
    """A tile geometry with ``subtileShapeK`` K slices per subtile.

    Tile i lives at VGPR 100 + 10*i so each read's destination identifies the
    tile it came from, and offset j addresses register 200 + j likewise.
    """
    return SimpleNamespace(
        tc=tc,
        loadWidthLR=16,                     # 16 bytes -> 4 VGPRs per ds_read
        subtileSize=subtileSize,
        globalSubtileGrid=[gridX, 1],
        localSubtileGrid=list(localGrid),
        subtileShape=[1, subtileShapeK],
        sharedVgprLROffset=[200 + j for j in range(64)],
        vgprTiles=[_tile(100 + 10 * i, numRegs) for i in range(numTiles)],
        getSubtileShapeLinearId=lambda du, m: du,
        lrTileIndexForSubtile=lambda s0, s1, mfmaId: mfmaId,
    )


_READ_RE = re.compile(r'ds_read_b128 v\[(\d+):\d+\], v(\d+)(?: offset:(\d+))?')


def _reads(module):
    """Each ds_read as (dst, addr, offset), in emission order."""
    out = []
    for inst in module.flatitems():
        m = _READ_RE.match(str(inst))
        assert m, str(inst)
        out.append((int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)))
    return out


class TestEmitSubtileDsRead:

    def test_one_read_per_k_slice(self):
        """Four destination VGPRs is exactly one ds_read, so the read count is
        the number of K slices."""
        module = emitSubtileDsRead(None, {}, _tileInfo(subtileShapeK=3), [0, 0])
        assert _reads(module) == [(100, 200, 0), (110, 201, 0), (120, 202, 0)]

    def test_wide_tiles_split_into_multiple_reads(self):
        """An 8-VGPR tile (fp8) needs two ds_read_b128, and the second starts
        four registers into the tile."""
        module = emitSubtileDsRead(None, {}, _tileInfo(subtileShapeK=2, numRegs=8), [0, 0])
        assert _reads(module) == [(100, 200, 0), (104, 201, 0),
                                  (110, 202, 0), (114, 203, 0)]

    def test_offsets_are_consumed_without_gaps(self):
        """Address registers advance once per read, not once per tile."""
        module = emitSubtileDsRead(None, {}, _tileInfo(subtileShapeK=3, numRegs=8), [0, 0])
        assert [addr for _, addr, _ in _reads(module)] == [200, 201, 202, 203, 204, 205]

    def test_lds_offset_places_the_subtile(self):
        """offset = sId0*subtileSize + sId1*gridX*subtileSize."""
        info = _tileInfo(subtileShapeK=1, subtileSize=64, gridX=2)
        assert _reads(emitSubtileDsRead(None, {}, info, [0, 0]))[0][2] == 0
        assert _reads(emitSubtileDsRead(None, {}, info, [1, 0]))[0][2] == 64
        assert _reads(emitSubtileDsRead(None, {}, info, [0, 1]))[0][2] == 128
        assert _reads(emitSubtileDsRead(None, {}, info, [3, 2]))[0][2] == 448

    def test_comment_names_the_tensor_and_subtile(self):
        module = emitSubtileDsRead(None, {}, _tileInfo(tc='B', subtileShapeK=1), [1, 2])
        assert 'SubtileB[1, 2] subIterK=0 read=0' in str(module.flatitems()[0])


class TestLocalReadDoSubtile:

    @staticmethod
    def _writer(tileInfoA, tileInfoB):
        return SimpleNamespace(states=SimpleNamespace(
            a=SimpleNamespace(tileInfo=tileInfoA),
            b=SimpleNamespace(tileInfo=tileInfoB)))

    def test_covers_the_whole_local_grid(self):
        """One emitSubtileDsRead per grid cell, row-major over (i, j)."""
        info = _tileInfo(tc='A', subtileShapeK=1, localGrid=(2, 3))
        module = localReadDoSubtile('A', self._writer(info, None), {})
        assert len(_reads(module)) == 6
        assert [off for _, _, off in _reads(module)] == [0, 128, 256, 64, 192, 320]

    def test_selects_the_tensor_tile_info(self):
        a = _tileInfo(tc='A', subtileShapeK=1)
        b = _tileInfo(tc='B', subtileShapeK=1)
        writer = self._writer(a, b)
        assert 'SubtileA' in str(localReadDoSubtile('A', writer, {}))
        assert 'SubtileB' in str(localReadDoSubtile('B', writer, {}))

    def test_empty_grid_emits_nothing(self):
        info = _tileInfo(subtileShapeK=1, localGrid=(0, 4))
        module = localReadDoSubtile('A', self._writer(info, None), {})
        assert module.flatitems() == []
