# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ReuseAcrossPersistent names a tile's A by the M-tile DefaultWGM gives the tile.

RAP keeps A resident across the tiles of a persistent workgroup and refills it
when a tile reads another one. A tile reads the A of its WorkGroup0 after
DefaultWGM, which folds the N-tile in, so the entry guard -- which runs before
the tile is claimed -- has to arrive at the same M-tile from the tile cursor.
These tests run both emitters on the CPU and compare them tile by tile.
"""

import itertools
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from rocisa.container import sgpr

import Tensile.KernelWriterAssembly  # noqa: F401  (loads the components in order)
import Tensile.Components.WorkAssignment as work_assignment
import Tensile.Components.WorkGroupMappingAlgos as wgm_algos
from Tensile.Component import Component
from Tensile.Components.TileProcessingStrategy import DataParallel
from Tensile.Components.WorkAssignment import StaticGrid
from Tensile.Components.WorkGroupMappingAlgos import DefaultWGM
from scalar_exec_helpers import exact_scalar_divides, run_scalar

pytestmark = pytest.mark.unit

_MASK = 0xFFFFFFFF


class _Pool:
    """Hands out fresh indices, so no two live temporaries can alias."""

    def __init__(self, start):
        self.next = start

    def checkOut(self, size, tag="", *args, **kwargs):
        idx = self.next
        self.next += size
        return idx

    def checkOutAligned(self, size, alignment, tag="", *args, **kwargs):
        self.next += (-self.next) % alignment
        return self.checkOut(size, tag)

    def checkIn(self, idx):
        pass


class _Labels:
    def __init__(self):
        self.count = 0

    def getNameInc(self, name):
        self.count += 1
        return "%s_%d" % (name, self.count)


class _Writer:
    def __init__(self):
        self.labels = _Labels()
        self.sgprPool = _Pool(100)
        self.vgprPool = _Pool(0)
        self.states = SimpleNamespace(WGMTransformLevels=-1)

    @contextmanager
    def allocTmpSgpr(self, size, alignment=1, tag=""):
        yield SimpleNamespace(idx=self.sgprPool.checkOutAligned(size, alignment or 1), size=size)

    @contextmanager
    def allocTmpSgprList(self, nums, alignmentList=None, tag=""):
        yield [SimpleNamespace(idx=self.sgprPool.checkOut(n), size=n) for n in nums]


def _kernel(wgmxcc, supportCustomWGM=True):
    return {
        "InternalSupportParams": {"SupportCustomWGM": supportCustomWGM},
        "WorkGroupMappingXCC": wgmxcc,
        "ClusterDim": [1, 1],
        "WavefrontSize": 32,
    }


def _wgm_field(wgm, wgmxcc):
    """Pack WGM the way the host does, with a neighbouring field set to prove it is masked."""
    if wgmxcc == -1:
        return (0x3 << 10) | (wgm & 0x3FF)
    return (0x5 << 16) | (wgm & 0xFFFF)


def wgm_map(wg0, wg1, n0, n1, wgm):
    """DefaultWGM in Python: the coordinates a tile ends up with."""
    if wgm > 1:
        first, second, nFirst, nSecond = wg0, wg1, n0, n1
    elif wgm >= 0:
        return wg0, wg1
    else:
        first, second, nFirst, nSecond = wg1, wg0, n1, n0
    width = abs(wgm)
    block = second // width
    serial = (second % width) * nFirst + first
    fullBlocks = nSecond // width
    divisor = (nSecond - fullBlocks * width or width) if block >= fullBlocks else width
    newFirst, newSecond = serial // divisor, serial % divisor + block * width
    return (newFirst, newSecond) if wgm > 1 else (newSecond, newFirst)


@pytest.fixture
def _cpu(monkeypatch):
    exact_scalar_divides(monkeypatch, work_assignment, wgm_algos)
    monkeypatch.setattr(Component.TileProcessingStrategy, "find", lambda writer: DataParallel())


_CASES = list(itertools.product(
    [1, 2, 3, 5, 6],                                    # NumWorkGroups0
    [1, 4, 7, 16],                                      # NumWorkGroups1
    [-9, -4, -3, -2, -1, 0, 1, 2, 3, 4, 8, 16, 24],     # WGM, both signs and wider than the grid
    [-1, 1],                                            # WorkGroupMappingXCC: the two field layouts
))


@pytest.mark.parametrize("n0, n1, wgm, wgmxcc", _CASES)
def test_guard_peek_names_the_m_tile_default_wgm_gives_the_tile(_cpu, n0, n1, wgm, wgmxcc):
    batches = 2
    kernel = _kernel(wgmxcc)
    peek = StaticGrid().peekTileBatch(_Writer(), kernel, "RAPResidentBatch", "RAPResidentMTile")
    remap = DefaultWGM(_Writer(), kernel, "WGM")
    common = {
        str(sgpr("NumWorkGroups0")): n0,
        str(sgpr("NumWorkGroups1")): n1,
        str(sgpr("WGM")): _wgm_field(wgm, wgmxcc),
    }
    for tile in range(n0 * n1 * batches):
        peeked = run_scalar(peek, {**common, str(sgpr("NextTile")): tile})
        batch, inBatch = divmod(tile, n0 * n1)
        remapped = run_scalar(remap, {**common,
                                      str(sgpr("WorkGroup0")): inBatch % n0,
                                      str(sgpr("WorkGroup1")): inBatch // n0})
        mTile = remapped[str(sgpr("WorkGroup0"))]
        assert mTile == wgm_map(inBatch % n0, inBatch // n0, n0, n1, wgm)[0]
        assert peeked[str(sgpr("RAPResidentBatch"))] == batch
        assert peeked[str(sgpr("RAPResidentMTile"))] == mTile, (
            "tile %d: guard expects M-tile %d, the tile reads %d"
            % (tile, peeked[str(sgpr("RAPResidentMTile"))], mTile))
        for claimed in ("WorkGroup0", "WorkGroup1", "WorkGroup2"):
            assert str(sgpr(claimed)) not in peeked, "the peek claimed %s" % claimed


def test_without_the_runtime_field_the_peek_is_the_plain_remainder(_cpu):
    """No SupportCustomWGM, no remap: the host does not pack the field, so the peek never reads it."""
    n0, n1 = 3, 7
    peek = StaticGrid().peekTileBatch(_Writer(), _kernel(1, supportCustomWGM=False),
                                      "RAPResidentBatch", "RAPResidentMTile")
    for tile in range(n0 * n1):
        peeked = run_scalar(peek, {str(sgpr("NumWorkGroups0")): n0, str(sgpr("NumWorkGroups1")): n1,
                                   str(sgpr("NextTile")): tile})
        assert peeked[str(sgpr("RAPResidentMTile"))] == tile % n0
