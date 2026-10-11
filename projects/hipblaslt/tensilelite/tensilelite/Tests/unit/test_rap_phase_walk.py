# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ReuseAcrossPersistent walks each workgroup's tiles in phases.

A workgroup owns tiles rank + k*grid. RAP keeps the A of one M-tile resident,
so the order matters: the plain stride changes M-tile at almost every tile once
grid is not a multiple of the period of the (remapped) M-tile along the cursor.
StaticGrid.rapPhaseWalkInit picks a stride that keeps the M-tile and an order of
phases; these tests run the emitted init and per-tile advance on the CPU and
check them against a model of the same plan, and check what the plan is for:
the same tiles per workgroup, every tile exactly once, and one M-tile per phase.
"""

import itertools
import re

import pytest
from rocisa.container import sgpr

from Tensile.Components.TileProcessingStrategy import DataParallel
from Tensile.Components.WorkAssignment import StaticGrid
from scalar_exec_helpers import run_scalar
from test_rap_tile_identity import _Writer, _cpu, _kernel, _wgm_field, wgm_map  # noqa: F401  (_cpu is a fixture)

pytestmark = pytest.mark.unit

_PARTITION = DataParallel().staticPartition()
_CURSOR, _BOUND, _GRID = str(sgpr(_PARTITION.cursor)), str(sgpr(_PARTITION.bound)), str(sgpr(_PARTITION.grid))


def _extended_gcd(a, b):
    oldR, r, oldS, s = a, b, 1, 0
    while r:
        q = oldR // r
        oldR, r = r, oldR - q * r
        oldS, s = s, oldS - q * s
    return oldR, oldS


def decoded(wgm, wgmxcc):
    """The WGM the kernel reads back: the field is 10 bits wide when WorkGroupMappingXCC is -1."""
    bits = 10 if wgmxcc == -1 else 16
    value = wgm & ((1 << bits) - 1)
    return value - (1 << bits) if value >> (bits - 1) else value


def phase_plan(grid, tiles, n0, n1, wgm, live):
    """(stride, step) the kernel should derive; the plain walk is (grid, 0)."""
    period, negative = n0, False
    if live and wgm > 1:
        period = n0 * min(wgm, n1)
    elif live and wgm < -1:
        negative = True
        height = min(-wgm, n0)
        if n0 % height:
            return grid, 0
        period = n0 * height
    if n0 == 1 or period > 0xFFFFFF:
        return grid, 0
    gcd, coeff = _extended_gcd(grid, period)
    phases = period // gcd
    if tiles > n0 * n1:
        sparse = 2 * phases > -(-(n0 * n1) // grid)
    elif negative:
        sparse = 2 * phases > -(-tiles // grid)
    else:
        sparse = phases > 2 * -(-tiles // grid)
    if sparse:
        return grid, 0
    return phases * grid, (coeff % phases) * grid


def walk(rank, tiles, stride, step):
    order, cursor, offset = [], rank, 0
    while cursor < tiles:
        order.append(cursor)
        cursor += stride
        while cursor >= tiles:
            offset = (offset + step) % stride
            if offset == 0:
                return order
            cursor = rank + offset
    return order


def _emit(kernel):
    writer = _Writer()
    init = StaticGrid().rapPhaseWalkInit(writer, kernel, _PARTITION)
    advance = StaticGrid().rapPhaseWalkAdvance(writer, kernel, _PARTITION)
    return init, advance


def _run_walk(init, advance, rank, grid, tiles, n0, n1, field):
    regs = run_scalar(init, {
        _CURSOR: rank, _GRID: grid, _BOUND: tiles,
        str(sgpr("NumWorkGroups0")): n0, str(sgpr("NumWorkGroups1")): n1,
        str(sgpr("WGM")): field,
    })
    packed = regs[str(sgpr("RAPPhasePlan"))]
    plan = ((packed >> 16) * grid, (packed & 0xFFFF) * grid)
    order = []
    while regs[_CURSOR] < tiles:
        order.append(regs[_CURSOR])
        assert len(order) <= tiles, "the walk does not terminate"
        regs = run_scalar(advance, regs)
    assert regs[_CURSOR] == tiles, "an exhausted walk leaves the cursor at the bound"
    return plan, order


# (NumWorkGroups0, NumWorkGroups1, batches, grid, WGM)
_CASES = [
    (3, 16, 1, 256, 1), (3, 16, 1, 32, 1), (3, 16, 1, 32, 8), (3, 16, 1, 31, 8),
    (3, 16, 2, 32, 8), (5, 10, 1, 32, 4), (6, 8, 1, 17, 1), (6, 8, 1, 17, 16),
    (6, 8, 2, 20, 3), (2, 16, 1, 32, 8), (4, 9, 1, 23, 3), (1, 40, 1, 7, 8),
    (3, 37, 1, 16, 8), (3, 37, 1, 16, 1000), (5, 12, 1, 9, -5), (6, 7, 1, 13, -3),
    (6, 7, 1, 13, -4), (6, 7, 1, 13, -1), (6, 7, 1, 13, 0), (3, 8, 1, 64, 8),
    (3, 8, 1, 1, 8), (7, 5, 2, 11, -2), (4, 25, 1, 30, 24), (3, 300, 1, 255, 300),
]


@pytest.mark.parametrize("wgmxcc", [-1, 1])
@pytest.mark.parametrize("live", [True, False], ids=["runtime_wgm", "no_runtime_wgm"])
@pytest.mark.parametrize("n0, n1, batches, grid, wgm", _CASES)
def test_phase_walk_matches_the_plan_and_keeps_each_workgroups_tiles(
    _cpu, n0, n1, batches, grid, wgm, live, wgmxcc
):
    tiles = n0 * n1 * batches
    init, advance = _emit(_kernel(wgmxcc, supportCustomWGM=live))
    stride, step = phase_plan(grid, tiles, n0, n1, decoded(wgm, wgmxcc), live)
    assert step < stride
    seen = []
    for rank in range(min(grid, tiles)):
        plan, order = _run_walk(init, advance, rank, grid, tiles, n0, n1, _wgm_field(wgm, wgmxcc))
        assert plan == (stride, step)
        assert order == walk(rank, tiles, stride, step)
        assert sorted(order) == list(range(rank, tiles, grid)), "the walk changed which tiles a workgroup owns"
        seen.extend(order)
    assert sorted(seen) == list(range(tiles)), "some tile is not processed exactly once"


def _m_tile(tile, n0, n1, wgm):
    batch, inBatch = divmod(tile, n0 * n1)
    return batch, wgm_map(inBatch % n0, inBatch // n0, n0, n1, wgm)[0]


# Full WGM blocks and one batch: every phase reads a single M-tile, and with
# WGM > 1 the phases come in M-tile order, wrapping around once.
@pytest.mark.parametrize("n0, n1, grid, wgm", [
    (3, 16, 32, 1), (3, 16, 32, 8), (3, 128, 31, 8), (5, 64, 32, 4), (6, 64, 20, 16),
    (6, 8, 20, 4), (3, 32, 24, 16), (4, 66, 23, 3), (6, 80, 13, -3), (5, 90, 9, -5),
])
def test_each_phase_reads_one_m_tile(_cpu, n0, n1, grid, wgm):
    tiles = n0 * n1
    init, advance = _emit(_kernel(-1))
    stride, step = phase_plan(grid, tiles, n0, n1, wgm, True)
    assert stride > grid, "this case is meant to take the phased walk"
    for rank in range(min(grid, tiles)):
        _, order = _run_walk(init, advance, rank, grid, tiles, n0, n1, _wgm_field(wgm, -1))
        phases = {}
        for tile in order:
            phases.setdefault((tile - rank) % stride, []).append(_m_tile(tile, n0, n1, wgm))
        for offset, mTiles in phases.items():
            assert len(set(mTiles)) == 1, "rank %d, phase at +%d reads %s" % (rank, offset, mTiles)
        if wgm > 1:
            visited = [mTiles[0][1] for mTiles in phases.values()]
            drops = sum(later < earlier for earlier, later in zip(visited, visited[1:]))
            assert drops <= 1, "rank %d visits M-tiles out of order: %s" % (rank, visited)


def test_the_plain_stride_reloads_where_the_phased_walk_does_not(_cpu):
    """The case that motivated the walk: grid 32 against NumWorkGroups0 3, six tiles each."""
    n0, n1, grid = 3, 64, 32
    tiles = n0 * n1
    init, advance = _emit(_kernel(-1))

    def reloads(order):
        mTiles = [_m_tile(tile, n0, n1, 1) for tile in order]
        return 1 + sum(a != b for a, b in zip(mTiles, mTiles[1:]))

    phased = plain = 0
    for rank in range(grid):
        _, order = _run_walk(init, advance, rank, grid, tiles, n0, n1, _wgm_field(1, -1))
        phased += reloads(order)
        plain += reloads(list(range(rank, tiles, grid)))
    assert (plain, phased) == (tiles, 3 * grid)


def test_the_walk_keeps_to_its_sgpr_budget(_cpu):
    """Two sgprs for the whole kernel, and four temporaries at kernel start.

    The init runs with every persistent register and kernel argument live, where
    a RAP kernel's sgpr high-water mark tends to be, so whatever it borrows there
    adds to the kernel's sgpr count one for one.
    """
    writer = _Writer()
    first = writer.sgprPool.next
    init = StaticGrid().rapPhaseWalkInit(writer, _kernel(-1), _PARTITION)
    assert writer.sgprPool.next - first <= 4
    advance = StaticGrid().rapPhaseWalkAdvance(writer, _kernel(-1), _PARTITION)
    assert set(re.findall(r"sgpr(RAPPhase\w+)", str(init) + str(advance))) == {"RAPPhasePlan", "RAPPhaseStart"}


def test_non_rap_activation_keeps_the_plain_stride():
    """Only RAP kernels carry the phase walk; every other StaticGrid kernel is unchanged."""
    from rocisa.code import Module

    # Stubbed on the instance, not in a subclass: Component registers every
    # subclass as an implementation, and a second DataParallel would make
    # TileProcessingStrategy.find fail for every later test in the process.
    processing = DataParallel()
    processing.materializeTile = lambda *args, **kwargs: Module("materialize stub")
    writer = _Writer()
    rendered = str(StaticGrid().activateReservedOrAcquire(
        writer, {"ReuseAcrossPersistent": 0, "SpaceFillingAlgo": []}, processing, None, None))
    lines = [line.split("//")[0].strip() for line in rendered.splitlines()
             if line.strip() and not line.strip().startswith("/*")]
    assert lines == [
        "s_add_u32 s[sgprNextTile], s[sgprNextTile], s[sgprPersistentGrid]",
        "s_cselect_b32 s[sgprNextTile], s[sgprTotalTiles], s[sgprNextTile]",
    ]
