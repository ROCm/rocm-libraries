# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""DPP store-coalesce batch-sizing unit tests.

Two pure helpers decide how the fused-store elements are grouped into store batches so the
DPP coalesce (which joins 4 M-adjacent blocks = two M-pairs into one all-lanes store) always
has complete groups co-resident in a batch:

* ``_coalesceBatchFloorMult4`` floors the batch to a multiple of 4 (>=4). A batch that is not
  a multiple of 4 strands the trailing <4 remainder, which coalesces with nothing (measured:
  MT256x256 at batchLen 6 -> WA 1.15 vs 1.00 at 4/8/16).
* ``_dppCoalesceBatchBounds`` splits the element list into (start, stop) sub-batches that never
  straddle an N-column and isolate the orphan tail when MIWaveTile[0] % 4 != 0 (e.g. the two
  orphan M-tiles of a MIWT[0]=10 / MT320x256 kernel).

Both are driven here in isolation -- no GPU, no kernel build -- against synthetic inputs.
"""

import pytest

from tensilelite.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit


# ── _coalesceBatchFloorMult4 ──────────────────────────────────────────────────

def test_floor_rounds_down_to_multiple_of_4():
    # 6 = one 4-block group + a stranded pair -> drop the stranded pair.
    assert KernelWriterAssembly._coalesceBatchFloorMult4(6, 64) == 4
    # 10 -> 8 (two groups), 15 -> 12, 7 -> 4.
    assert KernelWriterAssembly._coalesceBatchFloorMult4(10, 64) == 8
    assert KernelWriterAssembly._coalesceBatchFloorMult4(15, 64) == 12
    assert KernelWriterAssembly._coalesceBatchFloorMult4(7, 64) == 4


def test_floor_leaves_exact_multiples_of_4_unchanged():
    for n in (4, 8, 12, 16):
        assert KernelWriterAssembly._coalesceBatchFloorMult4(n, 64) == n


def test_floor_below_one_group_uses_coalesce_minimum():
    # Below 4 the sizer would strand everything; floor UP to one full group (min with
    # the element count, so a tiny tile is not over-batched).
    assert KernelWriterAssembly._coalesceBatchFloorMult4(1, 64) == 4
    assert KernelWriterAssembly._coalesceBatchFloorMult4(2, 64) == 4
    assert KernelWriterAssembly._coalesceBatchFloorMult4(3, 64) == 4
    # Fewer than 4 elements total: cannot reach a full group, so cap at the element count.
    assert KernelWriterAssembly._coalesceBatchFloorMult4(2, 3) == 3
    assert KernelWriterAssembly._coalesceBatchFloorMult4(1, 2) == 2


def test_floor_never_raises_the_batch_size():
    # Rounding DOWN is the invariant that guarantees no VGPR overflow: for nepb >= 4 the
    # result is always <= nepb.
    for nepb in range(4, 40):
        assert KernelWriterAssembly._coalesceBatchFloorMult4(nepb, 64) <= nepb


# ── _dppCoalesceBatchBounds ───────────────────────────────────────────────────

def _ncol(n, count):
    """`count` synthetic elements all in N-column `n` (only element[i][0] is read)."""
    return [(n,)] * count


def test_bounds_isolate_orphan_tail_miwt10():
    # One N-column of 10 M-tiles (MIWT[0]=10): two complete DPP groups (8) + a 2-tile orphan.
    # With nepb=8 the split must be (0,8) + (8,10) -- orphan in its own sub-batch.
    assert KernelWriterAssembly._dppCoalesceBatchBounds(_ncol(0, 10), 8) == [(0, 8), (8, 10)]


def test_bounds_orphan_isolated_regardless_of_nepb():
    el = _ncol(0, 10)
    # nepb=4 -> (0,4),(4,8),(8,10); nepb=20 (> N-col) -> still caps the clean part at 8.
    assert KernelWriterAssembly._dppCoalesceBatchBounds(el, 4) == [(0, 4), (4, 8), (8, 10)]
    assert KernelWriterAssembly._dppCoalesceBatchBounds(el, 20) == [(0, 8), (8, 10)]


def test_bounds_clean_miwt8_column_not_split_by_orphan_logic():
    # 8 M-tiles (MIWT[0]=8): exactly two groups, no orphan -> a single (0,8) at nepb=8.
    assert KernelWriterAssembly._dppCoalesceBatchBounds(_ncol(0, 8), 8) == [(0, 8)]
    # nepb=4 just halves it into two clean groups.
    assert KernelWriterAssembly._dppCoalesceBatchBounds(_ncol(0, 8), 4) == [(0, 4), (4, 8)]


def test_bounds_never_straddle_an_n_column():
    # Two N-columns of 10 each; a sub-batch must never cross the N=0 -> N=1 boundary at idx 10.
    el = _ncol(0, 10) + _ncol(1, 10)
    bounds = KernelWriterAssembly._dppCoalesceBatchBounds(el, 8)
    assert bounds == [(0, 8), (8, 10), (10, 18), (18, 20)]
    # No bound spans the N-column boundary (10).
    for lo, hi in bounds:
        assert not (lo < 10 < hi)


def test_bounds_cover_every_element_contiguously():
    for cols in [((0, 10),), ((0, 8),), ((0, 10), (1, 6)), ((0, 4), (1, 10), (2, 2))]:
        el = []
        for n, c in cols:
            el += _ncol(n, c)
        bounds = KernelWriterAssembly._dppCoalesceBatchBounds(el, 8)
        # contiguous, non-overlapping, exhaustive
        assert bounds[0][0] == 0
        assert bounds[-1][1] == len(el)
        for (a_lo, a_hi), (b_lo, b_hi) in zip(bounds, bounds[1:]):
            assert a_hi == b_lo
            assert a_lo < a_hi
