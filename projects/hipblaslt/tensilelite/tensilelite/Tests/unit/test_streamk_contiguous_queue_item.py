# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Codegen tests for the contiguous per-XCD work-item order (SK5 Hybrid).

The dynamic sub-path pops work item i = cnt * numQueues + q from queue q.
StreamKHybrid._contiguousQueueItem renumbers it to
c = q * floor(T / numQueues) + min(q, T % numQueues) + cnt (T = TotalItems)
when no workgroup gets more than one item and every tile is whole, so each
XCD runs a contiguous range of tiles, as on the static path. These tests run
the emitted SALU on a small interpreter and check the renumbering is a
bijection that gives every queue its contiguous range, and that split tiles
and launches with more items than workgroups keep the interleaved order.
"""

import itertools
from types import SimpleNamespace

import pytest

# Prime the component registry before StreamK imports (avoids circular import).
from Tensile.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401

from rocisa.code import Label
from rocisa.instruction import (
    SAddU32,
    SAndB32,
    SCBranchSCC0,
    SCBranchSCC1,
    SCmpEQU32,
    SCmpGtU32,
    SCmpLtU32,
    SLShiftRightB32,
    SMinU32,
    SMulI32,
)

from Tensile.Component import Component
from Tensile.Components.StreamK import StreamKHybrid

pytestmark = pytest.mark.unit

_NUM_QUEUES = 8
_ITEM = 40  # SGPR the caller's popped work item lives in


class _Pool:
    def __init__(self):
        self._next = 60
        self.out = set()

    def checkOut(self, size, tag="", preventOverflow=True):
        idx = self._next
        self._next += size
        self.out.add(idx)
        return idx

    def checkIn(self, idx):
        self.out.remove(idx)


def _writer():
    counter = itertools.count()
    return SimpleNamespace(
        labels=SimpleNamespace(getNameInc=lambda n: "%s_%d" % (n, next(counter))),
        sgprPool=_Pool(),
    )


@pytest.fixture(autouse=True)
def _queues(monkeypatch):
    wa = SimpleNamespace(queueConstants=lambda w, k: (_NUM_QUEUES, _NUM_QUEUES - 1, 3, 7))
    monkeypatch.setattr(Component.WorkAssignment, "find", classmethod(lambda cls, w, *a, **k: wa))


def _emit():
    w = _writer()
    module = StreamKHybrid()._contiguousQueueItem(w, {"WavefrontSize": 64}, _ITEM)
    assert not w.sgprPool.out, "every temporary is checked back in"
    # PAP emits this near the SGPR high-water mark: two temporaries at most.
    assert w.sgprPool._next - 60 <= 2, "at most two temporary SGPRs"
    return list(module.flatitems())


def _reg(text, regs):
    text = str(text)
    if text.startswith("s[sgpr"):
        return regs[text[len("s[sgpr"):-1]]
    if text.startswith("s"):
        return regs[int(text[1:])]
    return int(text, 0)


def _run(items, item, total, sk_tiles, sk_split, grid=None):
    """Interpret the emitted SALU; return the renumbered item."""
    regs = {_ITEM: item, "TotalItems": total, "SKTiles": sk_tiles, "SKSplit": sk_split,
            "SKGrid": total if grid is None else grid}
    labels = {i.getLabelName(): n for n, i in enumerate(items) if isinstance(i, Label)}
    scc, pc = 0, 0
    while pc < len(items):
        i = items[pc]
        pc += 1
        if isinstance(i, Label):
            continue
        p = [str(x) for x in i.getParams()]
        dst = lambda v: regs.__setitem__(
            p[0][len("s[sgpr"):-1] if p[0].startswith("s[sgpr") else int(p[0][1:]), v & 0xFFFFFFFF)
        if isinstance(i, SCmpLtU32):
            scc = int(_reg(p[0], regs) < _reg(p[1], regs))
        elif isinstance(i, SCmpGtU32):
            scc = int(_reg(p[0], regs) > _reg(p[1], regs))
        elif isinstance(i, SCmpEQU32):
            scc = int(_reg(p[0], regs) == _reg(p[1], regs))
        elif isinstance(i, (SCBranchSCC0, SCBranchSCC1)):
            if scc == (1 if isinstance(i, SCBranchSCC1) else 0):
                pc = labels[p[0]]
        elif isinstance(i, SAndB32):
            dst(_reg(p[1], regs) & _reg(p[2], regs))
        elif isinstance(i, SLShiftRightB32):
            dst(_reg(p[1], regs) >> _reg(p[2], regs))
        elif isinstance(i, SMulI32):
            dst(_reg(p[1], regs) * _reg(p[2], regs))
        elif isinstance(i, SAddU32):
            dst(_reg(p[1], regs) + _reg(p[2], regs))
        elif isinstance(i, SMinU32):
            dst(min(_reg(p[1], regs), _reg(p[2], regs)))
        else:
            raise AssertionError("unexpected instruction %s" % type(i).__name__)
    return regs[_ITEM]


def _queue_items(total):
    # What fetchAndBroadcast hands queue q: cnt * numQueues + q, in pop order.
    return {q: list(range(q, total, _NUM_QUEUES)) for q in range(_NUM_QUEUES)}


@pytest.mark.parametrize("total", [1, 7, 8, 9, 63, 256, 300, 8192, 8197])
@pytest.mark.parametrize("sk_tiles,sk_split", [(0, 2), (0, 1), (0, 16)])
@pytest.mark.parametrize("extra_grid", [0, 5])
def test_contiguous_ranges(total, sk_tiles, sk_split, extra_grid):
    # At most one item per workgroup (grid >= TotalItems).
    items = _emit()
    seen, start = [], 0
    for q, popped in _queue_items(total).items():
        mapped = [_run(items, i, total, sk_tiles, sk_split, total + extra_grid) for i in popped]
        # Queue q runs the next count_q items, in order.
        assert mapped == list(range(start, start + len(popped))), (q, popped[:4], mapped[:4])
        start += len(popped)
        seen += mapped
    assert sorted(seen) == list(range(total)), "a bijection on [0, TotalItems)"


@pytest.mark.parametrize("sk_tiles,sk_split", [(64, 2), (40, 7), (3, 4), (28, 8), (14, 16), (24, 10), (1, 256), (5, 9)])
def test_split_tiles_keep_the_interleaved_order(sk_tiles, sk_split):
    items = _emit()
    total = sk_tiles * sk_split
    for i in range(total):
        assert _run(items, i, total, sk_tiles, sk_split) == i


@pytest.mark.parametrize("total,grid", [(300, 256), (8192, 768), (9, 8)])
def test_more_items_than_workgroups_keep_the_interleaved_order(total, grid):
    # Several rounds: the static path walks the tiles round by round, which
    # the interleaved order follows more closely than one block per XCD.
    items = _emit()
    for i in range(total):
        assert _run(items, i, total, 0, 2, grid) == i


def test_one_queue_emits_nothing(monkeypatch):
    wa = SimpleNamespace(queueConstants=lambda w, k: (1, 0, 0, 7))
    monkeypatch.setattr(Component.WorkAssignment, "find", classmethod(lambda cls, w, *a, **k: wa))
    assert not list(StreamKHybrid()._contiguousQueueItem(_writer(), {}, _ITEM).flatitems())


def test_tile_identity_reads_the_renumbered_item(monkeypatch):
    # The renumbering runs first in _computeNextTileIdentity, on the popped
    # item's SGPR, before the full/partial split reads it.
    from rocisa.code import Module, TextBlock

    def marker(text):
        m = Module(text)
        m.add(TextBlock("// %s\n" % text))
        return m

    monkeypatch.setattr(StreamKHybrid, "_contiguousQueueItem",
                        lambda self, w, k, sidx: marker("renumber s%d" % sidx))
    monkeypatch.setattr(StreamKHybrid, "computeTotalTiles", lambda self, w, k, dst: marker("total tiles"))
    w = _writer()
    w.vgprPool = _Pool()
    w.sgprPool.out.add(_ITEM)  # the caller's popped item, checked in by the helper
    text = str(StreamKHybrid()._computeNextTileIdentity(w, {"WavefrontSize": 64}, _ITEM))
    assert "renumber s%d" % _ITEM in text
    assert text.index("renumber s%d" % _ITEM) < text.index("total tiles")
