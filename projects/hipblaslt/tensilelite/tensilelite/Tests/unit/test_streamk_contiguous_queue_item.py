# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Codegen tests for the contiguous per-XCD work-item order of
TileProcessingStrategy=StreamK with WorkAssignment=Hybrid (dynamic sub-mode).

Queue q pops item i = cnt * numQueues + q. StreamKHybrid._contiguousQueueItem
renumbers it to c = q * floor(T / numQueues) + min(q, T % numQueues) + cnt
(T = TotalItems), so each XCD runs a contiguous range of tiles, like the
static sub-mode's contiguous WGMXCC remap. The tests check the emitted
instructions, the runtime gate and the register use, and check the formula
itself on a plain Python reference.
"""

import itertools
from types import SimpleNamespace

import pytest

# Prime the component registry before StreamK imports (avoids circular import).
from tensilelite.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401

from rocisa.code import Label, Module, TextBlock
from rocisa.instruction import (
    SAddU32,
    SAndB32,
    SCBranchSCC0,
    SCBranchSCC1,
    SCmpEQU32,
    SCmpGtU32,
    SLShiftRightB32,
    SMinU32,
    SMulI32,
)

from tensilelite.Component import Component
from tensilelite.Components.StreamK import StreamKHybrid

pytestmark = pytest.mark.unit

_NUM_QUEUES = 8
_ITEM = 40  # SGPR holding the popped work item
_KERNEL = {"WorkGroupMappingXCC": -1, "WavefrontSize": 64}


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


def _queues(monkeypatch, numQueues):
    wa = SimpleNamespace(queueConstants=lambda w, k: (numQueues, numQueues - 1, numQueues.bit_length() - 1, 7))
    monkeypatch.setattr(Component.WorkAssignment, "find", classmethod(lambda cls, w, *a, **k: wa))


@pytest.fixture(autouse=True)
def _eight_queues(monkeypatch):
    _queues(monkeypatch, _NUM_QUEUES)


def _emit(kernel=_KERNEL):
    w = _writer()
    items = list(StreamKHybrid()._contiguousQueueItem(w, kernel, _ITEM).flatitems())
    return w, items


def _params(inst):
    return [str(p) for p in inst.getParams()]


def _imm_in(inst, value):
    for p in inst.getParams():
        try:
            if int(str(p), 0) == value:
                return True
        except (TypeError, ValueError):
            continue
    return False


def test_emitted_sequence():
    _, items = _emit()
    assert [type(i) for i in items] == [
        SCmpGtU32, SCBranchSCC1,             # TotalItems > SKGrid
        SCmpEQU32, SCBranchSCC0,             # SKTiles == 0
        SAndB32, SCmpEQU32, SCBranchSCC0,    # static uses the contiguous remap
        SAndB32, SLShiftRightB32,            # q, cnt
        SLShiftRightB32, SMulI32, SAddU32,   # + q * floor(T / numQueues)
        SAndB32, SMinU32, SAddU32,           # + min(q, T % numQueues)
        Label,
    ]


def test_gate_branches_to_the_end():
    _, items = _emit()
    done = items[-1].getLabelName()
    branches = [i for i in items if isinstance(i, (SCBranchSCC0, SCBranchSCC1))]
    assert len(branches) == 3
    assert all(_params(b)[0] == done for b in branches)


def test_gate_operands():
    _, items = _emit()
    rounds, tiles, wgm, mode = items[0], items[2], items[4], items[5]
    assert _params(rounds) == ["s[sgprTotalItems]", "s[sgprSKGrid]"]
    assert isinstance(items[1], SCBranchSCC1), "more items than workgroups: skip"
    assert _params(tiles)[0] == "s[sgprSKTiles]" and _imm_in(tiles, 0)
    assert isinstance(items[3], SCBranchSCC0), "split tiles: skip"
    # WGM bits: WGMXCC [13:10], chunk [21:14] with bit 14 masked out.
    assert "s[sgprWGM]" in _params(wgm) and _imm_in(wgm, 0x3FBC00)
    assert _params(mode)[0] == _params(wgm)[0] and _imm_in(mode, _NUM_QUEUES << 10)
    assert isinstance(items[6], SCBranchSCC0), "other WGMXCC modes: skip"


def test_queue_and_count_immediates():
    _, items = _emit()
    q, cnt, floorT, modT = items[7], items[8], items[9], items[12]
    assert _params(q)[1] == "s%d" % _ITEM and _imm_in(q, _NUM_QUEUES - 1)
    assert _params(cnt)[:2] == ["s%d" % _ITEM, "s%d" % _ITEM] and _imm_in(cnt, 3)
    assert "s[sgprTotalItems]" in _params(floorT) and _imm_in(floorT, 3)
    assert "s[sgprTotalItems]" in _params(modT) and _imm_in(modT, _NUM_QUEUES - 1)
    item, sQ, sTmp = "s%d" % _ITEM, _params(q)[0], _params(floorT)[0]
    assert sQ not in (item, sTmp) and sTmp != item
    assert _params(items[10]) == [sTmp, sQ, sTmp], "q * floor(T / numQueues)"
    assert _params(items[11]) == [item, item, sTmp]
    assert _params(items[13]) == [sTmp, sTmp, sQ], "min(q, T % numQueues)"
    assert _params(items[14]) == [item, item, sTmp], "the renumbered item replaces the popped one"


def test_temporaries_checked_back_in():
    w, _ = _emit()
    assert not w.sgprPool.out


def test_one_queue_emits_nothing(monkeypatch):
    _queues(monkeypatch, 1)
    w, items = _emit()
    assert not items and not w.sgprPool.out


@pytest.mark.parametrize("wgmxcc", [1, 8])
def test_fixed_wgmxcc_emits_nothing(wgmxcc):
    # Only auto-WGMXCC kernels use the runtime contiguous remap on the static sub-mode.
    w, items = _emit({"WorkGroupMappingXCC": wgmxcc, "WavefrontSize": 64})
    assert not items and not w.sgprPool.out


def _reference(i, total, numQueues=_NUM_QUEUES):
    q, cnt = i % numQueues, i // numQueues
    return q * (total // numQueues) + min(q, total % numQueues) + cnt


@pytest.mark.parametrize("total", [1, 7, 8, 9, 63, 256, 300, 8197])
def test_reference_gives_each_queue_a_contiguous_range(total):
    start, seen = 0, []
    for q in range(_NUM_QUEUES):
        mapped = [_reference(i, total) for i in range(q, total, _NUM_QUEUES)]
        assert mapped == list(range(start, start + len(mapped)))
        start += len(mapped)
        seen += mapped
    assert sorted(seen) == list(range(total))


def test_tile_identity_reads_the_renumbered_item(monkeypatch):
    # The renumbering runs first, on the popped item's SGPR, before the tile decode.
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
    text = str(StreamKHybrid()._computeNextTileIdentity(w, _KERNEL, _ITEM))
    assert text.index("renumber s%d" % _ITEM) < text.index("total tiles")
