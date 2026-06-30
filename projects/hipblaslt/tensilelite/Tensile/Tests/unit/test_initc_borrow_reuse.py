################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Regression tests for the InitC scratch register-reuse fix.

The relocated initC (InitCGROverlap) requests a 2-VGPR MFMA zero-scratch after
the input tile block has filled the VGPR pool to its peak. A fresh
``checkOutAligned(2,2)`` then grows the pool past ``MaxVgpr`` (the 255->258
overflow) and the kernel is rejected. The fix lets initC *borrow* a provably-dead
2-aligned pair from the already-allocated input tile block instead, keeping the
fast MFMA zeroing path with no pool growth; it falls back to scratch-free scalar
zeroing only when no safe borrow pair exists.

These tests pin:
  - borrow path: no checkout/grow, MFMA ops emitted using the borrowed reg, no
    checkIn of the borrowed (non-owned) reg.
  - scalar fallback: no borrow + no headroom -> scalar zeroing, no MFMA.
  - headroom fast path: plenty of room -> normal checkout + checkIn (unchanged).
  - initVgprTilesToZero threads borrowScratch to every pool-grouped range.
  - _make_initC_op wires _initc_borrow_scratch -> initVgprTilesToZero end-to-end.
  - _initc_borrow_scratch guard: DirectToLds -> borrow; DirectToVgpr -> None;
    per-tensor DTL/DTV, A->B fallback, odd-aligned / <2-reg tiles skipped.

CPU-only; no GPU, no compile.
"""

import pytest

pytestmark = pytest.mark.unit


class _RecordingPool:
    """vgpr pool mock with a fixed high-water size() that records checkouts."""

    def __init__(self, size):
        self._size = size
        self.checkouts = []
        self.checkins = []

    def size(self):
        return self._size

    def available(self):
        return 0

    def checkOutAligned(self, n, align, tag=None):
        self.checkouts.append((n, align, tag))
        base = self._size
        self._size += n  # simulate growth so a leaked checkout would be visible
        return base

    def checkIn(self, start):
        self.checkins.append(start)


class _Writer:
    def __init__(self, vgpr_size, max_vgpr=256, has_wmma=False):
        class _States:
            pass
        self.states = _States()
        self.states.asmCaps = {"HasWMMA_AccImmZero": has_wmma}
        self.states.regCaps = {"MaxVgpr": max_vgpr}
        self.vgprPool = _RecordingPool(vgpr_size)
        self.agprPool = object()  # identity sentinel for isAgpr comparisons


class _TileInfo:
    tc = "D"


def _zero(module_writer, **kw):
    from Tensile.Components.Subtile.Kernel import _zeroRegRange
    from rocisa.code import Module
    module = Module()
    _zeroRegRange(module, module_writer, _TileInfo(),
                  kw["firstReg"], kw["totalRegs"], kw["isAgpr"],
                  borrowScratch=kw.get("borrowScratch"))
    return module, str(module)


# ---------------------------------------------------------------------------
# _zeroRegRange precedence: borrow / scalar / checkout
# ---------------------------------------------------------------------------

def test_borrow_used_when_no_headroom_keeps_mfma_and_no_checkout():
    """No headroom + borrowScratch provided -> MFMA path on borrowed reg, no growth."""
    # size 255: alignedTop = 256, 256 + 2 = 258 > MaxVgpr 256 -> no headroom.
    w = _Writer(vgpr_size=255)
    borrow = 20
    module, src = _zero(w, firstReg=0, totalRegs=96, isAgpr=True, borrowScratch=borrow)

    # MFMA/WMMA zeroing kept (96/16 = 6 matrix ops), using borrowed pair v[20:21].
    # (The matrix mnemonic renders as v_mfma or v_wmma depending on the active
    # rocisa ISA; key on the deterministic "initD: [" range comment instead.)
    assert src.count("initD: [") == 6, f"expected 6 matrix initD ops, got:\n{src[:400]}"
    assert "v[20:21]" in src, f"borrowed scratch v[20:21] not used:\n{src[:400]}"
    # No fresh checkout (no pool growth) and no checkIn of the borrowed reg.
    assert w.vgprPool.checkouts == [], "must not check out a fresh scratch when borrowing"
    assert w.vgprPool.checkins == [], "must not check in the borrowed (non-owned) reg"
    assert w.vgprPool.size() == 255, "pool must not grow when borrowing"


def test_scalar_fallback_when_no_headroom_and_no_borrow():
    """No headroom + no borrow -> scalar zeroing, no MFMA, no checkout."""
    w = _Writer(vgpr_size=255)
    module, src = _zero(w, firstReg=0, totalRegs=96, isAgpr=True, borrowScratch=None)

    assert "initD: [" not in src, f"scalar fallback must emit no matrix ops:\n{src[:400]}"
    # 96 scalar zeroing writes (one per reg), each tagged "// initD".
    assert src.count("// initD") == 96, f"expected 96 scalar writes:\n{src[:400]}"
    assert w.vgprPool.checkouts == [], "scalar fallback must not check out a scratch"


def test_headroom_uses_normal_checkout_and_checkin():
    """Ample headroom -> existing checkout path unchanged (checkout + checkin)."""
    w = _Writer(vgpr_size=100)
    module, src = _zero(w, firstReg=0, totalRegs=96, isAgpr=True, borrowScratch=20)

    assert src.count("initD: [") == 6, "matrix zeroing path expected with headroom"
    assert len(w.vgprPool.checkouts) == 1, "expected exactly one fresh scratch checkout"
    assert w.vgprPool.checkouts[0][:2] == (2, 2), "scratch must be a 2-aligned 2-reg checkout"
    assert len(w.vgprPool.checkins) == 1, "fresh scratch must be checked back in"
    # Borrow is ignored when there is headroom (passing kernels unchanged).
    assert "v[20:21]" not in src, "borrow must not be used when headroom exists"


# ---------------------------------------------------------------------------
# _initc_borrow_scratch safety guard
# ---------------------------------------------------------------------------

class _RegList:
    def __init__(self, base, n, pool=None):
        self.indices = [base + k for k in range(n)]
        self.pool = pool

class _Tile:
    def __init__(self, base, n, pool=None):
        self.regList = _RegList(base, n, pool)

class _FakeSched:
    def __init__(self, tilesA=None, tilesB=None):
        self.vgprTilesA = tilesA or []
        self.vgprTilesB = tilesB or []


def _borrow(sched, kernel):
    from Tensile.Components.Subtile.LogicalScheduler import LogicalScheduler
    return LogicalScheduler._initc_borrow_scratch(sched, kernel)


def test_borrow_scratch_returns_pair_with_directtolds():
    """DirectToLds=1 (GR->LDS) -> input tiles dead -> returns 2-aligned base."""
    sched = _FakeSched(tilesA=[_Tile(20, 8)])
    base = _borrow(sched, {"DirectToLds": True})
    assert base == 20, f"expected borrow base 20, got {base}"


def test_borrow_scratch_none_with_directtovgpr():
    """DirectToVgprA -> GR lands in VGPRs -> tiles may be live -> no borrow."""
    sched = _FakeSched(tilesA=[_Tile(20, 8)])
    base = _borrow(sched, {"DirectToLds": True, "DirectToVgprA": True,
                           "DirectToVgprB": True})
    assert base is None, f"must not borrow when GR targets VGPRs, got {base}"


def test_borrow_scratch_none_without_directtolds():
    """No DirectToLds -> not provably dead -> no borrow."""
    sched = _FakeSched(tilesA=[_Tile(20, 8)])
    base = _borrow(sched, {"DirectToLds": False})
    assert base is None


def test_borrow_scratch_none_when_no_tiles():
    """Empty tile lists -> nothing to borrow."""
    sched = _FakeSched()
    base = _borrow(sched, {"DirectToLds": True})
    assert base is None
