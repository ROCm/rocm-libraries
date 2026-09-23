# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Adversarial tests for LDS liveness analysis and interval-based smem packing.

Companion to ``test_smem_liveness_packing.py`` for PR #8844 ("common LDS
allocation for A/B/CShuffle").  Where the sibling file checks the intended happy
paths, this file stress-tests the *safety* invariant that actually protects the
kernel from data corruption:

    Two allocations whose live intervals OVERLAP must be assigned DISJOINT byte
    ranges in the shared @smem_pool.  If overlapping allocations ever share a
    byte, one kernel write silently clobbers another's live data.

Everything here is CPU-only: it drives ``_Lowerer._collect_smem`` /
``_collect_smem_liveness`` / ``_compute_smem_layout`` directly (no GPU, no
compilation), matching the style of the sibling test file.

Tests 5 and 7 originally exposed two real packer bugs (nested-loop liveness
under-extension and the slot-expansion collision).  Both are now fixed
(``_collect_smem_liveness`` extends inner-loop uses to the enclosing loop end,
and ``_compute_smem_layout`` rejects a slot-reuse candidate whose expanded range
would overlap a still-live slot), so these tests are now hard regression guards.
"""

from __future__ import annotations

import random

import pytest

from rocke.core.ir import (
    BF16,
    BF8E5M2,
    F16,
    F32,
    FP8E4M3,
    I8,
    KernelDef,
    Op,
    Region,
    SmemType,
    Value,
)
from rocke.core.lower_llvm import _Lowerer


# ---------------------------------------------------------------------------
# helpers (mirrors test_smem_liveness_packing.py)
# ---------------------------------------------------------------------------


def _smem(elem, *shape: int) -> SmemType:
    return SmemType(elem, list(shape))


def _make_op(name: str, operands=(), results=()):
    op = Op(name=name, operands=list(operands), results=list(results))
    for r in results:
        r.op = op
    return op


def _alloc(name: str, stype: SmemType):
    v = Value(name=name, type=stype)
    op = _make_op("tile.smem_alloc", results=[v])
    return op, v


def _use(*vals: Value) -> Op:
    return _make_op("tile.lds_store", operands=list(vals))


def _for(*ops: Op) -> Op:
    """A minimal scf.for whose body is the given ops."""
    return Op(
        name="scf.for",
        operands=[],
        results=[],
        regions=[Region(label="body", ops=list(ops))],
    )


def _lowerer(*ops: Op) -> _Lowerer:
    body = Region(label="body", ops=list(ops))
    kernel = KernelDef(name="k", params=[], body=body)
    low = _Lowerer(kernel)
    low._collect_smem(body)
    return low


def _pack(*ops: Op):
    """Run the full layout pass; return ({short_name: offset}, pool_size)."""
    low = _lowerer(*ops)
    low._compute_smem_layout()
    offsets = {
        g.lstrip("@").split(".")[0]: low._smem_offsets[g] for g in low._smem_offsets
    }
    return offsets, low._smem_pool_size


def _overlaps(o1: int, s1: int, o2: int, s2: int) -> bool:
    """True iff byte ranges [o1,o1+s1) and [o2,o2+s2) intersect."""
    return o1 < o2 + s2 and o2 < o1 + s1


def _intervals_interfere(a, b) -> bool:
    """Closed-interval interference, matching the packer's reuse rule.

    The packer reuses a slot only when ``slot_last < new_first`` (strict), so
    two allocations interfere (cannot share a byte) exactly when their closed
    live intervals [first, last] intersect.
    """
    return a[0] <= b[1] and b[0] <= a[1]


# byte sizes
_KB = 1024
_4KB = 4 * _KB  # f32 [32x32]
_8KB = 8 * _KB  # f16 [64x64]
_12KB = 12 * _KB  # f32 [128x24]
_16KB = 16 * _KB  # i8  [128x128]
_64KB = 64 * _KB  # f32 [128x128] (CShuffle)


def _round16(n: int) -> int:
    return (n + 15) & ~15


# ===========================================================================
# 1. Non-interference safety — the core anti-corruption invariant
# ===========================================================================


def test_overlapping_allocs_get_disjoint_byte_ranges():
    """CORE INVARIANT: allocations with overlapping live intervals never
    share a byte.

    Two allocations that are both read by the same op are simultaneously live;
    the packer must give them disjoint byte ranges, otherwise one kernel store
    corrupts the other's data.  This is the single most important safety
    property of the shared-LDS packer.
    """
    # A and B are both live at the shared use -> must not overlap.
    op_a, va = _alloc("%A", _smem(F32, 128, 24))  # 12KB
    op_b, vb = _alloc("%B", _smem(F32, 128, 24))  # 12KB
    op_u = _use(va, vb)
    offsets, pool = _pack(op_a, op_b, op_u)
    assert not _overlaps(
        offsets["A"], _12KB, offsets["B"], _12KB
    ), "A and B are simultaneously live and MUST occupy disjoint byte ranges"
    # With two mutually interfering 12KB tiles the pool holds both.
    assert pool == _round16(2 * _12KB)

    # Three mutually-interfering tiles: every pair must be disjoint.
    op_a, va = _alloc("%A", _smem(F16, 64, 64))  # 8KB
    op_b, vb = _alloc("%B", _smem(F16, 64, 64))  # 8KB
    op_c, vc = _alloc("%C", _smem(F16, 64, 64))  # 8KB
    op_u = _use(va, vb, vc)
    offsets, pool = _pack(op_a, op_b, op_c, op_u)
    for n1, n2 in (("A", "B"), ("A", "C"), ("B", "C")):
        assert not _overlaps(
            offsets[n1], _8KB, offsets[n2], _8KB
        ), f"{n1} and {n2} are simultaneously live and must be disjoint"
    assert pool == _round16(3 * _8KB)


# ===========================================================================
# 2. Reuse actually happens for non-overlapping intervals
# ===========================================================================


def test_nonoverlapping_allocs_reuse_same_offset():
    """Allocations whose live ranges are strictly ordered (one fully before the
    other) share the SAME base offset, and the pool size collapses to
    max(size), not sum(size)."""
    # A is used only in the first phase; C only in the second phase.
    op_a, va = _alloc("%A", _smem(F16, 64, 64))  # 8KB, live [0,1]
    op_u1 = _use(va)
    op_c, vc = _alloc("%C", _smem(F16, 64, 64))  # 8KB, live [2,3]
    op_u2 = _use(vc)
    offsets, pool = _pack(op_a, op_u1, op_c, op_u2)
    assert offsets["A"] == 0
    assert offsets["C"] == 0, "C's interval is after A's -> C must reuse A's slot"
    assert pool == _round16(_8KB), "pool must collapse to max(size), not sum(size)"


# ===========================================================================
# 3. A/B/C GEMM shape — the pattern the PR actually optimizes
# ===========================================================================


def test_abc_gemm_shape_pool_is_max_not_sum():
    """Model the real GEMM: A and B live inside the K-loop, CShuffle live only
    after the loop.

    Expected: pool == max(A+B, C) (A and B coexist in the loop, C reuses the
    freed A/B region), and A and B — both live in the loop — never alias.
    """
    op_a, va = _alloc("%A", _smem(F32, 128, 24))  # 12KB
    op_b, vb = _alloc("%B", _smem(F32, 128, 24))  # 12KB
    # K-loop reads both A and B every iteration.
    loop = _for(_use(va, vb))
    # CShuffle after the loop.
    op_c, vc = _alloc("%C", _smem(F32, 128, 128))  # 64KB
    op_cu = _use(vc)
    offsets, pool = _pack(op_a, op_b, loop, op_c, op_cu)

    ab_total = _12KB + _12KB
    assert not _overlaps(
        offsets["A"], _12KB, offsets["B"], _12KB
    ), "A and B are both live inside the K-loop and must never alias"
    assert pool == _round16(max(ab_total, _64KB)), "pool must be max(A+B, C), not A+B+C"
    assert pool == _round16(_64KB)


# ===========================================================================
# 4. Alignment
# ===========================================================================


@pytest.mark.parametrize("byte_elem", [I8, FP8E4M3, BF8E5M2])
def test_byte_elem_allocs_are_16b_aligned(byte_elem):
    """Byte-element (i8/fp8/bf8) allocations get 16-byte-aligned offsets;
    16-bit (f16/bf16) allocations get >=4-byte-aligned offsets; and the whole
    pool size is a multiple of 16."""
    op_a, va = _alloc("%A", _smem(byte_elem, 128, 128))  # 16KB, align 16
    op_b, vb = _alloc("%B", _smem(F16, 64, 64))  # 8KB, align 4
    op_u = _use(va, vb)
    offsets, pool = _pack(op_a, op_b, op_u)
    assert offsets["A"] % 16 == 0, f"{byte_elem.name} alloc must be 16-byte aligned"
    assert offsets["B"] % 4 == 0, "f16 alloc must be >=4-byte aligned"
    assert pool % 16 == 0, "pool size must be a multiple of 16"


@pytest.mark.parametrize("half_elem", [F16, BF16])
def test_byte_alloc_after_odd_half_alloc_is_16b_aligned(half_elem):
    """Mixed-dtype pool: a 16-bit alloc whose size is NOT a multiple of 16
    precedes an interfering byte alloc.  The byte alloc must still land on a
    16-byte boundary (padding is inserted), never on the raw end of the f16
    region."""
    # f16 [130] = 260 bytes -> 260 % 16 == 4, so a naive bump would misalign.
    op_h, vh = _alloc("%H", _smem(half_elem, 130))  # 260 bytes, align 4
    op_b, vb = _alloc("%B", _smem(I8, 4096))  # 4KB, align 16
    op_u = _use(vh, vb)  # both live -> byte alloc opens a fresh, aligned slot
    offsets, pool = _pack(op_h, op_b, op_u)
    assert offsets["H"] % 4 == 0
    assert offsets["B"] % 16 == 0, "byte alloc must be 16-byte aligned even after f16"
    assert offsets["B"] >= 260, "byte alloc must sit past the live f16 region"
    assert pool % 16 == 0


# ===========================================================================
# 5. Nested-loop conservative liveness (adversarial) — XFAIL, real bug
# ===========================================================================


def test_nested_loop_inner_alloc_interferes_with_outer_body_alloc():
    """Outer scf.for { inner scf.for { use X } ; use Y }.

    X is used only in the inner loop; Y only in the outer body after the inner
    loop.  Within a single outer iteration X's use precedes Y's, so naively they
    do not overlap.  But because the outer loop repeats, on iteration N+1 the
    inner loop re-reads (re-produces/consumes) X while Y from iteration N may
    still be live -> X and Y are loop-carried interfering and MUST get disjoint
    byte ranges.

    Regression guard for the nested-loop liveness fix: ``_collect_smem_liveness``
    now extends an inner-loop use to the enclosing loop's end, so X's live range
    reaches the outer loop end and interferes with Y.
    """
    op_x, vx = _alloc("%X", _smem(F16, 64, 64))  # 8KB
    inner = _for(op_x, _use(vx))
    op_y, vy = _alloc("%Y", _smem(F16, 64, 64))  # 8KB
    outer = _for(inner, op_y, _use(vy))
    offsets, pool = _pack(outer)
    # Correct (conservative) behavior: X and Y interfere -> disjoint.
    assert not _overlaps(offsets["X"], _8KB, offsets["Y"], _8KB), (
        "X (inner-loop) and Y (outer-body) are loop-carried interfering and "
        "must not share LDS"
    )


# ===========================================================================
# 6. Fallback — single alloc, zero allocs
# ===========================================================================


def test_single_alloc_offset_zero():
    """A lone *referenced* allocation lands at offset 0 with a 16-rounded pool."""
    op_a, va = _alloc("%A", _smem(F32, 128, 24))  # 12KB
    op_u = _use(va)
    offsets, pool = _pack(op_a, op_u)
    assert offsets["A"] == 0
    assert pool == _round16(_12KB)


def test_zero_allocs_pool_is_zero():
    """A kernel with no smem allocations produces a zero-size pool."""
    low = _lowerer(_make_op("arith.constant"))
    low._compute_smem_layout()
    assert low._smem_pool_size == 0


# ===========================================================================
# 7. Property / fuzz — the strongest anti-corruption guarantee
# ===========================================================================


def _pack_from_intervals(sizes, intervals, elems=None):
    """Drive the REAL packer with caller-supplied sizes and live intervals.

    Builds one smem_alloc per entry (so ``_smem_globals`` is populated the same
    way real IR would), then monkeypatches ``_collect_smem_liveness`` to return
    the requested intervals.  ``_compute_smem_layout`` then runs unchanged.
    Returns the list of assigned byte offsets in input order.
    """
    n = len(sizes)
    if elems is None:
        elems = [I8] * n  # i8 -> 1 byte/elem, exact byte size via shape [size]
    ops = []
    gnames = []
    for i, (elem, sz) in enumerate(zip(elems, sizes)):
        eb = {"i8": 1, "f16": 2, "f32": 4}[elem.name]
        assert sz % eb == 0
        op, _ = _alloc(f"%a{i}", _smem(elem, sz // eb))
        ops.append(op)
        gnames.append(f"@a{i}.k")
    low = _lowerer(*ops)
    live = {gnames[i]: tuple(intervals[i]) for i in range(n)}
    # Return (intervals, used): every allocation with an injected interval is
    # considered referenced, so the packer places all of them (this harness
    # tests interval packing, not dead-allocation stripping).
    low._collect_smem_liveness = lambda region, _live=live: (dict(_live), set(_live))
    low._compute_smem_layout()
    return [low._smem_offsets[g] for g in gnames]


def test_fuzz_deterministic_expansion_collision():
    """Deterministic regression guard for the slot-expansion collision.

    A(12KB, live[0,3]), B(12KB, live[1,10]), C(64KB, live[4,10]).
    When C is packed, A's slot (offset 0) is free (A dead at seq>3).  The old
    packer reused it and expanded the slot to 64KB WITHOUT checking that B — still
    live at offset 12KB — lay inside the expanded range, so C and B (which
    interfere) ended up sharing bytes [12KB, 24KB).  The fixed packer rejects that
    reuse candidate and places C so it stays disjoint from B.
    """
    sizes = [_12KB, _12KB, _64KB]
    intervals = [(0, 3), (1, 10), (4, 10)]
    offs = _pack_from_intervals(sizes, intervals)
    # B and C interfere -> must be disjoint.
    assert _intervals_interfere(intervals[1], intervals[2])
    assert not _overlaps(offs[1], sizes[1], offs[2], sizes[2]), (
        f"slot-expansion collision: B@{offs[1]}(+{sizes[1]}) overlaps "
        f"C@{offs[2]}(+{sizes[2]}) though intervals {intervals[1]} & "
        f"{intervals[2]} interfere"
    )


def test_fuzz_overlapping_allocs_never_share_bytes():
    """UNIVERSAL INVARIANT (fuzz): across 200 randomized allocation sets, every
    pair of allocations whose live intervals overlap is assigned disjoint byte
    ranges.

    This is the strongest guarantee that the packer never lets one allocation's
    stores corrupt another's live data.  Uses a fixed seed for determinism.
    """
    rng = random.Random(1234)
    elem_choices = [(I8, 1), (F16, 2), (F32, 4)]
    violations = []

    for trial in range(200):
        n = rng.randint(2, 6)
        elems = []
        sizes = []
        for _ in range(n):
            elem, eb = rng.choice(elem_choices)
            elems.append(elem)
            sizes.append(rng.randint(1, 20) * 1024)  # multiple of 4 -> valid for all
        intervals = []
        for _ in range(n):
            s = rng.randint(0, 20)
            e = s + rng.randint(0, 20)
            intervals.append((s, e))

        offs = _pack_from_intervals(sizes, intervals, elems=elems)

        for i in range(n):
            for j in range(i + 1, n):
                if _intervals_interfere(intervals[i], intervals[j]) and _overlaps(
                    offs[i], sizes[i], offs[j], sizes[j]
                ):
                    violations.append(
                        {
                            "trial": trial,
                            "i": (intervals[i], offs[i], sizes[i]),
                            "j": (intervals[j], offs[j], sizes[j]),
                        }
                    )

    assert not violations, (
        f"{len(violations)} interfering allocation pairs were assigned "
        f"OVERLAPPING byte ranges (LDS corruption). First: {violations[0]}"
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
