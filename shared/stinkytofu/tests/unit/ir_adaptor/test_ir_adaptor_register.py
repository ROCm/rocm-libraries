################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
"""Standalone tests for ``ir_adaptor.register.RegisterPool``.

Run from any working directory:

    python3 shared/stinkytofu/tests/unit/ir_adaptor/test_ir_adaptor_register.py

Or with pytest if available:

    pytest shared/stinkytofu/tests/unit/ir_adaptor/test_ir_adaptor_register.py

The tests assert behaviors that Tensile's KernelWriter implicitly depends on
(AMDGPU ABI, byte-for-byte parity with rocisa's allocator, etc.). Treat any
failure here as a regression that will silently corrupt generated asm.
"""

from __future__ import annotations

import copy
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

# ---------------------------------------------------------------------------
# Self-contained sys.path bootstrap so the test runs without any install /
# editable-mode setup. ir_adaptor lives at:
#     shared/stinkytofu/python_module/ir_adaptor/
# This file lives at:
#     shared/stinkytofu/tests/unit/ir_adaptor/test_ir_adaptor_register.py
# So python_module is three levels up + python_module.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_MODULE = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "python_module"))
if _PY_MODULE not in sys.path:
    sys.path.insert(0, _PY_MODULE)

from ir_adaptor.enum import RegisterType  # noqa: E402
from ir_adaptor.register import RegisterPool  # noqa: E402


# ---------------------------------------------------------------------------
# Status / Register inner types
# ---------------------------------------------------------------------------


class TestStatusEnum(unittest.TestCase):
    """The Status enum values are wire-compatible with the rocisa C++ enum."""

    def test_values_match_rocisa(self):
        self.assertEqual(int(RegisterPool.Status.Unavailable), 0)
        self.assertEqual(int(RegisterPool.Status.Available), 1)
        self.assertEqual(int(RegisterPool.Status.InUse), 2)

    def test_attached_to_pool_class(self):
        self.assertTrue(hasattr(RegisterPool, "Status"))
        self.assertTrue(hasattr(RegisterPool, "Register"))


class TestRegisterRecord(unittest.TestCase):
    def test_construct_and_repr(self):
        r = RegisterPool.Register(RegisterPool.Status.Available, "tagX")
        self.assertEqual(r.status, RegisterPool.Status.Available)
        self.assertEqual(r.tag, "tagX")
        self.assertIn("tagX", repr(r))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction(unittest.TestCase):
    def test_empty_pool(self):
        p = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=True)
        self.assertEqual(p.size(), 0)
        self.assertEqual(p.available(), 0)

    def test_initial_size_marks_unavailable(self):
        p = RegisterPool(4, RegisterType.Vgpr, defaultPreventOverflow=False)
        self.assertEqual(p.size(), 4)
        self.assertEqual(p.available(), 0)
        for r in p.getPool():
            self.assertEqual(r.status, RegisterPool.Status.Unavailable)
            self.assertEqual(r.tag, "init")


# ---------------------------------------------------------------------------
# AMDGPU ABI: KernArgAddress sequence (the assertion that motivates Phase 2)
# ---------------------------------------------------------------------------


class TestKernArgAddressABI(unittest.TestCase):
    """Tensile's _initKernel asserts SGPR0 == KernArgAddress.

    Reproduces the exact call sequence at KernelWriter.py:7456-7466.
    """

    def test_kernarg_address_at_index_0(self):
        pool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=True)

        # KernArgAddress is 2 sgprs (64-bit pointer), no alignment requirement
        # at this call site; preventOverflow=False allows tail-grow.
        idx = pool.checkOutAligned(2, 1, "KernArgAddress", preventOverflow=False)
        self.assertEqual(idx, 0)
        self.assertEqual(pool.size(), 2)

    def test_initial_sgpr_layout(self):
        pool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=True)

        idx_kern = pool.checkOutAligned(2, 1, "KernArgAddress", preventOverflow=False)
        idx_wg0 = pool.checkOutAligned(1, 1, "WorkGroup0", preventOverflow=False)
        idx_wg1 = pool.checkOutAligned(1, 1, "WorkGroup1", preventOverflow=False)

        self.assertEqual(idx_kern, 0)
        self.assertEqual(idx_wg0, 2)
        self.assertEqual(idx_wg1, 3)
        self.assertEqual(pool.size(), 4)


# ---------------------------------------------------------------------------
# checkOut / checkOutAligned algorithm
# ---------------------------------------------------------------------------


class TestCheckOutAligned(unittest.TestCase):
    def test_zero_size_raises(self):
        pool = RegisterPool(4, RegisterType.Sgpr, defaultPreventOverflow=False)
        pool.add(0, 4)
        with self.assertRaises(ValueError):
            pool.checkOutAligned(0, 1, "x")

    def test_first_fit_within_existing_pool(self):
        pool = RegisterPool(8, RegisterType.Sgpr, defaultPreventOverflow=False)
        pool.add(0, 8)
        self.assertEqual(pool.available(), 8)

        idx = pool.checkOutAligned(2, 1, "a")
        self.assertEqual(idx, 0)
        self.assertEqual(pool.available(), 6)

    def test_alignment_grows_with_padding(self):
        """alignment=4 forces a 4-aligned start, padding intermediate slots
        with Available so subsequent allocs can reuse them."""
        pool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False)
        # Take 3 slots first so the pool is at size=3 with [0,1,2] InUse.
        pool.checkOutAligned(1, 1, "x", preventOverflow=False)
        pool.checkOutAligned(1, 1, "y", preventOverflow=False)
        pool.checkOutAligned(1, 1, "z", preventOverflow=False)
        self.assertEqual(pool.size(), 3)

        # Now ask for 4 vgprs aligned to 4: must start at idx 4 (not 3).
        idx = pool.checkOutAligned(4, 4, "valuC", preventOverflow=False)
        self.assertEqual(idx, 4)
        # pool[3] is the padding slot kept as Available for future reuse.
        self.assertEqual(pool._pool[3].status, RegisterPool.Status.Available)
        # pool[4..7] are the actual InUse slots.
        for k in range(4, 8):
            self.assertEqual(pool._pool[k].status, RegisterPool.Status.InUse)

    def test_reuse_after_checkin(self):
        pool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False)
        a = pool.checkOutAligned(2, 1, "A", preventOverflow=False)
        b = pool.checkOutAligned(2, 1, "B", preventOverflow=False)
        self.assertEqual(a, 0)
        self.assertEqual(b, 2)

        pool.checkIn(a)
        c = pool.checkOutAligned(2, 1, "C", preventOverflow=False)
        # First-fit must reclaim the freed [0,1].
        self.assertEqual(c, 0)
        self.assertEqual(pool._pool[0].tag, "C")

    def test_skip_inuse_blocks(self):
        """A run shorter than size should be skipped over by the i+=j hack."""
        pool = RegisterPool(8, RegisterType.Sgpr, defaultPreventOverflow=False)
        pool.add(0, 8)
        # Make a 2-slot hole at [3,4] by allocating non-contiguous tmps.
        pool._pool[2].status = RegisterPool.Status.InUse
        pool._pool[2].tag = "blocker"
        pool._pool[5].status = RegisterPool.Status.InUse
        pool._pool[5].tag = "blocker"
        # Available now: [0,1] [3,4] [6,7]; ask for 3 contiguous.
        idx = pool.checkOutAligned(3, 1, "needs3")
        # Cannot fit at 0,1,2 (2 is blocked); cannot fit at 3,4,5 (5 blocked);
        # must land at 6,7 + tail-grow 1.
        self.assertEqual(idx, 6)
        self.assertEqual(pool.size(), 9)

    def test_prevent_overflow_raises(self):
        pool = RegisterPool(2, RegisterType.Sgpr, defaultPreventOverflow=True)
        pool.add(0, 2)
        # Available = 2; ask for 4 with preventOverflow inherited from default.
        with self.assertRaises(RuntimeError):
            pool.checkOutAligned(4, 1, "x")

    def test_explicit_prevent_overflow_overrides_default(self):
        """preventOverflow=False (the int 0, passed by Tensile's defineSgpr)
        must permit tail-grow even when defaultPreventOverflow=True.

        rocisa's tail-walk loop is ``for (i = size-1; i > 0; --i)`` — index 0
        is intentionally NOT visited, so a 2-slot all-Available pool grown by
        4 lands at start=1 and ends at size=5 (NOT 0 / 4). We mirror this
        quirk 1:1 to keep byte-for-byte parity with the nanobind backend.
        """
        pool = RegisterPool(2, RegisterType.Sgpr, defaultPreventOverflow=True)
        pool.add(0, 2)
        idx = pool.checkOutAligned(4, 1, "x", preventOverflow=False)
        self.assertEqual(idx, 1)
        self.assertEqual(pool.size(), 5)
        # pool[0] remains Available (untouched by the tail-walk).
        self.assertEqual(pool._pool[0].status, RegisterPool.Status.Available)


# ---------------------------------------------------------------------------
# checkIn lifecycle
# ---------------------------------------------------------------------------


class TestCheckIn(unittest.TestCase):
    def test_checkin_releases_slots(self):
        pool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False)
        idx = pool.checkOutAligned(3, 1, "tmp", preventOverflow=False)
        for k in range(idx, idx + 3):
            self.assertEqual(pool._pool[k].status, RegisterPool.Status.InUse)

        pool.checkIn(idx)
        for k in range(idx, idx + 3):
            self.assertEqual(pool._pool[k].status, RegisterPool.Status.Available)

    def test_checkin_unknown_warns_does_not_raise(self):
        pool = RegisterPool(4, RegisterType.Sgpr, defaultPreventOverflow=False)
        pool.add(0, 4)
        buf = io.StringIO()
        with redirect_stdout(buf):
            pool.checkIn(99)  # never checked out
        self.assertIn("never checked out", buf.getvalue())


# ---------------------------------------------------------------------------
# checkOutMulti
# ---------------------------------------------------------------------------


class TestCheckOutMulti(unittest.TestCase):
    def test_partition_lump_sum(self):
        pool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False)
        idx_vec = pool.checkOutMulti([2, 1, 3], 1, ["A", "B", "C"])
        self.assertEqual(idx_vec, [0, 2, 3])
        self.assertEqual(pool._pool[0].tag, "A")
        self.assertEqual(pool._pool[1].tag, "A")
        self.assertEqual(pool._pool[2].tag, "B")
        self.assertEqual(pool._pool[3].tag, "C")
        self.assertEqual(pool._pool[5].tag, "C")

        # Each partition should be independently checkInable.
        pool.checkIn(2)
        self.assertEqual(pool._pool[2].status, RegisterPool.Status.Available)
        self.assertEqual(pool._pool[3].status, RegisterPool.Status.InUse)

    def test_length_mismatch_raises(self):
        pool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False)
        with self.assertRaises(ValueError):
            pool.checkOutMulti([1, 2], 1, ["only one tag"])


# ---------------------------------------------------------------------------
# add / remove / addRange
# ---------------------------------------------------------------------------


class TestAddRemove(unittest.TestCase):
    def test_add_marks_available(self):
        pool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False)
        pool.add(0, 4, "manual")
        self.assertEqual(pool.size(), 4)
        self.assertEqual(pool.available(), 4)
        for r in pool.getPool():
            self.assertEqual(r.status, RegisterPool.Status.Available)
            self.assertEqual(r.tag, "manual")

    def test_addRange_returns_string(self):
        pool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False)
        self.assertEqual(pool.addRange(2, 5, "rr"), "2-5")
        self.assertEqual(pool.addRange(7, 7, "single"), "7")

    def test_remove_flips_to_unavailable(self):
        pool = RegisterPool(4, RegisterType.Sgpr, defaultPreventOverflow=False)
        pool.add(0, 4)
        pool.remove(1, 2, "rm")
        states = [r.status for r in pool.getPool()]
        self.assertEqual(states[0], RegisterPool.Status.Available)
        self.assertEqual(states[1], RegisterPool.Status.Unavailable)
        self.assertEqual(states[2], RegisterPool.Status.Unavailable)
        self.assertEqual(states[3], RegisterPool.Status.Available)


# ---------------------------------------------------------------------------
# Free-state toggle: addFromCheckOut / removeFromCheckOut
# ---------------------------------------------------------------------------


class TestFreeStateToggle(unittest.TestCase):
    """Used by KernelWriter.freeSgprVarPool to temporarily lend out a slot
    while keeping the name binding intact."""

    def test_round_trip(self):
        pool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False)
        idx = pool.checkOutAligned(2, 1, "A", preventOverflow=False)
        self.assertEqual(pool._pool[idx].status, RegisterPool.Status.InUse)

        pool.addFromCheckOut(idx)
        # Slots become Available; bookkeeping moved to Temp side.
        self.assertEqual(pool._pool[idx].status, RegisterPool.Status.Available)
        self.assertNotIn(idx, pool._checkOutSize)
        self.assertIn(idx, pool._checkOutSizeTemp)

        pool.removeFromCheckOut(idx)
        # Back to InUse, with the original tag restored.
        self.assertEqual(pool._pool[idx].status, RegisterPool.Status.InUse)
        self.assertEqual(pool._pool[idx].tag, "A")
        self.assertIn(idx, pool._checkOutSize)
        self.assertNotIn(idx, pool._checkOutSizeTemp)

    def test_addFromCheckOut_unknown_raises(self):
        pool = RegisterPool(4, RegisterType.Sgpr, defaultPreventOverflow=False)
        with self.assertRaises(RuntimeError):
            pool.addFromCheckOut(0)


# ---------------------------------------------------------------------------
# checkFinalState
# ---------------------------------------------------------------------------


class TestCheckFinalState(unittest.TestCase):
    def test_clean_state_passes(self):
        pool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False)
        idx = pool.checkOutAligned(2, 1, "tmp", preventOverflow=False)
        pool.checkIn(idx)
        pool.checkFinalState()  # must not raise

    def test_dirty_state_raises(self):
        pool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False)
        pool.checkOutAligned(2, 1, "leaked", preventOverflow=False)
        with self.assertRaises(RuntimeError):
            pool.checkFinalState()


# ---------------------------------------------------------------------------
# availableBlock* introspection
# ---------------------------------------------------------------------------


class TestAvailability(unittest.TestCase):
    def test_available_count(self):
        pool = RegisterPool(8, RegisterType.Sgpr, defaultPreventOverflow=False)
        pool.add(0, 8)
        self.assertEqual(pool.available(), 8)
        pool.checkOutAligned(3, 1, "x")
        self.assertEqual(pool.available(), 5)

    def test_available_block_at_end(self):
        pool = RegisterPool(8, RegisterType.Sgpr, defaultPreventOverflow=False)
        pool.add(0, 8)
        pool.checkOutAligned(3, 1, "x")  # idx 0..2 InUse
        # Tail is 8-3 = 5 contiguous Available slots.
        self.assertEqual(pool.availableBlockAtEnd(), 5)

    def test_available_block_aligned(self):
        pool = RegisterPool(8, RegisterType.Sgpr, defaultPreventOverflow=False)
        pool.add(0, 8)
        # Available [0..7]; ask blocks of size 4 aligned to 4.
        # 4-aligned starts: 0, 4. Two blocks of 4 fit. Total = 8.
        self.assertEqual(pool.availableBlock(4, 4), 8)


# ---------------------------------------------------------------------------
# Pool growth helpers
# ---------------------------------------------------------------------------


class TestPoolGrowth(unittest.TestCase):
    def test_appendPool_adds_available(self):
        pool = RegisterPool(2, RegisterType.Sgpr, defaultPreventOverflow=False)
        self.assertEqual(pool.size(), 2)
        pool.appendPool(6)
        self.assertEqual(pool.size(), 6)
        # Original 2 stay Unavailable; appended 4 are Available.
        for r in pool.getPool()[:2]:
            self.assertEqual(r.status, RegisterPool.Status.Unavailable)
        for r in pool.getPool()[2:]:
            self.assertEqual(r.status, RegisterPool.Status.Available)
            self.assertEqual(r.tag, "append pool")

    def test_appendPool_smaller_is_noop(self):
        pool = RegisterPool(4, RegisterType.Sgpr, defaultPreventOverflow=False)
        pool.appendPool(2)
        self.assertEqual(pool.size(), 4)


# ---------------------------------------------------------------------------
# Occupancy limits
# ---------------------------------------------------------------------------


class TestOccupancyLimit(unittest.TestCase):
    def test_set_and_reset(self):
        pool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False)
        pool.setOccupancyLimit(maxSize=10, size=8)
        self.assertEqual(pool._occupancyLimitSize, 8)
        self.assertEqual(pool._occupancyLimitMaxSize, 10)
        pool.resetOccupancyLimit()
        self.assertEqual(pool._occupancyLimitSize, 0)
        self.assertEqual(pool._occupancyLimitMaxSize, 0)


# ---------------------------------------------------------------------------
# deepcopy
# ---------------------------------------------------------------------------


class TestDeepCopy(unittest.TestCase):
    def test_deepcopy_independent_state(self):
        pool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False)
        pool.checkOutAligned(2, 1, "A", preventOverflow=False)
        pool.checkOutAligned(1, 1, "B", preventOverflow=False)

        clone = copy.deepcopy(pool)
        self.assertEqual(clone.size(), pool.size())
        self.assertEqual(clone.available(), pool.available())
        self.assertEqual(
            [r.status for r in clone.getPool()],
            [r.status for r in pool.getPool()],
        )

        # Mutate clone; original must not change.
        clone.checkOutAligned(2, 1, "C", preventOverflow=False)
        self.assertNotEqual(clone.size(), pool.size())


# ---------------------------------------------------------------------------
# Tensile-realistic mini scenario: replay the head of _initKernel
# ---------------------------------------------------------------------------


class TestTensileInitKernelHead(unittest.TestCase):
    """Simulates KernelWriter.py:7383 + 7456-7466 to ensure ABI parity."""

    def test_replay(self):
        sgprPool = RegisterPool(
            0, RegisterType.Sgpr, defaultPreventOverflow=True
        )
        sgprs = {}

        def defineSgpr(name, n, align=1):
            idx = sgprPool.checkOutAligned(
                n, align, tag=name, preventOverflow=False
            )
            sgprs[name] = idx
            return idx

        defineSgpr("KernArgAddress", 2)        # rpga = 2 (64-bit pointer)
        defineSgpr("WorkGroup0", 1)
        defineSgpr("WorkGroup1", 1)
        defineSgpr("ArgType", 1)
        defineSgpr("StaggerU", 1)
        defineSgpr("WGM", 1)

        # AMDGPU ABI: kernarg pointer must live in SGPR0.
        self.assertEqual(sgprs["KernArgAddress"], 0)
        # The rest are packed densely with no padding.
        self.assertEqual(sgprs["WorkGroup0"], 2)
        self.assertEqual(sgprs["WorkGroup1"], 3)
        self.assertEqual(sgprs["ArgType"], 4)
        self.assertEqual(sgprs["StaggerU"], 5)
        self.assertEqual(sgprs["WGM"], 6)
        self.assertEqual(sgprPool.size(), 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
