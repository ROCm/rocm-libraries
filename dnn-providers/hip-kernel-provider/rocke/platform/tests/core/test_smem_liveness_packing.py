# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for LDS liveness analysis and interval-based smem packing.

Tests cover:
  - _collect_smem_liveness: correct live-interval computation for sequential
    ops, nested scf.for loops, and mixed regions.
  - _compute_smem_layout: interval packing produces the correct byte offsets
    and pool size for non-interfering, interfering, and mixed allocations.
"""

from __future__ import annotations

import unittest

from rocke.core.ir import F16, F32, I8, KernelDef, Op, Region, SmemType, Value
from rocke.core.lower_llvm import _Lowerer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _smem(elem, *shape: int) -> SmemType:
    return SmemType(elem, list(shape))


def _make_op(name: str, operands=(), results=()):
    op = Op(name=name, operands=list(operands), results=list(results))
    for r in results:
        r.op = op
    return op


def _alloc(name: str, stype: SmemType) -> tuple[Op, Value]:
    v = Value(name=name, type=stype)
    op = _make_op("tile.smem_alloc", results=[v])
    return op, v


def _use(*vals: Value) -> Op:
    return _make_op("tile.lds_store", operands=list(vals))


def _lowerer(*ops: Op) -> _Lowerer:
    body = Region(label="body", ops=list(ops))
    kernel = KernelDef(name="k", params=[], body=body)
    low = _Lowerer(kernel)
    low._collect_smem(body)
    return low


# byte sizes
_KB = 1024
_12KB = 12 * _KB  # f32 [128x24]
_24KB = 24 * _KB  # two 12KB tiles
_64KB = 64 * _KB  # f32 [128x128]  (CShuffle)
_8KB = 8 * _KB  # f16 [64x64]
_4KB = 4 * _KB  # f32 [32x32]


# ---------------------------------------------------------------------------
# liveness tests
# ---------------------------------------------------------------------------


class TestSmemLiveness(unittest.TestCase):

    def _live(self, *ops: Op) -> dict:
        low = _lowerer(*ops)
        raw, _ = low._collect_smem_liveness(low.kernel.body)
        # Strip the kernel-name suffix for readability: "@A.k" -> "A"
        return {k.lstrip("@").split(".")[0]: v for k, v in raw.items()}

    def test_single_alloc_no_uses(self):
        op_a, _ = _alloc("%A", _smem(F16, 64, 64))
        live = self._live(op_a)
        # Defined at seq 0, no downstream uses → last = 0
        self.assertEqual(live["A"], (0, 0))

    def test_alloc_with_single_use(self):
        op_a, va = _alloc("%A", _smem(F16, 64, 64))
        op_u = _use(va)
        live = self._live(op_a, op_u)
        # alloc at 0, use at 1 → (0, 1)
        self.assertEqual(live["A"], (0, 1))

    def test_two_allocs_disjoint(self):
        # A used only in first half, C used only in second half
        op_a, va = _alloc("%A", _smem(F32, 128, 24))
        op_b, vb = _alloc("%B", _smem(F32, 128, 24))
        op_u1 = _use(va, vb)
        op_u2 = _use(va, vb)
        op_c, vc = _alloc("%C", _smem(F32, 128, 128))
        op_u3 = _use(vc)
        live = self._live(op_a, op_b, op_u1, op_u2, op_c, op_u3)
        # A: def=0, last use=3; B: def=1, last use=3; C: def=4, last use=5
        self.assertEqual(live["A"], (0, 3))
        self.assertEqual(live["B"], (1, 3))
        self.assertEqual(live["C"], (4, 5))
        # A and C must not interfere (C starts after A ends)
        a_first, a_last = live["A"]
        c_first, c_last = live["C"]
        self.assertGreater(c_first, a_last)

    def test_scf_for_extends_last_to_loop_end(self):
        # A used inside a loop → last_seq extended to the loop's *last* preorder
        # index (end of the scf.for subtree), not the for-op's own index.
        op_a, va = _alloc("%A", _smem(F16, 64, 64))
        inner_use = _use(va)
        loop_body = Region(label="body", ops=[inner_use])
        for_op = Op(
            name="scf.for",
            operands=[],
            results=[],
            regions=[loop_body],
        )
        low = _lowerer(op_a, for_op)
        raw, _ = low._collect_smem_liveness(low.kernel.body)
        live = {k.lstrip("@").split(".")[0]: v for k, v in raw.items()}
        # seq: op_a=0, for_op=1, inner_use=2. The loop subtree spans [1, 2], so
        # the use inside extends A's last_seq to 2 (the loop end).
        self.assertEqual(live["A"][0], 0)
        self.assertEqual(live["A"][1], 2)

    def test_alloc_inside_non_loop_region(self):
        # alloc inside an if-region (not scf.for) — no conservative extension
        op_a, va = _alloc("%A", _smem(F16, 64, 64))
        inner_use = _use(va)
        if_body = Region(label="then", ops=[inner_use])
        if_op = Op(name="scf.if", operands=[], results=[], regions=[if_body])
        low = _lowerer(op_a, if_op)
        raw, _ = low._collect_smem_liveness(low.kernel.body)
        live = {k.lstrip("@").split(".")[0]: v for k, v in raw.items()}
        # scf.if is not a loop → use inside extends last normally (to inner idx)
        # The inner use is at seq 2 (after op_a=0, if_op=1); but since
        # loop_end=None for scf.if, last = inner idx = 2.
        self.assertGreaterEqual(live["A"][1], 1)


# ---------------------------------------------------------------------------
# packing tests
# ---------------------------------------------------------------------------


class TestSmemPacking(unittest.TestCase):

    def _pack(self, *ops: Op):
        low = _lowerer(*ops)
        low._compute_smem_layout()
        offsets = {
            k.lstrip("@").split(".")[0]: low._smem_offsets[k] for k in low._smem_offsets
        }
        return offsets, low._smem_pool_size

    @staticmethod
    def _overlaps(o1, s1, o2, s2) -> bool:
        return o1 < o2 + s2 and o2 < o1 + s1

    # -- single alloc ---------------------------------------------------------

    def test_single_alloc(self):
        # A single *referenced* allocation lands at offset 0 with a 16-rounded
        # pool. (A never-referenced allocation is dead-stripped and costs 0 --
        # see TestSmemPacking.test_dead_alloc_costs_nothing -- so give A a use.)
        op_a, va = _alloc("%A", _smem(F32, 128, 24))
        op_u = _use(va)
        offsets, pool = self._pack(op_a, op_u)
        self.assertEqual(offsets["A"], 0)
        self.assertEqual(pool, (_12KB + 15) & ~15)

    def test_dead_alloc_costs_nothing(self):
        # An allocation that is never read/written/address-taken must not
        # consume pool bytes: the pre-pool lowering emitted it as its own
        # addrspace(3) global that the AMDGPU backend dead-strips. Folding it
        # into the pool used to inflate LDS and overflow the 64 KB limit (the
        # fp16 D128 nw=4 attention Acc_lds regression).
        op_a, _ = _alloc("%A", _smem(F32, 128, 24))  # 12KB, no uses -> dead
        op_b, vb = _alloc("%B", _smem(F32, 128, 24))  # 12KB, used -> live
        op_u = _use(vb)
        offsets, pool = self._pack(op_a, op_b, op_u)
        self.assertEqual(offsets["B"], 0)
        self.assertEqual(pool, (_12KB + 15) & ~15)  # only B counts

    def test_all_dead_allocs_emit_no_pool(self):
        # When EVERY allocation is dead, the pool is empty AND _smem_globals is
        # cleared so finalize() emits no pool global. Otherwise Python would
        # emit a zero-length `[0 x i8]` addrspace(3) global while the C++ engine
        # (gated on pool_size > 0) emits nothing -- a byte-identity divergence,
        # and some LLVM/AMDGPU pipelines reject a zero-length global.
        op_a, _ = _alloc("%A", _smem(F32, 128, 24))  # dead
        op_b, _ = _alloc("%B", _smem(F16, 64, 64))  # dead
        low = _lowerer(op_a, op_b)
        low._compute_smem_layout()
        self.assertEqual(low._smem_pool_size, 0)
        self.assertEqual(low._smem_globals, [])

    # -- interfering allocs must not overlap ----------------------------------

    def test_two_interfering_no_overlap(self):
        op_a, va = _alloc("%A", _smem(F32, 128, 24))  # 12KB
        op_b, vb = _alloc("%B", _smem(F32, 128, 24))  # 12KB
        op_u = _use(va, vb)
        offsets, pool = self._pack(op_a, op_b, op_u)
        self.assertFalse(
            self._overlaps(offsets["A"], _12KB, offsets["B"], _12KB),
            "A and B interfere and must not share memory",
        )
        # Pool = A + B = 24KB
        self.assertEqual(pool, (_24KB + 15) & ~15)

    # -- non-interfering: C smaller than A+B ----------------------------------

    def test_non_interfering_c_smaller_reuses_a_slot(self):
        op_a, va = _alloc("%A", _smem(F16, 64, 64))  # 8KB
        op_b, vb = _alloc("%B", _smem(F16, 64, 64))  # 8KB
        op_u1 = _use(va, vb)
        op_u2 = _use(va, vb)
        op_c, vc = _alloc("%C", _smem(F32, 32, 32))  # 4KB — fits in A's slot
        op_u3 = _use(vc)
        offsets, pool = self._pack(op_a, op_b, op_u1, op_u2, op_c, op_u3)
        # C must reuse slot A (offset 0)
        self.assertEqual(offsets["C"], 0, "C should reuse A's slot at offset 0")
        # Pool = max(A+B, C@0) = 16KB
        self.assertEqual(pool, (2 * _8KB + 15) & ~15)
        # A and B must not overlap
        self.assertFalse(self._overlaps(offsets["A"], _8KB, offsets["B"], _8KB))

    # -- non-interfering: C larger than A+B (the CShuffle case) --------------

    def test_non_interfering_c_larger_reuses_a_slot_expands_pool(self):
        """CShuffle (64KB) reuses A's slot even though it's larger than A (12KB).

        Before the fix the packer opened a fresh slot at offset 24KB
        producing pool=88KB.  After the fix C starts at offset 0 and
        pool = max(A+B=24KB, C=64KB) = 64KB.
        """
        op_a, va = _alloc("%A", _smem(F32, 128, 24))  # 12KB
        op_b, vb = _alloc("%B", _smem(F32, 128, 24))  # 12KB
        op_u1 = _use(va, vb)
        op_u2 = _use(va, vb)
        op_c, vc = _alloc("%C", _smem(F32, 128, 128))  # 64KB
        op_u3 = _use(vc)
        offsets, pool = self._pack(op_a, op_b, op_u1, op_u2, op_c, op_u3)
        # C must sit at offset 0
        self.assertEqual(offsets["C"], 0, "C should reuse A's slot at offset 0")
        # Pool = 64KB (not 88KB)
        self.assertEqual(pool, (_64KB + 15) & ~15)
        # A and B must not overlap each other
        self.assertFalse(self._overlaps(offsets["A"], _12KB, offsets["B"], _12KB))

    # -- all three interfere --------------------------------------------------

    def test_three_interfering_allocs(self):
        op_a, va = _alloc("%A", _smem(F16, 64, 64))  # 8KB
        op_b, vb = _alloc("%B", _smem(F16, 64, 64))  # 8KB
        op_c, vc = _alloc("%C", _smem(F16, 64, 64))  # 8KB
        # All three are defined before the single op that reads them, so their
        # live intervals overlap at the use and none may share memory. (Ops are
        # passed in valid SSA order: every %value is defined before it is used.)
        op_u = _use(va, vb, vc)
        offsets, pool = self._pack(op_a, op_b, op_c, op_u)
        for n1, o1, n2, o2 in [
            ("A", offsets["A"], "B", offsets["B"]),
            ("A", offsets["A"], "C", offsets["C"]),
            ("B", offsets["B"], "C", offsets["C"]),
        ]:
            self.assertFalse(
                self._overlaps(o1, _8KB, o2, _8KB),
                f"{n1} and {n2} interfere and must not overlap",
            )
        # Three mutually interfering 8KB tiles → pool = 24KB
        self.assertEqual(pool, (3 * _8KB + 15) & ~15)

    # -- double-buffer: A and A2 interfere with each other --------------------

    def test_double_buffer_ping_pong(self):
        # compv4 allocates A_smem and A_smem2 for ping-pong; both live in loop
        op_a, va = _alloc("%A", _smem(F16, 64, 64))  # 8KB
        op_a2, va2 = _alloc("%A2", _smem(F16, 64, 64))  # 8KB
        op_b, vb = _alloc("%B", _smem(F16, 64, 64))  # 8KB
        op_b2, vb2 = _alloc("%B2", _smem(F16, 64, 64))  # 8KB
        # All four live simultaneously in the loop body
        loop_use = _use(va, va2, vb, vb2)
        loop_body = Region(label="body", ops=[loop_use])
        for_op = Op(name="scf.for", operands=[], results=[], regions=[loop_body])
        # CShuffle after the loop — non-interfering
        op_c, vc = _alloc("%C", _smem(F32, 32, 32))  # 4KB
        op_cu = _use(vc)
        low = _lowerer(op_a, op_a2, op_b, op_b2, for_op, op_c, op_cu)
        low._compute_smem_layout()
        oA = low._smem_offsets["@A.k"]
        oA2 = low._smem_offsets["@A2.k"]
        oB = low._smem_offsets["@B.k"]
        oB2 = low._smem_offsets["@B2.k"]
        oC = low._smem_offsets["@C.k"]
        pool = low._smem_pool_size
        # None of the four loop tiles may overlap
        for n1, o1, s1, n2, o2, s2 in [
            ("A", oA, _8KB, "A2", oA2, _8KB),
            ("A", oA, _8KB, "B", oB, _8KB),
            ("A", oA, _8KB, "B2", oB2, _8KB),
            ("A2", oA2, _8KB, "B", oB, _8KB),
            ("A2", oA2, _8KB, "B2", oB2, _8KB),
            ("B", oB, _8KB, "B2", oB2, _8KB),
        ]:
            self.assertFalse(
                self._overlaps(o1, s1, o2, s2),
                f"{n1} and {n2} must not overlap",
            )
        # Pool must be exactly 4 × 8KB = 32KB; C reuses one of those slots
        self.assertEqual(pool, (4 * _8KB + 15) & ~15)
        # C must sit within the existing pool (reusing a freed slot)
        self.assertLess(oC + _4KB, pool + 1)

    # -- byte-element alignment (i8 gets 16-byte align) -----------------------

    def test_i8_alignment(self):
        op_a, va = _alloc("%A", _smem(I8, 128, 128))  # 16KB, align=16
        op_b, vb = _alloc("%B", _smem(F16, 64, 64))  # 8KB,  align=4
        op_u = _use(va, vb)
        offsets, pool = self._pack(op_a, op_b, op_u)
        self.assertEqual(offsets["A"] % 16, 0, "i8 alloc must be 16-byte aligned")
        self.assertEqual(offsets["B"] % 4, 0, "f16 alloc must be 4-byte aligned")

    # -- empty kernel ---------------------------------------------------------

    def test_no_allocs(self):
        op_u = _make_op("arith.constant")
        low = _lowerer(op_u)
        low._compute_smem_layout()
        self.assertEqual(low._smem_pool_size, 0)


if __name__ == "__main__":
    unittest.main()
