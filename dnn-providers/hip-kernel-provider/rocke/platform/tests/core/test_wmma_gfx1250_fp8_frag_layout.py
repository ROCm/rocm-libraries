# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx1250 16x16x64 FP8/BF8 A and B operand fragment layouts.

These maps say which (row, K) element of A -- and which (K, col) of B -- a given
lane's fragment slot holds. Get one wrong and the kernel still builds, still
launches, and returns plausible numbers; the only signal is a numeric mismatch on
silicon, and only for a shape that happens to distinguish the wrong map from the
right one. A square tile does not distinguish an M/N transposition, and a map
that is wrong only in its K stride survives any K below the stride.

The layout is checkable on the host because it is a *bijection*. Each lane holds
``<8 x i32>``, each slot is the dword covering four K-contiguous low-bit
elements, so a wave covers 32 lanes x 8 slots x 4 bytes = 1024 positions -- and
the operand is 16 x 64 = 1024. Every position must be hit exactly once. A
duplicate means two lanes fight over one element; a hole means an element never
reaches the MMA. That catches a bad map without a GPU, which is the point.

The accumulator half of this op is already covered by
``test_wmma_gfx12_acc_layout.py``. This file deliberately carries its own
evaluator rather than importing that module's: a test importing another test's
private helper couples two files that have no reason to move together, and the
evaluator is five lines.

CPU-only: no GPU, compile, or launch required.
"""

from __future__ import annotations

import unittest
from typing import Callable

from rocke.core.arch.target import (
    _MMA_FRAGMENT_INFO,
    _wmma_gfx1250_a_16x16x32,
    _wmma_gfx1250_a_16x16x64_f8,
    _wmma_gfx1250_b_16x16x32,
    _wmma_gfx1250_b_16x16x64_f8,
)

# The four K=64 low-bit op_ids. All of them run the same lane math: the operand
# element type changes what the bytes *mean*, not where they live.
_FP8_OP_IDS = (
    "wmma_gfx1250_f32_16x16x64_fp8_fp8",
    "wmma_gfx1250_f32_16x16x64_fp8_bf8",
    "wmma_gfx1250_f32_16x16x64_bf8_fp8",
    "wmma_gfx1250_f32_16x16x64_bf8_bf8",
)

_M = _N = 16
_K = 64
_WAVE = 32
_FRAG_LEN = 8
_PER_SLOT = 4  # low-bit elements packed into one i32 slot


class _ConstBuilder:
    """Toy IR builder that evaluates coordinate expressions as plain integers."""

    def const_i32(self, v: int) -> int:
        return int(v)

    def mod(self, a: int, b: int) -> int:
        return a % b

    def div(self, a: int, b: int) -> int:
        return a // b

    def add(self, a: int, b: int) -> int:
        return a + b

    def mul(self, a: int, b: int) -> int:
        return a * b


def _coords(fn: Callable, lane: int, slot: int) -> tuple[int, int]:
    return fn(_ConstBuilder(), lane, slot)


def _positions(fn: Callable, *, k_index: int) -> list[tuple[int, int]]:
    """Expand a slot-granular map to one entry per element it covers.

    ``k_index`` says which half of the returned pair is K, since A returns
    ``(row, k)`` and B returns ``(k, col)``.
    """
    out = []
    for lane in range(_WAVE):
        for slot in range(_FRAG_LEN):
            pair = list(_coords(fn, lane, slot))
            base = pair[k_index]
            for byte in range(_PER_SLOT):
                pair[k_index] = base + byte
                out.append((pair[0], pair[1]))
    return out


class TestOperandBijection(unittest.TestCase):
    def test_a_covers_every_element_exactly_once(self):
        got = _positions(_wmma_gfx1250_a_16x16x64_f8, k_index=1)
        want = {(row, k) for row in range(_M) for k in range(_K)}
        self.assertEqual(len(got), _M * _K, "wave must cover exactly M*K elements")
        self.assertEqual(set(got), want, "A map has a hole or lands outside 16x64")
        self.assertEqual(len(set(got)), len(got), "A map assigns one element twice")

    def test_b_covers_every_element_exactly_once(self):
        got = _positions(_wmma_gfx1250_b_16x16x64_f8, k_index=0)
        want = {(k, col) for k in range(_K) for col in range(_N)}
        self.assertEqual(len(got), _N * _K)
        self.assertEqual(set(got), want, "B map has a hole or lands outside 64x16")
        self.assertEqual(len(set(got)), len(got), "B map assigns one element twice")

    def test_a_and_b_use_the_documented_coordinate_order(self):
        # A returns (row, k) and B returns (k, col). Swapping them is the error
        # the bijection cannot see -- both orders are bijective over a square
        # operand -- so state the convention separately.
        row, k = _coords(_wmma_gfx1250_a_16x16x64_f8, lane=17, slot=3)
        self.assertEqual(row, 1, "A's first coordinate is the row, = lane % 16")
        self.assertEqual(k, 32 + 12, "A's second coordinate is K")
        k_b, col = _coords(_wmma_gfx1250_b_16x16x64_f8, lane=17, slot=3)
        self.assertEqual(k_b, 32 + 12, "B's first coordinate is K")
        self.assertEqual(col, 1, "B's second coordinate is the column")


class TestOperandStructure(unittest.TestCase):
    def test_row_and_col_depend_on_the_lane_alone(self):
        # The fragment slot walks K, never M or N. If a slot ever moved the row,
        # a lane would need rows from two different tiles.
        for lane in range(_WAVE):
            rows = {_coords(_wmma_gfx1250_a_16x16x64_f8, lane, s)[0] for s in range(8)}
            cols = {_coords(_wmma_gfx1250_b_16x16x64_f8, lane, s)[1] for s in range(8)}
            self.assertEqual(rows, {lane % 16}, f"lane {lane} A row moved with slot")
            self.assertEqual(cols, {lane % 16}, f"lane {lane} B col moved with slot")

    def test_slots_step_k_by_one_dword(self):
        # Slot i is the dword holding K..K+3, so consecutive slots must be
        # exactly four apart: a gap would skip elements the bijection then
        # reports as holes, but the stride is the thing actually being asserted.
        for lane in range(_WAVE):
            ks = [_coords(_wmma_gfx1250_a_16x16x64_f8, lane, s)[1] for s in range(8)]
            self.assertEqual(ks, [ks[0] + 4 * i for i in range(8)], f"lane {lane}")

    def test_the_upper_lane_half_holds_the_upper_k_half(self):
        # lane and lane+16 are the same row, 32 K-elements apart. This is the
        # one fact that makes a 16x16x64 atom fit in a 32-lane wave.
        for lane in range(16):
            for slot in range(8):
                lo = _coords(_wmma_gfx1250_a_16x16x64_f8, lane, slot)
                hi = _coords(_wmma_gfx1250_a_16x16x64_f8, lane + 16, slot)
                self.assertEqual(lo[0], hi[0], "lane halves must share a row")
                self.assertEqual(hi[1] - lo[1], 32, "lane halves are 32 K apart")


class TestRelationToTheKEquals32Map(unittest.TestCase):
    """The K=64 map claims to be the K=32 f16 map with the K span doubled.

    That claim is load-bearing -- it is why the fp8 map was trusted before it had
    been run -- so it is asserted rather than left in a docstring.
    """

    def test_same_row_and_col_assignment(self):
        for lane in range(_WAVE):
            self.assertEqual(
                _coords(_wmma_gfx1250_a_16x16x32, lane, 0)[0],
                _coords(_wmma_gfx1250_a_16x16x64_f8, lane, 0)[0],
            )
            self.assertEqual(
                _coords(_wmma_gfx1250_b_16x16x32, lane, 0)[1],
                _coords(_wmma_gfx1250_b_16x16x64_f8, lane, 0)[1],
            )

    def test_k_base_is_exactly_doubled(self):
        # The f16 map is element-granular over 16 K per lane-half; the fp8 map is
        # dword-granular over 32. Slot 0 pins the lane-half base of each.
        for lane in range(_WAVE):
            k32 = _coords(_wmma_gfx1250_a_16x16x32, lane, 0)[1]
            k64 = _coords(_wmma_gfx1250_a_16x16x64_f8, lane, 0)[1]
            self.assertEqual(k64, 2 * k32, f"lane {lane}: K base not doubled")


class TestCatalogWiring(unittest.TestCase):
    def test_all_four_low_bit_op_ids_are_registered(self):
        # Without this the layout tests above pass while the catalog still hands
        # the lowerer no map at all -- which is the state these entries replaced.
        for op_id in _FP8_OP_IDS:
            with self.subTest(op_id=op_id):
                self.assertIn(op_id, _MMA_FRAGMENT_INFO)

    def test_every_low_bit_op_shares_one_pair_of_maps(self):
        # fp8_bf8 / bf8_fp8 / bf8_bf8 have no numeric coverage anywhere, so they
        # are trusted purely by being the same function object as fp8_fp8, which
        # is verified on silicon. Identity, not equivalence: a copy could drift.
        for op_id in _FP8_OP_IDS:
            info = _MMA_FRAGMENT_INFO[op_id]
            with self.subTest(op_id=op_id):
                self.assertIs(info.a_fn, _wmma_gfx1250_a_16x16x64_f8)
                self.assertIs(info.b_fn, _wmma_gfx1250_b_16x16x64_f8)

    def test_fragment_metadata_matches_the_abi(self):
        # <8 x i32> per operand on a 32-lane wave. The bijection is computed from
        # these numbers, so if they drift the coverage test silently changes
        # what it proves.
        for op_id in _FP8_OP_IDS:
            info = _MMA_FRAGMENT_INFO[op_id]
            with self.subTest(op_id=op_id):
                self.assertEqual(info.a_frag_len, _FRAG_LEN)
                self.assertEqual(info.b_frag_len, _FRAG_LEN)
                self.assertEqual(info.wave_size, _WAVE)
                self.assertEqual(
                    _WAVE * info.a_frag_len * _PER_SLOT,
                    _M * _K,
                    "a wave must hold exactly one 16x64 A operand",
                )


if __name__ == "__main__":
    unittest.main()
