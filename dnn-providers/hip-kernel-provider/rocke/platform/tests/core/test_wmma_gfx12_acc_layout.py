# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validation tests: gfx1201 and gfx1250 share the same accumulator fragment layout.

Three layers of validation:

1. **Metadata identity** — every gfx1250 WMMA op carries the same ``c_frag_len``,
   ``wave_size``, and ``c_fn`` object as its gfx1201 counterparts.

2. **Coordinate exhaustion** — the coordinate function produces identical (row, col)
   pairs for all 256 (lane, slot) inputs on both sets of op_ids, using the same
   concrete evaluator helper.

3. **Coverage** — at least one gfx1201 op and all known gfx1250 WMMA ops are present
   in ``_MMA_FRAGMENT_INFO`` so the tests are not silently vacuous.

CPU-only: no GPU, compile, or launch required.
"""

from __future__ import annotations

import unittest
from typing import Callable, Tuple

from rocke.core.arch.target import _MMA_FRAGMENT_INFO, _wmma_gfx12_acc_16x16


# ---------------------------------------------------------------------------
# Minimal constant-folding IR builder — evaluates lane-coord functions on the
# host without the full IR machinery.
# ---------------------------------------------------------------------------


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


def _eval_acc(fn: Callable, lane: int, slot: int) -> Tuple[int, int]:
    b = _ConstBuilder()
    return fn(b, lane, slot)


def _full_layout(fn: Callable, frag_len: int, wave_size: int):
    """Return the complete (lane, slot) -> (row, col) mapping as a dict."""
    return {
        (lane, slot): _eval_acc(fn, lane, slot)
        for lane in range(wave_size)
        for slot in range(frag_len)
    }


# ---------------------------------------------------------------------------
# Op-id sets
# ---------------------------------------------------------------------------

_GFX1201_WMMA_OPS = {
    "wmma_gfx12_f32_16x16x16_f16",
    "wmma_gfx12_f32_16x16x16_bf16",
}

_GFX1250_WMMA_OPS = {
    "wmma_gfx1250_f32_16x16x4_f32",
    "wmma_gfx1250_f32_16x16x32_f16",
    "wmma_gfx1250_f32_16x16x32_bf16",
    "wmma_gfx1250_f32_16x16x64_fp8_fp8",
    "wmma_gfx1250_f32_16x16x64_fp8_bf8",
    "wmma_gfx1250_f32_16x16x64_bf8_fp8",
    "wmma_gfx1250_f32_16x16x64_bf8_bf8",
}

# Accumulator params shared by both arches: 16x16 output tile, wave32,
# <8 x float> per lane.
_EXPECTED_C_FRAG_LEN = 8
_EXPECTED_WAVE_SIZE = 32


class TestWmmaGfx12AccLayoutCoverage(unittest.TestCase):
    """Guard that the op-id sets are non-empty and fully registered."""

    def test_gfx1201_ops_in_ssot(self):
        for op_id in _GFX1201_WMMA_OPS:
            self.assertIn(
                op_id,
                _MMA_FRAGMENT_INFO,
                msg=f"gfx1201 op_id {op_id!r} missing from _MMA_FRAGMENT_INFO",
            )

    def test_gfx1250_ops_in_ssot(self):
        for op_id in _GFX1250_WMMA_OPS:
            self.assertIn(
                op_id,
                _MMA_FRAGMENT_INFO,
                msg=f"gfx1250 op_id {op_id!r} missing from _MMA_FRAGMENT_INFO",
            )


class TestWmmaGfx12AccFragMetadata(unittest.TestCase):
    """gfx1201 and gfx1250 WMMA ops must carry identical accumulator metadata."""

    def _check_acc_metadata(self, op_id: str):
        info = _MMA_FRAGMENT_INFO[op_id]
        self.assertEqual(
            info.c_frag_len,
            _EXPECTED_C_FRAG_LEN,
            msg=f"{op_id!r}: c_frag_len {info.c_frag_len} != {_EXPECTED_C_FRAG_LEN}",
        )
        self.assertEqual(
            info.wave_size,
            _EXPECTED_WAVE_SIZE,
            msg=f"{op_id!r}: wave_size {info.wave_size} != {_EXPECTED_WAVE_SIZE}",
        )
        self.assertIs(
            info.c_fn,
            _wmma_gfx12_acc_16x16,
            msg=f"{op_id!r}: c_fn is not _wmma_gfx12_acc_16x16 — "
            f"accumulator layout has been forked from the shared gfx12 map",
        )

    def test_gfx1201_acc_metadata(self):
        for op_id in _GFX1201_WMMA_OPS:
            with self.subTest(op_id=op_id):
                self._check_acc_metadata(op_id)

    def test_gfx1250_acc_metadata(self):
        for op_id in _GFX1250_WMMA_OPS:
            with self.subTest(op_id=op_id):
                self._check_acc_metadata(op_id)


class TestWmmaGfx12AccCoordinateFormula(unittest.TestCase):
    """The shared coordinate function must encode the expected formula for all inputs.

    Expected: slot ``i`` of lane ``l`` (0..31) maps to
        row = (l // 16) * 8 + i,  col = l % 16
    This covers the full 16x16 output tile: rows 0..15, cols 0..15.
    """

    def _expected_coord(self, lane: int, slot: int) -> Tuple[int, int]:
        row = (lane // 16) * 8 + slot
        col = lane % 16
        return row, col

    def test_coordinate_formula(self):
        builder = _ConstBuilder()
        for lane in range(_EXPECTED_WAVE_SIZE):
            for slot in range(_EXPECTED_C_FRAG_LEN):
                got = _wmma_gfx12_acc_16x16(builder, lane, slot)
                want = self._expected_coord(lane, slot)
                self.assertEqual(
                    got,
                    want,
                    msg=f"_wmma_gfx12_acc_16x16(lane={lane}, slot={slot}): "
                    f"got {got}, expected {want}",
                )

    def test_output_tile_is_fully_covered(self):
        """Every (row, col) in the 16x16 tile must be reachable."""
        coords = {
            self._expected_coord(lane, slot)
            for lane in range(_EXPECTED_WAVE_SIZE)
            for slot in range(_EXPECTED_C_FRAG_LEN)
        }
        expected = {(r, c) for r in range(16) for c in range(16)}
        self.assertEqual(
            coords,
            expected,
            msg="Accumulator layout does not cover the full 16x16 output tile",
        )

    def test_no_coordinate_collisions(self):
        """Each (lane, slot) pair must map to a unique matrix element."""
        seen: dict = {}
        for lane in range(_EXPECTED_WAVE_SIZE):
            for slot in range(_EXPECTED_C_FRAG_LEN):
                coord = self._expected_coord(lane, slot)
                if coord in seen:
                    prev_lane, prev_slot = seen[coord]
                    self.fail(
                        f"Coordinate {coord} claimed by both "
                        f"(lane={prev_lane}, slot={prev_slot}) and "
                        f"(lane={lane}, slot={slot})"
                    )
                seen[coord] = (lane, slot)


class TestWmmaGfx12AccLayoutIdentical(unittest.TestCase):
    """The concrete (lane,slot)->(row,col) mapping must be bit-for-bit identical
    across every gfx1201 op and every gfx1250 op that shares ``lm_wmma_gfx12_c``."""

    def _layout_for(self, op_id: str) -> dict:
        info = _MMA_FRAGMENT_INFO[op_id]
        if info.c_fn is None:
            self.skipTest(f"{op_id!r} has no verified accumulator lane map (c_fn=None)")
        return _full_layout(info.c_fn, info.c_frag_len, info.wave_size)

    def test_gfx1250_layout_matches_gfx1201_reference(self):
        # Use wmma_gfx12_f32_16x16x16_f16 as the canonical gfx1201 reference.
        reference_op = "wmma_gfx12_f32_16x16x16_f16"
        ref_layout = self._layout_for(reference_op)

        for op_id in _GFX1250_WMMA_OPS:
            with self.subTest(op_id=op_id):
                info = _MMA_FRAGMENT_INFO[op_id]
                if info.c_fn is None:
                    # op has no verified map; identity is enforced via the
                    # metadata test (same c_frag_len / wave_size / c_fn object).
                    continue
                candidate = _full_layout(info.c_fn, info.c_frag_len, info.wave_size)
                self.assertEqual(
                    candidate,
                    ref_layout,
                    msg=f"{op_id!r} accumulator layout differs from {reference_op!r}",
                )

    def test_gfx1201_ops_have_identical_layouts(self):
        layouts = {op: self._layout_for(op) for op in _GFX1201_WMMA_OPS}
        ops = sorted(layouts)
        for i in range(1, len(ops)):
            self.assertEqual(
                layouts[ops[i]],
                layouts[ops[0]],
                msg=f"gfx1201 accumulator layout mismatch: {ops[i]!r} vs {ops[0]!r}",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
