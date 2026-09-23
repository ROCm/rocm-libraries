# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Consistency guards for the SSOT-backed MMA metadata consumed by ``IRBuilder.mma``.

After the arch-SSOT cleanup, ``rocke.core.ir`` keeps *no* private frag-length or
int-accumulator tables. ``IRBuilder.mma`` sizes a ``tile.mma`` result vector from
a bare ``op_id`` string via three pieces:

* ``_mma_c_frag_len`` — accumulator fragment length, resolved from the arch SSOT
  ``core.arch.target._MMA_FRAGMENT_INFO`` (no ir-side copy);
* ``_mma_c_is_int`` — whether the atom accumulates in i32, resolved from the JSON
  catalog accumulator dtype (``target._op_id_c_dtype``);
* ``_MMA_RESULT_HINT`` — the ir-side SSA result-name hints (naming only, kept in
  ir.py to preserve byte-identical value numbering).

These must stay mutually consistent with the SSOT: any op_id that accumulates in
i32, or that carries a result-name hint, must also have a fragment length in the
SSOT, or ``IRBuilder.mma`` would raise on an otherwise-valid atom.

CPU-only: imports the accessors and the arch SSOT directly, no GPU / compile /
launch required.
"""

from __future__ import annotations

import unittest

from rocke.core.arch.target import _MMA_FRAGMENT_INFO, _op_id_c_dtype
from rocke.core.ir import (
    _MMA_RESULT_HINT,
    _mma_c_frag_len,
    _mma_c_is_int,
)


def _sizable_op_ids():
    """Op_ids the SSOT can actually size (positive accumulator frag length)."""
    return {
        op_id: info.c_frag_len
        for op_id, info in _MMA_FRAGMENT_INFO.items()
        if info.c_frag_len > 0
    }


class TestMmaFragTables(unittest.TestCase):
    def test_frag_lengths_are_positive(self):
        sizable = _sizable_op_ids()
        self.assertTrue(sizable, "the frag-length SSOT must expose at least one atom")
        for op_id, frag in sizable.items():
            self.assertIsInstance(frag, int)
            self.assertGreater(
                frag, 0, msg=f"c_frag_len for {op_id!r} must be positive"
            )

    def test_accessor_matches_ssot_and_raises_on_unknown(self):
        for op_id, frag in _sizable_op_ids().items():
            self.assertEqual(_mma_c_frag_len(op_id), frag)
        with self.assertRaises(ValueError):
            _mma_c_frag_len("not_a_real_op_id")

    def test_int_acc_op_ids_have_frag_lengths(self):
        int_op_ids = [op for op, dtype in _op_id_c_dtype().items() if dtype == "i32"]
        self.assertTrue(
            int_op_ids, "expected at least one i32-accumulator atom in the catalog"
        )
        for op_id in int_op_ids:
            self.assertTrue(
                _mma_c_is_int(op_id),
                msg=f"{op_id!r} is i32 in the catalog but _mma_c_is_int disagrees",
            )
            self.assertGreater(
                _mma_c_frag_len(op_id),
                0,
                msg=f"int-accumulator op_id {op_id!r} has no SSOT frag length, so "
                f"IRBuilder.mma would raise for it",
            )

    def test_result_hint_op_ids_have_frag_lengths(self):
        for op_id in _MMA_RESULT_HINT:
            self.assertGreater(
                _mma_c_frag_len(op_id),
                0,
                msg=f"result-hint op_id {op_id!r} has no SSOT frag length",
            )

    def test_known_frag_lengths(self):
        # Spot-check representative 16x16 vs 32x32 accumulator widths.
        self.assertEqual(_mma_c_frag_len("mfma_f32_16x16x16_f16"), 4)
        self.assertEqual(_mma_c_frag_len("mfma_f32_32x32x8_f16"), 16)
        self.assertEqual(_mma_c_frag_len("wmma_f32_16x16x16_f16"), 8)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
