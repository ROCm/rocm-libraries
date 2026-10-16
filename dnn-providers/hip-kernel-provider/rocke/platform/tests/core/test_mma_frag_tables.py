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

from rocke.core.arch.target import (
    _MMA_FRAGMENT_INFO,
    _frag_info,
    _load_specs,
    _op_id_c_dtype,
    _op_id_physical_op_id,
)
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
    def test_every_scaled_catalog_atom_has_the_same_padded_abi(self):
        rows = [
            row
            for row in _load_specs()["gfx1250"]["mma"]
            if row["family"] == "wmma_scaled"
        ]
        self.assertEqual(len(rows), 86)
        for row in rows:
            info = _MMA_FRAGMENT_INFO[row["op_id"]]
            self.assertEqual(
                (info.a_frag_len, info.b_frag_len, info.c_frag_len, info.wave_size),
                (16, 16, 8, 32),
            )
            self.assertEqual(info.a_scale_frag_len, 128 // row["scale_block_k"])
            self.assertEqual(info.b_scale_frag_len, info.a_scale_frag_len)
            self.assertEqual(_mma_c_frag_len(row["op_id"]), 8)
            with self.assertRaises(ValueError):
                _mma_c_frag_len(row["op_id"] + "_unknown")

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

    def test_emulated_op_ids_size_through_their_physical_atom(self):
        # An emulated atom borrows its fragment from the instruction it issues
        # rather than restating it, so its bare op_id must size exactly as that
        # physical atom does -- or IRBuilder.mma("<logical op_id>") would raise.
        emulated = _op_id_physical_op_id()
        self.assertTrue(emulated, "expected at least one emulated atom in the catalog")
        for op_id, physical in emulated.items():
            with self.subTest(op_id=op_id):
                self.assertNotIn(op_id, _MMA_FRAGMENT_INFO)
                self.assertIs(_frag_info(op_id), _MMA_FRAGMENT_INFO[physical])
                self.assertEqual(
                    _mma_c_frag_len(op_id), _MMA_FRAGMENT_INFO[physical].c_frag_len
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
