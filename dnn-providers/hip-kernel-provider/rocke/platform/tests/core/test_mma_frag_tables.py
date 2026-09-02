# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Consistency guards for the SSOT-backed MMA metadata consumed by ``IRBuilder.mma``.

After the arch-SSOT cleanup, ``rocke.core.ir`` keeps *no* private frag-length or
int-accumulator tables. ``IRBuilder.mma`` sizes a ``tile.mma`` result vector from
a bare ``op_id`` string via three pieces:

* ``_mma_d_frag_len`` — result fragment length, resolved from the arch SSOT
  ``core.arch.target._MMA_FRAGMENT_INFO`` (no ir-side copy);
* ``_mma_d_is_int`` — whether the atom produces i32, resolved from the JSON
  catalog result dtype (``target._op_id_d_dtype``);
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

from rocke.core.arch.target import MmaOp, _MMA_FRAGMENT_INFO, _op_id_d_dtype
from rocke.core.ir import (
    F32,
    I32,
    IRBuilder,
    _MMA_RESULT_HINT,
    _mma_d_frag_len,
    _mma_d_is_int,
)


def _sizable_op_ids():
    """Op_ids the SSOT can actually size (positive accumulator frag length)."""
    return {
        op_id: info.d_frag_len
        for op_id, info in _MMA_FRAGMENT_INFO.items()
        if info.d_frag_len > 0
    }


class TestMmaFragTables(unittest.TestCase):
    def test_frag_lengths_are_positive(self):
        sizable = _sizable_op_ids()
        self.assertTrue(sizable, "the frag-length SSOT must expose at least one atom")
        for op_id, frag in sizable.items():
            self.assertIsInstance(frag, int)
            self.assertGreater(
                frag, 0, msg=f"d_frag_len for {op_id!r} must be positive"
            )

    def test_accessor_matches_ssot_and_raises_on_unknown(self):
        for op_id, frag in _sizable_op_ids().items():
            self.assertEqual(_mma_d_frag_len(op_id), frag)
        with self.assertRaises(ValueError):
            _mma_d_frag_len("not_a_real_op_id")

    def test_int_acc_op_ids_have_frag_lengths(self):
        int_op_ids = [op for op, dtype in _op_id_d_dtype().items() if dtype == "i32"]
        self.assertTrue(
            int_op_ids, "expected at least one i32-accumulator atom in the catalog"
        )
        for op_id in int_op_ids:
            self.assertTrue(
                _mma_d_is_int(op_id),
                msg=f"{op_id!r} is i32 in the catalog but _mma_d_is_int disagrees",
            )
            self.assertGreater(
                _mma_d_frag_len(op_id),
                0,
                msg=f"int-accumulator op_id {op_id!r} has no SSOT frag length, so "
                f"IRBuilder.mma would raise for it",
            )

    def test_result_hint_op_ids_have_frag_lengths(self):
        for op_id in _MMA_RESULT_HINT:
            self.assertGreater(
                _mma_d_frag_len(op_id),
                0,
                msg=f"result-hint op_id {op_id!r} has no SSOT frag length",
            )

    def test_known_frag_lengths(self):
        # Spot-check representative 16x16 vs 32x32 accumulator widths.
        self.assertEqual(_mma_d_frag_len("mfma_f32_16x16x16_f16"), 4)
        self.assertEqual(_mma_d_frag_len("mfma_f32_32x32x8_f16"), 16)
        self.assertEqual(_mma_d_frag_len("wmma_f32_16x16x16_f16"), 8)

    def test_mma_uses_d_metadata_for_result_and_keeps_three_sources(self):
        op = MmaOp(
            family="mma",
            a_dtype="fp16",
            b_dtype="fp16",
            c_dtype="i32",
            d_dtype="fp32",
            m=1,
            n=1,
            k=1,
            op_id="synthetic_four_role",
            c_frag_len=7,
            d_frag_len=3,
        )
        builder = IRBuilder("synthetic_four_role")
        a = builder.const_i32(1)
        b = builder.const_i32(2)
        c = builder.const_i32(3)
        d = builder.mma(op, a, b, c)

        self.assertEqual(d.type.count, 3)
        self.assertIs(d.type.elem, F32)
        self.assertEqual(d.op.name, "tile.mma")
        self.assertEqual(d.op.operands, [a, b, c])

        int_result = builder.mma(
            MmaOp(
                family="mma",
                a_dtype="fp16",
                b_dtype="fp16",
                c_dtype="fp32",
                d_dtype="i32",
                m=1,
                n=1,
                k=1,
                op_id="synthetic_i32_result",
                c_frag_len=2,
                d_frag_len=5,
            ),
            a,
            b,
            c,
        )
        self.assertEqual(int_result.type.count, 5)
        self.assertIs(int_result.type.elem, I32)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
