# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Consistency guards for the IR-layer MMA metadata tables in ``rocke.core.ir``.

``IRBuilder.mma`` sizes a ``tile.mma`` result vector from a bare ``op_id`` string
via the module-level tables ``_MMA_C_FRAG_LEN`` (accumulator fragment length),
``_MMA_C_INT_OP_IDS`` (which atoms accumulate in i32), and ``_MMA_RESULT_HINT``
(SSA result-name hints). These tables are hand-maintained and keyed by op_id, so
the tables must stay mutually consistent: any op_id used by the int-accumulator
set or the result-hint map must also have a fragment length registered, or
``IRBuilder.mma`` would raise on an otherwise-valid atom.

CPU-only: imports the tables and the ``_mma_c_frag_len`` accessor directly, no
GPU / compile / launch required.
"""

from __future__ import annotations

import unittest

from rocke.core.ir import (
    _MMA_C_FRAG_LEN,
    _MMA_C_INT_OP_IDS,
    _MMA_RESULT_HINT,
    _mma_c_frag_len,
)


class TestMmaFragTables(unittest.TestCase):
    def test_frag_lengths_are_positive(self):
        self.assertTrue(_MMA_C_FRAG_LEN, "the frag-length table must not be empty")
        for op_id, frag in _MMA_C_FRAG_LEN.items():
            self.assertIsInstance(frag, int)
            self.assertGreater(
                frag, 0, msg=f"c_frag_len for {op_id!r} must be positive"
            )

    def test_accessor_matches_table_and_raises_on_unknown(self):
        for op_id, frag in _MMA_C_FRAG_LEN.items():
            self.assertEqual(_mma_c_frag_len(op_id), frag)
        with self.assertRaises(ValueError):
            _mma_c_frag_len("not_a_real_op_id")

    def test_int_acc_op_ids_have_frag_lengths(self):
        for op_id in _MMA_C_INT_OP_IDS:
            self.assertIn(
                op_id,
                _MMA_C_FRAG_LEN,
                msg=f"int-accumulator op_id {op_id!r} is missing from "
                f"_MMA_C_FRAG_LEN, so IRBuilder.mma would raise for it",
            )

    def test_result_hint_op_ids_have_frag_lengths(self):
        for op_id in _MMA_RESULT_HINT:
            self.assertIn(
                op_id,
                _MMA_C_FRAG_LEN,
                msg=f"result-hint op_id {op_id!r} is missing from _MMA_C_FRAG_LEN",
            )

    def test_known_frag_lengths(self):
        # Spot-check representative 16x16 vs 32x32 accumulator widths.
        self.assertEqual(_mma_c_frag_len("mfma_f32_16x16x16_f16"), 4)
        self.assertEqual(_mma_c_frag_len("mfma_f32_32x32x8_f16"), 16)
        self.assertEqual(_mma_c_frag_len("wmma_f32_16x16x16_f16"), 8)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
