# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Consistency guards for the SSOT-backed MMA metadata consumed by ``IRBuilder.mma``.

After the arch-SSOT cleanup, ``rocke.core.ir`` keeps *no* private frag-length or
int-accumulator tables. ``IRBuilder.mma`` sizes a ``tile.mma`` result vector from
a bare ``op_id`` string via three pieces:

* ``_mma_c_frag_len`` — accumulator fragment length, resolved from the arch SSOT
  ``core.arch.target._MMA_FRAGMENT_INFO`` (no ir-side copy);
* ``_mma_c_dtype`` -- the atom's accumulator dtype, resolved from the JSON catalog
  (``target._op_id_c_dtype``); every catalog accumulator has an IR type;
* ``_MMA_RESULT_HINT`` — the ir-side SSA result-name hints (naming only, kept in
  ir.py to preserve byte-identical value numbering).

These must stay mutually consistent with the SSOT: any op_id that accumulates in
i32, or that carries a result-name hint, must also have a fragment length in the
SSOT, or ``IRBuilder.mma`` would raise on an otherwise-valid atom.

The C++ ``rocke_b_mma`` is reached through the ``rocke_ir_types`` test binary
(``ROCKE_IR_TYPES_TEST``): for every op_id both builders give the same result
type, or refuse the call with the same text.

CPU-only: imports the accessors and the arch SSOT directly, no GPU / compile /
launch required.
"""

from __future__ import annotations

import os
import subprocess
import unittest

from rocke.core.arch import ArchTarget, known_arches
from rocke.core.arch.target import (
    _MMA_FRAGMENT_INFO,
    _frag_info,
    _load_specs,
    _op_id_c_dtype,
    _op_id_physical_op_id,
)
from rocke.core.ir import (
    _MMA_RESULT_HINT,
    F16,
    F32,
    I32,
    IRBuilder,
    _mma_c_dtype,
    _mma_c_frag_len,
    dtype_to_ir_type,
)


def _sizable_op_ids():
    """Op_ids the SSOT can actually size (positive accumulator frag length)."""
    return {
        op_id: info.c_frag_len
        for op_id, info in _MMA_FRAGMENT_INFO.items()
        if info.c_frag_len > 0
    }


def _python_mma(op_id: str) -> str:
    """Build one MMA of ``op_id`` on i32 zero operands with the Python builder.

    The twin of ``rocke_ir_types --mma``: the result type, or ``error: <text>``
    when the builder refuses the call. A refusal must leave the kernel without a
    new op.
    """
    b = IRBuilder("mma_case")
    zero = b.const_i32(0)
    ops_before = len(b.kernel.body.ops)
    try:
        return b.mma(op_id, zero, zero, zero).type.name
    except ValueError as error:
        assert len(b.kernel.body.ops) == ops_before
        return f"error: {error}"


def _native_mma(executable: str, op_id: str) -> str:
    """The C++ builder's answer for ``op_id``, from ``rocke_ir_types --mma``."""
    return subprocess.run(
        [executable, "--mma", op_id], check=True, capture_output=True, text=True
    ).stdout.rstrip("\n")


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
            self.assertEqual(_mma_c_dtype(op_id), "i32")
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

    def test_an_op_id_outside_the_catalog_accumulates_in_fp32(self):
        # mfma_f32_4x4x4_f16 has fragment information but no catalog row.
        self.assertNotIn("mfma_f32_4x4x4_f16", _op_id_c_dtype())
        self.assertEqual(_mma_c_dtype("mfma_f32_4x4x4_f16"), "fp32")

    def test_mma_results_take_the_accumulator_dtype(self):
        for gfx, op_id, c, a_type, result in (
            ("gfx942", "mfma_f32_16x16x16_f16", (F32, 4), (F16, 4), "vec<f32x4>"),
            ("gfx1151", "wmma_i32_16x16x16_iu8", (I32, 8), (I32, 4), "vec<i32x8>"),
        ):
            with self.subTest(op_id=op_id):
                b = IRBuilder(f"result_{op_id}")
                a = b.zero_vec(*a_type)
                op = ArchTarget.from_gfx(gfx).mma.by_op_id(op_id)
                self.assertEqual(b.mma(op, a, a, b.zero_vec(*c)).type.name, result)
                self.assertEqual(b.mma(op_id, a, a, b.zero_vec(*c)).type.name, result)

    def test_every_catalog_accumulator_has_an_ir_type(self):
        for gfx in known_arches():
            for op in ArchTarget.from_gfx(gfx).mma.ops:
                with self.subTest(gfx=gfx, op_id=op.op_id):
                    dtype_to_ir_type(op.c_dtype)
                    self.assertEqual(_mma_c_dtype(op.op_id), op.c_dtype)

    def test_native_mma_results_match_the_python_builder(self):
        executable = os.environ.get("ROCKE_IR_TYPES_TEST")
        if not executable:
            self.skipTest("set ROCKE_IR_TYPES_TEST to the built rocke_ir_types test")
        catalog_op_ids = {
            op.op_id
            for gfx in known_arches()
            for op in ArchTarget.from_gfx(gfx).mma.ops
        }
        for op_id in sorted(catalog_op_ids | set(_MMA_FRAGMENT_INFO)):
            with self.subTest(op_id=op_id):
                self.assertEqual(_native_mma(executable, op_id), _python_mma(op_id))
        # An unknown op_id is refused by both. The hint after the op_id differs:
        # Python can take an MmaOp and lists the known op_ids, C cannot.
        native = _native_mma(executable, "not_a_real_op_id")
        refusal = "error: unknown MMA op_id 'not_a_real_op_id'; "
        self.assertTrue(native.startswith(refusal), native)
        self.assertTrue(_python_mma("not_a_real_op_id").startswith(refusal))

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
