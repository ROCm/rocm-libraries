# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The two shipped gfx950 XF32 atoms.

These rows are the first users of the emulation descriptor. What matters about
them is not that they exist but that they are *faithful*: the fragment they
describe must be the physical bf16 atom's fragment, or the lowering will hand
the instruction the wrong number of elements. Since the C catalog restates the
fragment metadata rather than deriving it (a static table has no build step),
that equality is pinned here on both engines.
"""

import unittest

from rocke.core.arch import ArchTarget, MmaEmulationRecipe, known_arches

EMULATED = {
    "xf32_bf16x3_f32_16x16x32": ("mfma_f32_16x16x32_bf16", (16, 16, 32)),
    "xf32_bf16x3_f32_32x32x16": ("mfma_f32_32x32x16_bf16", (32, 32, 16)),
}


def _ops(gfx="gfx950"):
    return {op.op_id: op for op in ArchTarget.from_gfx(gfx).mma.ops}


class TestShippedRows(unittest.TestCase):
    def test_both_rows_are_present_on_gfx950(self):
        ops = _ops()
        for op_id, (_, shape) in EMULATED.items():
            with self.subTest(op_id=op_id):
                self.assertIn(op_id, ops)
                op = ops[op_id]
                self.assertEqual((op.m, op.n, op.k), shape)
                self.assertEqual(op.family, "mma")

    def test_rows_are_bf16x3_emulation_over_the_named_physical_atom(self):
        ops = _ops()
        for op_id, (physical, _) in EMULATED.items():
            with self.subTest(op_id=op_id):
                op = ops[op_id]
                self.assertFalse(op.is_native)
                self.assertEqual(op.emulation.recipe, MmaEmulationRecipe.BF16X3)
                self.assertEqual(op.emulation.physical_op_id, physical)

    def test_fragment_metadata_equals_the_physical_atom(self):
        """The one way these rows can be wrong and still load."""
        ops = _ops()
        for op_id, (physical, _) in EMULATED.items():
            with self.subTest(op_id=op_id):
                emulated, real = ops[op_id], ops[physical]
                self.assertEqual(emulated.a_frag_len, real.a_frag_len)
                self.assertEqual(emulated.b_frag_len, real.b_frag_len)
                self.assertEqual(emulated.c_frag_len, real.c_frag_len)
                self.assertEqual(emulated.wave_size, real.wave_size)
                self.assertEqual(emulated.a_layout(), real.a_layout())
                self.assertEqual(emulated.b_layout(), real.b_layout())
                self.assertEqual(emulated.c_layout(), real.c_layout())

    def test_the_physical_atom_is_itself_native(self):
        """No chains: an emulated atom must expand into real hardware in one hop."""
        ops = _ops()
        for physical, _ in EMULATED.values():
            with self.subTest(physical=physical):
                self.assertTrue(ops[physical].is_native)

    def test_rows_accumulate_in_fp32_and_are_unscaled(self):
        ops = _ops()
        for op_id in EMULATED:
            with self.subTest(op_id=op_id):
                op = ops[op_id]
                self.assertEqual(op.c_dtype, "fp32")
                # D is not spelled in the row: it equals the accumulator, and
                # saying so explicitly would only create a second place to drift.
                self.assertIsNone(op.d_dtype)
                self.assertEqual(op.result_dtype, "fp32")
                self.assertIsNone(op.a_scale_dtype)
                self.assertIsNone(op.b_scale_dtype)


class TestSelection(unittest.TestCase):
    def test_xf32_queries_reach_both_rows_on_gfx950(self):
        got = ArchTarget.from_gfx("gfx950").mma.enumerate(
            family="mma", a_dtype="xf32", b_dtype="xf32", c_dtype="fp32"
        )
        self.assertEqual({op.op_id for op in got}, set(EMULATED))

    def test_largest_k_picks_the_16x16_row(self):
        catalog = ArchTarget.from_gfx("gfx950").mma
        shape = {"a_dtype": "xf32", "b_dtype": "xf32", "c_dtype": "fp32", "m": 16, "n": 16}
        self.assertTrue(catalog.has_shape(k=32, **shape))
        self.assertEqual(
            catalog.select_largest_k(**shape).op_id, "xf32_bf16x3_f32_16x16x32"
        )

    def test_no_other_arch_offers_xf32(self):
        """Absence of a row is an explicit rejection -- there is no fallback."""
        for gfx in known_arches():
            if gfx == "gfx950":
                continue
            with self.subTest(gfx=gfx):
                self.assertFalse(
                    ArchTarget.from_gfx(gfx).mma.enumerate(
                        family="mma", a_dtype="xf32", b_dtype="xf32", c_dtype="fp32"
                    )
                )

    def test_fp32_selection_on_gfx950_is_untouched(self):
        """The rows must not leak into the native fp32 MFMA path."""
        got = ArchTarget.from_gfx("gfx950").mma.enumerate(
            family="mma", a_dtype="fp32", b_dtype="fp32", c_dtype="fp32"
        )
        self.assertEqual(
            {op.op_id for op in got},
            {"mfma_f32_16x16x4_f32", "mfma_f32_32x32x2_f32"},
        )

    def test_bf16_selection_on_gfx950_is_untouched(self):
        got = ArchTarget.from_gfx("gfx950").mma.enumerate(
            family="mma", a_dtype="bf16", b_dtype="bf16", c_dtype="fp32"
        )
        self.assertEqual(
            {op.op_id for op in got},
            {
                "mfma_f32_16x16x16_bf16",
                "mfma_f32_16x16x32_bf16",
                "mfma_f32_32x32x8_bf16",
                "mfma_f32_32x32x16_bf16",
            },
        )


if __name__ == "__main__":
    unittest.main()
