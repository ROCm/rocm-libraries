# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The MMA catalog's `emulation` descriptor, and the native-preferred rule.

An emulated atom has no native encoding: the lowering expands it into repeated
issues of another atom. It therefore has no fragment layout of its own and
borrows the physical atom's, and it must lose to a native atom of the same shape
whenever one exists. These tests pin both, plus the load-time validation that
keeps a malformed descriptor from reaching the lowering.
"""

import unittest
from unittest import mock

import pytest
from rocke.core.arch import ArchTarget, MmaEmulation, MmaEmulationRecipe, known_arches
from rocke.core.arch.target import (
    _MMA_FRAGMENT_INFO,
    _build_mma_op,
    _prefer_native,
)

# A real bf16 atom, so derived fragment metadata has something true to point at.
PHYSICAL = "mfma_f32_16x16x32_bf16"


def _row(**over):
    row = {
        "family": "mma",
        "a": "xf32",
        "b": "xf32",
        "c": "fp32",
        "m": 16,
        "n": 16,
        "k": 32,
        "op_id": "xf32_bf16x3_f32_16x16x32",
        "emulation": {"recipe": "bf16x3", "physical_op_id": PHYSICAL},
    }
    row.update(over)
    return row


class TestEmulationDescriptor(unittest.TestCase):
    def test_absent_key_means_native(self):
        op = _build_mma_op(
            {
                "family": "mma",
                "a": "bf16",
                "b": "bf16",
                "c": "fp32",
                "m": 16,
                "n": 16,
                "k": 32,
                "op_id": PHYSICAL,
            }
        )
        self.assertIsNone(op.emulation)
        self.assertTrue(op.is_native)

    def test_descriptor_is_parsed_and_typed(self):
        op = _build_mma_op(_row())
        self.assertFalse(op.is_native)
        self.assertEqual(op.emulation.recipe, MmaEmulationRecipe.BF16X3)
        self.assertEqual(op.emulation.physical_op_id, PHYSICAL)
        # Spelled as a plain string in JSON, but enum-typed once loaded.
        self.assertIsInstance(op.emulation.recipe, MmaEmulationRecipe)

    def test_fragment_metadata_is_inherited_from_the_physical_atom(self):
        """The whole point of physical_op_id: derive, never restate."""
        emulated = _build_mma_op(_row())
        physical = _MMA_FRAGMENT_INFO[PHYSICAL]
        self.assertEqual(emulated.a_frag_len, physical.a_frag_len)
        self.assertEqual(emulated.b_frag_len, physical.b_frag_len)
        self.assertEqual(emulated.c_frag_len, physical.c_frag_len)
        self.assertEqual(emulated.wave_size, physical.wave_size)
        # Layout maps too -- the logical fragment occupies the same coordinates.
        self.assertIsNotNone(emulated.a_layout())
        self.assertIsNotNone(emulated.c_layout())

    def test_unknown_recipe_is_rejected_at_load(self):
        with pytest.raises(ValueError, match="unknown MMA emulation recipe"):
            _build_mma_op(_row(emulation={"recipe": "fp8x4", "physical_op_id": PHYSICAL}))

    def test_missing_physical_op_id_is_rejected(self):
        with pytest.raises(ValueError, match="missing required key"):
            _build_mma_op(_row(emulation={"recipe": "bf16x3"}))

    def test_unknown_descriptor_keys_are_rejected(self):
        with pytest.raises(ValueError, match="unknown MMA emulation keys"):
            _build_mma_op(
                _row(
                    emulation={
                        "recipe": "bf16x3",
                        "physical_op_id": PHYSICAL,
                        "rounding": "rne",
                    }
                )
            )

    def test_unregistered_physical_op_id_is_rejected(self):
        """Otherwise the row silently inherits zero-length fragments."""
        with pytest.raises(ValueError, match="not .*registered in _MMA_FRAGMENT_INFO"):
            _build_mma_op(
                _row(emulation={"recipe": "bf16x3", "physical_op_id": "mfma_nonexistent"})
            )

    def test_descriptor_requires_a_physical_op_id_string(self):
        with pytest.raises(ValueError, match="requires a physical_op_id"):
            MmaEmulation(recipe="bf16x3", physical_op_id="")


class TestPreferNative(unittest.TestCase):
    """`_prefer_native` is the whole selection policy; test it directly."""

    def _emulated(self, k=32):
        return _build_mma_op(_row(k=k, op_id=f"xf32_bf16x3_f32_16x16x{k}"))

    def _native(self, k=32, a="xf32"):
        return _build_mma_op(
            {
                "family": "mma",
                "a": a,
                "b": a,
                "c": "fp32",
                "m": 16,
                "n": 16,
                "k": k,
                "op_id": f"native_{a}_16x16x{k}",
            }
        )

    def test_native_wins_at_equal_shape(self):
        native, emulated = self._native(), self._emulated()
        self.assertEqual(_prefer_native([emulated, native]), [native])

    def test_emulated_survives_when_no_native_covers_that_shape(self):
        native, emulated = self._native(k=32), self._emulated(k=16)
        # Same triple, different K: the emulated atom is the only way to get K=16.
        self.assertEqual(_prefer_native([native, emulated]), [native, emulated])

    def test_all_native_list_is_returned_unchanged(self):
        """The no-emulation fast path -- this is every pre-XF32 query."""
        ops = [self._native(k=16), self._native(k=32)]
        self.assertIs(_prefer_native(ops), ops)

    def test_order_is_preserved(self):
        a, b = self._native(k=16), self._native(k=32)
        self.assertEqual(_prefer_native([b, a]), [b, a])


class TestShippedCatalogUnaffected(unittest.TestCase):
    def test_every_shipped_atom_is_native(self):
        """No catalog row declares emulation yet, so selection cannot have moved."""
        for gfx in known_arches():
            for op in ArchTarget.from_gfx(gfx).mma.ops:
                with self.subTest(gfx=gfx, op_id=op.op_id):
                    self.assertTrue(op.is_native)
                    self.assertIsNone(op.emulation)

    def test_enumerate_is_unchanged_when_nothing_is_emulated(self):
        """Guards the comparator against perturbing existing selection.

        Compares live enumerate() output against the same filter with the
        preference rule bypassed entirely.
        """
        for gfx in known_arches():
            catalog = ArchTarget.from_gfx(gfx).mma
            for op in catalog.ops:
                with self.subTest(gfx=gfx, op_id=op.op_id):
                    with mock.patch(
                        "rocke.core.arch.target._prefer_native", side_effect=lambda x: x
                    ):
                        unfiltered = catalog.enumerate(
                            family=op.family,
                            a_dtype=op.a_dtype,
                            b_dtype=op.b_dtype,
                            c_dtype=op.c_dtype,
                            scales=(
                                op.a_scale_dtype,
                                op.b_scale_dtype,
                                op.scale_block_k,
                            ),
                        )
                    filtered = catalog.enumerate(
                        family=op.family,
                        a_dtype=op.a_dtype,
                        b_dtype=op.b_dtype,
                        c_dtype=op.c_dtype,
                        scales=(op.a_scale_dtype, op.b_scale_dtype, op.scale_block_k),
                    )
                    self.assertEqual(filtered, unfiltered)


if __name__ == "__main__":
    unittest.main()
