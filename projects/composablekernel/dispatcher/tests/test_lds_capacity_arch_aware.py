#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Regression tests for the architecture-aware LDS staging budget.

The budget for the A+B staging tiles used to be a single table keyed on the
pipeline alone. That held every architecture to gfx942's 64 KB, so gfx950
(160 KB of LDS) and gfx1250 (320 KB) silently lost the largest and deepest
tiles at codegen time -- they were never generated, never benchmarked and
never selectable.

These tests pin the property that makes the defect impossible to reintroduce:
the effective budget must differ between architectures with different LDS
capacities, and must never exceed what the silicon actually has.

Can be run as:
    python3 tests/test_lds_capacity_arch_aware.py
    ctest -R test_lds_capacity_arch_aware
"""

import sys
import unittest
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

from arch_filter import ArchFilter, KernelConfig  # noqa: E402
from arch_specs_generated import (  # noqa: E402
    LDS_CAPACITY_LIMITS_BY_ARCH,
    get_lds_limit,
)

# Source of truth: get_lds_size() in include/ck_tile/core/arch/arch.hpp
HARDWARE_LDS_KB = {
    "gfx908": 64,
    "gfx90a": 64,
    "gfx942": 64,
    "gfx950": 160,
    "gfx1100": 64,
    "gfx1200": 64,
    "gfx1201": 64,
    "gfx1250": 320,
}

# gfx942's budget is already correct, so this change must leave it untouched.
# Any diff here means the change stopped being a pure widening.
GFX942_FROZEN = {
    "mem": 65536,
    "compv1": 65536,
    "compv2": 65536,
    "compv3": 65536,
    "compv4": 32768,
    "compv5": 65536,
    "compv6": 32768,
    "preshufflev1": 32768,
    "preshufflev2": 32768,
    "default": 65536,
}

# Pipelines that get half the capacity. compv4 and preshufflev2 are genuinely
# double-buffered, so half is the exact model rather than a safety margin.
HALF_CAPACITY_PIPELINES = ("compv4", "compv6", "preshufflev1", "preshufflev2")


class TestLdsBudgetIsArchAware(unittest.TestCase):
    """The core guard: the budget must depend on the architecture."""

    def test_budget_differs_across_architectures(self):
        for pipeline in ("mem", "compv3", "compv4", "default"):
            with self.subTest(pipeline=pipeline):
                gfx942 = get_lds_limit("gfx942", pipeline)
                gfx950 = get_lds_limit("gfx950", pipeline)
                gfx1250 = get_lds_limit("gfx1250", pipeline)

                self.assertLess(
                    gfx942,
                    gfx950,
                    f"{pipeline}: gfx950 has 160 KB of LDS but is budgeted like "
                    f"gfx942 ({gfx942} B). The budget is architecture-blind again.",
                )
                self.assertLess(
                    gfx950,
                    gfx1250,
                    f"{pipeline}: gfx1250 has 320 KB of LDS but is budgeted at or "
                    f"below gfx950 ({gfx950} B).",
                )

    def test_budget_tracks_hardware_capacity(self):
        """Architectures with equal LDS get equal budgets, and vice versa."""
        for arch, capacity_kb in HARDWARE_LDS_KB.items():
            with self.subTest(arch=arch):
                expected = HARDWARE_LDS_KB["gfx942"] == capacity_kb
                actual = get_lds_limit(arch, "default") == get_lds_limit(
                    "gfx942", "default"
                )
                self.assertEqual(
                    expected,
                    actual,
                    f"{arch} has {capacity_kb} KB of LDS; its default budget "
                    f"should match gfx942's only when its capacity does.",
                )


class TestLdsBudgetIsSafe(unittest.TestCase):
    """Guards that keep a mis-edited budget from reaching a GPU."""

    def test_budget_never_exceeds_hardware(self):
        for arch, per_pipeline in LDS_CAPACITY_LIMITS_BY_ARCH.items():
            capacity = HARDWARE_LDS_KB[arch] * 1024
            for pipeline, budget in per_pipeline.items():
                with self.subTest(arch=arch, pipeline=pipeline):
                    self.assertLessEqual(
                        budget,
                        capacity,
                        f"{arch}/{pipeline}: budget {budget} B exceeds the "
                        f"{capacity} B the hardware has.",
                    )

    def test_gfx942_budget_is_unchanged(self):
        self.assertEqual(LDS_CAPACITY_LIMITS_BY_ARCH["gfx942"], GFX942_FROZEN)

    def test_every_architecture_is_present(self):
        """A new GPU must not silently inherit another architecture's budget."""
        self.assertEqual(set(LDS_CAPACITY_LIMITS_BY_ARCH), set(HARDWARE_LDS_KB))

    def test_double_buffered_pipelines_get_half(self):
        """Pipelines that stage two LDS buffers get half the capacity.

        Only the ordering and the ratio are pinned, not a byte count, so the
        test keeps holding when an architecture's capacity changes.
        """
        for arch, per_pipeline in LDS_CAPACITY_LIMITS_BY_ARCH.items():
            capacity = HARDWARE_LDS_KB[arch] * 1024
            for pipeline in HALF_CAPACITY_PIPELINES:
                with self.subTest(arch=arch, pipeline=pipeline):
                    self.assertEqual(per_pipeline[pipeline], capacity // 2)

    def test_unknown_arch_gets_the_smallest_budget(self):
        """An unrecognised target must not be handed more LDS than it may have."""
        smallest = min(p["default"] for p in LDS_CAPACITY_LIMITS_BY_ARCH.values())
        self.assertEqual(get_lds_limit("gfx9999", "default"), smallest)


class TestLdsValidationEndToEnd(unittest.TestCase):
    """The budget has to actually reach the validator, not just the table."""

    @staticmethod
    def _config(pipeline, tile_m=128, tile_n=256, tile_k=128):
        # fp16 A and B: 128x128x2 + 256x128x2 = 96 KB of staging.
        return KernelConfig(
            datatype_a="fp16",
            datatype_b="fp16",
            datatype_c="fp16",
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=8,
            pipeline=pipeline,
        )

    def _lds_errors(self, arch, config):
        result = ArchFilter(arch, strict_mode=False).validate_kernel(config)
        return [e for e in result.errors if "LDS capacity exceeded" in e]

    def test_96kb_tile_rejected_on_gfx942_accepted_on_gfx950(self):
        """The tile at the heart of the defect: fits gfx950, not gfx942."""
        config = self._config("compv3")
        self.assertTrue(
            self._lds_errors("gfx942", config),
            "96 KB of staging must not fit gfx942's 64 KB budget.",
        )
        self.assertFalse(
            self._lds_errors("gfx950", config),
            "96 KB of staging fits comfortably in gfx950's 160 KB.",
        )

    def test_error_message_names_the_architecture(self):
        """The old message was architecture-neutral, which hid the defect."""
        errors = self._lds_errors("gfx942", self._config("compv3"))
        self.assertTrue(errors)
        self.assertIn("gfx942", errors[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
