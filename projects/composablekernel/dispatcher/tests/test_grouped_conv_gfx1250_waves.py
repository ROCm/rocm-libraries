#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
gfx1250 grouped convolution keeps the wave grids it had before the GEMM-only
16/32-wave grids were added.

arch_specs.json lists 16-wave (4x4, 8x2, 2x8) and 32-wave (8x4, 4x8) grids for
gfx1250. They are GEMM-only: grouped conv drops them
(codegen_common.grouped_conv_wave_configs), so its instance sets and its
config validation are unchanged by them. This test pins that.

Run: python3 -m pytest tests/test_grouped_conv_gfx1250_waves.py -v
"""

import collections
import logging
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

from arch_specs_generated import WARP_SUPPORTED_COMBINATIONS  # noqa: E402
from codegen_common import (  # noqa: E402
    GEMM_ONLY_WAVE_CONFIGS,
    grouped_conv_wave_configs,
)
from grouped_conv_utils import (  # noqa: E402
    auto_correct_grouped_conv_config,
    validate_grouped_conv_config,
)
from unified_grouped_conv_codegen import (  # noqa: E402
    GroupedConvVariant,
    get_default_configs,
)

GEMM_ONLY_GRIDS = [[4, 4, 1], [8, 2, 1], [2, 8, 1], [8, 4, 1], [4, 8, 1]]


def _wave_histogram(rule_set):
    """Count gfx1250 grouped-conv instances per (wave_m, wave_n, wave_k)."""
    logging.disable(logging.WARNING)
    try:
        cfgs = get_default_configs(
            arch="gfx1250",
            variants=[
                GroupedConvVariant.FORWARD,
                GroupedConvVariant.BACKWARD_DATA,
                GroupedConvVariant.BACKWARD_WEIGHT,
            ],
            ndims=[2, 3],
            datatypes=["fp16", "bf16", "fp32"],
            rule_set=rule_set,
        )
    finally:
        logging.disable(logging.NOTSET)
    hist = collections.Counter()
    for c in cfgs:
        t = getattr(c, "tile", None)
        if t is None:  # depthwise configs carry no wave grid
            continue
        hist[(t.warp_m, t.warp_n, t.warp_k)] += 1
    return dict(hist)


class TestGroupedConvWaveFilter(unittest.TestCase):
    def test_gemm_only_grids_listed_for_gfx1250(self):
        for g in GEMM_ONLY_GRIDS:
            self.assertIn(g, WARP_SUPPORTED_COMBINATIONS["gfx1250"])

    def test_gemm_only_table_is_gfx1250_new_grids(self):
        self.assertEqual(GEMM_ONLY_WAVE_CONFIGS, {"gfx1250": GEMM_ONLY_GRIDS})

    def test_filter_drops_only_gemm_only_grids(self):
        for arch, combos in WARP_SUPPORTED_COMBINATIONS.items():
            kept = grouped_conv_wave_configs(arch, combos)
            if arch != "gfx1250":
                # e.g. gfx950 keeps its pre-existing [8,2,1]/[4,4,1]
                self.assertEqual(kept, combos, arch)
        self.assertEqual(
            grouped_conv_wave_configs(
                "gfx1250", WARP_SUPPORTED_COMBINATIONS["gfx1250"]
            ),
            [
                [2, 4, 1],
                [1, 8, 1],
                [8, 1, 1],
                [4, 2, 1],
                [2, 1, 1],
                [1, 2, 2],
                [4, 1, 1],
                [1, 4, 1],
                [2, 2, 1],
            ],
        )
        # tuples (tile_math) are filtered too
        self.assertEqual(
            grouped_conv_wave_configs("gfx1250", [(4, 4, 1), (2, 2, 1)]), [(2, 2, 1)]
        )


class TestGfx1250GroupedConvInstanceSets(unittest.TestCase):
    """Wave histograms of the gfx1250 grouped-conv rule sets (pre-16/32-wave)."""

    def test_tiny(self):
        self.assertEqual(
            _wave_histogram("tiny"),
            {(1, 2, 2): 8, (2, 2, 1): 8, (8, 1, 1): 4},
        )

    def test_default(self):
        self.assertEqual(
            _wave_histogram("default"),
            {(1, 4, 1): 32, (2, 2, 1): 32, (4, 1, 1): 32},
        )


def _gfx1250_conv_config(wave):
    return {
        "tile_config": {
            "tile_k": 128,
            "tile_c": 128,
            "wave_m": wave[0],
            "wave_n": wave[1],
            "wave_k": wave[2],
            "warp_m": 16,
            "warp_n": 16,
            "warp_k": 32,
        },
        "trait_config": {
            "pipeline": "compv3",
            "epilogue": "cshuffle",
            "scheduler": "intrawave",
        },
        "variant": "2d_fwd",
        "ndim_spatial": 2,
        "arch": "gfx1250",
        "layout": "nhwgc",
        "dtype": "fp16",
    }


class TestGfx1250GroupedConvValidation(unittest.TestCase):
    def test_gemm_only_grid_rejected(self):
        for g in GEMM_ONLY_GRIDS:
            result = validate_grouped_conv_config(_gfx1250_conv_config(g))
            self.assertFalse(result.is_valid, g)
            self.assertTrue(any("GEMM-only" in e for e in result.errors), g)

    def test_auto_correct_picks_conv_grid(self):
        corrected, _ = auto_correct_grouped_conv_config(_gfx1250_conv_config([4, 4, 1]))
        tc = corrected["tile_config"]
        self.assertIn(
            [tc["wave_m"], tc["wave_n"], tc["wave_k"]],
            grouped_conv_wave_configs(
                "gfx1250", WARP_SUPPORTED_COMBINATIONS["gfx1250"]
            ),
        )

    def test_eight_wave_grid_still_accepted(self):
        result = validate_grouped_conv_config(_gfx1250_conv_config([2, 4, 1]))
        self.assertFalse(any("wave" in e.lower() for e in result.errors), result.errors)


if __name__ == "__main__":
    unittest.main()
