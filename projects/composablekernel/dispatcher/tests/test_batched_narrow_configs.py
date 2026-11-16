#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests for the gfx950 batched GEMM narrow-tile sweep configs.

default_config.json sweeps M/N blocks and tile_k from 64 up and pads nothing.
The *_config_gfx950.json files below add the blocks it leaves out: an M or N
block of 16 or 32, tile_k 32, and padded kernels at native vector widths. Each
test expands a config through the batched bridge (no GPU, hipcc or build) and
checks that it lists those kernels and only those.

Run: python3 -m pytest tests/test_batched_narrow_configs.py -v
"""

import functools
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

from batched_gemm_utils import expand_sweep  # noqa: E402

_CONFIG_DIR = DISPATCHER_DIR.parent / "tile_engine/ops/gemm/batched_gemm/configs"
# The gfx950 wave maps of the Tile Engine table (gemm_validation_utils).
_TE_WAVES = {(1, 4, 1), (2, 2, 1), (4, 1, 1)}


@functools.lru_cache(maxsize=None)
def _kernels(path, layout="rcr"):
    return expand_sweep(str(path), arch="gfx950", dtype="bf16", layout=layout)


def _waves(c):
    return (c.wave_m, c.wave_n, c.wave_k)


def _warp_tile(c):
    return (c.warp_tile_m, c.warp_tile_n, c.warp_tile_k)


class TestNarrowConfigs(unittest.TestCase):
    # config -> (rule every listed kernel meets, warp tile, M/N/K blocks it must list)
    CASES = {
        "narrow_m": (
            lambda c: c.tile_m <= 32,
            (16, 16, 32),
            {(16, 64, 32), (32, 32, 32), (32, 256, 128)},
        ),
        "narrow_n": (
            lambda c: c.tile_n <= 32 and c.tile_m >= 64,
            (16, 16, 32),
            {(64, 16, 32), (256, 32, 128)},
        ),
        "k32": (
            lambda c: c.tile_k == 32 and min(c.tile_m, c.tile_n) >= 64,
            (16, 16, 32),
            {(128, 128, 32)},
        ),
        "pads": (
            lambda c: c.tile_k == 64 and min(c.tile_m, c.tile_n) >= 64,
            (32, 32, 16),
            {(64, 64, 64), (192, 128, 64)},
        ),
    }

    def _check(self, check):
        for name, case in self.CASES.items():
            with self.subTest(config=name):
                check(_kernels(_CONFIG_DIR / f"{name}_config_gfx950.json"), *case)

    def test_lists_only_its_blocks(self):
        def check(kernels, rule, warp_tile, blocks):
            self.assertTrue(kernels)
            self.assertEqual([c.name for c in kernels if not rule(c)], [])
            self.assertEqual({_warp_tile(c) for c in kernels}, {warp_tile})
            self.assertLessEqual(blocks, {(c.tile_m, c.tile_n, c.tile_k) for c in kernels})

        self._check(check)

    def test_tile_engine_wave_maps(self):
        def check(kernels, *_):
            self.assertLessEqual({_waves(c) for c in kernels}, _TE_WAVES)

        self._check(check)

    def test_padded_kernels_at_native_widths(self):
        # Every pad_m/n/k combination, all at native vector widths: a problem that is
        # not a tile multiple in one extent gets a kernel padded only there.
        def check(kernels, *_):
            self.assertEqual(len({(c.pad_m, c.pad_n, c.pad_k) for c in kernels}), 8)
            widths = {(c.vector_size_a, c.vector_size_b, c.vector_size_c) for c in kernels}
            self.assertEqual(widths, {(0, 0, 0)})

        self._check(check)

    def test_every_layout_expands(self):
        for layout in ("ccr", "crr", "rrr"):
            with self.subTest(layout=layout):
                self.assertTrue(_kernels(_CONFIG_DIR / "narrow_m_config_gfx950.json", layout))

    def test_default_config_keeps_its_blocks(self):
        # The new files are separate sweeps: the default one still starts at 64.
        kernels = _kernels(_CONFIG_DIR / "default_config.json")
        self.assertEqual(min(min(c.tile_m, c.tile_n, c.tile_k) for c in kernels), 64)
        self.assertFalse([c.name for c in kernels if c.pad_m or c.pad_n or c.pad_k])
        self.assertLessEqual({_waves(c) for c in kernels}, _TE_WAVES)


if __name__ == "__main__":
    unittest.main()
