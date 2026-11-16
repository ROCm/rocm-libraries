#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests for the gfx950 batched GEMM narrow-tile sweep configs.

default_config.json sweeps M/N blocks and tile_k from 64 up and pads nothing.
The *_config_gfx950.json files below add the blocks it leaves out: an M or N
block of 16 or 32, tile_k 32 (both bf16 warp tiles), padded kernels at native
vector widths, and (with the batched-only extra_wave_maps key) 1- and 2-wave
blocks. Each test expands a
config through the batched bridge (no GPU, hipcc or build) and checks that it
lists those kernels and only those.

Run: python3 -m pytest tests/test_batched_narrow_configs.py -v
"""

import functools
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

from batched_gemm_utils import _extra_wave_maps, expand_sweep  # noqa: E402
from gemm_utils import _warp_config_supported, _warp_supported_table  # noqa: E402

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
            {(128, 128, 32), (128, 192, 32), (192, 64, 32)},
        ),
        "k32_w32": (
            lambda c: c.tile_k == 32 and min(c.tile_m, c.tile_n) >= 64,
            (32, 32, 16),
            {(64, 64, 32), (256, 128, 32)},
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


class TestBatchedExtraWaveMaps(unittest.TestCase):
    """narrow_1w2w_*: extra_wave_maps adds 1- and 2-wave blocks for that config only."""

    EXTRA = {(1, 1, 1), (1, 2, 1), (2, 1, 1)}

    @staticmethod
    def _path(half):
        return _CONFIG_DIR / f"narrow_1w2w_{half}_config_gfx950.json"

    def _expand_with(self, half, extra):
        cfg = json.loads(self._path(half).read_text())
        cfg.pop("extra_wave_maps")
        if extra is not None:
            cfg["extra_wave_maps"] = {"gfx950": extra}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps(cfg))
            return expand_sweep(str(path), arch="gfx950", dtype="bf16", layout="rcr")

    def test_lists_the_extra_maps(self):
        for half, rule in (("m", lambda c: c.tile_m <= 32), ("n", lambda c: c.tile_n <= 32)):
            with self.subTest(half=half):
                kernels = _kernels(self._path(half))
                self.assertEqual({_waves(c) for c in kernels}, self.EXTRA | {(2, 2, 1)})
                self.assertEqual([c.name for c in kernels if not rule(c)], [])
        blocks = {(c.tile_m, c.tile_n, _waves(c)) for c in _kernels(self._path("m"))}
        self.assertIn((16, 16, (1, 1, 1)), blocks)

    def test_narrow_w32_one_wave_on_the_32_side(self):
        for half, rule in (("m", lambda c: c.tile_m == 32), ("n", lambda c: c.tile_n == 32)):
            with self.subTest(half=half):
                kernels = _kernels(_CONFIG_DIR / f"narrow_w32_{half}_config_gfx950.json")
                self.assertEqual([c.name for c in kernels if not rule(c)], [])
                self.assertEqual({_warp_tile(c) for c in kernels}, {(32, 32, 16)})
                self.assertLessEqual({_waves(c) for c in kernels}, self.EXTRA | _TE_WAVES)
        blocks = {(c.tile_m, c.tile_n, _waves(c)) for c in _kernels(
            _CONFIG_DIR / "narrow_w32_n_config_gfx950.json")}
        self.assertLessEqual(
            {(128, 32, (4, 1, 1)), (64, 32, (2, 1, 1)), (256, 32, (1, 1, 1))}, blocks
        )

    def test_without_the_key_only_the_table_maps(self):
        kernels = self._expand_with("m", None)
        self.assertEqual({_waves(c) for c in kernels}, {(2, 2, 1)})

    def test_rejects_maps_batched_does_not_take(self):
        for extra in ([[4, 2, 1]], [[2, 1, 2]], [[1, 1, 1], [8, 2, 1]]):
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    self._expand_with("m", extra)

    def test_maps_are_per_arch(self):
        self.assertEqual(_extra_wave_maps(str(self._path("m")), "gfx942"), ())
        self.assertEqual(set(_extra_wave_maps(str(self._path("m")), "gfx950")), self.EXTRA)

    def test_shared_gate_without_extra_is_the_table(self):
        for arch, table in _warp_supported_table().items():
            for wave in itertools.product((1, 2, 4, 8), (1, 2, 4, 8), (1, 2)):
                with self.subTest(arch=arch, wave=wave):
                    self.assertEqual(_warp_config_supported(*wave, arch), list(wave) in table)


if __name__ == "__main__":
    unittest.main()
