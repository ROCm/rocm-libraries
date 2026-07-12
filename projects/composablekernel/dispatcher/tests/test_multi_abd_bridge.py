#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the multi_abd GEMM bridge.

The Tile Engine -> Dispatcher multi_abd bridge relies on one hard invariant:
``GemmKernelConfig.name`` must reproduce, byte-for-byte, the kernel stem that
``unified_gemm_codegen.py`` bakes into the generated kernel (and that the .so
reports at runtime). For multi_abd that stem carries the 4-char (A,B,E,D)
layout plus a ``_multiabd_a{na}_b{nb}_d{nd}_{aop}_{bop}_{cdeop}`` suffix, so
distinct tensor counts / element-wise ops can never collapse onto one kernel.

These tests exercise only pure host-side logic (name generation, the codegen
JSON projection, and the shipped configs/*.json). No GPU, hipcc, or build is
required, so the suite runs green in CPU-only CI.

Run: python3 -m pytest tests/test_multi_abd_bridge.py -v
"""

import json
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
REPO_ROOT = DISPATCHER_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

from gemm_utils import (  # noqa: E402
    GemmKernelConfig,
    _output_dtype,
    _dtype_from_kernel_name,
)

# The shipped multi_abd sweep configs the bridge codegens from.
_CONFIG_DIR = (
    REPO_ROOT
    / "tile_engine"
    / "ops"
    / "gemm"
    / "gemm_multi_abd"
    / "configs"
)


def _make_config(**overrides):
    """A canonical multi_abd config; overrides tweak individual fields."""
    kw = dict(
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_c="fp16",
        dtype_acc="fp32",
        layout_a="row",
        layout_b="col",
        layout_c="row",
        layout_d="row",
        variant="multi_abd",
        num_a_tensors=2,
        num_b_tensors=2,
        num_d_tensors=2,
    )
    kw.update(overrides)
    return GemmKernelConfig(**kw)


class TestMultiAbdName(unittest.TestCase):
    """The multi_abd kernel-name contract (the byte-parity invariant)."""

    def test_name_carries_multiabd_suffix(self):
        cfg = _make_config()
        name = cfg.name
        # 4-char (A,B,E,D) layout, not the 3-char C layout.
        self.assertIn("_rcrr_", name)
        # multiabd tensor-count + op suffix, exactly as codegen emits it.
        self.assertTrue(
            name.endswith(
                "_multiabd_a2_b2_d2_PassThrough_PassThrough_PassThrough"
            ),
            name,
        )

    def test_full_stem_is_stable(self):
        # Pin the entire stem so any drift in the naming scheme is caught.
        cfg = _make_config(
            tile_m=128, tile_n=128, tile_k=32,
            wave_m=2, wave_n=2, wave_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pipeline="compv4", scheduler="intrawave", epilogue="cshuffle",
            pad_m=True, pad_n=True, pad_k=True, persistent=False,
        )
        expected = (
            "gemm_fp16_rcrr_compv4_cshuffle_intrawave"
            "_True_True_True_False"
            "_128x128x32_2x2x1_32x32x16"
            "_multiabd_a2_b2_d2_PassThrough_PassThrough_PassThrough"
        )
        self.assertEqual(cfg.name, expected)

    def test_tensor_counts_change_the_name(self):
        # Distinct tensor counts must not collapse onto one kernel name.
        base = _make_config().name
        more = _make_config(num_a_tensors=3, num_d_tensors=1).name
        self.assertNotEqual(base, more)
        self.assertIn("_a3_", more)
        self.assertIn("_d1_", more)

    def test_elementwise_ops_change_the_name(self):
        base = _make_config().name
        scaled = _make_config(cde_elementwise_op="AddScale").name
        self.assertNotEqual(base, scaled)
        self.assertTrue(scaled.endswith("_PassThrough_PassThrough_AddScale"))

    def test_dtype_recovers_from_name(self):
        # The runner reads the input dtype straight out of the compiled .so
        # name, so every dtype the bridge builds must round-trip.
        for dtype in ("fp16", "bf16"):
            cfg = _make_config(
                dtype_a=dtype, dtype_b=dtype, dtype_c=_output_dtype(dtype),
            )
            self.assertEqual(_dtype_from_kernel_name(cfg.name), dtype)

    def test_name_carries_four_char_layout(self):
        # multi_abd uses the 4-char (A,B,E,D) layout code in the stem; the D
        # char must reflect layout_d independently of the C layout.
        for la, lb, lc, ld in (
            ("row", "col", "row", "row"),
            ("row", "row", "row", "col"),
            ("col", "col", "row", "row"),
        ):
            cfg = _make_config(
                layout_a=la, layout_b=lb, layout_c=lc, layout_d=ld,
            )
            self.assertIn(f"_{cfg.layout4}_", cfg.name)


class TestMultiAbdCodegenJson(unittest.TestCase):
    """The codegen JSON projection must carry the multi_abd block."""

    def test_codegen_json_has_multi_abd_block(self):
        cfg = _make_config(
            num_a_tensors=2, num_b_tensors=2, num_d_tensors=2,
            cde_elementwise_op="AddScale",
        )
        j = cfg.to_codegen_json()
        self.assertIn("multi_abd_config", j)
        mabd = j["multi_abd_config"]
        self.assertEqual(mabd["num_a_tensors"], 2)
        self.assertEqual(mabd["num_b_tensors"], 2)
        self.assertEqual(mabd["num_d_tensors"], 2)
        self.assertEqual(mabd["cde_elementwise_op"], "AddScale")

    def test_codegen_json_core_blocks(self):
        j = _make_config().to_codegen_json()
        self.assertIn("tile_config", j)
        self.assertIn("trait_config", j)
        # dispatcher wave_* is projected onto codegen warp_* (warps per block).
        self.assertEqual(j["tile_config"]["warp_m"], [2])


class TestMultiAbdShippedConfigs(unittest.TestCase):
    """The configs/*.json the bridge codegens from must be well-formed sweeps."""

    def test_config_dir_exists(self):
        self.assertTrue(_CONFIG_DIR.is_dir(), _CONFIG_DIR)

    def test_configs_are_valid_sweeps(self):
        configs = sorted(_CONFIG_DIR.glob("*.json"))
        self.assertGreater(len(configs), 0, "no multi_abd configs shipped")
        for path in configs:
            with self.subTest(config=path.name):
                with open(path) as f:
                    data = json.load(f)
                self.assertIn("tile_config", data)
                self.assertIn("trait_config", data)
                tc = data["tile_config"]
                for key in (
                    "tile_m", "tile_n", "tile_k",
                    "warp_m", "warp_n", "warp_k",
                    "warp_tile_m", "warp_tile_n", "warp_tile_k",
                ):
                    self.assertIn(key, tc, f"{path.name} missing {key}")
                tr = data["trait_config"]
                for key in ("pipeline", "scheduler", "epilogue"):
                    self.assertIn(key, tr, f"{path.name} missing {key}")


if __name__ == "__main__":
    unittest.main()
