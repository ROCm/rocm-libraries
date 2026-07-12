#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the preshuffle GEMM bridge.

The preshuffle bridge pre-permutes the B operand, so its kernel differs from
plain GEMM by a ``_preshuffle`` name token (plus ``_permuteN`` when the
permute-N B-shuffle is selected). That token is the byte-parity invariant tying
config -> codegen -> the compiled kernel name the runtime reports, so a
preshuffle kernel can never collapse onto its plain-GEMM sibling. The
``permute_n`` knob is also surfaced at the top level of the codegen JSON, which
is where unified_gemm_codegen selects shuffle_b_permuteN vs shuffle_b.

Everything under test is pure host-side logic (name generation, the codegen
JSON projection, and the shipped configs/*.json). No GPU, hipcc, or dispatcher
build is required.

Run: python3 -m pytest tests/test_preshuffle_bridge.py -v
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
    _layout_from_kernel_name,
)

_CONFIG_DIR = (
    REPO_ROOT
    / "tile_engine"
    / "ops"
    / "gemm"
    / "gemm_preshuffle"
    / "configs"
)


def _make_config(**overrides):
    kw = dict(
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_c="fp16",
        dtype_acc="fp32",
        layout_a="row",
        layout_b="col",
        layout_c="row",
        variant="preshuffle",
    )
    kw.update(overrides)
    return GemmKernelConfig(**kw)


class TestPreshuffleName(unittest.TestCase):
    """The preshuffle kernel-name contract (the byte-parity invariant)."""

    def test_name_gains_preshuffle_suffix(self):
        cfg = _make_config()
        self.assertTrue(cfg.name.endswith("_preshuffle"), cfg.name)

    def test_name_is_plain_gemm_plus_suffix(self):
        # The preshuffle name must be exactly the plain-GEMM name +
        # "_preshuffle", so it shares codegen but never collides.
        common = dict(
            dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32",
            layout_a="row", layout_b="row", layout_c="row",
            tile_m=128, tile_n=128, tile_k=32,
            wave_m=2, wave_n=2, wave_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pipeline="compv4", scheduler="intrawave", epilogue="cshuffle",
            pad_m=True, pad_n=True, pad_k=True, persistent=False,
        )
        plain = GemmKernelConfig(variant="standard", **common)
        pre = GemmKernelConfig(variant="preshuffle", **common)
        self.assertEqual(pre.name, plain.name + "_preshuffle")

    def test_permute_n_appends_token(self):
        # permute_n selects the shuffle_b_permuteN pipeline; it must be visible
        # in the kernel name so the two shuffles never collapse.
        base = _make_config(permute_n=False)
        permuted = _make_config(permute_n=True)
        self.assertFalse(base.name.endswith("_permuteN"))
        self.assertTrue(permuted.name.endswith("_preshuffle_permuteN"), permuted.name)
        self.assertNotEqual(base.name, permuted.name)

    def test_dtype_and_layout_recover_from_name(self):
        for dtype in ("fp16", "bf16"):
            for la, lb, lc in (
                ("row", "col", "row"),
                ("row", "row", "row"),
                ("col", "col", "row"),
            ):
                cfg = _make_config(
                    dtype_a=dtype, dtype_b=dtype, dtype_c=_output_dtype(dtype),
                    layout_a=la, layout_b=lb, layout_c=lc,
                )
                name = cfg.name
                self.assertEqual(_dtype_from_kernel_name(name), dtype)
                self.assertEqual(_layout_from_kernel_name(name), cfg.layout)


class TestPreshuffleCodegenJson(unittest.TestCase):
    """The codegen JSON must surface the top-level permute_n knob."""

    def test_permute_n_in_codegen_json(self):
        self.assertEqual(_make_config(permute_n=True).to_codegen_json()["permute_n"], True)
        self.assertEqual(_make_config(permute_n=False).to_codegen_json()["permute_n"], False)

    def test_codegen_json_core_blocks(self):
        j = _make_config().to_codegen_json()
        self.assertIn("tile_config", j)
        self.assertIn("trait_config", j)
        # dispatcher wave_* is projected onto codegen warp_* (warps per block).
        self.assertEqual(j["tile_config"]["warp_m"], [2])


class TestPreshuffleShippedConfigs(unittest.TestCase):
    """The configs/*.json the bridge codegens from must be well-formed sweeps."""

    def test_config_dir_exists(self):
        self.assertTrue(_CONFIG_DIR.is_dir(), _CONFIG_DIR)

    def test_configs_are_valid_sweeps(self):
        configs = sorted(_CONFIG_DIR.glob("*.json"))
        self.assertGreater(len(configs), 0, "no preshuffle configs shipped")
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
