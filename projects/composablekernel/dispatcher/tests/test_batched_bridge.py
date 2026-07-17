#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the batched GEMM bridge.

Batched GEMM reuses the plain-GEMM config/codegen machinery but pins the
variant to ``"batched"`` so ``BatchedGemmKernelConfig.name`` gains a trailing
``_batched`` token. That token is the byte-parity invariant tying config ->
codegen -> the compiled kernel name the runtime reports, so distinct batched
kernels never collapse onto the plain-GEMM name.

Everything under test is pure host-side logic (name generation, the codegen
JSON projection, the batched problem flop count, and the shipped
configs/*.json). No GPU, hipcc, or dispatcher build is required.

Run: python3 -m pytest tests/test_batched_bridge.py -v
"""

import json
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
REPO_ROOT = DISPATCHER_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

from batched_gemm_utils import (  # noqa: E402
    BatchedGemmKernelConfig,
    BatchedGemmProblem,
)
from gemm_utils import (  # noqa: E402
    GemmKernelConfig,
    _output_dtype,
    _dtype_from_kernel_name,
)

_CONFIG_DIR = (
    REPO_ROOT
    / "tile_engine"
    / "ops"
    / "gemm"
    / "batched_gemm"
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
    )
    kw.update(overrides)
    return BatchedGemmKernelConfig(**kw)


class TestBatchedName(unittest.TestCase):
    """The batched kernel-name contract (the byte-parity invariant)."""

    def test_variant_is_batched(self):
        self.assertEqual(_make_config().variant, "batched")

    def test_name_gains_batched_suffix(self):
        cfg = _make_config()
        self.assertTrue(cfg.name.endswith("_batched"), cfg.name)

    def test_name_is_plain_gemm_plus_suffix(self):
        # The batched name must be exactly the plain-GEMM name + "_batched",
        # so the two bridges share codegen but never collide.
        common = dict(
            dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32",
            layout_a="row", layout_b="row", layout_c="row",
            tile_m=128, tile_n=128, tile_k=32,
            wave_m=2, wave_n=2, wave_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
            pipeline="compv4", scheduler="intrawave", epilogue="cshuffle",
            pad_m=True, pad_n=True, pad_k=True, persistent=False,
        )
        plain = GemmKernelConfig(**common)
        batched = BatchedGemmKernelConfig(**common)
        self.assertEqual(batched.name, plain.name + "_batched")

    def test_suffix_not_doubled(self):
        # Re-deriving the name must not append a second "_batched".
        cfg = _make_config()
        self.assertEqual(cfg.name.count("_batched"), 1)

    def test_dtype_recovers_from_name(self):
        for dtype in ("fp16", "bf16"):
            cfg = _make_config(
                dtype_a=dtype, dtype_b=dtype, dtype_c=_output_dtype(dtype),
            )
            self.assertEqual(_dtype_from_kernel_name(cfg.name), dtype)


class TestBatchedCodegenJson(unittest.TestCase):
    """The codegen JSON projection is inherited from the plain-GEMM config."""

    def test_codegen_json_core_blocks(self):
        j = _make_config().to_codegen_json()
        self.assertIn("tile_config", j)
        self.assertIn("trait_config", j)
        # dispatcher wave_* is projected onto codegen warp_* (warps per block).
        self.assertEqual(j["tile_config"]["warp_m"], [2])


class TestBatchedProblem(unittest.TestCase):
    """The batched problem carries the batch dimension into the flop count."""

    def test_flops_scale_with_batch(self):
        one = BatchedGemmProblem(batch_count=1, M=128, N=128, K=64)
        eight = BatchedGemmProblem(batch_count=8, M=128, N=128, K=64)
        self.assertEqual(eight.flops, 8 * one.flops)

    def test_flops_formula(self):
        p = BatchedGemmProblem(batch_count=4, M=32, N=16, K=8)
        self.assertEqual(p.flops, 2.0 * 4 * 32 * 16 * 8)

    def test_dict_roundtrip(self):
        p = BatchedGemmProblem(batch_count=3, M=64, N=48, K=16)
        back = BatchedGemmProblem.from_dict(p.to_dict())
        self.assertEqual(back.batch_count, 3)
        self.assertEqual((back.M, back.N, back.K), (64, 48, 16))


class TestBatchedShippedConfigs(unittest.TestCase):
    """The configs/*.json the bridge codegens from must be well-formed sweeps."""

    def test_config_dir_exists(self):
        self.assertTrue(_CONFIG_DIR.is_dir(), _CONFIG_DIR)

    def test_configs_are_valid_sweeps(self):
        configs = sorted(_CONFIG_DIR.glob("*.json"))
        self.assertGreater(len(configs), 0, "no batched configs shipped")
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
