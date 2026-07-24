#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the tensor_quant GEMM TileEngine -> Dispatcher bridge.

Locks the config name format, the byte-exact codegen<->utils kernel-name contract,
the codegen-JSON projection, and the fp8/bf8 x rcr scope (the exact dtype/layout set
that the Old-TE gemm_quant_tensor.cpp instance builder registers). No GPU, no hipcc,
no Old-TE builder import required.
"""

import sys
import unittest
from pathlib import Path

_DISP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))

from gemm_tensor_quant_utils import (  # noqa: E402
    TensorQuantGemmProblem,
    default_fp8_config,
    default_bf8_config,
)
from unified_gemm_tensor_quant_codegen import (  # noqa: E402
    make_tensor_quant_kernel_name,
)


class TestConfigName(unittest.TestCase):
    def test_fp8_name_prefix(self):
        cfg = default_fp8_config()
        self.assertTrue(cfg.name.startswith("gemm_tensor_quant_fp8_rcr_"), cfg.name)

    def test_bf8_name_prefix(self):
        cfg = default_bf8_config()
        self.assertTrue(cfg.name.startswith("gemm_tensor_quant_bf8_rcr_"), cfg.name)

    def test_name_encodes_tiles(self):
        cfg = default_fp8_config()
        tile = f"{cfg.tile_m}x{cfg.tile_n}x{cfg.tile_k}"
        warp_tile = f"{cfg.warp_tile_m}x{cfg.warp_tile_n}x{cfg.warp_tile_k}"
        self.assertIn(tile, cfg.name)
        self.assertIn(warp_tile, cfg.name)


class TestNameContract(unittest.TestCase):
    """utils .name must be byte-exact with the codegen name builder."""

    def _assert_contract(self, cfg):
        expected = make_tensor_quant_kernel_name(
            variant_key=cfg.variant_key,
            layout=cfg.layout,
            pipeline=cfg.pipeline,
            epilogue=cfg.epilogue,
            scheduler=cfg.scheduler,
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            warp_tile_m=cfg.warp_tile_m, warp_tile_n=cfg.warp_tile_n,
            warp_tile_k=cfg.warp_tile_k,
        )
        self.assertEqual(cfg.name, expected)

    def test_fp8_contract(self):
        self._assert_contract(default_fp8_config())

    def test_bf8_contract(self):
        self._assert_contract(default_bf8_config())


class TestCodegenProjection(unittest.TestCase):
    def test_to_codegen_config_roundtrip(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        self.assertEqual(d["variant_keys"], [cfg.variant_key])
        self.assertEqual(d["layouts"], [cfg.layout])
        self.assertEqual(d["pipeline"], cfg.pipeline)
        self.assertEqual(d["scheduler"], cfg.scheduler)
        tc = d["tile_configs"][0]
        self.assertEqual(tc["tile_m"], cfg.tile_m)
        self.assertEqual(tc["warp_tile_k"], cfg.warp_tile_k)


class TestScope(unittest.TestCase):
    """gemm_quant_tensor.cpp registers exactly fp8/tensor and bf8/tensor, rcr only."""

    def test_default_variants(self):
        self.assertEqual(default_fp8_config().variant_key, "fp8")
        self.assertEqual(default_bf8_config().variant_key, "bf8")

    def test_layout_is_rcr(self):
        self.assertEqual(default_fp8_config().layout, "rcr")
        self.assertEqual(default_bf8_config().layout, "rcr")


class TestProblem(unittest.TestCase):
    def test_problem_defaults(self):
        p = TensorQuantGemmProblem(M=256, N=256, K=256)
        self.assertEqual(p.k_batch, 1)


if __name__ == "__main__":
    unittest.main()
