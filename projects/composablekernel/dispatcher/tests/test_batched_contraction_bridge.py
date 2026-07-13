#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the batched-contraction TileEngine -> Dispatcher bridge.

Lock the byte-exact name contract between codegen and utils, the codegen-JSON
projection, problem flop counting, and sweep expansion. No GPU required.
"""

import sys
import unittest
from pathlib import Path

_DISP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))

from batched_contraction_utils import (  # noqa: E402
    BatchedContractionKernelConfig,
    BatchedContractionProblem,
    expand_sweep,
)
from unified_batched_contraction_codegen import (  # noqa: E402
    make_batched_contraction_kernel_name,
    _spec_from_config,
)


class TestNameContract(unittest.TestCase):
    def _cfg(self, **kw):
        base = dict(dtype="fp16", layout="rcr", pipeline="compv3", epilogue="cshuffle",
                    scheduler="intrawave", tile_m=64, tile_n=64, tile_k=64,
                    warp_m=2, warp_n=2, warp_k=1, warp_tile_m=32, warp_tile_n=32, warp_tile_k=16)
        base.update(kw)
        return BatchedContractionKernelConfig(**base)

    def test_name_prefix(self):
        self.assertTrue(self._cfg().name.startswith("batched_contraction_fp16_rcr_"))

    def test_config_name_equals_codegen_name(self):
        # utils .name and codegen _spec_from_config(...).name must be byte-identical
        cfg = self._cfg(num_dim_g=2, num_dim_m=1, num_dim_n=1, num_dim_k=1)
        spec = _spec_from_config(cfg.to_codegen_config())
        self.assertEqual(cfg.name, spec.name)

    def test_num_dim_changes_name(self):
        a = self._cfg(num_dim_g=1).name
        b = self._cfg(num_dim_g=2).name
        self.assertNotEqual(a, b)
        self.assertIn("g1m1n1k1", a)
        self.assertIn("g2m1n1k1", b)

    def test_dtype_layout_in_name(self):
        self.assertIn("_bf16_rrr_", self._cfg(dtype="bf16", layout="rrr").name)

    def test_no_spaces(self):
        self.assertNotIn(" ", self._cfg().name)

    def test_elementwise_suffix(self):
        self.assertTrue(self._cfg(elementwise="MultiDAdd", num_d_tensors=1).name.endswith("_MultiDAdd"))
        self.assertNotIn("PassThrough", self._cfg().name)


class TestCodegenJson(unittest.TestCase):
    def test_roundtrip_tile(self):
        cfg = BatchedContractionKernelConfig(tile_m=128, tile_n=256, tile_k=64,
                                             warp_m=2, warp_n=2, warp_k=1,
                                             warp_tile_m=32, warp_tile_n=32, warp_tile_k=16)
        j = cfg.to_codegen_config()
        self.assertEqual(j["tile_config"]["tile_m"], 128)
        self.assertEqual(j["tile_config"]["tile_n"], 256)
        self.assertEqual(j["tile_config"]["warp_tile_k"], 16)
        self.assertEqual(j["datatype"], "fp16")

    def test_num_dim_projection(self):
        cfg = BatchedContractionKernelConfig(num_dim_g=2, num_dim_k=3)
        j = cfg.to_codegen_config()
        self.assertEqual(j["num_dim_g"], 2)
        self.assertEqual(j["num_dim_k"], 3)


class TestProblem(unittest.TestCase):
    def test_products(self):
        p = BatchedContractionProblem(g_dims=[2, 3], m_dims=[4, 16], n_dims=[128], k_dims=[4, 16])
        self.assertEqual(p.G, 6)
        self.assertEqual(p.M, 64)
        self.assertEqual(p.N, 128)
        self.assertEqual(p.K, 64)

    def test_flops(self):
        p = BatchedContractionProblem(g_dims=[3], m_dims=[128], n_dims=[128], k_dims=[128])
        self.assertEqual(p.flops, 2 * 3 * 128 * 128 * 128)

    def test_roundtrip(self):
        p = BatchedContractionProblem(g_dims=[2], m_dims=[64], n_dims=[64], k_dims=[64], k_batch=2)
        self.assertEqual(BatchedContractionProblem.from_dict(p.to_dict()).to_dict(), p.to_dict())


class TestValidity(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(BatchedContractionKernelConfig(
            tile_m=64, tile_n=64, tile_k=64, warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16).is_valid())

    def test_invalid_divisibility(self):
        self.assertFalse(BatchedContractionKernelConfig(
            tile_m=48, tile_n=64, tile_k=64, warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=32, warp_tile_n=32, warp_tile_k=16).is_valid())


class TestSweep(unittest.TestCase):
    def test_expand_dedup_and_valid(self):
        config = {
            "tile_config": {
                "tile_m": {"values": [64, 128]}, "tile_n": {"values": [64]},
                "tile_k": {"values": [64]}, "warp_m": {"values": [2]},
                "warp_n": {"values": [2]}, "warp_k": {"values": [1]},
                "warp_tile_m": {"values": [32]}, "warp_tile_n": {"values": [32]},
                "warp_tile_k": {"values": [16]},
            },
            "trait_config": {"pipeline": {"values": ["compv3", "mem"]},
                             "scheduler": {"values": ["intrawave"]},
                             "epilogue": {"values": ["cshuffle"]}},
            "num_dim_g": 1, "num_dim_m": 1, "num_dim_n": 1, "num_dim_k": 1,
        }
        cfgs = expand_sweep(config)
        names = [c.name for c in cfgs]
        self.assertEqual(len(names), len(set(names)))  # deduped
        self.assertTrue(all(c.is_valid() for c in cfgs))
        self.assertEqual(len(cfgs), 2 * 2)  # (tile_m 2) x (pipeline 2)


if __name__ == "__main__":
    unittest.main()
