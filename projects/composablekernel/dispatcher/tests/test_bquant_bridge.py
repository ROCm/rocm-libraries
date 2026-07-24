#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the (non-grouped) bquant GEMM TileEngine -> Dispatcher bridge.

Locks the config name format (distinct `gemm_bquant` prefix, NOT the grouped bridge), the
byte-exact codegen<->utils kernel-name contract, the codegen-JSON projection, and the
fp8/bf8/fp8i4/bf8i4 + MX(bf16bf16/bf16bf8/bf16fp4) scope with preshuffleB / preshuffleQuant
families that Old-TE gemm_bquant_quantgrouped*.cpp register. No GPU / hipcc.
"""

import sys
import unittest
from pathlib import Path

_DISP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))

from gemm_bquant_utils import (  # noqa: E402
    NAME_PREFIX,
    BQuantGemmProblem,
    default_fp8_config,
    default_bf8_config,
    default_fp8i4_config,
    default_bf8i4_config,
    default_fp8_preshuffleb_config,
    default_fp8_preshufflequant_config,
    default_fp8_preshuffleb_bquant_config,
    default_mx_bf16bf16_config,
    default_mx_bf16bf8_config,
    default_mx_bf16fp4_config,
)
from codegen_common import make_bquant_kernel_name  # noqa: E402

_BASE = [default_fp8_config, default_bf8_config, default_fp8i4_config, default_bf8i4_config]
_MX = [default_mx_bf16bf16_config, default_mx_bf16bf8_config, default_mx_bf16fp4_config]
_ALL = _BASE + [
    default_fp8_preshuffleb_config,
    default_fp8_preshufflequant_config,
    default_fp8_preshuffleb_bquant_config,
] + _MX


class TestPrefix(unittest.TestCase):
    def test_name_prefix_is_gemm_bquant(self):
        self.assertEqual(NAME_PREFIX, "gemm_bquant")
        for ctor in _ALL:
            self.assertTrue(ctor().name.startswith("gemm_bquant_"), ctor().name)

    def test_not_grouped_prefix(self):
        # Must NOT collide with the grouped_gemm_bquant bridge namespace.
        for ctor in _ALL:
            self.assertFalse(ctor().name.startswith("grouped_"), ctor().name)


class TestNameContract(unittest.TestCase):
    def _assert_contract(self, cfg):
        expected = make_bquant_kernel_name(
            variant_key=cfg.variant_key,
            layout=cfg.layout,
            pipeline=cfg.pipeline,
            epilogue=cfg.epilogue,
            scheduler=cfg.scheduler,
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            warp_tile_m=cfg.warp_tile_m, warp_tile_n=cfg.warp_tile_n,
            warp_tile_k=cfg.warp_tile_k,
            quant_group_m=cfg.quant_group_m,
            quant_group_n=cfg.quant_group_n,
            quant_group_k=cfg.quant_group_k,
            preshuffle_b=cfg.preshuffle_b,
            preshuffle_bquant=cfg.preshuffle_bquant,
            name_prefix=NAME_PREFIX,
        )
        self.assertEqual(cfg.name, expected)

    def test_all_contracts(self):
        for ctor in _ALL:
            self._assert_contract(ctor())


class TestScope(unittest.TestCase):
    def test_base_variants(self):
        self.assertEqual([c().variant_key for c in _BASE],
                         ["fp8", "bf8", "fp8i4", "bf8i4"])

    def test_layout_is_rcr(self):
        for ctor in _ALL:
            self.assertEqual(ctor().layout, "rcr")

    def test_mx_pipeline_is_microscale(self):
        for ctor in _MX:
            self.assertEqual(ctor().pipeline, "microscale")

    def test_preshuffle_flags(self):
        self.assertFalse(default_fp8_config().preshuffle_b)
        self.assertTrue(default_fp8_preshuffleb_config().preshuffle_b)
        self.assertTrue(default_fp8_preshufflequant_config().preshuffle_bquant)
        pq = default_fp8_preshuffleb_bquant_config()
        self.assertTrue(pq.preshuffle_b and pq.preshuffle_bquant)


class TestCodegenProjection(unittest.TestCase):
    def test_to_codegen_config_roundtrip(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        self.assertEqual(d["variant_keys"], [cfg.variant_key])
        self.assertEqual(d["layouts"], [cfg.layout])
        self.assertEqual(d["quant_groups"][0]["quant_group_k"], cfg.quant_group_k)
        self.assertEqual(d["preshuffle_b"], cfg.preshuffle_b)


class TestProblem(unittest.TestCase):
    def test_problem_defaults(self):
        p = BQuantGemmProblem(M=256, N=256, K=256)
        self.assertEqual(p.k_batch, 1)
        self.assertEqual(p.quant_group_k, 128)


if __name__ == "__main__":
    unittest.main()
