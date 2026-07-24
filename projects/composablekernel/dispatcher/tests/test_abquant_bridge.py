#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the abquant (A+B both quantized) GEMM TileEngine -> Dispatcher bridge.

Locks the config name format, the byte-exact codegen<->utils kernel-name contract, the
codegen-JSON projection, and the fp8/bf8/fp4 x rcr scope with the preshuffleB /
preshuffleQuant families that Old-TE gemm_abquant_quantgrouped*.cpp register. No GPU / hipcc.
"""

import sys
import unittest
from pathlib import Path

_DISP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))

from gemm_abquant_utils import (  # noqa: E402
    default_fp8_config,
    default_bf8_config,
    default_fp4_config,
    default_fp8_preshufflequant_config,
    default_fp8_preshuffleb_config,
    default_bf8_preshuffleb_config,
    default_fp4_preshuffleb_config,
    default_fp8_preshuffleb_preshufflequant_config,
)
from codegen_common import make_abquant_kernel_name  # noqa: E402

_ALL = [
    default_fp8_config,
    default_bf8_config,
    default_fp4_config,
    default_fp8_preshufflequant_config,
    default_fp8_preshuffleb_config,
    default_bf8_preshuffleb_config,
    default_fp4_preshuffleb_config,
    default_fp8_preshuffleb_preshufflequant_config,
]


class TestConfigName(unittest.TestCase):
    def test_name_prefix_and_layout(self):
        for ctor in _ALL:
            cfg = ctor()
            self.assertTrue(cfg.name.startswith(f"gemm_abquant_{cfg.variant_key}"), cfg.name)
            self.assertIn("rcr", cfg.name)

    def test_name_encodes_tiles(self):
        cfg = default_fp8_config()
        self.assertIn(f"{cfg.tile_m}x{cfg.tile_n}x{cfg.tile_k}", cfg.name)
        self.assertIn(f"{cfg.warp_tile_m}x{cfg.warp_tile_n}x{cfg.warp_tile_k}", cfg.name)


class TestNameContract(unittest.TestCase):
    def _assert_contract(self, cfg):
        expected = make_abquant_kernel_name(
            variant_key=cfg.variant_key,
            layout=cfg.layout,
            pipeline=cfg.pipeline,
            epilogue=cfg.epilogue,
            scheduler=cfg.scheduler,
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            warp_tile_m=cfg.warp_tile_m, warp_tile_n=cfg.warp_tile_n,
            warp_tile_k=cfg.warp_tile_k,
            aquant_group_k=cfg.aquant_group_k,
            bquant_group_n=cfg.bquant_group_n,
            bquant_group_k=cfg.bquant_group_k,
            preshuffle_b=cfg.preshuffle_b,
            preshuffle_bquant=cfg.preshuffle_bquant,
            eight_waves=cfg.eight_waves,
        )
        self.assertEqual(cfg.name, expected)

    def test_all_contracts(self):
        for ctor in _ALL:
            self._assert_contract(ctor())


class TestScope(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(default_fp8_config().variant_key, "fp8")
        self.assertEqual(default_bf8_config().variant_key, "bf8")
        self.assertEqual(default_fp4_config().variant_key, "fp4")

    def test_layout_is_rcr(self):
        for ctor in _ALL:
            self.assertEqual(ctor().layout, "rcr")

    def test_preshuffle_flags(self):
        self.assertFalse(default_fp8_config().preshuffle_b)
        self.assertTrue(default_fp8_preshuffleb_config().preshuffle_b)
        self.assertTrue(default_fp8_preshufflequant_config().preshuffle_bquant)
        pq = default_fp8_preshuffleb_preshufflequant_config()
        self.assertTrue(pq.preshuffle_b and pq.preshuffle_bquant)


class TestGfx950WarpTileK(unittest.TestCase):
    """Finding #1/#2: on gfx950 fp8/bf8 use K_Warp_Tile=128 (get_k_warp_tile),
    fp4 stays 32. Locks the *compiled shape*, not just the name string."""

    def test_fp8_bf8_warp_tile_k_is_128_on_gfx950(self):
        for ctor in (
            default_fp8_config,
            default_bf8_config,
            default_fp8_preshufflequant_config,
            default_fp8_preshuffleb_preshufflequant_config,
        ):
            cfg = ctor(gfx_arch="gfx950")
            self.assertEqual(cfg.warp_tile_k, 128, ctor.__name__)

    def test_fp4_warp_tile_k_is_32_on_gfx950(self):
        self.assertEqual(default_fp4_config(gfx_arch="gfx950").warp_tile_k, 32)
        self.assertEqual(default_fp4_preshuffleb_config(gfx_arch="gfx950").warp_tile_k, 32)

    def test_warp_tile_k_is_32_on_gfx942(self):
        for ctor in _ALL:
            cfg = ctor(gfx_arch="gfx942")
            self.assertEqual(cfg.warp_tile_k, 32, ctor.__name__)
            self.assertFalse(cfg.eight_waves, ctor.__name__)


class TestGfx950EightWaves(unittest.TestCase):
    """Finding #1: exactly the 6 fp8/bf8 kernels that route through the
    GemmConfig / GemmConfigPrefill aliases become EightWaves on gfx950:
      non-preshuffleb non-pq 1x128x128 (fp8, bf8)
      preshuffleb            {1,128}   (fp8, bf8)
    All other kernels (fp8 1x1x128 non-pq, all preshufflequant, all fp4) do not."""

    def _ew(self, cfg):
        # An eight_waves kernel must carry the flag, the 192x256x128 tile,
        # the 4x2x1 warps, warp_tile_k=128, the eightwaves pipeline and name tag.
        self.assertTrue(cfg.eight_waves, cfg.name)
        self.assertEqual((cfg.tile_m, cfg.tile_n, cfg.tile_k), (192, 256, 128), cfg.name)
        self.assertEqual((cfg.warp_m, cfg.warp_n, cfg.warp_k), (4, 2, 1), cfg.name)
        self.assertEqual(cfg.warp_tile_k, 128, cfg.name)
        self.assertEqual(cfg.pipeline, "eightwaves", cfg.name)
        self.assertIn("eightwaves", cfg.name)
        # eight_waves always uses the CShuffle epilogue (TiledMMAPermuteN=false).
        self.assertNotIn("permute_n", cfg.name, cfg.name)

    def test_the_six_eight_waves_kernels(self):
        ew = [
            default_fp8_config(bquant_group_n=128, gfx_arch="gfx950"),
            default_bf8_config(bquant_group_n=128, gfx_arch="gfx950"),
            default_fp8_preshuffleb_config(bquant_group_n=1, gfx_arch="gfx950"),
            default_fp8_preshuffleb_config(bquant_group_n=128, gfx_arch="gfx950"),
            default_bf8_preshuffleb_config(bquant_group_n=1, gfx_arch="gfx950"),
            default_bf8_preshuffleb_config(bquant_group_n=128, gfx_arch="gfx950"),
        ]
        self.assertEqual(len(ew), 6)
        for cfg in ew:
            self._ew(cfg)
        # preshuffleb eight_waves still carries preshuffle_b / double_smem.
        for cfg in ew[2:]:
            self.assertTrue(cfg.preshuffle_b and cfg.double_smem_buffer, cfg.name)

    def test_non_eight_waves_kernels(self):
        not_ew = [
            default_fp8_config(bquant_group_n=1, gfx_arch="gfx950"),   # hardcoded ABQuantPrefill
            default_fp4_config(gfx_arch="gfx950"),
            default_fp4_preshuffleb_config(gfx_arch="gfx950"),
            default_fp8_preshufflequant_config(bquant_group_n=1, gfx_arch="gfx950"),
            default_fp8_preshufflequant_config(bquant_group_n=128, gfx_arch="gfx950"),
            default_fp8_preshuffleb_preshufflequant_config(bquant_group_n=1, gfx_arch="gfx950"),
            default_fp8_preshuffleb_preshufflequant_config(bquant_group_n=128, gfx_arch="gfx950"),
        ]
        for cfg in not_ew:
            self.assertFalse(cfg.eight_waves, cfg.name)
            self.assertNotIn("eightwaves", cfg.name, cfg.name)


class TestCodegenProjection(unittest.TestCase):
    def test_to_codegen_config_roundtrip(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        self.assertEqual(d["variant_keys"], [cfg.variant_key])
        self.assertEqual(d["layouts"], [cfg.layout])
        self.assertEqual(d["aquant_group_k"], cfg.aquant_group_k)
        self.assertEqual(d["bquant_groups"][0]["bquant_group_n"], cfg.bquant_group_n)
        self.assertEqual(d["preshuffle_b"], cfg.preshuffle_b)


if __name__ == "__main__":
    unittest.main()
