#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the aquant (A-only quantized) GEMM TileEngine -> Dispatcher bridge.

Locks the config name format, the byte-exact codegen<->utils kernel-name contract, the
codegen-JSON projection, and the dtype/pipeline scope (fp8/bf8/fp8i4/bf8i4, decode via the
mem/interwave pipeline and preshufflequant via the compv3/intrawave pipeline) that Old-TE
gemm_aquant_quantgrouped{,_preshufflequant}.cpp register. No GPU, no hipcc required.
"""

import sys
import unittest
from pathlib import Path

_DISP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))

from gemm_aquant_utils import (  # noqa: E402
    AQuantGemmProblem,
    _aq_stride,
    _validate_arch,
    _LAYOUTS_DECODE,
    _LAYOUTS_PRESHUFFLEQUANT,
    _LAYOUTS_AQ_COLMAJOR,
    _SUPPORTED_ARCHS,
    _VARIANT_META,
    default_fp8_config,
    default_bf8_config,
    default_fp8i4_config,
    default_bf8i4_config,
    default_fp8_preshufflequant_config,
    default_bf8i4_preshufflequant_config,
)
from codegen_common import make_aquant_kernel_name  # noqa: E402

_DECODE = [
    ("fp8", default_fp8_config),
    ("bf8", default_bf8_config),
    ("fp8i4", default_fp8i4_config),
    ("bf8i4", default_bf8i4_config),
]


class TestConfigName(unittest.TestCase):
    def test_decode_name_prefix(self):
        for variant, ctor in _DECODE:
            cfg = ctor()
            self.assertTrue(cfg.name.startswith(f"gemm_aquant_{variant}_"), cfg.name)

    def test_name_encodes_tiles(self):
        cfg = default_fp8_config()
        self.assertIn(f"{cfg.tile_m}x{cfg.tile_n}x{cfg.tile_k}", cfg.name)
        self.assertIn(f"{cfg.warp_tile_m}x{cfg.warp_tile_n}x{cfg.warp_tile_k}", cfg.name)

    def test_pipeline_key_reflects_preshuffle(self):
        self.assertEqual(default_fp8_config().pipeline_key, "mem")
        self.assertEqual(default_fp8_preshufflequant_config().pipeline_key, "compv3")


class TestNameContract(unittest.TestCase):
    def _assert_contract(self, cfg):
        expected = make_aquant_kernel_name(
            variant_key=cfg.variant_key,
            layout=cfg.layout,
            pipeline=cfg.pipeline_key,
            epilogue="cshuffle",
            scheduler=cfg.scheduler,
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            warp_tile_m=cfg.warp_tile_m, warp_tile_n=cfg.warp_tile_n,
            warp_tile_k=cfg.warp_tile_k,
            quant_group_m=cfg.quant_group_m,
            quant_group_n=cfg.quant_group_n,
            quant_group_k=cfg.quant_group_k,
            preshuffle_aquant=cfg.preshuffle_aquant,
        )
        self.assertEqual(cfg.name, expected)

    def test_decode_contracts(self):
        for _, ctor in _DECODE:
            self._assert_contract(ctor())

    def test_preshufflequant_contracts(self):
        self._assert_contract(default_fp8_preshufflequant_config())
        self._assert_contract(default_bf8i4_preshufflequant_config())


class TestScope(unittest.TestCase):
    def test_decode_variants(self):
        self.assertEqual([v for v, _ in _DECODE],
                         [ctor().variant_key for _, ctor in _DECODE])

    def test_decode_uses_mem_pipeline(self):
        for _, ctor in _DECODE:
            self.assertFalse(ctor().preshuffle_aquant)

    def test_preshufflequant_flag(self):
        self.assertTrue(default_fp8_preshufflequant_config().preshuffle_aquant)


class TestCodegenProjection(unittest.TestCase):
    def test_to_codegen_config_roundtrip(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        self.assertEqual(d["variant_keys"], [cfg.variant_key])
        self.assertEqual(d["layouts"], [cfg.layout])
        self.assertEqual(d["quant_groups"][0]["quant_group_k"], cfg.quant_group_k)
        self.assertEqual(d["preshuffle_aquant"], cfg.preshuffle_aquant)


class TestProblem(unittest.TestCase):
    def test_problem_constructs(self):
        p = AQuantGemmProblem(M=128, N=256, K=512)
        self.assertEqual((p.M, p.N, p.K), (128, 256, 512))


class TestAQStride(unittest.TestCase):
    """ccr must carry the column-major AQ stride (M), all others row-major (QK_A)."""

    M, QK_A = 96, 4

    def test_ccr_is_column_major(self):
        # ccr: A=C B=C -> AQ column-major, leading dim = M (row count), not QK_A.
        self.assertIn("ccr", _LAYOUTS_AQ_COLMAJOR)
        self.assertEqual(_aq_stride("ccr", self.M, self.QK_A), self.M)

    def test_row_major_layouts_use_qk_a(self):
        for layout in ("rcr", "rrr", "crr"):
            self.assertNotIn(layout, _LAYOUTS_AQ_COLMAJOR)
            self.assertEqual(_aq_stride(layout, self.M, self.QK_A), self.QK_A)

    def test_ccr_stride_differs_from_row_major(self):
        # Guards against the round-1 bug where stride_AQ was always QK_A.
        self.assertNotEqual(
            _aq_stride("ccr", self.M, self.QK_A),
            _aq_stride("rcr", self.M, self.QK_A),
        )


class TestLayoutScope(unittest.TestCase):
    """Lock the layout scope and the 28-kernel (4 variant x 7 layout) count."""

    def test_decode_layouts(self):
        self.assertEqual(_LAYOUTS_DECODE, ("rcr", "rrr", "crr", "ccr"))

    def test_preshufflequant_excludes_ccr(self):
        self.assertEqual(_LAYOUTS_PRESHUFFLEQUANT, ("rcr", "rrr", "crr"))
        self.assertNotIn("ccr", _LAYOUTS_PRESHUFFLEQUANT)

    def test_full_kernel_count_is_28(self):
        # 4 variants x (4 decode layouts + 3 preshufflequant layouts) = 28.
        variants = len(_VARIANT_META)
        total = variants * (len(_LAYOUTS_DECODE) + len(_LAYOUTS_PRESHUFFLEQUANT))
        self.assertEqual(variants, 4)
        self.assertEqual(total, 28)


class TestSupportedArchs(unittest.TestCase):
    def test_gfx90a_supported_and_validated(self):
        self.assertIn("gfx90a", _SUPPORTED_ARCHS)
        self.assertEqual(_validate_arch("gfx90a"), "gfx90a")
        self.assertEqual(_validate_arch("gfx90a:sramecc+:xnack-"), "gfx90a:sramecc+:xnack-")

    def test_unsupported_arch_raises(self):
        with self.assertRaises(ValueError):
            _validate_arch("gfx1030")


if __name__ == "__main__":
    unittest.main()
