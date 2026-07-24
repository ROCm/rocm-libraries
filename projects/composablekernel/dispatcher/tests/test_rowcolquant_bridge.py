#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the rowcolquant GEMM TileEngine -> Dispatcher bridge.

Locks the config name format, the byte-exact codegen<->utils kernel-name contract,
the codegen-JSON projection, and the fp8/bf8 x rcr scope (the exact dtype/layout set
that Old-TE gemm_quant_rowcol.cpp registers: per-row scale on A, per-col scale on B).
No GPU, no hipcc, no Old-TE builder import required.
"""

import sys
import unittest
from pathlib import Path

_DISP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))

from gemm_rowcolquant_utils import (  # noqa: E402
    RowColQuantGemmProblem,
    default_fp8_config,
    default_bf8_config,
    encode_fp8_bytes,
    quantize_dequantize_fp8,
    fp8_encoding_available,
)
from codegen_common import make_rowcolquant_kernel_name  # noqa: E402


class TestConfigName(unittest.TestCase):
    def test_fp8_name_prefix(self):
        self.assertTrue(default_fp8_config().name.startswith("gemm_rowcolquant_fp8_rcr_"),
                        default_fp8_config().name)

    def test_bf8_name_prefix(self):
        self.assertTrue(default_bf8_config().name.startswith("gemm_rowcolquant_bf8_rcr_"),
                        default_bf8_config().name)

    def test_name_encodes_tiles(self):
        cfg = default_fp8_config()
        self.assertIn(f"{cfg.tile_m}x{cfg.tile_n}x{cfg.tile_k}", cfg.name)
        self.assertIn(f"{cfg.warp_tile_m}x{cfg.warp_tile_n}x{cfg.warp_tile_k}", cfg.name)


class TestNameContract(unittest.TestCase):
    def _assert_contract(self, cfg):
        expected = make_rowcolquant_kernel_name(
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
        self.assertEqual(d["tile_configs"][0]["tile_m"], cfg.tile_m)
        self.assertEqual(d["tile_configs"][0]["warp_tile_k"], cfg.warp_tile_k)


class TestScope(unittest.TestCase):
    """gemm_quant_rowcol.cpp registers exactly fp8/rowcol and bf8/rowcol, rcr only."""

    def test_default_variants(self):
        self.assertEqual(default_fp8_config().variant_key, "fp8")
        self.assertEqual(default_bf8_config().variant_key, "bf8")

    def test_layout_is_rcr(self):
        self.assertEqual(default_fp8_config().layout, "rcr")
        self.assertEqual(default_bf8_config().layout, "rcr")


class TestProblem(unittest.TestCase):
    def test_problem_defaults(self):
        p = RowColQuantGemmProblem(M=256, N=256, K=256)
        self.assertEqual(p.k_batch, 1)


@unittest.skipUnless(fp8_encoding_available(), "ml_dtypes fp8 not installed")
class TestFp8Encode(unittest.TestCase):
    """The self-test's genuine numeric path depends on these host-side encoders.

    Encoded fp8/bf8 must be exactly 1 byte per element (the ctypes lib reads
    A/B as const fp8_t*/bf8_t*), and the reference-side quantize->dequantize
    must produce values consistent with that same encoding.
    """

    def test_encode_is_one_byte_per_element(self):
        import numpy as np

        a = np.array([[0.5, -1.25, 2.0, -0.03]], dtype=np.float32)
        for variant in ("fp8", "bf8"):
            enc = encode_fp8_bytes(a, variant)
            self.assertEqual(enc.dtype, np.uint8)
            self.assertEqual(enc.shape, a.shape)
            self.assertEqual(enc.nbytes, a.size)  # 1 byte/element, not 4

    def test_quant_dequant_matches_encoding(self):
        import numpy as np

        a = np.array([0.5, -1.25, 2.0, 1.0], dtype=np.float32)
        # Exactly representable e4m3 values survive the round-trip.
        qd = quantize_dequantize_fp8(a, "fp8")
        np.testing.assert_allclose(qd, a, rtol=0, atol=0)

    def test_fp8_and_bf8_round_differently(self):
        import numpy as np

        # 0.03 is not exactly representable; e4m3 and e5m2 round it differently.
        a = np.array([0.03], dtype=np.float32)
        self.assertNotEqual(
            float(quantize_dequantize_fp8(a, "fp8")[0]),
            float(quantize_dequantize_fp8(a, "bf8")[0]),
        )


if __name__ == "__main__":
    unittest.main()
