#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU regressions for gfx1250 Stream-K config selection and input verification.

GPU correctness is exercised by the native gfx1250 WMMA suite and the bridge
worker. These tests cover the host-side boundaries that choose a valid kernel
and compare its output against the same quantized inputs the device receives.
"""

import contextlib
import io
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
CK_ROOT = DISPATCHER_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

from gemm_utils import _SUPPORTED_ARCHES, expand_sweep  # noqa: E402

GFX1250_CONFIG = (
    CK_ROOT
    / "tile_engine"
    / "ops"
    / "gemm"
    / "gemm_streamk"
    / "configs"
    / "gfx1250_config.json"
)


class TestGfx1250Supported(unittest.TestCase):
    """gfx1250 must be a first-class supported arch (from the parent branch)."""

    def test_gfx1250_in_supported_arches(self):
        self.assertIn("gfx1250", _SUPPORTED_ARCHES)


class TestGfx1250StreamKConfig(unittest.TestCase):
    """The gfx1250 sweep uses WMMA 16x16x32 or 16x16x64 for its input type."""

    def setUp(self):
        self.assertTrue(
            GFX1250_CONFIG.exists(),
            f"missing gfx1250 stream-K config: {GFX1250_CONFIG}",
        )
        with open(GFX1250_CONFIG) as f:
            self.cfg = json.load(f)

    def test_config_uses_wmma_warp_tile(self):
        tc = self.cfg["tile_config"]
        # gfx1250 fp16/bf16 WMMA warp tile is 16x16x32; the MFMA 32x32x16 tile of
        # the default stream-K config does not run on RDNA4.
        self.assertEqual(tc["warp_tile_m"]["values"], [16])
        self.assertEqual(tc["warp_tile_n"]["values"], [16])
        self.assertEqual(tc["warp_tile_k"]["values"], [32, 64])

    def test_config_warp_combo_is_gfx1250_supported(self):
        # 2x2x1 (4-warp) is in the gfx1250 wave-combo list.
        tc = self.cfg["tile_config"]
        self.assertEqual(tc["warp_m"]["values"], [2])
        self.assertEqual(tc["warp_n"]["values"], [2])
        self.assertEqual(tc["warp_k"]["values"], [1])


class TestGfx1250StreamKExpansion(unittest.TestCase):
    """expand_sweep drives the whole gfx1250 stream-K host path (no GPU)."""

    def _expand(self, dtype="fp16"):
        return expand_sweep(
            str(GFX1250_CONFIG),
            "gfx1250",
            dtype=dtype,
            layout="rcr",
            variant="stream_k",
        )

    def test_expands_to_streamk_kernels_for_gfx1250(self):
        configs = self._expand("fp16")
        self.assertGreater(len(configs), 0)
        for c in configs:
            # arch is stamped concretely (never a silent gfx942 default).
            self.assertEqual(c.gfx_arch, "gfx1250")
            # stream-K identity preserved end-to-end.
            self.assertEqual(c.variant, "stream_k")
            self.assertTrue(c.name.endswith("_streamk"))
            # WMMA warp tile carried through into the kernel name.
            self.assertIn("16x16x32", c.name)

    def test_bf16_also_expands_for_gfx1250(self):
        configs = self._expand("bf16")
        self.assertGreater(len(configs), 0)
        for c in configs:
            self.assertEqual(c.gfx_arch, "gfx1250")
            self.assertTrue(c.name.endswith("_streamk"))

    def test_all_input_types_expand_to_supported_wmma_tiles(self):
        for dtype in ("fp16", "bf16", "fp8", "bf8"):
            with self.subTest(dtype=dtype):
                configs = self._expand(dtype)
                expected_k = 32 if dtype in ("fp16", "bf16") else 64
                self.assertEqual(len(configs), 4 if expected_k == 32 else 2)
                self.assertEqual({c.pipeline for c in configs}, {"compv3", "compv4"})
                for config in configs:
                    self.assertEqual(config.warp_tile_k, expected_k)
                    self.assertGreaterEqual(config.tile_k, expected_k)
                    self.assertEqual(
                        config.dtype_c, "bf16" if dtype == "bf16" else "fp16"
                    )

    def test_unsupported_arch_still_rejected(self):
        # The gfx1250 addition must not weaken the arch guard for bogus archs.
        with self.assertRaises(ValueError):
            expand_sweep(
                str(GFX1250_CONFIG),
                "gfx999",
                dtype="fp16",
                layout="rcr",
                variant="stream_k",
            )


class TestGfx1250StreamKDriver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(CK_ROOT / "tile_engine" / "ops" / "gemm"))
        import streamk_gemm_full_benchmark
        import run_one_streamk_gemm_kernel

        cls.driver = streamk_gemm_full_benchmark
        cls.worker = run_one_streamk_gemm_kernel

    def test_implicit_gfx1250_config_selects_wmma(self):
        args = SimpleNamespace(configs=[], arch="gfx1250")
        self.assertEqual(self.driver.resolve_configs(args), [str(GFX1250_CONFIG)])

    def test_explicit_config_is_preserved(self):
        args = SimpleNamespace(configs=["custom.json"], arch="gfx1250")
        self.assertEqual(self.driver.resolve_configs(args), ["custom.json"])

    def test_other_arch_default_is_preserved(self):
        args = SimpleNamespace(configs=[], arch="gfx942")
        self.assertEqual(
            self.driver.resolve_configs(args), [str(self.driver.DEFAULT_CONFIG)]
        )

    def test_ocp_reference_matches_device_input_quantization(self):
        import ml_dtypes

        for dtype, numpy_type, tiny in (
            ("fp8", ml_dtypes.float8_e4m3fn, 2**-10),
            ("bf8", ml_dtypes.float8_e5m2, 2**-17),
        ):
            for corrupt in (False, True):
                with self.subTest(dtype=dtype, corrupt=corrupt):
                    # OCP/FNUZ subnormal boundaries differ. A reference that
                    # silently encodes FNUZ must fail this comparison.
                    a = np.array([[tiny, 2 * tiny], [1, 0.5]], dtype=np.float32)
                    b = np.ones((2, 2), dtype=np.float32)
                    expected = a.astype(numpy_type).astype(np.float32) @ b
                    if corrupt:
                        expected = expected + 1
                    runner = SimpleNamespace(
                        kernel_name=f"gemm_{dtype}_rcr_streamk",
                        run=lambda *args: SimpleNamespace(
                            success=True, output=expected, time_ms=1, tflops=1
                        ),
                    )
                    output = io.StringIO()
                    with patch.object(
                        self.worker, "GpuGemmRunner", return_value=runner
                    ), patch.object(
                        sys.modules["gemm_utils"], "_default_use_ocp", return_value=True
                    ), patch.object(
                        np.random, "randn", side_effect=[a / 0.1, b / 0.1]
                    ), contextlib.redirect_stdout(output):
                        self.worker._run_one(
                            0,
                            "unused.so",
                            {"M": 2, "N": 2, "K": 2},
                            runner.kernel_name,
                            True,
                            1e-8,
                        )
                    self.assertEqual(
                        json.loads(output.getvalue())["verified"], not corrupt
                    )

    def test_regenerated_specs_preserve_gfx1250_wmma_types(self):
        import runpy

        sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
        from generate_arch_specs import load_arch_specs, generate_python_module
        import arch_specs_generated

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "arch_specs_generated.py"
            generate_python_module(
                load_arch_specs(DISPATCHER_DIR / "codegen" / "arch_specs.json"), target
            )
            generated = runpy.run_path(str(target))
        self.assertEqual(
            generated["WARP_TILE_SUPPORTED_COMBINATIONS"]["gfx1250"],
            arch_specs_generated.WARP_TILE_SUPPORTED_COMBINATIONS["gfx1250"],
        )


if __name__ == "__main__":
    unittest.main()
