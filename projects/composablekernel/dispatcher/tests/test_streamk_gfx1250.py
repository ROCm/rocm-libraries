#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for stream-K GEMM bridge enablement on gfx1250 (MI400).

The shared gfx1250 GEMM enablement lands in the parent branch
(``users/muozturk/ck/gemm-universal-gfx1250-enable``): ``gfx1250`` in
``gemm_utils._SUPPORTED_ARCHES`` + wave-combo fallback, the gfx1250 wave/warp
combos in ``gemm_validation_utils``, and the gfx1250 fp16/bf16 ``[[16,16,32]]``
(WMMA) warp tiles in ``arch_specs_generated``. This PR is the stream-K-specific
layer on top: a gfx1250 sweep config with WMMA (16x16x32) warp tiles, since the
default stream-K sweep config uses MFMA (32x32x16) tiles that do NOT run on
gfx1250's RDNA4 WMMA engine.

These tests lock in the pure host-side plumbing that must hold for stream-K on
gfx1250 -- all reachable without a GPU:

  * ``expand_sweep`` accepts ``arch='gfx1250'`` for the stream-K variant and
    stamps that arch onto every produced kernel (so the compile command's
    ``-DGFX_ARCH`` / ``--offload-arch`` is a concrete supported arch, never a
    silent gfx942 default).
  * The bundled ``gfx1250_config.json`` uses the WMMA 16x16x32 warp tile
    (16/16/32), NOT the MFMA 32x32x16 tile of the default config -- MFMA tiles
    silently produce no runnable kernel on gfx1250.
  * Every expanded config keeps the stream-K identity (``_streamk`` suffix and
    ``stream_k`` variant) so the runtime registry lookup key still matches.

NOTE (runtime, out of scope for this CPU test): on the MI400/gfx1250 node the
stream-K kernels codegen and compile cleanly with these tiles, but the kernel
launch itself hangs (the cross-workgroup K-reduction coherency handshake in
streamk_gemm_coherency.hpp does not complete on RDNA4). That is a C++ kernel
issue tracked separately; this PR delivers the Python/JSON enablement and pins
the host-side contract.

No GPU is touched. Run:
    python3 -m pytest tests/test_streamk_gfx1250.py -v
"""

import json
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
    """The bundled gfx1250 stream-K sweep config uses WMMA (16x16x32) tiles."""

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
        self.assertEqual(tc["warp_tile_k"]["values"], [32])

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


if __name__ == "__main__":
    unittest.main()
