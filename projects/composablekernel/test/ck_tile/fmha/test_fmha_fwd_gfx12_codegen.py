#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate actual gfx12 forward sources; verify the BF16 D256 tile and that other tiles are unchanged."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_EXAMPLE = (
    Path(__file__).resolve().parent / "../../../example/ck_tile/01_fmha"
).resolve()


class TestGfx12ForwardCodegen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        out = Path(cls.tmp.name)
        result = subprocess.run(
            [
                sys.executable,
                str(_EXAMPLE / "generate.py"),
                "--targets",
                "gfx1201",
                "--api",
                "fwd",
                "--receipt",
                "2",
                "--optdim",
                "32,64,128,256",
                "--output_dir",
                str(out),
            ],
            cwd=_EXAMPLE,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        cls.names = [p.name for p in out.glob("fmha_fwd_*.cpp")]

    def check_tiles(self, dtype, expected):
        for dim, shape in expected.items():
            with self.subTest(dtype=dtype, dim=dim):
                names = [
                    n
                    for n in self.names
                    if n.startswith(f"fmha_fwd_d{dim}_{dtype}_batch_")
                ]
                self.assertTrue(names, f"No batch kernels for {dtype} D{dim}")
                self.assertTrue(all(f"_{shape}_" in n for n in names), names)
                self.assertTrue(any("_nmask_" in n for n in names))
                self.assertTrue(any("_mask_" in n for n in names))

    def test_bf16_tiles(self):
        self.check_tiles(
            "bf16",
            {
                32: "b64x64x16x32x32x32",
                64: "b64x64x32x64x32x64",
                256: "b64x64x32x256x16x256",
            },
        )

    def test_fp16_tiles_unchanged(self):
        self.check_tiles(
            "fp16",
            {
                32: "b64x64x16x32x32x32",
                64: "b64x64x32x64x32x64",
                256: "b64x64x32x256x32x256",
            },
        )

    def test_d128_upstream_tile_unchanged(self):
        for dtype in ("bf16", "fp16"):
            self.check_tiles(dtype, {128: "b128x64x32x128x32x128"})


if __name__ == "__main__":
    unittest.main()
