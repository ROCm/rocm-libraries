#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Codegen test for the gfx950 FMHA-forward d128 tile selection.

On gfx950 the kM0=64 qr_async tile is not generated for batch-mode problems without a mask,
logits soft-cap, bias, dropout or skip, because qr_async_trload covers them and is faster on
small grids. It must still be generated where trload is not (dropout, bias) and for masked or
group-mode problems, and other gfx9 targets keep it everywhere.

Pure Python (no GPU, no HIP toolchain): it only lists the blobs `generate.py` would emit.
"""

import os
import subprocess
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_FMHA_EX = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "example", "ck_tile", "01_fmha")
)
_GENERATE_PY = os.path.join(_FMHA_EX, "generate.py")

_TILE64 = "b64x128x32x128x32x128"


def _list_blobs(target):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "blobs.txt")
        subprocess.run(
            [
                sys.executable,
                _GENERATE_PY,
                "--targets",
                target,
                "--api",
                "fwd",
                "--list_blobs",
                out,
            ],
            check=True,
            cwd=_FMHA_EX,
        )
        with open(out) as f:
            return [os.path.basename(line.strip()) for line in f if line.strip()]


def _d128_16bit(blobs, *parts):
    return [
        b
        for b in blobs
        if (b.startswith("fmha_fwd_d128_bf16_") or b.startswith("fmha_fwd_d128_fp16_"))
        and all(p in b for p in parts)
    ]


class TestGfx950TileSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gfx950 = _list_blobs("gfx950")
        cls.gfx942 = _list_blobs("gfx942")

    def test_async64_dropped_for_plain_batch(self):
        self.assertEqual(
            _d128_16bit(
                self.gfx950,
                "_batch_",
                _TILE64,
                "_qr_async_vr_",
                "_nlogits_nbias_nmask_",
                "_ndropout_nskip_",
            ),
            [],
        )

    def test_trload_covers_plain_batch(self):
        self.assertTrue(
            _d128_16bit(
                self.gfx950,
                "_batch_",
                "_qr_async_trload_",
                "_nlogits_nbias_nmask_",
                "_ndropout_nskip_",
            )
        )

    def test_async64_kept_where_trload_is_not_used(self):
        for parts in (
            ("_batch_", "_mask_"),  # masked
            ("_batch_", "_dropout_"),  # dropout
            ("_batch_", "_alibi_"),  # bias
            ("_group_", "_nmask_"),  # group mode
        ):
            with self.subTest(parts=parts):
                self.assertTrue(
                    _d128_16bit(self.gfx950, _TILE64, "_qr_async_vr_", *parts)
                )

    def test_other_gfx9_unchanged(self):
        self.assertTrue(
            _d128_16bit(
                self.gfx942,
                "_batch_",
                _TILE64,
                "_qr_async_vr_",
                "_nlogits_nbias_nmask_",
                "_ndropout_nskip_",
            )
        )


if __name__ == "__main__":
    unittest.main()
