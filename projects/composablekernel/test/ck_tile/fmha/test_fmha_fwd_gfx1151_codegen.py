# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""FMHA forward codegen tests for the gfx1151 tile table (CPU only)."""

import sys
import unittest
from pathlib import Path

_CK_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_CK_ROOT / "example" / "ck_tile" / "01_fmha"))

from codegen.ops.fmha_fwd import get_fwd_blobs  # noqa: E402


def _kernels(targets, kernel_filter="*_bf16_*"):
    _, kernels = get_fwd_blobs(
        targets=targets,
        kernel_filter=kernel_filter,
        receipt=0,
        optdim_list=[32, 64, 128, 256],
        mask_impl="simplified",
    )
    return kernels


def _whole_kv(kernel):
    t = kernel.F_tile
    return t.F_bk0 == t.F_bk0max and t.F_bk1 == t.F_bn0


def _plain(kernel):
    return kernel.F_pipeline.F_dropout == "f" and kernel.F_pipeline.F_dpad == "f"


class TestGfx1151Codegen(unittest.TestCase):
    def test_gfx1151_has_its_own_arch(self):
        archs = {k.F_arch.name for k in _kernels(["gfx1151"])}
        self.assertEqual(archs, {"gfx1151"})

    def test_other_gfx115_targets_keep_the_family_tiles(self):
        for target in ("gfx1150", "gfx1152", "gfx1153"):
            with self.subTest(target=target):
                kernels = _kernels([target])
                self.assertEqual({k.F_arch.name for k in kernels}, {"gfx115"})
                self.assertFalse(
                    any(_whole_kv(k) for k in kernels if k.F_hdim in (64, 128))
                )

    def test_family_arch_is_not_compiled_for_gfx1151(self):
        kernels = _kernels(["gfx1150"])
        self.assertIn("!defined(__gfx1151__)", kernels[0].F_arch.preprocessor_check)

    def test_family_target_covers_gfx1151(self):
        # A build that passes only the family target must still get gfx1151 kernels.
        archs = {k.F_arch.name for k in _kernels(["gfx115"])}
        self.assertEqual(archs, {"gfx115", "gfx1151"})

    def test_whole_kv_tiles_only_pair_with_plain_pipelines(self):
        kernels = _kernels(["gfx1151"])
        whole = [k for k in kernels if k.F_hdim in (64, 128) and _whole_kv(k)]
        self.assertTrue(whole)
        self.assertTrue(all(_plain(k) for k in whole))

    def test_d128_plain_uses_only_the_whole_kv_tile(self):
        d128 = [k for k in _kernels(["gfx1151"]) if k.F_hdim == 128 and _plain(k)]
        self.assertTrue(d128)
        self.assertEqual(
            {(k.F_tile.F_bm0, k.F_tile.F_bn0, k.F_tile.F_bk0) for k in d128},
            {(64, 32, 128)},
        )

    def test_non_plain_pipelines_match_the_family_tiles(self):
        def tiles(target):
            return {
                (k.F_hdim, k.F_tile.name, k.F_pipeline.name)
                for k in _kernels([target])
                if not _plain(k)
            }

        self.assertEqual(tiles("gfx1151"), tiles("gfx1150"))


if __name__ == "__main__":
    unittest.main()
