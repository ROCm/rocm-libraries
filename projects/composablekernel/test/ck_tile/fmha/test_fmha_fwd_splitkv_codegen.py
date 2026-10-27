#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_FMHA_EXAMPLE = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "example", "ck_tile", "01_fmha")
)
sys.path.insert(0, _FMHA_EXAMPLE)

from codegen.ops.fmha_fwd_splitkv import (  # noqa: E402
    FmhaFwdSplitKVApiTrait,
    get_fwd_splitkv_blobs,
)


def _pagecheck(bn0: int, page_shorter_than_tile: bool) -> str:
    trait = object.__new__(FmhaFwdSplitKVApiTrait)
    object.__setattr__(trait, "pagedkv", "t")
    object.__setattr__(trait, "page_shorter_than_tile", page_shorter_than_tile)
    object.__setattr__(trait, "bn0", bn0)
    object.__setattr__(trait, "constraint", "true")
    return trait.pagecheck


class SplitKvPageCheckTest(unittest.TestCase):
    def test_non_merge_tile_requires_page_aligned_origins(self):
        self.assertEqual(
            _pagecheck(64, False),
            "64 <= a.page_block_size && a.page_block_size % 64 == 0",
        )

    def test_merge_tile_requires_half_tile_aligned_pages(self):
        self.assertEqual(
            _pagecheck(32, True),
            "32 <= 2 * a.page_block_size && a.page_block_size % 16 == 0",
        )


class SplitKvCrossPageTileTest(unittest.TestCase):
    def test_cross_page_tile_is_generated_for_paged_kv_only(self):
        # Non-paged dispatch always reaches the bn0=64 tile listed first, so a
        # non-paged cross-page instance would be compiled and never selected.
        kernels = get_fwd_splitkv_blobs(["gfx1100"], "", 0, "simplified", [-1])
        cross_page = [k for k in kernels if k.F_tile.F_page_shorter_than_tile]
        self.assertTrue(cross_page)
        self.assertEqual(
            {k.F_pipeline.F_pagedkv for k in cross_page},
            {"t"},
        )


if __name__ == "__main__":
    unittest.main()
