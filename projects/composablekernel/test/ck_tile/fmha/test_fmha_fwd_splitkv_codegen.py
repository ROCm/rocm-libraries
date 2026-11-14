#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import dataclasses
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_FMHA_EXAMPLE = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "example", "ck_tile", "01_fmha")
)
sys.path.insert(0, _FMHA_EXAMPLE)

from codegen.ops.fmha_fwd_splitkv import (  # noqa: E402
    get_fwd_splitkv_api_trait,
    get_fwd_splitkv_blobs,
)

_GFX1100_KERNELS = get_fwd_splitkv_blobs(["gfx1100"], "", 0, "simplified", [-1])


def _paged_trait(bn0: int):
    # The dispatch entry codegen emits for a paged gfx1100 hdim128 kernel with
    # this bn0. pagecheck does not read the combine tile, so any width will do.
    kernel = next(
        k
        for k in _GFX1100_KERNELS
        if k.F_pipeline.F_pagedkv == "t" and k.F_hdim == 128 and k.F_tile.F_bn0 == bn0
    )
    return get_fwd_splitkv_api_trait(kernel, bn1comb=32)


class SplitKvPageCheckTest(unittest.TestCase):
    def test_non_merge_tile_requires_page_aligned_origins(self):
        trait = _paged_trait(64)
        self.assertFalse(trait.page_shorter_than_tile)
        self.assertEqual(
            trait.pagecheck,
            "64 <= a.page_block_size && a.page_block_size % 64 == 0",
        )

    def test_merge_tile_requires_half_tile_aligned_pages(self):
        trait = _paged_trait(32)
        self.assertTrue(trait.page_shorter_than_tile)
        self.assertEqual(
            dataclasses.replace(trait, constraint="true").pagecheck,
            "32 <= 2 * a.page_block_size && a.page_block_size % 16 == 0",
        )

    def test_merge_tile_keeps_its_device_constraint(self):
        trait = _paged_trait(32)
        self.assertEqual(
            trait.pagecheck,
            "(32 <= 2 * a.page_block_size && a.page_block_size % 16 == 0) && "
            '(device_name.compare(0, 7, "gfx1100") == 0)',
        )


class SplitKvCrossPageTileTest(unittest.TestCase):
    def test_cross_page_tile_is_generated_for_paged_kv_only(self):
        # Non-paged dispatch always reaches the bn0=64 tile listed first, so a
        # non-paged cross-page instance would be compiled and never selected.
        cross_page = [k for k in _GFX1100_KERNELS if k.F_tile.F_page_shorter_than_tile]
        self.assertTrue(cross_page)
        self.assertEqual(
            {k.F_pipeline.F_pagedkv for k in cross_page},
            {"t"},
        )


if __name__ == "__main__":
    unittest.main()
