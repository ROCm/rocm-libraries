# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``supports_tiled_{2d,3d}_spec``: the spec-first predicates ``hkp_pack`` calls judge
a spec by the fields it carries, not by the wrapped predicate's defaults."""

from __future__ import annotations

import unittest

import kernels.gfx942.attention_tiled_2d as gfx942_2d
import kernels.gfx942.attention_tiled_3d as gfx942_3d
import kernels.gfx950.attention_tiled_2d as gfx950_2d
import kernels.gfx950.attention_tiled_3d as gfx950_3d

_PROBLEM = dict(
    block_size=16,
    num_query_heads=8,
    num_kv_heads=1,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
)
_X8 = dict(use_mfma_32x32x8=True, use_transposed_qk_32x32=True)
_ONE_WARP = dict(num_warps=1, block_m_per_warp=32, tile_size=128)
_FOUR_WARP = dict(num_warps=4, block_m_per_warp=32, tile_size=64)


class TiledSpecWrappers(unittest.TestCase):
    def test_gfx942_2d_forwards_the_lds_sizing_flags(self):
        # Each case fits gfx942's 64 KiB LDS only through the flags it carries.
        cases = (
            (dict(**_FOUR_WARP, **_X8), True),
            (dict(**_ONE_WARP, use_k_single_buffer=True, **_X8), True),
            (
                dict(
                    **_ONE_WARP,
                    use_conflict_free_v_store=True,
                    use_k_sliced_ring=True,
                    **_X8,
                ),
                True,
            ),
            (_FOUR_WARP, False),
            (_ONE_WARP, False),
            (
                dict(
                    **_ONE_WARP,
                    use_k_single_buffer=True,
                    use_conflict_free_v_store=True,
                    **_X8,
                ),
                False,
            ),
        )
        for fields, expected in cases:
            with self.subTest(fields):
                spec = gfx942_2d.UnifiedAttention2DTiledSpec(
                    head_size=128, **_PROBLEM, **fields
                )
                self.assertEqual(gfx942_2d.supports_tiled_2d_spec(spec)[0], expected)

    def test_gfx942_4warp_gqa_wrapper_accepts_the_d256_cohort(self):
        # The spec the dispatcher builds for D256 bf16 GQA 16/2 bs16, which
        # routes to build_gfx942_4warp_gqa; only the use_d256_fast LDS model fits it.
        spec = gfx942_2d.UnifiedAttention2DTiledSpec(
            head_size=256,
            **dict(_PROBLEM, num_query_heads=16, num_kv_heads=2),
            num_warps=1,
            block_m_per_warp=32,
            tile_size=32,
            ring_depth=3,
            k_slice_hd=32,
        )
        self.assertTrue(gfx942_2d.supports_gfx942_4warp_gqa_spec(spec)[0])
        self.assertFalse(gfx942_2d.supports_tiled_2d_spec(spec)[0])

    def test_every_wrapper_judges_head_size(self):
        seg = {"num_segments": 16}
        cases = ((gfx950_2d, "2d", {}), (gfx942_3d, "3d", seg), (gfx950_3d, "3d", seg))
        for module, dim, extra in cases:
            spec_cls = getattr(module, f"UnifiedAttention{dim.upper()}TiledSpec")
            wrapper = getattr(module, f"supports_tiled_{dim}_spec")
            with self.subTest(module.__name__):
                ok = spec_cls(head_size=128, **_PROBLEM, **extra)
                self.assertTrue(wrapper(ok)[0])
                bad = spec_cls(head_size=96, **_PROBLEM, **extra)
                accepted, reason = wrapper(bad)
                self.assertFalse(accepted)
                self.assertIn("head_size", reason)


if __name__ == "__main__":
    unittest.main()
