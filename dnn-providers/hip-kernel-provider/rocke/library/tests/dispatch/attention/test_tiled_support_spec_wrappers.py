# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``supports_tiled_{2d,3d}_spec``: the spec-first predicates ``hkp_pack`` calls.

``hkp_pack`` swallows an exception from a predicate and packs anyway, so a wrapper
that stops agreeing with the kernel's own predicate fails open. These tests call the
wrappers directly: a spec is judged by the layout flags it carries, never by the
predicate's defaults.
"""

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

_TRANSPOSED_X8 = dict(use_mfma_32x32x8=True, use_transposed_qk_32x32=True)


class Gfx942TiledTwoDimensionalSpecWrapper(unittest.TestCase):
    """On gfx942 the pipeline flags size the LDS estimate, so they decide admission."""

    def _verdict(self, *, head_size=128, **fields):
        spec = gfx942_2d.UnifiedAttention2DTiledSpec(
            head_size=head_size, **_PROBLEM, **fields
        )
        return gfx942_2d.supports_tiled_2d_spec(spec)[0]

    def test_flags_that_make_the_tile_fit_are_honoured(self):
        # Each spec fits gfx942's 64 KiB LDS only because of the flags it carries;
        # the same shape with the predicate's default flags does not (see the
        # rejected control below).
        fits = {
            "transposed x8 shrinks the 4-warp tile": dict(
                num_warps=4, block_m_per_warp=32, tile_size=64, **_TRANSPOSED_X8
            ),
            "single K buffer": dict(
                num_warps=1,
                block_m_per_warp=32,
                tile_size=128,
                use_k_single_buffer=True,
                **_TRANSPOSED_X8,
            ),
            "sliced K ring": dict(
                num_warps=1,
                block_m_per_warp=32,
                tile_size=128,
                use_conflict_free_v_store=True,
                use_k_sliced_ring=True,
                **_TRANSPOSED_X8,
            ),
        }
        for label, fields in fits.items():
            with self.subTest(label):
                self.assertTrue(self._verdict(**fields))

    def test_the_same_shapes_with_default_flags_are_rejected(self):
        for fields in (
            dict(num_warps=4, block_m_per_warp=32, tile_size=64),
            dict(num_warps=1, block_m_per_warp=32, tile_size=128),
        ):
            with self.subTest(fields):
                self.assertFalse(self._verdict(**fields))

    def test_a_flag_combination_over_the_lds_budget_is_rejected(self):
        # Conflict-free V store adds padding that tips the single-buffer layout
        # past 64 KiB (67,584 B).
        self.assertFalse(
            self._verdict(
                num_warps=1,
                block_m_per_warp=32,
                tile_size=128,
                use_k_single_buffer=True,
                use_conflict_free_v_store=True,
                **_TRANSPOSED_X8,
            )
        )

    def test_the_wrapper_matches_the_predicate_it_wraps(self):
        spec = gfx942_2d.UnifiedAttention2DTiledSpec(
            head_size=128,
            **_PROBLEM,
            num_warps=1,
            block_m_per_warp=32,
            tile_size=128,
            use_k_single_buffer=True,
            **_TRANSPOSED_X8,
        )
        direct = gfx942_2d.supports_tiled_2d(
            head_size=128,
            block_size=16,
            dtype="bf16",
            num_queries_per_kv=spec.num_queries_per_kv,
            use_alibi=False,
            use_qq_bias=False,
            use_fp8=False,
            q_dtype="bf16",
            num_warps=1,
            block_m_per_warp=32,
            tile_size=128,
            use_k_single_buffer=True,
            **_TRANSPOSED_X8,
        )
        self.assertEqual(gfx942_2d.supports_tiled_2d_spec(spec), direct)


class Gfx950TiledSpecWrappers(unittest.TestCase):
    def test_two_dimensional_accepts_a_supported_spec_and_names_the_reason(self):
        ok = gfx950_2d.UnifiedAttention2DTiledSpec(head_size=128, **_PROBLEM)
        self.assertTrue(gfx950_2d.supports_tiled_2d_spec(ok)[0])
        bad = gfx950_2d.UnifiedAttention2DTiledSpec(head_size=96, **_PROBLEM)
        accepted, reason = gfx950_2d.supports_tiled_2d_spec(bad)
        self.assertFalse(accepted)
        self.assertIn("head_size", reason)


class TiledThreeDimensionalSpecWrappers(unittest.TestCase):
    def test_every_arch_judges_the_spec_it_is_given(self):
        for arch, module in (("gfx942", gfx942_3d), ("gfx950", gfx950_3d)):
            with self.subTest(arch):
                spec_cls = module.UnifiedAttention3DTiledSpec
                ok = spec_cls(head_size=128, num_segments=16, **_PROBLEM)
                self.assertTrue(module.supports_tiled_3d_spec(ok, arch=arch)[0])
                bad = spec_cls(head_size=96, num_segments=16, **_PROBLEM)
                accepted, reason = module.supports_tiled_3d_spec(bad, arch=arch)
                self.assertFalse(accepted)
                self.assertIn("head_size", reason)


if __name__ == "__main__":
    unittest.main()
