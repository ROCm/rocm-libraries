# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import unittest
from dataclasses import replace
from itertools import combinations

from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec, wmma_fmha_fwd_grid


class TestWmmaFmhaFwdSpec(unittest.TestCase):
    def test_mha_ok(self):
        spec = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4)
        self.assertEqual(spec.kv_heads, 4)

    def test_divisible_gqa_ok(self):
        spec = WmmaFmhaFwdSpec(head_size=64, num_query_heads=8, num_kv_heads=2)
        self.assertEqual(spec.kv_heads, 2)

    def test_non_divisible_gqa_rejected(self):
        with self.assertRaises(ValueError):
            WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, num_kv_heads=3)

    def test_dtype_cache_keys(self):
        fp16 = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, dtype="fp16")
        alias = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, dtype="f16")
        bf16 = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, dtype="bf16")
        self.assertEqual(fp16.kernel_name(), alias.kernel_name())
        self.assertNotEqual(fp16.kernel_name(), bf16.kernel_name())

    def test_unsupported_dtype_rejected(self):
        with self.assertRaises(ValueError):
            WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, dtype="fp32")

    def test_bottom_right_requires_causal_mask(self):
        with self.assertRaises(ValueError):
            WmmaFmhaFwdSpec(
                head_size=64, num_query_heads=4, causal_bottom_right=True,
            )

    def test_causal_alignment_cache_keys(self):
        top_left = WmmaFmhaFwdSpec(
            head_size=64, num_query_heads=4, mask_mode="causal",
        )
        bottom_right = WmmaFmhaFwdSpec(
            head_size=64, num_query_heads=4, mask_mode="causal",
            causal_bottom_right=True,
        )
        self.assertNotEqual(top_left.kernel_name(), bottom_right.kernel_name())

    def test_partial_query_grid_requires_specialization(self):
        aligned = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4)
        with self.assertRaises(ValueError):
            wmma_fmha_fwd_grid(aligned, seqlen_q=1, batch=2)
        tail = replace(aligned, query_tail=True)
        for length, tiles in ((1, 1), (16, 1), (17, 2)):
            with self.subTest(length=length):
                self.assertEqual(
                    wmma_fmha_fwd_grid(tail, seqlen_q=length, batch=2),
                    (tiles, 4, 2),
                )

    def test_tail_specializations_do_not_alias(self):
        base = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4)
        variants = [
            replace(base, query_tail=q, kv_tail=k)
            for q, k in ((False, False), (True, False), (False, True), (True, True))
        ]
        for left, right in combinations(variants, 2):
            self.assertNotEqual(left.kernel_name(), right.kernel_name())

    def test_window_requires_causal_mask(self):
        for mask, width in (("none", 1), ("causal", -1)):
            with self.subTest(mask=mask, width=width), self.assertRaises(ValueError):
                WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, mask_mode=mask, sliding_window=width)

    def test_score_features_have_distinct_cache_keys(self):
        base = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, mask_mode="causal")
        variants = [base]
        variants += [
            replace(base, **{flag: True})
            for flag in ("use_softcap", "use_sinks", "use_alibi", "use_qq_bias")
        ]
        variants += [replace(base, sliding_window=width) for width in (1, 64)]
        for left, right in combinations(variants, 2):
            self.assertNotEqual(left.kernel_name(), right.kernel_name())

    def test_layout_page_size_contract(self):
        for layout, page in (
            ("unknown", 0), ("dense", 16), ("ragged", 16),
            ("paged", 0), ("paged", -16), ("paged", 3),
        ):
            with self.subTest(layout=layout, page=page), self.assertRaises(ValueError):
                WmmaFmhaFwdSpec(
                    head_size=64, num_query_heads=4, layout=layout, page_block_size=page,
                )

    def test_packed_grids_cover_longest_sequence(self):
        for layout, page in (("ragged", 0), ("paged", 16)):
            spec = WmmaFmhaFwdSpec(
                head_size=64, num_query_heads=4, layout=layout, page_block_size=page,
            )
            for length, tiles in ((1, 1), (16, 1), (17, 2)):
                with self.subTest(layout=layout, length=length):
                    self.assertEqual(wmma_fmha_fwd_grid(spec, seqlen_q=length, batch=3), (tiles, 4, 3))

    def test_layout_cache_keys_separate_addressing(self):
        base = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4)
        variants = [base, replace(base, layout="ragged")]
        variants += [replace(base, layout="paged", page_block_size=page) for page in (16, 32, 64)]
        variants += [replace(spec, kv_dtype="fp8e4m3") for spec in variants]
        for left, right in combinations(variants, 2):
            self.assertNotEqual(left.kernel_name(), right.kernel_name())

    def test_unsupported_kv_storage_rejected(self):
        for storage in ("bf8e5m2", "fp8e4m3fnuz", "fp32"):
            with self.subTest(storage=storage), self.assertRaises(ValueError):
                WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, kv_dtype=storage)

    def test_transposed_qk_rejects_unimplemented_combinations(self):
        base = dict(head_size=64, num_query_heads=4, transposed_qk=True)
        for changes in (
            {"dtype": "bf16"}, {"head_size": 256}, {"block_n": 16}, {"num_waves": 4},
            {"layout": "ragged"}, {"kv_dtype": "fp8e4m3"}, {"query_tail": True},
            {"kv_tail": True}, {"v_lds_stage": True}, {"use_sinks": True},
            {"use_softcap": True}, {"use_alibi": True}, {"use_qq_bias": True},
            {"mask_mode": "causal", "sliding_window": 32},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                WmmaFmhaFwdSpec(**dict(base, **changes))

    def test_transposed_query_grid_covers_whole_wave_groups(self):
        spec = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, transposed_qk=True, num_waves=2)
        with self.assertRaises(ValueError):
            wmma_fmha_fwd_grid(spec, seqlen_q=48, batch=2)
        self.assertEqual(wmma_fmha_fwd_grid(spec, seqlen_q=64, batch=2), (2, 4, 2))

    def test_transposed_geometry_does_not_alias_other_kernels(self):
        base = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4)
        variants = [base] + [
            replace(base, transposed_qk=True, block_n=block, num_waves=waves)
            for block in (32, 64) for waves in (1, 2)
        ]
        for left, right in combinations(variants, 2):
            self.assertNotEqual(left.kernel_name(), right.kernel_name())

    def test_scheduler_policies_do_not_alias_compiled_kernels(self):
        from rocke.core.codegen_policy import SchedulerStrategy

        base = WmmaFmhaFwdSpec(head_size=64, num_query_heads=4)
        variants = [base] + [replace(base, scheduler_strategy=strategy) for strategy in SchedulerStrategy]
        for left, right in combinations(variants, 2):
            self.assertNotEqual(left.kernel_name(), right.kernel_name())

    def test_unknown_scheduler_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, scheduler_strategy="unknown")

    def test_output_tiles_require_complete_nonoverlapping_head_partition(self):
        for tile in (-16, 1, 48, 96, 256, 512):
            with self.subTest(tile=tile), self.assertRaises(ValueError):
                WmmaFmhaFwdSpec(head_size=256, num_query_heads=4, value_tile_size=tile)
        with self.assertRaises(ValueError):
            WmmaFmhaFwdSpec(head_size=64, num_query_heads=4, transposed_qk=True, value_tile_size=32)

    def test_output_tile_grid_preserves_batch_and_bounds(self):
        spec = WmmaFmhaFwdSpec(head_size=256, num_query_heads=4, value_tile_size=64, query_tail=True)
        self.assertEqual(wmma_fmha_fwd_grid(spec, seqlen_q=17, batch=3), (2, 4, 12))
        for batch in (-1, 0x7FFFFFFF // 4 + 1):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                wmma_fmha_fwd_grid(spec, seqlen_q=17, batch=batch)

    def test_output_tile_cache_keys_separate_partitions(self):
        base = WmmaFmhaFwdSpec(head_size=256, num_query_heads=4)
        variants = [replace(base, value_tile_size=tile) for tile in (0, 16, 32, 64, 128)]
        for left, right in combinations(variants, 2):
            self.assertNotEqual(left.kernel_name(), right.kernel_name())


if __name__ == "__main__":
    unittest.main()
