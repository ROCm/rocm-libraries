# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import unittest
from dataclasses import replace
from itertools import combinations

from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec, wmma_fmha_fwd_grid


class TestWmmaFmhaFwdSpec(unittest.TestCase):
    def test_runtime_shape_is_not_part_of_spec_identity(self):
        spec = WmmaFmhaFwdSpec(head_size=96, mask_mode="window")
        name = spec.kernel_name()
        self.assertIn("H96", name)
        self.assertNotIn("HQ", name)
        self.assertNotIn("HK", name)
        self.assertNotIn("sw", name)

    def test_dtype_cache_keys(self):
        fp16 = WmmaFmhaFwdSpec(head_size=64, dtype="fp16")
        alias = WmmaFmhaFwdSpec(head_size=64, dtype="f16")
        bf16 = WmmaFmhaFwdSpec(head_size=64, dtype="bf16")
        self.assertEqual(fp16.kernel_name(), alias.kernel_name())
        self.assertNotEqual(fp16.kernel_name(), bf16.kernel_name())

    def test_unsupported_dtype_rejected(self):
        with self.assertRaises(ValueError):
            WmmaFmhaFwdSpec(head_size=64, dtype="fp32")

    def test_head_size_must_be_supported_multiple(self):
        for size in (0, 8, 272):
            with self.subTest(head_size=size), self.assertRaises(ValueError):
                WmmaFmhaFwdSpec(head_size=size)

    def test_mask_modes_have_distinct_cache_keys(self):
        variants = [
            WmmaFmhaFwdSpec(head_size=64, mask_mode=mode)
            for mode in ("none", "causal", "window")
        ]
        for left, right in combinations(variants, 2):
            self.assertNotEqual(left.kernel_name(), right.kernel_name())

    def test_unknown_mask_mode_rejected(self):
        with self.assertRaises(ValueError):
            WmmaFmhaFwdSpec(head_size=64, mask_mode="unknown")

    def test_partial_query_grid_requires_specialization(self):
        aligned = WmmaFmhaFwdSpec(head_size=64)
        with self.assertRaises(ValueError):
            wmma_fmha_fwd_grid(aligned, seqlen_q=1, num_query_heads=4, batch=2)
        tail = replace(aligned, query_tail=True)
        for length, tiles in ((1, 1), (16, 1), (17, 2)):
            with self.subTest(length=length):
                self.assertEqual(
                    wmma_fmha_fwd_grid(
                        tail, seqlen_q=length, num_query_heads=4, batch=2
                    ),
                    (tiles, 4, 2),
                )

    def test_tail_specializations_do_not_alias(self):
        base = WmmaFmhaFwdSpec(head_size=64)
        variants = [
            replace(base, query_tail=q, kv_tail=k)
            for q, k in ((False, False), (True, False), (False, True), (True, True))
        ]
        for left, right in combinations(variants, 2):
            self.assertNotEqual(left.kernel_name(), right.kernel_name())

    def test_score_features_have_distinct_cache_keys(self):
        base = WmmaFmhaFwdSpec(head_size=64, mask_mode="causal")
        variants = [base]
        variants += [
            replace(base, **{flag: True})
            for flag in ("use_softcap", "use_sinks", "use_alibi", "use_qq_bias")
        ]
        variants += [
            replace(base, use_attn_bias=True, bias_dtype=dtype)
            for dtype in ("f32", "q")
        ]
        variants += [replace(base, mask_mode="window")]
        for left, right in combinations(variants, 2):
            self.assertNotEqual(left.kernel_name(), right.kernel_name())

    def test_layout_page_size_contract(self):
        for layout, page in (
            ("unknown", 0),
            ("dense", 16),
            ("ragged", 16),
            ("paged", 0),
            ("paged", -16),
            ("paged", 3),
        ):
            with self.subTest(layout=layout, page=page), self.assertRaises(ValueError):
                WmmaFmhaFwdSpec(
                    head_size=64,
                    layout=layout,
                    page_block_size=page,
                )

    def test_packed_grids_cover_longest_sequence(self):
        for layout, page in (("ragged", 0), ("paged", 16)):
            spec = WmmaFmhaFwdSpec(
                head_size=64,
                layout=layout,
                page_block_size=page,
            )
            for length, tiles in ((1, 1), (16, 1), (17, 2)):
                with self.subTest(layout=layout, length=length):
                    self.assertEqual(
                        wmma_fmha_fwd_grid(
                            spec,
                            seqlen_q=length,
                            num_query_heads=4,
                            batch=3,
                        ),
                        (tiles, 4, 3),
                    )

    def test_layout_cache_keys_separate_addressing(self):
        base = WmmaFmhaFwdSpec(head_size=64)
        variants = [base, replace(base, layout="ragged")]
        variants += [
            replace(base, layout="paged", page_block_size=page) for page in (16, 32, 64)
        ]
        variants += [replace(spec, kv_dtype="fp8e4m3") for spec in variants]
        for left, right in combinations(variants, 2):
            self.assertNotEqual(left.kernel_name(), right.kernel_name())

    def test_unsupported_kv_storage_rejected(self):
        for storage in ("bf8e5m2", "fp8e4m3fnuz", "fp32"):
            with self.subTest(storage=storage), self.assertRaises(ValueError):
                WmmaFmhaFwdSpec(head_size=64, kv_dtype=storage)

    def test_transposed_qk_supports_both_16_bit_dtypes(self):
        for dtype in ("fp16", "bf16"):
            spec = WmmaFmhaFwdSpec(
                head_size=64,
                dtype=dtype,
                transposed_qk=True,
            )
            self.assertIn("wmma_swapqk", spec.kernel_name())

    def test_transposed_qk_rejects_unimplemented_combinations(self):
        base = dict(head_size=64, transposed_qk=True)
        for changes in (
            {"head_size": 256},
            {"block_n": 16},
            {"num_waves": 4},
            {"layout": "ragged"},
            {"kv_dtype": "fp8e4m3"},
            {"query_tail": True},
            {"kv_tail": True},
            {"v_lds_stage": True},
            {"use_sinks": True},
            {"use_softcap": True},
            {"use_alibi": True},
            {"use_qq_bias": True},
            {"use_attn_bias": True},
            {"mask_mode": "window"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                WmmaFmhaFwdSpec(**dict(base, **changes))

    def test_transposed_query_grid_covers_whole_wave_groups(self):
        spec = WmmaFmhaFwdSpec(head_size=64, transposed_qk=True, num_waves=2)
        with self.assertRaises(ValueError):
            wmma_fmha_fwd_grid(spec, seqlen_q=48, num_query_heads=4, batch=2)
        self.assertEqual(
            wmma_fmha_fwd_grid(spec, seqlen_q=64, num_query_heads=4, batch=2),
            (2, 4, 2),
        )

    def test_transposed_geometry_does_not_alias_other_kernels(self):
        base = WmmaFmhaFwdSpec(head_size=64)
        variants = [base] + [
            replace(base, transposed_qk=True, block_n=block, num_waves=waves)
            for block in (32, 64)
            for waves in (1, 2)
        ]
        for left, right in combinations(variants, 2):
            self.assertNotEqual(left.kernel_name(), right.kernel_name())

    def test_scheduler_policies_do_not_alias_compiled_kernels(self):
        from rocke.core.codegen_policy import SchedulerStrategy

        base = WmmaFmhaFwdSpec(head_size=64)
        variants = [base] + [
            replace(base, scheduler_strategy=strategy) for strategy in SchedulerStrategy
        ]
        for left, right in combinations(variants, 2):
            self.assertNotEqual(left.kernel_name(), right.kernel_name())

    def test_unknown_scheduler_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            WmmaFmhaFwdSpec(head_size=64, scheduler_strategy="unknown")

    def test_output_tiles_require_complete_nonoverlapping_head_partition(self):
        for tile in (-16, 1, 48, 96, 256, 512):
            with self.subTest(tile=tile), self.assertRaises(ValueError):
                WmmaFmhaFwdSpec(head_size=256, value_tile_size=tile)
        with self.assertRaises(ValueError):
            WmmaFmhaFwdSpec(head_size=64, transposed_qk=True, value_tile_size=32)

    def test_output_tile_grid_preserves_batch_and_bounds(self):
        spec = WmmaFmhaFwdSpec(head_size=256, value_tile_size=64, query_tail=True)
        self.assertEqual(
            wmma_fmha_fwd_grid(spec, seqlen_q=17, num_query_heads=4, batch=3),
            (2, 4, 12),
        )
        for batch in (-1, 0x7FFFFFFF // 4 + 1):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                wmma_fmha_fwd_grid(spec, seqlen_q=17, num_query_heads=4, batch=batch)

    def test_output_tile_cache_keys_separate_partitions(self):
        base = WmmaFmhaFwdSpec(head_size=256)
        variants = [
            replace(base, value_tile_size=tile) for tile in (0, 16, 32, 64, 128)
        ]
        for left, right in combinations(variants, 2):
            self.assertNotEqual(left.kernel_name(), right.kernel_name())

    def test_causal_tile_skip_requires_plain_standard_causal(self):
        base = dict(head_size=64, causal_tile_skip=True)
        WmmaFmhaFwdSpec(**dict(base, mask_mode="causal"))
        for changes in (
            {},
            {"mask_mode": "window"},
            {"mask_mode": "causal", "transposed_qk": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                WmmaFmhaFwdSpec(**dict(base, **changes))

    def test_causal_tile_skip_cache_key_and_emission(self):
        from kernels.gfx1151.wmma_fmha_fwd import build_wmma_fmha_fwd
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        base = WmmaFmhaFwdSpec(
            head_size=64,
            mask_mode="causal",
            query_tail=True,
            kv_tail=True,
        )
        skipping = replace(base, causal_tile_skip=True)
        self.assertNotEqual(base.kernel_name(), skipping.kernel_name())
        plain = lower_kernel_to_llvm(
            build_wmma_fmha_fwd(base, "gfx1151"), arch="gfx1151"
        )
        bounded = lower_kernel_to_llvm(
            build_wmma_fmha_fwd(skipping, "gfx1151"), arch="gfx1151"
        )
        self.assertNotEqual(plain, bounded)

    def test_v_head_size_validation_and_naming(self):
        base = WmmaFmhaFwdSpec(head_size=128)
        self.assertEqual(base.v_dim, 128)
        wide = replace(base, v_head_size=256)
        self.assertEqual(wide.v_dim, 256)
        self.assertNotEqual(base.kernel_name(), wide.kernel_name())
        self.assertIn("vh256", wide.kernel_name())
        for bad in (-16, 8, 20, 272):
            with self.subTest(v_head_size=bad), self.assertRaises(ValueError):
                replace(base, v_head_size=bad)

    def test_v_head_size_emission_differs_and_is_wmma_only(self):
        from kernels.gfx1151.wmma_fmha_fwd import build_wmma_fmha_fwd, is_valid_spec
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        base = WmmaFmhaFwdSpec(head_size=128, v_head_size=64)
        ok, why = is_valid_spec(base, arch="gfx1151")
        self.assertTrue(ok, why)
        ok, _ = is_valid_spec(base, arch="gfx942")
        self.assertFalse(ok)
        equal = replace(base, v_head_size=0)
        self.assertNotEqual(
            lower_kernel_to_llvm(build_wmma_fmha_fwd(base, "gfx1151"), arch="gfx1151"),
            lower_kernel_to_llvm(build_wmma_fmha_fwd(equal, "gfx1151"), arch="gfx1151"),
        )

    def test_attn_bias_naming_params_and_default_is_unchanged(self):
        from kernels.gfx1151.wmma_fmha_fwd import build_wmma_fmha_fwd
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        def ir(spec):
            return lower_kernel_to_llvm(
                build_wmma_fmha_fwd(spec, "gfx1151"), arch="gfx1151"
            )

        def params(spec):
            return [p.name for p in build_wmma_fmha_fwd(spec, "gfx1151").params]

        bias = ["attn_bias_ptr", "bias_stride_b", "bias_stride_h", "bias_stride_q"]
        base = WmmaFmhaFwdSpec(head_size=64, mask_mode="none")
        self.assertFalse(base.use_attn_bias)
        self.assertNotIn("abias", base.kernel_name())
        for dtype in ("f32", "q"):
            with self.subTest(dtype=dtype):
                spec = replace(base, use_attn_bias=True, bias_dtype=dtype)
                self.assertIn(f"abias_{dtype}", spec.kernel_name())
                self.assertEqual(params(spec), params(base) + bias)
                self.assertNotEqual(ir(spec), ir(base))
        self.assertNotEqual(
            ir(replace(base, use_attn_bias=True, bias_dtype="f32")),
            ir(replace(base, use_attn_bias=True, bias_dtype="q")),
        )
        self.assertEqual(ir(base), ir(replace(base, use_attn_bias=False)))

    def test_attn_bias_rejects_unknown_dtype_and_transposed_qk(self):
        with self.assertRaises(ValueError):
            WmmaFmhaFwdSpec(head_size=64, use_attn_bias=True, bias_dtype="f64")
        with self.assertRaises(ValueError):
            WmmaFmhaFwdSpec(
                head_size=64,
                mask_mode="none",
                transposed_qk=True,
                use_attn_bias=True,
            )


if __name__ == "__main__":
    unittest.main()
