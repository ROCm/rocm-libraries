# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Public gfx1151 selection, unsupported requests, and cross-architecture isolation."""

import unittest
from unittest import mock

from dispatch.attention import AttentionMaskType, AttentionRequest, dispatch_attention
from dispatch.attention.gfx1151 import _make_wmma_fwd_candidate, _wmma_fwd_spec
from kernels.gfx1151.wmma_fmha_fwd import is_valid_spec


def _request(**changes):
    fields = dict(
        batch=2,
        nhead_q=16,
        nhead_k=4,
        seqlen_q=512,
        seqlen_k=512,
        hdim_q=128,
        hdim_v=128,
        arch="gfx1151",
        dtype="fp16",
        num_cus=64,
    )
    fields.update(changes)
    return AttentionRequest(**fields)


class TestGfx1151AttentionSelection(unittest.TestCase):
    def test_auto_and_explicit_selectors_reach_the_executable_kernel(self):
        for selectors in (
            {},
            {"algorithm": "wmma_fmha_fwd"},
            {"spec_id": "gfx1151_wmma_fmha_fwd"},
        ):
            with self.subTest(selectors=selectors):
                result = dispatch_attention(_request(layout="dense", **selectors))
                self.assertEqual(result.candidate.name, "attention_gfx1151_wmma")
        with self.assertRaises(ValueError):
            dispatch_attention(_request(layout="dense", algorithm="unified_2d"))

    def test_unsupported_requests_cannot_fall_back_to_a_different_operation(self):
        cases = (
            {"dtype": "fp32"},
            {"use_fp8": True, "fp8_fnuz": True},
            {"nhead_q": 15},
            {"hdim_q": 100, "hdim_v": 100},
            {"hdim_q": 64, "hdim_v": 272},
            {"layout": "sparse"},
            {"layout": "paged", "kv_block_size": 48},
            {"mask_type": 99},
            {"mask_type": AttentionMaskType.SLIDING_WINDOW},
            {"sliding_window": -1},
        )
        for changes in cases:
            fields = {"layout": "dense"}
            fields.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dispatch_attention(_request(**fields))

    def test_runtime_shape_fields_reuse_one_spec(self):
        first = dispatch_attention(_request(nhead_q=16, nhead_k=4)).spec
        second = dispatch_attention(
            _request(nhead_q=32, nhead_k=8, seqlen_q=1024, seqlen_k=1024)
        ).spec
        self.assertEqual(first, second)

    def test_head_dim_96_and_lse_are_supported(self):
        result = dispatch_attention(_request(hdim_q=96, hdim_v=96, return_lse=True))
        self.assertEqual(result.spec.head_size, 96)
        self.assertFalse(result.spec.transposed_qk)

    def test_runtime_window_bounds_do_not_change_code_object(self):
        left = dispatch_attention(
            _request(
                mask_type=AttentionMaskType.SLIDING_WINDOW,
                window_left=31,
                window_right=7,
            )
        ).spec
        right = dispatch_attention(
            _request(
                mask_type=AttentionMaskType.SLIDING_WINDOW,
                window_left=127,
                window_right=63,
                window_bottom_right=True,
            )
        ).spec
        self.assertEqual(left, right)
        self.assertEqual(left.mask_mode, "window")

    def test_bf16_uses_transposed_fast_path(self):
        result = dispatch_attention(_request(dtype="bf16", layout="dense"))
        self.assertTrue(result.spec.transposed_qk)

    def test_explicit_layout_and_score_requirements_are_not_ignored_elsewhere(self):
        for changes in (
            {"layout": "dense"},
            {"layout": "ragged"},
            {"layout": "paged"},
            {"use_softcap": True},
            {"use_alibi": True},
            {"use_qq_bias": True},
            {"use_attn_bias": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dispatch_attention(_request(arch="gfx950", **changes))

    def test_causal_requests_skip_masked_tiles_without_a_fake_window(self):
        for layout, mask in (
            ("ragged", AttentionMaskType.TOP_LEFT_CAUSAL),
            ("dense", AttentionMaskType.BOTTOM_RIGHT_CAUSAL),
            ("paged", AttentionMaskType.TOP_LEFT_CAUSAL),
        ):
            with self.subTest(layout=layout, mask=mask):
                spec = dispatch_attention(
                    _request(
                        layout=layout,
                        kv_block_size=16,
                        seqlen_q=100,
                        seqlen_k=100,
                        mask_type=mask,
                    )
                ).spec
                self.assertTrue(spec.causal_tile_skip)
                self.assertEqual(spec.mask_mode, "causal")

    def test_unmasked_and_windowed_requests_do_not_request_tile_skip(self):
        spec = dispatch_attention(_request(layout="ragged", seqlen_q=100)).spec
        self.assertFalse(spec.causal_tile_skip)
        spec = dispatch_attention(
            _request(
                layout="ragged",
                seqlen_q=100,
                mask_type=AttentionMaskType.SLIDING_WINDOW,
                sliding_window=32,
            )
        ).spec
        self.assertFalse(spec.causal_tile_skip)
        self.assertEqual(spec.mask_mode, "window")

    def test_right_window_selects_the_wmma_kernel_for_unmasked_requests(self):
        for mask in (
            AttentionMaskType.NO_MASK,
            AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        ):
            with self.subTest(mask=mask):
                spec = dispatch_attention(
                    _request(
                        layout="dense",
                        seqlen_q=100,
                        seqlen_k=100,
                        mask_type=mask,
                        window_right=16,
                    )
                ).spec
                self.assertEqual(spec.mask_mode, "window")
                self.assertFalse(spec.causal_tile_skip)
                self.assertFalse(spec.transposed_qk)

    def test_right_window_combines_with_a_left_window(self):
        spec = dispatch_attention(
            _request(
                layout="ragged",
                seqlen_q=100,
                mask_type=AttentionMaskType.SLIDING_WINDOW,
                sliding_window=32,
                window_right=8,
            )
        ).spec
        self.assertEqual(spec.mask_mode, "window")

    def test_right_window_rejects_unsupported_combinations(self):
        cases = (
            {"window_right": -2},
            {"window_right": 8, "mask_type": AttentionMaskType.TOP_LEFT_CAUSAL},
            {"window_right": 8, "arch": "gfx950"},
        )
        for changes in cases:
            fields = {"layout": "dense"}
            fields.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dispatch_attention(_request(**fields))

    def test_return_lse_reuses_the_same_gfx1151_code_object(self):
        for layout in ("dense", "ragged"):
            with self.subTest(layout=layout):
                plain = dispatch_attention(_request(layout=layout)).spec
                lse = dispatch_attention(_request(layout=layout, return_lse=True)).spec
                self.assertEqual(plain, lse)
        for arch in ("gfx950", "gfx942"):
            with self.subTest(arch=arch), self.assertRaises(ValueError):
                dispatch_attention(_request(arch=arch, return_lse=True))

    def test_attn_bias_selects_a_bias_kernel_only_on_gfx1151(self):
        for layout in ("dense", "ragged", "paged"):
            for dtype in ("f32", "q"):
                with self.subTest(layout=layout, dtype=dtype):
                    spec = dispatch_attention(
                        _request(
                            layout=layout, use_attn_bias=True, attn_bias_dtype=dtype
                        )
                    ).spec
                    self.assertTrue(spec.use_attn_bias)
                    self.assertEqual(spec.bias_dtype, dtype)
                    self.assertFalse(spec.transposed_qk)
        self.assertFalse(dispatch_attention(_request()).spec.use_attn_bias)
        for arch in ("gfx950", "gfx942"):
            with self.subTest(arch=arch), self.assertRaises(ValueError):
                dispatch_attention(_request(arch=arch, use_attn_bias=True))

    def test_attn_bias_composes_with_other_score_features(self):
        spec = dispatch_attention(
            _request(
                use_attn_bias=True,
                use_softcap=True,
                use_alibi=True,
                use_qq_bias=True,
                return_lse=True,
                mask_type=AttentionMaskType.TOP_LEFT_CAUSAL,
            )
        ).spec
        self.assertTrue(
            spec.use_attn_bias
            and spec.use_softcap
            and spec.use_alibi
            and spec.use_qq_bias
        )
        self.assertEqual(
            spec,
            dispatch_attention(
                _request(
                    use_attn_bias=True,
                    use_softcap=True,
                    use_alibi=True,
                    use_qq_bias=True,
                    return_lse=False,
                    mask_type=AttentionMaskType.TOP_LEFT_CAUSAL,
                )
            ).spec,
        )

    def test_output_column_tiling_scales_with_the_compute_unit_count(self):
        def tile(num_cus, nhead_q):
            return dispatch_attention(
                _request(
                    batch=1,
                    nhead_q=nhead_q,
                    nhead_k=nhead_q,
                    seqlen_q=512,
                    seqlen_k=512,
                    hdim_q=256,
                    hdim_v=256,
                    num_cus=num_cus,
                )
            ).spec.value_tile_size

        # 32 query groups per head: 64 heads -> 1024 groups would not fit 40 CUs.
        self.assertGreater(tile(40, 2), 0)
        self.assertEqual(tile(40, 8), 0)
        self.assertGreater(tile(160, 8), 0)
        self.assertEqual(tile(160, 32), 0)

    def _unset_cu_request(self):
        return _request(
            batch=1,
            nhead_q=4,
            nhead_k=4,
            seqlen_q=512,
            seqlen_k=512,
            hdim_q=256,
            hdim_v=256,
            num_cus=0,
        )

    def test_default_compute_units_match_the_reference_part(self):
        # 128 query groups is the boundary on the 40-CU reference part; pin the
        # live-device query away so the result does not depend on the host GPU.
        with mock.patch(
            "dispatch.attention.gfx1151._device_num_cus", return_value=None
        ):
            spec = dispatch_attention(self._unset_cu_request()).spec
        self.assertGreater(spec.value_tile_size, 0)

    def test_live_device_compute_units_are_doubled_for_wgp_mode(self):
        # A device reporting 20 WGPs is 40 CUs (tiles); one reporting 16 is 32 CUs
        # (128 groups no longer fit, so the full-head kernel is kept).
        with mock.patch(
            "rocke.runtime.hip_module.get_device_arch", return_value="gfx1151"
        ):
            with mock.patch(
                "dispatch.attention.gfx1151._device_num_cus", return_value=20
            ):
                self.assertGreater(
                    dispatch_attention(self._unset_cu_request()).spec.value_tile_size, 0
                )
            with mock.patch(
                "dispatch.attention.gfx1151._device_num_cus", return_value=16
            ):
                self.assertEqual(
                    dispatch_attention(self._unset_cu_request()).spec.value_tile_size, 0
                )

    def test_sweep_space_offers_distinct_valid_variants_baseline_first(self):
        for changes in (
            dict(seqlen_q=1024, seqlen_k=1024, hdim_q=64, hdim_v=64),
            dict(
                seqlen_q=256,
                seqlen_k=512,
                hdim_q=256,
                hdim_v=256,
                mask_type=AttentionMaskType.NO_MASK,
            ),
            dict(
                seqlen_q=100,
                seqlen_k=100,
                mask_type=AttentionMaskType.TOP_LEFT_CAUSAL,
            ),
        ):
            with self.subTest(**changes):
                request = _request(**changes)
                result = dispatch_attention(request)
                sweep = result.candidate.sweep_space(request)
                self.assertGreater(len(sweep), 1)
                self.assertEqual(sweep[0], result.spec)
                self.assertEqual(len(set(sweep)), len(sweep))
                for spec in sweep:
                    ok, why = is_valid_spec(spec, arch="gfx1151")
                    self.assertTrue(ok, why)

    def test_sweep_space_is_empty_for_inadmissible_requests(self):
        request = _request(hdim_q=100, hdim_v=128)
        self.assertEqual(_make_wmma_fwd_candidate().sweep_space(request), ())

    def test_unequal_and_non_power_of_two_head_dims_are_admitted(self):
        for hdim_q, hdim_v in ((96, 96), (64, 128), (192, 128), (128, 64)):
            with self.subTest(hdim_q=hdim_q, hdim_v=hdim_v):
                request = _request(hdim_q=hdim_q, hdim_v=hdim_v)
                spec = _wmma_fwd_spec(request)
                self.assertEqual(spec.head_size, hdim_q)
                self.assertEqual(spec.v_dim, hdim_v)
                self.assertEqual(spec.v_head_size, 0 if hdim_q == hdim_v else hdim_v)
                self.assertFalse(spec.transposed_qk and hdim_q != hdim_v)
                ok, why = is_valid_spec(spec, arch="gfx1151")
                self.assertTrue(ok, why)

    def test_other_arch_defaults_remain_unchanged(self):
        for arch in ("gfx942", "gfx950", "gfx1250"):
            with self.subTest(arch=arch):
                self.assertEqual(
                    dispatch_attention(_request(arch=arch)).candidate.name,
                    "attention_unified_2d",
                )


if __name__ == "__main__":
    unittest.main()
