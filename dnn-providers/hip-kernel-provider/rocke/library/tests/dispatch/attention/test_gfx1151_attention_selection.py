# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Public gfx1151 selection, unsupported requests, and cross-architecture isolation."""

import dataclasses
import json
import unittest
from unittest import mock

from dispatch.attention import (
    AttentionMaskType,
    AttentionRequest,
    attention_candidates,
    attention_tuning_spec,
    dispatch_attention,
)
from dispatch.attention.axes import WMMA_PATH_FIELDS, WMMA_PROBLEM_FIELDS, tuning_axes
from dispatch.attention.gfx1151 import (
    RDNA_WMMA_ARCHES,
    _make_wmma_fwd_candidate,
    _window_bounds,
    _wmma_fwd_spec,
    tuned_wmma_selectors,
)
from kernels.gfx1151.wmma_fmha_fwd import WmmaFmhaFwdSpec, is_valid_spec
from rocke.dispatch.core import PinRefused, pin_to_spec


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
    def test_generic_target_shares_kernels_with_gfx1151_except_tuned_tiles(self):
        for changes in (
            dict(seqlen_q=1024, seqlen_k=1024, hdim_q=64, hdim_v=64),
            dict(seqlen_q=37, seqlen_k=53, mask_type=AttentionMaskType.TOP_LEFT_CAUSAL),
            dict(hdim_q=256, hdim_v=256, batch=1, nhead_q=4, nhead_k=4),
            dict(hdim_q=64, hdim_v=128),
            dict(layout="paged", kv_block_size=16, use_fp8=True),
        ):
            with self.subTest(**changes):
                native = dispatch_attention(_request(**changes)).spec
                generic = dispatch_attention(_request(arch="gfx11-generic", **changes))
                self.assertEqual(generic.candidate.name, "attention_gfx1151_wmma")
                self.assertEqual(
                    dataclasses.replace(
                        generic.spec,
                        block_n=native.block_n,
                        value_tile_size=native.value_tile_size,
                    ),
                    native,
                )

    def test_generic_target_prefers_narrow_wide_launches_and_wide_output_tiles(self):
        wide = dict(seqlen_q=1024, seqlen_k=1024, hdim_q=128, hdim_v=128)
        native = dispatch_attention(_request(**wide)).spec
        generic = dispatch_attention(_request(arch="gfx11-generic", **wide)).spec
        self.assertEqual((native.block_n, native.num_waves), (64, 2))
        self.assertEqual((generic.block_n, generic.num_waves), (32, 2))
        noncausal_d256 = dict(hdim_q=256, hdim_v=256, batch=1, nhead_q=4, nhead_k=4)
        for arch, tile in (
            ("gfx1151", 32),
            ("gfx11-generic", 64),
            ("gfx12-generic", 32),
        ):
            with self.subTest(arch=arch):
                spec = dispatch_attention(_request(arch=arch, **noncausal_d256)).spec
                self.assertEqual(spec.value_tile_size, tile)

    def test_gfx12_targets_run_the_standard_wmma_path(self):
        for arch in ("gfx12-generic", "gfx1201"):
            for changes in (
                dict(seqlen_q=1024, seqlen_k=1024, hdim_q=64, hdim_v=64),
                dict(
                    hdim_q=128, hdim_v=128, mask_type=AttentionMaskType.TOP_LEFT_CAUSAL
                ),
                dict(hdim_q=256, hdim_v=256, batch=1, nhead_q=4, nhead_k=4),
                dict(hdim_q=64, hdim_v=128),
            ):
                with self.subTest(arch=arch, **changes):
                    result = dispatch_attention(_request(arch=arch, **changes))
                    self.assertEqual(result.candidate.name, "attention_gfx1151_wmma")
                    self.assertFalse(result.spec.transposed_qk)
                    ok, why = is_valid_spec(result.spec, arch=arch)
                    self.assertTrue(ok, why)

    def test_device_names_route_to_their_rdna_catalog_row(self):
        # A caller may pass the device name: members without a catalog row use
        # their generic row, processors with one keep it.
        for device, arch in (
            ("gfx1100", "gfx11-generic"),
            ("gfx1103:xnack-", "gfx11-generic"),
            ("gfx1151", "gfx1151"),
            ("gfx1200", "gfx12-generic"),
            ("gfx1200-strict", "gfx12-generic"),
            ("gfx1201", "gfx1201"),
        ):
            with self.subTest(device=device):
                result = dispatch_attention(
                    _request(arch=device, hdim_q=128, hdim_v=128)
                )
                self.assertEqual(result.request.arch, arch)
                self.assertEqual(result.candidate.name, "attention_gfx1151_wmma")
                ok, why = is_valid_spec(result.spec, arch=arch)
                self.assertTrue(ok, why)

    def test_fp8_value_staging_follows_the_wmma_generation(self):
        paged_fp8 = dict(layout="paged", kv_block_size=16, use_fp8=True)
        gfx11 = dispatch_attention(_request(arch="gfx11-generic", **paged_fp8)).spec
        gfx12 = dispatch_attention(_request(arch="gfx12-generic", **paged_fp8)).spec
        self.assertTrue(gfx11.v_lds_stage)
        self.assertFalse(gfx12.v_lds_stage)
        self.assertIsNone(gfx11.scheduler_strategy)
        self.assertIsNone(gfx12.scheduler_strategy)

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
        result = dispatch_attention(_request(hdim_q=96, hdim_v=96, emit_lse=True))
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

    def test_causal_mask_with_window_left_keeps_the_diagonal(self):
        # cuDNN spells a causal sliding window as causal + window_left; an
        # unbounded left (-1) must stay plain causal, not become "no mask".
        for mask in (
            AttentionMaskType.TOP_LEFT_CAUSAL,
            AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        ):
            for left, expected in ((-1, (-1, 0)), (9, (9, 0))):
                with self.subTest(mask=mask, left=left):
                    request = _request(mask_type=mask, window_left=left)
                    self.assertEqual(_window_bounds(request), expected)

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
                lse = dispatch_attention(_request(layout=layout, emit_lse=True)).spec
                self.assertEqual(plain, lse)
        for arch in ("gfx950", "gfx942"):
            with self.subTest(arch=arch), self.assertRaises(ValueError):
                dispatch_attention(_request(arch=arch, emit_lse=True))

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
                emit_lse=True,
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
                    emit_lse=False,
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
        # HIP reports WGPs on RDNA: 20 WGPs is 40 CUs (tiles); 16 is 32 CUs (128
        # groups no longer fit, so the full-head kernel is kept). A generic
        # request reads any device of its family; a processor request its own.
        for device, arch in (("gfx1151", "gfx1151"), ("gfx1100", "gfx11-generic")):
            request = dataclasses.replace(self._unset_cu_request(), arch=arch)
            with self.subTest(device=device, arch=arch), mock.patch(
                "rocke.runtime.hip_module.get_device_target_id", return_value=device
            ):
                with mock.patch(
                    "dispatch.attention.gfx1151._device_num_cus", return_value=20
                ):
                    self.assertGreater(
                        dispatch_attention(request).spec.value_tile_size, 0
                    )
                with mock.patch(
                    "dispatch.attention.gfx1151._device_num_cus", return_value=16
                ):
                    self.assertEqual(
                        dispatch_attention(request).spec.value_tile_size, 0
                    )

    def test_device_that_cannot_run_the_object_keeps_the_reference_count(self):
        # A gfx942 host for a generic object, and a sibling gfx1100 host for a
        # gfx1151-specific object, must not lend their CU count.
        for device, arch in (("gfx942", "gfx11-generic"), ("gfx1100", "gfx1151")):
            request = dataclasses.replace(self._unset_cu_request(), arch=arch)
            with self.subTest(device=device, arch=arch), mock.patch(
                "rocke.runtime.hip_module.get_device_target_id", return_value=device
            ), mock.patch(
                "dispatch.attention.gfx1151._device_num_cus", return_value=16
            ):
                self.assertGreater(dispatch_attention(request).spec.value_tile_size, 0)

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


def _pinned(request, **changes):
    """``request`` pinned to its arch's tuned WMMA forward candidate."""
    return dataclasses.replace(
        request,
        algorithm="wmma_fmha_fwd",
        spec_id=tuned_wmma_selectors(request.arch)[1],
        **changes,
    )


# Transposed wide launch (gfx11), causal with tails, output-column tiling, and an
# unequal-width band: one of each policy branch the knobs hang off.
_TUNED_CASES = (
    dict(seqlen_q=1024, seqlen_k=1024),
    dict(seqlen_q=37, seqlen_k=53, mask_type=AttentionMaskType.TOP_LEFT_CAUSAL),
    dict(hdim_q=256, hdim_v=256, batch=1, nhead_q=4, nhead_k=4),
    dict(
        hdim_q=64,
        hdim_v=128,
        mask_type=AttentionMaskType.SLIDING_WINDOW,
        window_left=32,
        window_right=0,
    ),
)
_STANDARD_KNOBS = {
    "v_lds_stage",
    "causal_tile_skip",
    "value_tile_size",
    "scheduler_strategy",
}


class TestRdnaWmmaTunedCandidates(unittest.TestCase):
    def test_pin_selects_the_tuned_candidate_whose_default_is_the_auto_spec(self):
        for arch in RDNA_WMMA_ARCHES:
            for changes in _TUNED_CASES:
                with self.subTest(arch=arch, **changes):
                    request = _request(arch=arch, **changes)
                    auto = dispatch_attention(request)
                    self.assertEqual(auto.candidate.name, "attention_gfx1151_wmma")
                    tuned = dispatch_attention(_pinned(request))
                    self.assertEqual(
                        tuned.candidate.name, tuned_wmma_selectors(arch)[0]
                    )
                    self.assertEqual(tuned.spec.kernel_spec, auto.spec)
                    self.assertEqual(tuned.request.tuning_id, tuned.spec.tuning_id)

    def test_transposed_geometry_is_tunable_only_on_gfx11(self):
        def axes(arch):
            return {axis.name for axis in tuning_axes(arch, "wmma")}

        for arch in ("gfx11-generic", "gfx1151"):
            self.assertEqual(axes(arch), _STANDARD_KNOBS | {"block_n", "num_waves"})
        for arch in ("gfx12-generic", "gfx1201"):
            self.assertEqual(axes(arch), _STANDARD_KNOBS)
        # A configuration records exactly the knobs its body reads.
        wide = dispatch_attention(
            _pinned(_request(arch="gfx11-generic", seqlen_q=1024, seqlen_k=1024))
        )
        self.assertEqual(dict(wide.spec.knobs), {"block_n": 32, "num_waves": 2})
        standard = _pinned(
            _request(
                arch="gfx12-generic",
                seqlen_q=1024,
                seqlen_k=1024,
                mask_type=AttentionMaskType.TOP_LEFT_CAUSAL,
            )
        )
        result = dispatch_attention(standard)
        self.assertEqual({k for k, _ in result.spec.knobs}, _STANDARD_KNOBS)
        for spec in result.candidate.sweep_space(standard):
            self.assertFalse(spec.kernel_spec.transposed_qk)
            self.assertEqual(
                (spec.kernel_spec.block_n, spec.kernel_spec.num_waves), (32, 1)
            )

    def test_tuning_knobs_round_trip_through_stored_json(self):
        for arch in ("gfx11-generic", "gfx12-generic"):
            for changes in _TUNED_CASES:
                with self.subTest(arch=arch, **changes):
                    request = _request(arch=arch, **changes)
                    spec_id = tuned_wmma_selectors(arch)[1]
                    pinned = _pinned(request)
                    candidate = dispatch_attention(pinned).candidate
                    specs = list(candidate.sweep_space(pinned))
                    self.assertGreater(len(specs), 1)
                    for spec in specs:
                        stored = json.loads(
                            json.dumps({"id": spec.tuning_id, "knobs": spec.knobs})
                        )
                        replayed = attention_tuning_spec(
                            request, spec_id, stored["id"], dict(stored["knobs"])
                        )
                        self.assertEqual(replayed, spec)
                        again = pin_to_spec(request, candidate, spec)
                        self.assertEqual(dispatch_attention(again).spec, spec)

    def test_runtime_head_dims_is_served_only_by_rdna_wmma(self):
        for arch in ("gfx942", "gfx950", "gfx1250"):
            plain = _request(arch=arch, hdim_q=128, hdim_v=128)
            flagged = dataclasses.replace(plain, runtime_head_dims=True)
            checked = 0
            for candidate in attention_candidates():
                capability = candidate.capability
                if arch not in capability.arches or not capability.check(plain)[0]:
                    continue
                with self.subTest(arch=arch, candidate=candidate.name):
                    ok, why = capability.check(flagged)
                    self.assertFalse(ok)
                    self.assertIn("runtime_head_dims", why)
                    checked += 1
            self.assertGreater(checked, 0)
            with self.assertRaises(ValueError):
                dispatch_attention(flagged)

    def test_runtime_head_dims_selects_an_untiled_standard_bucket_object(self):
        # Each request takes, without the flag, the branch the flag turns off:
        # transposed QK, output-column tiling, and fp16 D64 V-LDS staging.
        for changes, exact_only in (
            (dict(seqlen_q=1024, seqlen_k=1024), "transposed_qk"),
            (dict(hdim_q=256, hdim_v=256, batch=1, nhead_q=4, nhead_k=4), None),
            (dict(hdim_q=64, hdim_v=64, seqlen_q=17, seqlen_k=17), "v_lds_stage"),
        ):
            for arch in ("gfx11-generic", "gfx1151"):
                with self.subTest(arch=arch, **changes):
                    request = _request(arch=arch, **changes)
                    exact = dispatch_attention(request).spec
                    bucket_request = dataclasses.replace(
                        request, runtime_head_dims=True
                    )
                    bucket = dispatch_attention(bucket_request).spec
                    if exact_only:
                        self.assertTrue(getattr(exact, exact_only))
                    else:
                        self.assertGreater(exact.value_tile_size, 0)
                    self.assertTrue(bucket.runtime_head_dims)
                    self.assertEqual(bucket.head_size, request.hdim_q)
                    self.assertFalse(bucket.transposed_qk)
                    self.assertFalse(bucket.v_lds_stage)
                    self.assertEqual(bucket.value_tile_size, 0)
                    self.assertEqual(bucket.scheduler_strategy, "max-ilp")
                    tuned = dispatch_attention(_pinned(bucket_request)).spec
                    self.assertEqual(tuned.kernel_spec, bucket)

    def test_illegal_knob_settings_are_refused(self):
        d256 = dict(hdim_q=256, hdim_v=256, batch=1, nhead_q=4, nhead_k=4)
        bucket = _request(arch="gfx11-generic", runtime_head_dims=True, **d256)
        exact = _request(arch="gfx11-generic", **d256)
        unaligned_keys = _request(arch="gfx11-generic", seqlen_q=1024, seqlen_k=1056)
        gfx12 = _request(arch="gfx12-generic")
        for request, knobs, reason in (
            (bucket, {"value_tile_size": 64}, "runtime_head_dims requires"),
            (exact, {"value_tile_size": 48}, "value_tile_size must be"),
            (exact, {"scheduler_strategy": "bogus"}, "unsupported scheduler_strategy"),
            (unaligned_keys, {"block_n": 64}, "must divide seqlen_k"),
            (gfx12, {"block_n": 64}, "not tunable"),
            (gfx12, {"not_a_knob": 1}, "not tunable"),
        ):
            with self.subTest(arch=request.arch, knobs=knobs):
                spec_id = tuned_wmma_selectors(request.arch)[1]
                with self.assertRaises(PinRefused) as caught:
                    attention_tuning_spec(request, spec_id, knobs=knobs)
                self.assertIn(reason, str(caught.exception))

    def test_cu_aware_selection_honors_the_request_over_the_live_device(self):
        d256 = dict(arch="gfx1151", batch=1, nhead_q=4, nhead_k=4, hdim_q=256)
        d256["hdim_v"] = 256
        with mock.patch(
            "rocke.runtime.hip_module.get_device_target_id", return_value="gfx1151"
        ), mock.patch("dispatch.attention.gfx1151._device_num_cus", return_value=16):
            # 16 WGPs (32 CUs) keep the full head on the auto route...
            unset = _request(num_cus=0, **d256)
            self.assertEqual(dispatch_attention(unset).spec.value_tile_size, 0)
            # ...an explicit count overrides the device...
            explicit = _request(num_cus=40, **d256)
            self.assertGreater(dispatch_attention(explicit).spec.value_tile_size, 0)
            # ...and a tuned base never reads it: the reference part's 40 CUs.
            tuned = dispatch_attention(_pinned(unset)).spec.kernel_spec
            self.assertEqual(tuned, dispatch_attention(explicit).spec)

    def test_every_wmma_spec_field_is_a_problem_path_or_knob_field(self):
        fields = {f.name for f in dataclasses.fields(WmmaFmhaFwdSpec)}
        swept = {
            name
            for axis in tuning_axes("gfx1151", "wmma")
            for choice in axis.choices
            for name, _value in choice
        }
        fixed = WMMA_PROBLEM_FIELDS | WMMA_PATH_FIELDS
        self.assertEqual(fields, swept | fixed)
        self.assertFalse(swept & fixed)


if __name__ == "__main__":
    unittest.main()
