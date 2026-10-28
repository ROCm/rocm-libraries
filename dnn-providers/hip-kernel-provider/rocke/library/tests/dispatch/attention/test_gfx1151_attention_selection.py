# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""RDNA (gfx11/gfx12) attention selection: prefill vs decode routing, the
request -> spec policy, tuned candidates, and cross-architecture isolation."""

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
from dispatch.attention.axes import (
    WMMA_DECODE_PATH_FIELDS,
    WMMA_DECODE_PROBLEM_FIELDS,
    WMMA_PROBLEM_FIELDS,
    tuning_axes,
)
from dispatch.attention.gfx1151 import (
    RDNA_WMMA_ARCHES,
    _make_auto_candidate,
    _resolve_num_cus,
    _window_bounds,
    mask_arguments,
    prefill_head_buckets,
    rdna_attention_use_decode,
    request_decode_plan,
    tuned_wmma_selectors,
)
from kernels.gfx1151.wmma_fmha_decode import (
    WmmaFmhaDecodeSpec,
    is_valid_spec as decode_is_valid,
)
from kernels.gfx1151.wmma_fmha_fwd_v2 import (
    WmmaFmhaFwdV2Spec,
    is_valid_spec as prefill_is_valid,
)
from rocke.dispatch.core import PinRefused, pin_to_spec

PREFILL = "attention_rdna_wmma_prefill"
DECODE = "attention_rdna_wmma_decode"


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


def _decode_request(**changes):
    fields = dict(
        seqlen_q=1,
        seqlen_k=4096,
        mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
    )
    fields.update(changes)
    return _request(**fields)


def _is_valid(spec, arch):
    if isinstance(spec, WmmaFmhaDecodeSpec):
        return decode_is_valid(spec, arch=arch)
    return prefill_is_valid(spec, arch=arch)


class TestRdnaAttentionSelection(unittest.TestCase):
    def test_every_rdna_request_runs_one_of_the_two_kernels(self):
        for arch in RDNA_WMMA_ARCHES:
            for request, name, spec_type in (
                (_request(arch=arch), PREFILL, WmmaFmhaFwdV2Spec),
                (_decode_request(arch=arch), DECODE, WmmaFmhaDecodeSpec),
            ):
                with self.subTest(arch=arch, candidate=name):
                    result = dispatch_attention(request)
                    self.assertEqual(result.candidate.name, name)
                    self.assertIsInstance(result.spec, spec_type)
                    ok, why = _is_valid(result.spec, arch)
                    self.assertTrue(ok, why)

    def test_generic_targets_share_objects_with_their_processors(self):
        for generic, processor in (
            ("gfx11-generic", "gfx1151"),
            ("gfx12-generic", "gfx1201"),
        ):
            for request in (
                _request(seqlen_q=1024, seqlen_k=1024, hdim_q=64, hdim_v=64),
                _request(hdim_q=192, hdim_v=128),
                _request(layout="paged", kv_block_size=16, use_fp8=True),
                _decode_request(),
                _decode_request(hdim_q=256, hdim_v=256, use_softcap=True),
            ):
                with self.subTest(arch=generic, request=request):
                    a = dispatch_attention(dataclasses.replace(request, arch=generic))
                    b = dispatch_attention(dataclasses.replace(request, arch=processor))
                    self.assertEqual(a.candidate.name, b.candidate.name)
                    self.assertEqual(a.spec, b.spec)

    def test_device_names_route_to_their_rdna_catalog_row(self):
        for device, arch in (
            ("gfx1100", "gfx11-generic"),
            ("gfx1103:xnack-", "gfx11-generic"),
            ("gfx1151", "gfx1151"),
            ("gfx1200", "gfx12-generic"),
            ("gfx1200-strict", "gfx12-generic"),
            ("gfx1201", "gfx1201"),
        ):
            with self.subTest(device=device):
                result = dispatch_attention(_request(arch=device))
                self.assertEqual(result.request.arch, arch)
                self.assertEqual(result.candidate.name, PREFILL)
                ok, why = _is_valid(result.spec, arch)
                self.assertTrue(ok, why)

    def test_auto_and_explicit_selectors_reach_the_kernels(self):
        for request, selectors, name in (
            (_request(), {}, PREFILL),
            (_request(), {"algorithm": "wmma_fmha_fwd_v2"}, PREFILL),
            (_request(), {"spec_id": "rdna_wmma_fmha_fwd_v2"}, PREFILL),
            (_decode_request(), {}, DECODE),
            (_decode_request(), {"algorithm": "wmma_fmha_decode"}, DECODE),
            (_decode_request(), {"spec_id": "rdna_wmma_fmha_decode"}, DECODE),
            # A pin runs its kernel whatever the decode/prefill rule says.
            (_decode_request(), {"algorithm": "wmma_fmha_fwd_v2"}, PREFILL),
            (_request(), {"algorithm": "wmma_fmha_decode"}, DECODE),
        ):
            with self.subTest(request=request, selectors=selectors):
                result = dispatch_attention(dataclasses.replace(request, **selectors))
                self.assertEqual(result.candidate.name, name)
        with self.assertRaises(ValueError):
            dispatch_attention(_request(layout="dense", algorithm="unified_2d"))

    def test_decode_route_follows_the_rule(self):
        for request, name in (
            (_decode_request(), DECODE),
            (_decode_request(seqlen_q=4), DECODE),
            (_request(seqlen_q=512), PREFILL),
            # gfx11: 4 x 64 packed rows stay on the decode kernel...
            (_request(seqlen_q=64, seqlen_k=32768, batch=1), DECODE),
            # ...512 over a long key range do not.
            (_request(seqlen_q=64, seqlen_k=32768, batch=1, nhead_q=32), PREFILL),
            # A narrow window is a short key range.
            (
                _request(
                    seqlen_q=64,
                    seqlen_k=32768,
                    batch=1,
                    nhead_q=32,
                    mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
                    window_left=127,
                ),
                DECODE,
            ),
            # gfx12: short key ranges send Sq 64 GQA to the prefill.
            (_request(arch="gfx12-generic", seqlen_q=64, seqlen_k=512), PREFILL),
            (_request(arch="gfx12-generic", seqlen_q=64, seqlen_k=4096), DECODE),
            (_decode_request(hdim_q=64, hdim_v=128), PREFILL),  # no decode bucket
            (_decode_request(runtime_head_dims=True), PREFILL),
        ):
            with self.subTest(request=request):
                self.assertEqual(dispatch_attention(request).candidate.name, name)

    def test_decode_rule_vectors(self):
        # (Sq, visible keys, Hq, Hkv, B, family, CUs) -> decode; the hipDNN
        # native pins the same vectors.
        for args, decode in (
            ((1, 4096, 32, 8, 1, "gfx11", 60), True),
            ((32, 32768, 64, 8, 8, "gfx11", 60), True),
            ((64, 4096, 64, 8, 1, "gfx11", 60), True),
            ((64, 4096, 64, 8, 8, "gfx11", 60), False),
            ((64, 2048, 64, 8, 32, "gfx11", 60), True),
            ((64, 32768, 64, 8, 1, "gfx11", 60), False),
            ((64, 32768, 32, 8, 8, "gfx11", 60), True),
            # The gfx11 long-key clause scales with the part: 96 CUs keep
            # these two on the decode kernel.
            ((64, 4096, 64, 8, 8, "gfx11", 96), True),
            ((64, 32768, 64, 8, 1, "gfx11", 96), True),
            ((64, 32768, 64, 8, 8, "gfx11", 96), False),
            ((32, 512, 64, 8, 32, "gfx12", 64), True),
            ((64, 512, 8, 8, 32, "gfx12", 64), True),
            ((64, 512, 32, 8, 8, "gfx12", 64), False),
            ((64, 1024, 64, 8, 1, "gfx12", 64), False),
            ((64, 2048, 64, 8, 8, "gfx12", 64), True),
            ((96, 4096, 8, 8, 1, "gfx12", 64), True),
            ((128, 4096, 8, 8, 8, "gfx12", 64), False),
            ((256, 4096, 8, 8, 1, "gfx12", 64), False),
            ((128, 4096, 8, 8, 1, "gfx11", 60), True),
            ((96, 4096, 32, 8, 1, "gfx11", 60), False),
        ):
            sq, keys, hq, hkv, batch, family, cus = args
            with self.subTest(args=args):
                self.assertEqual(
                    rdna_attention_use_decode(
                        seqlen_q=sq,
                        visible_keys=keys,
                        num_query_heads=hq,
                        num_kv_heads=hkv,
                        batch=batch,
                        wmma_family=family,
                        num_cus=cus,
                    ),
                    decode,
                )

    def test_decode_pin_refuses_widths_no_bucket_serves(self):
        request = _decode_request(hdim_q=64, hdim_v=128, algorithm="wmma_fmha_decode")
        with self.assertRaisesRegex(ValueError, "no decode bucket"):
            dispatch_attention(request)

    def test_unsupported_requests_cannot_fall_back_to_a_different_operation(self):
        cases = (
            {"dtype": "fp32"},
            {"use_fp8": True, "fp8_fnuz": True},
            {"nhead_q": 15},
            {"hdim_q": 100, "hdim_v": 100},
            {"hdim_q": 12, "hdim_v": 12},
            {"hdim_q": 64, "hdim_v": 272},
            {"layout": "sparse"},
            {"layout": "paged", "kv_block_size": 48},
            {"mask_type": 99},
            {"mask_type": AttentionMaskType.SLIDING_WINDOW},
            {"sliding_window": -1},
        )
        for make in (_request, _decode_request):
            for changes in cases:
                fields = {"layout": "dense"}
                fields.update(changes)
                with self.subTest(make=make.__name__, changes=changes):
                    with self.assertRaises(ValueError):
                        dispatch_attention(make(**fields))

    def test_runtime_shape_fields_reuse_one_spec(self):
        for make in (_request, _decode_request):
            with self.subTest(make=make.__name__):
                first = dispatch_attention(make(nhead_q=16, nhead_k=4)).spec
                second = dispatch_attention(
                    make(nhead_q=32, nhead_k=8, seqlen_k=8192, batch=3, num_cus=40)
                ).spec
                self.assertEqual(first, second)

    def test_prefill_head_width_buckets(self):
        for (dq, dv), buckets in {
            (64, 64): (64, 64),
            (40, 56): (64, 64),
            (8, 8): (64, 64),
            (72, 96): (96, 96),
            (112, 128): (128, 128),
            (64, 128): (64, 128),
            (128, 64): (128, 64),
            (192, 128): (192, 128),
            (200, 120): (256, 128),
            (136, 136): (256, 256),
            (160, 160): (256, 256),
            (192, 192): (256, 256),
            (256, 256): (256, 256),
        }.items():
            with self.subTest(dq=dq, dv=dv):
                self.assertEqual(prefill_head_buckets(dq, dv), buckets)
                for arch in ("gfx11-generic", "gfx12-generic"):
                    spec = dispatch_attention(
                        _request(arch=arch, hdim_q=dq, hdim_v=dv)
                    ).spec
                    self.assertEqual((spec.head_dim, spec.head_dim_v), buckets)
                    ok, why = prefill_is_valid(spec, arch=arch)
                    self.assertTrue(ok, why)
        with self.assertRaises(ValueError):
            prefill_head_buckets(264, 64)

    def test_prefill_geometry_follows_the_wmma_family(self):
        # gfx12 keeps Q/K up to 256 in one wave while V/O fits 128; gfx11
        # splits every width above 128 across a wave pair (64-row CTAs).
        for (dq, dv), (gfx11_pair, gfx12_pair) in {
            (128, 128): (False, False),
            (192, 128): (True, False),
            (256, 128): (True, False),
            (256, 256): (True, True),
        }.items():
            for arch, pair in (
                ("gfx11-generic", gfx11_pair),
                ("gfx12-generic", gfx12_pair),
            ):
                with self.subTest(arch=arch, dq=dq, dv=dv):
                    spec = dispatch_attention(
                        _request(arch=arch, hdim_q=dq, hdim_v=dv)
                    ).spec
                    self.assertEqual(spec.wave_pair, pair)
                    self.assertEqual(spec.block_m, 64 if pair else 128)
                    self.assertEqual(
                        (spec.block_n, spec.waves_per_eu, spec.scheduler_strategy),
                        (32, 2, None),
                    )

    def test_decode_buckets(self):
        for (dq, dv), bucket in {
            (64, 64): 64,
            (16, 64): 64,
            (96, 64): 128,
            (128, 128): 128,
            (72, 128): 128,
            (192, 128): 256,
            (256, 256): 256,
        }.items():
            with self.subTest(dq=dq, dv=dv):
                result = dispatch_attention(_decode_request(hdim_q=dq, hdim_v=dv))
                self.assertEqual(result.candidate.name, DECODE)
                self.assertEqual(result.spec.head_size, bucket)
        for dq, dv in ((64, 128), (128, 256), (8, 128)):
            with self.subTest(dq=dq, dv=dv):
                result = dispatch_attention(_decode_request(hdim_q=dq, hdim_v=dv))
                self.assertEqual(result.candidate.name, PREFILL)

    def test_decode_spec_is_resolved_for_its_family(self):
        for arch, family in (("gfx11-generic", "gfx11"), ("gfx12-generic", "gfx12")):
            with self.subTest(arch=arch):
                spec = dispatch_attention(_decode_request(arch=arch)).spec
                self.assertEqual(spec.wmma_family, family)
                self.assertTrue(spec.dv_split and spec.q_mode)

    def test_mask_class_and_runtime_bounds(self):
        none = dispatch_attention(_request()).spec
        self.assertEqual(none.mask, "none")
        for changes, bounds in (
            (dict(mask_type=AttentionMaskType.TOP_LEFT_CAUSAL), (False, -1, 0)),
            (dict(mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL), (True, -1, 0)),
            (
                dict(
                    mask_type=AttentionMaskType.SLIDING_WINDOW,
                    window_left=31,
                    window_right=7,
                ),
                (False, 31, 7),
            ),
            (
                dict(
                    mask_type=AttentionMaskType.SLIDING_WINDOW,
                    window_left=127,
                    window_right=63,
                    window_bottom_right=True,
                ),
                (True, 127, 63),
            ),
            (dict(window_right=16), (False, -1, 16)),
            (
                dict(mask_type=AttentionMaskType.TOP_LEFT_CAUSAL, window_left=9),
                (False, 9, 0),
            ),
        ):
            with self.subTest(**changes):
                request = _request(**changes)
                spec = dispatch_attention(request).spec
                # One masked object serves every bound: they are arguments.
                self.assertEqual(spec, dataclasses.replace(none, mask="causal"))
                self.assertEqual(mask_arguments(request), bounds)

    def test_causal_mask_with_window_left_keeps_the_diagonal(self):
        for mask in (
            AttentionMaskType.TOP_LEFT_CAUSAL,
            AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        ):
            for left, expected in ((-1, (-1, 0)), (9, (9, 0))):
                with self.subTest(mask=mask, left=left):
                    request = _request(mask_type=mask, window_left=left)
                    self.assertEqual(_window_bounds(request), expected)

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

    def test_lse_reuses_the_same_code_object(self):
        for make in (_request, _decode_request):
            for layout in ("dense", "ragged"):
                with self.subTest(make=make.__name__, layout=layout):
                    plain = dispatch_attention(make(layout=layout)).spec
                    lse = dispatch_attention(make(layout=layout, emit_lse=True)).spec
                    self.assertEqual(plain, lse)
        for arch in ("gfx950", "gfx942"):
            with self.subTest(arch=arch), self.assertRaises(ValueError):
                dispatch_attention(_request(arch=arch, emit_lse=True))

    def test_score_features_layouts_and_fp8_reach_both_kernels(self):
        features = dict(
            use_attn_bias=True,
            attn_bias_dtype="q",
            use_softcap=True,
            use_sinks=True,
            use_alibi=True,
            use_qq_bias=True,
            use_fp8=True,
            mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        )
        for layout, page in (("dense", 16), ("ragged", 16), ("paged", 32)):
            with self.subTest(layout=layout):
                prefill = dispatch_attention(
                    _request(layout=layout, kv_block_size=page, **features)
                ).spec
                self.assertEqual(
                    (
                        prefill.bias,
                        prefill.use_softcap,
                        prefill.use_sinks,
                        prefill.use_alibi,
                        prefill.use_qq_bias,
                        prefill.kv_dtype,
                        prefill.layout,
                        prefill.page_block_size,
                    ),
                    (
                        "q",
                        True,
                        True,
                        True,
                        True,
                        "fp8e4m3",
                        layout,
                        page if layout == "paged" else 0,
                    ),
                )
                decode = dispatch_attention(
                    _decode_request(layout=layout, kv_block_size=page, **features)
                ).spec
                self.assertEqual(
                    (
                        decode.bias,
                        decode.softcap,
                        decode.sinks,
                        decode.alibi,
                        decode.qq_bias,
                        decode.kv_dtype,
                        decode.layout,
                        decode.page_block_size,
                    ),
                    (
                        "q",
                        True,
                        True,
                        True,
                        True,
                        "fp8e4m3",
                        layout,
                        page if layout == "paged" else 0,
                    ),
                )
        for make in (_request, _decode_request):
            with self.subTest(make=make.__name__):
                f32 = dispatch_attention(make(use_attn_bias=True)).spec
                self.assertEqual(f32.bias, "f32")
                plain = dispatch_attention(make()).spec
                self.assertFalse(plain.bias)

    def test_other_arch_defaults_remain_unchanged(self):
        for arch in ("gfx942", "gfx950", "gfx1250"):
            with self.subTest(arch=arch):
                self.assertEqual(
                    dispatch_attention(_request(arch=arch)).candidate.name,
                    "attention_unified_2d",
                )

    def test_sweep_space_offers_distinct_valid_variants_baseline_first(self):
        for request in (
            _request(seqlen_q=1024, seqlen_k=1024, hdim_q=64, hdim_v=64),
            _request(hdim_q=256, hdim_v=256, mask_type=AttentionMaskType.NO_MASK),
            _decode_request(),
            _decode_request(hdim_q=256, hdim_v=256, use_fp8=True, use_softcap=True),
        ):
            for arch in ("gfx11-generic", "gfx12-generic"):
                with self.subTest(arch=arch, request=request):
                    request = dataclasses.replace(request, arch=arch)
                    result = dispatch_attention(request)
                    sweep = result.candidate.sweep_space(request)
                    self.assertGreater(len(sweep), 1)
                    self.assertEqual(sweep[0], result.spec)
                    self.assertEqual(len(set(sweep)), len(sweep))
                    for spec in sweep:
                        self.assertIs(type(spec), type(result.spec))
                        ok, why = _is_valid(spec, arch)
                        self.assertTrue(ok, why)

    def test_sweep_space_is_empty_for_inadmissible_requests(self):
        request = _request(hdim_q=100, hdim_v=128)
        for kind in ("prefill", "decode"):
            self.assertEqual(_make_auto_candidate(kind).sweep_space(request), ())


class TestRdnaComputeUnits(unittest.TestCase):
    """The CU count feeds the decode split count (``num_cus // 2`` HIP
    multiprocessors) and the decode/prefill rule."""

    def _unset(self, arch):
        return _decode_request(arch=arch, num_cus=0, batch=1)

    def test_default_compute_units_match_the_reference_part(self):
        with mock.patch(
            "rocke.runtime.hip_module.get_device_target_id", return_value=""
        ):
            self.assertEqual(_resolve_num_cus(self._unset("gfx1151")), 40)

    def test_live_device_compute_units_are_doubled_for_wgp_mode(self):
        for device, arch in (("gfx1151", "gfx1151"), ("gfx1100", "gfx11-generic")):
            with self.subTest(device=device, arch=arch), mock.patch(
                "rocke.runtime.hip_module.get_device_target_id", return_value=device
            ), mock.patch(
                "dispatch.attention.gfx1151._device_num_cus", return_value=20
            ):
                self.assertEqual(_resolve_num_cus(self._unset(arch)), 40)

    def test_device_that_cannot_run_the_object_keeps_the_reference_count(self):
        for device, arch in (("gfx942", "gfx11-generic"), ("gfx1100", "gfx1151")):
            with self.subTest(device=device, arch=arch), mock.patch(
                "rocke.runtime.hip_module.get_device_target_id", return_value=device
            ), mock.patch(
                "dispatch.attention.gfx1151._device_num_cus", return_value=96
            ):
                self.assertEqual(_resolve_num_cus(self._unset(arch)), 40)

    def test_route_rule_gets_the_request_cu_count_over_the_live_device(self):
        # 64 query heads over 8 KV heads, 64 rows against 32K keys: the rule picks the
        # prefill kernel at 64 CUs and the decode kernel from 96 CUs (a WGP count
        # instead of a CU count would move each boundary by 2x).
        shape = dict(arch="gfx11-generic", batch=1, nhead_q=64, nhead_k=8, seqlen_q=64)
        shape["seqlen_k"] = 32768
        routed = {}
        for cus, spec_type in (
            (64, WmmaFmhaFwdV2Spec),
            (96, WmmaFmhaDecodeSpec),
            (128, WmmaFmhaDecodeSpec),
        ):
            with self.subTest(num_cus=cus), mock.patch(
                "rocke.runtime.hip_module.get_device_target_id",
                return_value="gfx1100",
            ), mock.patch(
                "dispatch.attention.gfx1151._device_num_cus", return_value=128
            ), mock.patch(
                "dispatch.attention.gfx1151.rdna_attention_use_decode",
                wraps=rdna_attention_use_decode,
            ) as rule:
                request = _decode_request(num_cus=cus, **shape)
                # A live device of 128 WGPs (256 CUs) would route every case to the
                # decode kernel; the explicit count wins.
                self.assertIsInstance(dispatch_attention(request).spec, spec_type)
                routed[cus] = {call.kwargs["num_cus"] for call in rule.call_args_list}
        self.assertEqual(routed, {64: {64}, 96: {96}, 128: {128}})

    def test_decode_splits_follow_the_multiprocessor_count(self):
        request = _decode_request(batch=1, nhead_q=32, nhead_k=8, seqlen_k=32768)
        spec = dispatch_attention(request).spec
        small = request_decode_plan(request, spec, num_cus=8)
        large = request_decode_plan(request, spec, num_cus=128)
        self.assertLess(small.num_splits, large.num_splits)
        # 8 pack groups x 1 row tile; 2 workgroups per multiprocessor.
        self.assertEqual(large.num_splits, 128 // 2 * 2 // 8)
        self.assertEqual(large.grid, (8 * large.num_splits, 1, 1))
        self.assertEqual(large.fused, 1 < large.num_splits <= 4)
        one = request_decode_plan(
            dataclasses.replace(request, seqlen_k=64), spec, num_cus=128
        )
        self.assertEqual(one.num_splits, 1)
        self.assertEqual(one.workspace["bytes"], 0)

    def test_decode_splits_count_only_the_visible_window(self):
        window = _decode_request(
            batch=1,
            nhead_q=32,
            nhead_k=8,
            seqlen_k=32768,
            mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
            window_left=127,
        )
        spec = dispatch_attention(window).spec
        self.assertEqual(request_decode_plan(window, spec, num_cus=128).num_splits, 1)


def _pinned(request, kind, **changes):
    """``request`` pinned to its arch's tuned ``kind`` candidate."""
    return dataclasses.replace(
        request,
        algorithm="wmma_fmha_decode" if kind == "decode" else "wmma_fmha_fwd_v2",
        spec_id=tuned_wmma_selectors(request.arch, kind)[1],
        **changes,
    )


_TUNED_CASES = (
    ("prefill", dict(seqlen_q=1024, seqlen_k=1024)),
    (
        "prefill",
        dict(seqlen_q=37, seqlen_k=53, mask_type=AttentionMaskType.TOP_LEFT_CAUSAL),
    ),
    ("prefill", dict(hdim_q=256, hdim_v=256, batch=1, nhead_q=4, nhead_k=4)),
    ("prefill", dict(hdim_q=192, hdim_v=128, use_attn_bias=True)),
    ("decode", dict(seqlen_q=1, seqlen_k=4096)),
    ("decode", dict(seqlen_q=4, seqlen_k=777, hdim_q=256, hdim_v=256)),
    ("decode", dict(seqlen_q=1, seqlen_k=4096, use_fp8=True, use_softcap=True)),
)
_PREFILL_KNOBS = {
    "block_m",
    "wave_pair",
    "block_n",
    "waves_per_eu",
    "scheduler_strategy",
    "lazy_rescale",
    "k_pad",
    "v_pad",
    "cta_order",
}
_DECODE_KNOBS = {
    "num_waves",
    "dv_split",
    "q_mode",
    "v_store_early",
    "prefetch",
    "lockstep",
    "scheduler_strategy",
    "waves_per_eu",
    "fp8_cvt",
    "tanh_fast",
}


class TestRdnaWmmaTunedCandidates(unittest.TestCase):
    def test_pin_selects_the_tuned_candidate_whose_default_is_the_auto_spec(self):
        for arch in RDNA_WMMA_ARCHES:
            for kind, changes in _TUNED_CASES:
                with self.subTest(arch=arch, kind=kind, **changes):
                    request = _request(arch=arch, **changes)
                    algorithm = (
                        "wmma_fmha_decode" if kind == "decode" else "wmma_fmha_fwd_v2"
                    )
                    auto = dispatch_attention(
                        dataclasses.replace(request, algorithm=algorithm)
                    )
                    self.assertEqual(auto.candidate.name, f"attention_rdna_wmma_{kind}")
                    tuned = dispatch_attention(_pinned(request, kind))
                    self.assertEqual(
                        tuned.candidate.name, tuned_wmma_selectors(arch, kind)[0]
                    )
                    self.assertEqual(
                        tuned.spec.path, "wmma_decode" if kind == "decode" else "wmma"
                    )
                    self.assertEqual(tuned.spec.kernel_spec, auto.spec)
                    self.assertEqual(tuned.request.tuning_id, tuned.spec.tuning_id)

    def test_axes_cover_each_kernels_knobs(self):
        for arch in RDNA_WMMA_ARCHES:
            with self.subTest(arch=arch):
                self.assertEqual(
                    {axis.name for axis in tuning_axes(arch, "wmma")}, _PREFILL_KNOBS
                )
                self.assertEqual(
                    {axis.name for axis in tuning_axes(arch, "wmma_decode")},
                    _DECODE_KNOBS,
                )
        # A configuration records exactly the knobs its kernel reads.
        prefill = dispatch_attention(_pinned(_request(arch="gfx12-generic"), "prefill"))
        self.assertEqual({k for k, _ in prefill.spec.knobs}, _PREFILL_KNOBS)
        decode = dispatch_attention(
            _pinned(_decode_request(arch="gfx11-generic"), "decode")
        )
        self.assertEqual(
            {k for k, _ in decode.spec.knobs}, _DECODE_KNOBS - {"fp8_cvt", "tanh_fast"}
        )
        fp8 = dispatch_attention(
            _pinned(_decode_request(arch="gfx12-generic", use_fp8=True), "decode")
        )
        self.assertEqual({k for k, _ in fp8.spec.knobs}, _DECODE_KNOBS - {"tanh_fast"})

    def test_tuning_knobs_round_trip_through_stored_json(self):
        for arch in ("gfx11-generic", "gfx12-generic"):
            for kind, changes in _TUNED_CASES:
                with self.subTest(arch=arch, kind=kind, **changes):
                    request = _request(arch=arch, **changes)
                    spec_id = tuned_wmma_selectors(arch, kind)[1]
                    pinned = _pinned(request, kind)
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

    def test_tuned_specs_build_their_kernels(self):
        for arch in ("gfx11-generic", "gfx12-generic"):
            for kind, changes in _TUNED_CASES[::3]:
                with self.subTest(arch=arch, kind=kind):
                    result = dispatch_attention(
                        _pinned(_request(arch=arch, **changes), kind)
                    )
                    built = result.candidate.built(result.spec, arch)
                    kernels = built if isinstance(built, tuple) else (built,)
                    self.assertEqual(len(kernels), 2 if kind == "decode" else 1)
                    self.assertEqual(
                        kernels[0].name, result.spec.kernel_spec.kernel_name()
                    )

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
        for arch in RDNA_WMMA_ARCHES:
            with self.subTest(arch=arch):
                bucket = dispatch_attention(
                    _decode_request(arch=arch, runtime_head_dims=True)
                )
                self.assertEqual(bucket.candidate.name, PREFILL)

    def test_illegal_knob_settings_are_refused(self):
        d256 = dict(hdim_q=256, hdim_v=256, batch=1, nhead_q=4, nhead_k=4)
        for request, kind, knobs, reason in (
            (
                _request(arch="gfx11-generic", **d256),
                "prefill",
                {"wave_pair": False},
                "unsplit waves",
            ),
            (
                _request(arch="gfx12-generic", hdim_q=96, hdim_v=96),
                "prefill",
                {"k_pad": 4},
                "16-byte row alignment",
            ),
            (
                _request(arch="gfx11-generic"),
                "prefill",
                {"block_n": 24},
                "multiple of 16",
            ),
            (
                _request(arch="gfx12-generic"),
                "prefill",
                {"not_a_knob": 1},
                "not tunable",
            ),
            (
                _decode_request(arch="gfx11-generic", hdim_q=64, hdim_v=64),
                "decode",
                {"dv_split": 4},
                "invalid dv_split",
            ),
            (
                _decode_request(arch="gfx12-generic"),
                "decode",
                {"block_m": 64},
                "not tunable",
            ),
        ):
            with self.subTest(arch=request.arch, knobs=knobs):
                spec_id = tuned_wmma_selectors(request.arch, kind)[1]
                with self.assertRaises(PinRefused) as caught:
                    attention_tuning_spec(request, spec_id, knobs=knobs)
                self.assertIn(reason, str(caught.exception))

    def test_tuned_base_never_reads_the_live_device(self):
        request = _decode_request(arch="gfx1151", num_cus=0)
        with mock.patch(
            "rocke.runtime.hip_module.get_device_target_id", return_value="gfx1151"
        ), mock.patch("dispatch.attention.gfx1151._device_num_cus", return_value=8):
            tuned = dispatch_attention(_pinned(request, "decode"))
        reference = dispatch_attention(
            _pinned(_decode_request(arch="gfx1151", num_cus=40), "decode")
        )
        self.assertEqual(tuned.spec, reference.spec)

    def test_tuned_decode_grid_is_the_launched_grid(self):
        # The binding plans with the live CU count; the reported grid must too.
        request = _decode_request(arch="gfx1151", num_cus=0)
        with mock.patch(
            "rocke.runtime.hip_module.get_device_target_id", return_value="gfx1151"
        ), mock.patch("dispatch.attention.gfx1151._device_num_cus", return_value=8):
            tuned = dispatch_attention(_pinned(request, "decode"))
            grid = tuned.candidate.grid(tuned.spec, tuned.request)
            launched = request_decode_plan(tuned.request, tuned.spec.kernel_spec).grid
        self.assertEqual(grid, launched)
        self.assertNotEqual(
            grid,
            request_decode_plan(tuned.request, tuned.spec.kernel_spec, num_cus=40).grid,
        )

    def test_every_spec_field_is_a_problem_path_or_knob_field(self):
        for spec_type, path, fixed in (
            (WmmaFmhaFwdV2Spec, "wmma", WMMA_PROBLEM_FIELDS),
            (
                WmmaFmhaDecodeSpec,
                "wmma_decode",
                WMMA_DECODE_PROBLEM_FIELDS | WMMA_DECODE_PATH_FIELDS,
            ),
        ):
            with self.subTest(path=path):
                fields = {f.name for f in dataclasses.fields(spec_type)}
                swept = {
                    name
                    for axis in tuning_axes("gfx1151", path)
                    for choice in axis.choices
                    for name, _value in choice
                }
                self.assertEqual(fields, swept | fixed)
                self.assertFalse(swept & fixed)


if __name__ == "__main__":
    unittest.main()
