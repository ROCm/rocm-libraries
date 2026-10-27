# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A gfx942/gfx950 ``AttentionRequest(use_attn_bias=True)`` reaches the unified
kernels.

CPU-only: routing, tuning candidates and the explicit binding (with a stub
runner). The numeric check is ``test_attention_additive_bias_numeric.py``.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

from dispatch.attention import (
    AttentionRequest,
    attention_execution_candidates,
    dispatch_attention,
)
from dispatch.attention.common import AttentionSpec, _problem, _tuning_problem
from dispatch.attention.tuning_specs import (
    ExplicitAttention2DConfig,
    _bias_fields,
    make_explicit_attention_2d_spec,
)

_BIAS_PARAMS = [
    "additive_bias_ptr",
    "additive_bias_batch_stride",
    "additive_bias_head_stride",
    "additive_bias_sq_stride",
]


def _req(arch="gfx950", **kw) -> AttentionRequest:
    base = {
        "batch": 1,
        "nhead_q": 32,
        "nhead_k": 8,
        "seqlen_q": 1024,
        "seqlen_k": 1024,
        "hdim_q": 128,
        "hdim_v": 128,
        "arch": arch,
        "dtype": "bf16",
        # Bottom-right causal: the mask every arch's unified kernels implement.
        "mask_type": 2,
        "num_cus": 256,
        "use_attn_bias": True,
    }
    base.update(kw)
    return AttentionRequest(**base)


def _candidate(prefix: str):
    return next(
        c for c in attention_execution_candidates() if c.name.startswith(prefix)
    )


def _pinned(candidate, **kw) -> AttentionRequest:
    kw.setdefault("arch", candidate.capability.arches[0])
    return _req(algorithm=candidate.algorithm, spec_id=candidate.spec_id, **kw)


class _Tensor:
    def __init__(self, shape, dtype, values=None, *, strides=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = "cuda:0"
        self._values = values
        self._strides = strides

    def is_contiguous(self):
        return self._strides is None

    def detach(self):
        return self

    def stride(self, dim):
        if self._strides is not None:
            return self._strides[dim]
        stride = 1
        values = [0] * len(self.shape)
        for index in range(len(self.shape) - 1, -1, -1):
            values[index] = stride
            stride *= self.shape[index]
        return values[dim]

    def cpu(self):
        return self

    def tolist(self):
        if self._values is None:
            raise AssertionError("no host values")
        return self._values


def _paged_tensors():
    return {
        "q": _Tensor((1024, 32, 128), "torch.bfloat16"),
        "out": _Tensor((1024, 32, 128), "torch.bfloat16"),
        "k": _Tensor((64, 16, 8, 128), "torch.bfloat16"),
        "v": _Tensor((64, 16, 8, 128), "torch.bfloat16"),
        "cu_seqlens_q": _Tensor((2,), "torch.int32", [0, 1024]),
        "seqused_k": _Tensor((1,), "torch.int32", [1024]),
        "block_table": _Tensor((1, 64), "torch.int32", [list(range(64))]),
    }


class TestRouting(unittest.TestCase):
    def test_request_maps_to_problem(self):
        self.assertTrue(_problem(_req()).use_additive_bias)
        self.assertTrue(_tuning_problem(_req()).use_additive_bias)
        self.assertFalse(_problem(_req(use_attn_bias=False)).use_additive_bias)

    def test_prefill_and_decode_admit_bias(self):
        for arch, path, kw in (
            (arch, path, kw)
            for arch in ("gfx942", "gfx950")
            for path, kw in (("2d", {}), ("3d", {"seqlen_q": 1, "seqlen_k": 8192}))
        ):
            with self.subTest(arch=arch, path=path):
                kw = {"arch": arch, **kw}
                result = dispatch_attention(_req(**kw))
                self.assertEqual(result.candidate.name, f"attention_unified_{path}")
                self.assertEqual(result.spec.path, path)
                self.assertTrue(result.spec.use_additive_bias)
                self.assertTrue(result.spec.kernel_name().endswith("_addb"))
                plain = dispatch_attention(_req(use_attn_bias=False, **kw)).spec
                self.assertNotEqual(plain.kernel_name(), result.spec.kernel_name())

    def test_bias_off_identity_is_unchanged(self):
        spec = dispatch_attention(_req(use_attn_bias=False)).spec
        self.assertNotIn("use_additive_bias", spec.identity())
        self.assertNotIn("addb", spec.kernel_name())
        biased = dispatch_attention(_req()).spec
        self.assertTrue(biased.identity()["use_additive_bias"])

    def test_spec_flag_must_be_a_bool(self):
        with self.assertRaisesRegex(ValueError, "use_additive_bias must be a bool"):
            AttentionSpec(
                path="2d",
                head_size=128,
                block_size=16,
                dtype="bf16",
                num_query_heads=32,
                num_kv_heads=8,
                use_additive_bias=1,
            )

    def test_unsupported_bias_requests_are_refused(self):
        cases = {
            "q_dtype_bias": _req(attn_bias_dtype="q"),
            "gfx1250": _req(arch="gfx1250"),
            "strided_decode": _req(seqlen_q=1, seqlen_k=8192, kv_layout="strided"),
        }
        for label, request in cases.items():
            with self.subTest(case=label), self.assertRaises(ValueError):
                dispatch_attention(request)


class TestTuningCandidates(unittest.TestCase):
    def test_bias_geometries_build_bias_specs(self):
        for prefix in (
            "attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm",
            "attention_gfx950_u2d_transposed32_nw2_mw32_t2xb_llvm",
            "attention_gfx950_u3d_splitkv_seg16_t1xb",
            "attention_gfx942_u2d_narrow_nw2_mw16_t4xb_llvm",
            "attention_gfx942_u2d_narrow_nw2_mw32_t1xb_llvm",
            "attention_gfx942_u3d_splitkv_seg16_t1xb",
        ):
            candidate = _candidate(prefix)
            with self.subTest(candidate=candidate.name):
                req = _pinned(candidate)
                ok, why = candidate.admits(req)
                self.assertTrue(ok, why)
                spec = candidate.select_spec(req)
                self.assertTrue(spec.kernel_spec.use_additive_bias)
                self.assertIn("addb", spec.kernel_spec.kernel_name())
                names = [item["name"] for item in candidate.signature(spec)]
                self.assertEqual(names[-4:], _BIAS_PARAMS)

                plain = candidate.select_spec(_pinned(candidate, use_attn_bias=False))
                plain_names = [item["name"] for item in candidate.signature(plain)]
                self.assertNotIn("additive_bias_ptr", plain_names)
                # One configuration, two kernels: the knob identity is shared,
                # the compiled-kernel identity is not.
                self.assertEqual(plain.tuning_id, spec.tuning_id)
                self.assertNotEqual(plain.cache_key(), spec.cache_key())

    def test_bias_off_gfx942_identity_omits_the_field(self):
        # gfx942 tuning ids and kernel cache keys were stored before its specs
        # had the field, so a bias-off key leaves it out; gfx950's keys were
        # stored with it and keep it.
        for prefix, has_field in (
            ("attention_gfx942_u2d_narrow_nw2_mw16_t4xb_llvm", False),
            ("attention_gfx942_u3d_splitkv_seg16_t1xb", False),
            ("attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm", True),
        ):
            candidate = _candidate(prefix)
            with self.subTest(candidate=candidate.name):
                plain = candidate.select_spec(_pinned(candidate, use_attn_bias=False))
                names = {name for name, _ in plain.cache_key()[4]}
                self.assertEqual("use_additive_bias" in names, has_field)
                biased = candidate.select_spec(_pinned(candidate))
                self.assertIn(("use_additive_bias", "True"), set(biased.cache_key()[4]))

    def test_non_bias_geometries_refuse_bias(self):
        for prefix in (
            "attention_gfx950_u2d_wide32_nw2_mw32_t2xb_llvm",
            "attention_gfx942_u2d_wide32x8_nw2_mw32_t2xb_llvm",
            "attention_gfx942_u2d_transposed_x8_nw2_mw32_t2xb_llvm",
            "attention_gfx942_u2d_gfx942_4warp",
        ):
            candidate = _candidate(prefix)
            with self.subTest(candidate=candidate.name):
                ok, _ = candidate.admits(_pinned(candidate))
                self.assertFalse(ok)

    def test_explicit_spec_on_gfx942_and_gfx950_only(self):
        problem = replace(_problem(_req()), use_additive_bias=True)
        config = ExplicitAttention2DConfig(
            num_warps=1, block_m_per_warp=16, tile_policy="1x"
        )
        for arch in ("gfx942", "gfx950"):
            with self.subTest(arch=arch):
                spec = make_explicit_attention_2d_spec(problem, config, arch=arch)
                self.assertTrue(spec.use_additive_bias)
        with self.assertRaisesRegex(ValueError, "gfx942/gfx950-only"):
            _bias_fields(problem, "gfx1201")


class TestBinding(unittest.TestCase):
    def setUp(self):
        self.candidate = _candidate("attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm")
        self.req = _pinned(self.candidate)
        self.spec = self.candidate.select_spec(self.req)
        self.captured = {}
        patcher = mock.patch(
            "kernels.common.attention_unified.run_unified_attention_torch",
            side_effect=lambda **kw: self.captured.update(kw),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _launch(self, bias):
        tensors = _paged_tensors()
        tensors["attn_bias"] = bias
        self.candidate.bound_torch(self.req, self.spec, tensors).launch()
        return self.captured

    def test_binds_on_gfx942(self):
        candidate = _candidate("attention_gfx942_u2d_narrow_nw2_mw16_t4xb_llvm")
        req = _pinned(candidate)
        spec = candidate.select_spec(req)
        tensors = _paged_tensors()
        bias = _Tensor((1, 1, 1, 1024), "torch.float32")
        tensors["attn_bias"] = bias
        candidate.bound_torch(req, spec, tensors).launch()
        self.assertIs(self.captured["additive_bias"], bias)
        self.assertTrue(self.captured["tuning_spec"].kernel_spec.use_additive_bias)

    def test_binds_bias_and_derives_strides(self):
        cases = {
            # [1, 1, 1, Sk] key-padding bias: every dimension broadcasts.
            "broadcast": ((1, 1, 1, 1024), (0, 0, 0)),
            "full": ((1, 32, 1024, 1024), (0, 1024 * 1024, 1024)),
            "per_head": ((1, 32, 1, 1024), (0, 1024, 0)),
        }
        for label, (shape, strides) in cases.items():
            with self.subTest(case=label):
                bias = _Tensor(shape, "torch.float32")
                kw = self._launch(bias)
                self.assertIs(kw["additive_bias"], bias)
                self.assertEqual(
                    (
                        kw["additive_bias_batch_stride"],
                        kw["additive_bias_head_stride"],
                        kw["additive_bias_sq_stride"],
                    ),
                    strides,
                )
                self.assertTrue(kw["problem"].use_additive_bias)

    def test_unified_binding_allows_more_than_2gib_of_bias_bytes(self):
        # The unified kernels index the bias in i32 *elements*; the runner
        # bounds that index, so the binding must not apply the RDNA byte cap.
        head_stride = 20_000_000  # 31 * 20M * 4 bytes > 2 GiB; index < 2**31
        bias = _Tensor(
            (1, 32, 1, 1024), "torch.float32", strides=(0, head_stride, 0, 1)
        )
        kw = self._launch(bias)
        self.assertEqual(kw["additive_bias_head_stride"], head_stride)

    def test_bias_tensor_is_required_and_validated(self):
        with self.assertRaisesRegex(ValueError, r"requires tensors\['attn_bias'\]"):
            self.candidate.bound_torch(self.req, self.spec, _paged_tensors())
        cases = {
            "dtype": (_Tensor((1, 1, 1, 1024), "torch.bfloat16"), "fp32"),
            "rank": (_Tensor((1, 1024), "torch.float32"), "rank-4"),
            "heads": (_Tensor((1, 4, 1, 1024), "torch.float32"), "head"),
            "keys": (_Tensor((1, 1, 1, 512), "torch.float32"), "seqlen_k"),
            "key_stride": (
                _Tensor((1, 1, 1, 1024), "torch.float32", strides=(0, 0, 0, 2)),
                "contiguous",
            ),
        }
        for label, (bias, message) in cases.items():
            with self.subTest(case=label), self.assertRaisesRegex(ValueError, message):
                self._launch(bias)

    def test_bias_off_request_refuses_a_bias_tensor(self):
        candidate = self.candidate
        req = _pinned(candidate, use_attn_bias=False)
        spec = candidate.select_spec(req)
        tensors = _paged_tensors()
        tensors["attn_bias"] = _Tensor((1, 1, 1, 1024), "torch.float32")
        with self.assertRaisesRegex(ValueError, "no additive bias"):
            candidate.bound_torch(req, spec, tensors)

    def test_request_problem_and_spec_must_agree(self):
        plain_spec = self.candidate.select_spec(
            _pinned(self.candidate, use_attn_bias=False)
        )
        tensors = _paged_tensors()
        tensors["attn_bias"] = _Tensor((1, 1, 1, 1024), "torch.float32")
        with self.assertRaisesRegex(ValueError, "additive bias disagrees"):
            self.candidate.bound_torch(self.req, plain_spec, tensors)
        # The other direction: a bias kernel for a request without bias.
        plain_req = _pinned(self.candidate, use_attn_bias=False)
        with self.assertRaisesRegex(ValueError, "additive bias disagrees"):
            self.candidate.bound_torch(plain_req, self.spec, _paged_tensors())
        tensors["problem"] = replace(_tuning_problem(self.req), use_additive_bias=False)
        with self.assertRaisesRegex(ValueError, "use_attn_bias=True disagrees"):
            self.candidate.bound_torch(self.req, self.spec, tensors)


if __name__ == "__main__":
    unittest.main()
