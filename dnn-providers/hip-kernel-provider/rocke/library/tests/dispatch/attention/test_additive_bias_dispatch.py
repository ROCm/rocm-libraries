# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A gfx942/gfx950 ``AttentionRequest(use_attn_bias=True)`` reaches the unified
kernels.

CPU-only: routing, tuning candidates and the explicit binding (with a stub
runner). The numeric check is ``test_attention_additive_bias_numeric.py``.
"""

from __future__ import annotations

import unittest
from collections import Counter
from dataclasses import replace
from unittest import mock

from dispatch.attention import (
    AttentionRequest,
    attention_execution_candidates,
    dispatch_attention,
)
from dispatch.attention.common import (
    AttentionSpec,
    attention_problem,
    _tuning_problem,
    _unified_path_coverage,
)
from dispatch.attention.tuning_specs import (
    ExplicitAttention2DConfig,
    ExplicitAttention3DConfig,
    _bias_fields,
    make_explicit_attention_2d_spec,
    make_explicit_attention_3d_specs,
)

from ._fake_paged_tensors import FakeTensor as _Tensor
from ._fake_paged_tensors import paged_tensors as _paged_tensors

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


class TestRouting(unittest.TestCase):
    def test_request_maps_to_problem(self):
        self.assertTrue(attention_problem(_req()).use_additive_bias)
        self.assertTrue(_tuning_problem(_req()).use_additive_bias)
        self.assertFalse(attention_problem(_req(use_attn_bias=False)).use_additive_bias)

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

    def test_request_flag_must_be_a_bool(self):
        # A truthy non-bool must not select a bias kernel, and a falsy one must
        # not silently select a bias-free kernel.
        for bad in ("False", "false", 0, 1, None):
            with self.subTest(value=bad), self.assertRaisesRegex(
                ValueError, "use_attn_bias must be a bool"
            ):
                dispatch_attention(_req(use_attn_bias=bad))

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
        # The dispatcher error lists every candidate's reason; pin the unified
        # 2D candidate's own reason so a refusal for an unrelated cause fails.
        cases = {
            "q_dtype_bias": (
                _req(attn_bias_dtype="q"),
                r"attention_unified_2d: [^;]*f32 additive bias only",
            ),
            "gfx1250": (
                _req(arch="gfx1250"),
                (
                    r"attention_unified_2d: [^;]*additive bias is only implemented "
                    r"on gfx942 and gfx950"
                ),
            ),
            # gfx950 decodes fp8 K/V as OCP; declaring fnuz-quantised K/V here
            # is a format mismatch that must be refused with or without bias
            # (the 2D tiled gate used to skip this check for bias requests).
            "fp8_fnuz_mismatch": (
                _req(use_fp8=True, fp8_fnuz=True),
                r"attention_unified_2d: fp8 K/V on gfx950 [^;]*fnuz-quantised",
            ),
        }
        for label, (request, reason) in cases.items():
            with self.subTest(case=label), self.assertRaisesRegex(ValueError, reason):
                dispatch_attention(request)

    def test_scalar_gate_refusals_apply_to_bias_problems(self):
        # A q_dtype override without fp8 is refused only by the scalar gate.
        # AttentionRequest has no q_dtype, so set it on the problem directly.
        for path, kw in (("2d", {}), ("3d", {"seqlen_q": 1, "seqlen_k": 8192})):
            for bias in (False, True):
                with self.subTest(path=path, bias=bias):
                    req = _req(use_attn_bias=bias, **kw)
                    ok, why = _unified_path_coverage(
                        req, attention_problem(req), path, "gfx950"
                    )
                    self.assertTrue(ok, why)
                    problem = replace(attention_problem(req), q_dtype="fp16")
                    ok, why = _unified_path_coverage(req, problem, path, "gfx950")
                    self.assertFalse(ok)
                    self.assertIn("q_dtype override requires use_fp8", why)

    def test_strided_decode_refuses_bias(self):
        # mask_type=1 (the _req default, top-left causal) is refused by the
        # strided-decode candidate regardless of bias, so it can't tell us
        # anything about bias specifically: use a mask the candidate admits
        # (mask_type=2, bottom-right causal) and a valid strided-KV shape.
        strided_kw = dict(
            mask_type=2, kv_layout="strided", kv_block_size=64, seqlen_q=1, seqlen_k=65
        )
        plain = dispatch_attention(_req(use_attn_bias=False, **strided_kw))
        self.assertEqual(plain.candidate.name, "attention_strided_decode")
        with self.assertRaisesRegex(
            ValueError,
            r"attention_strided_decode: capability: cannot serve features \['attn_bias'\]",
        ):
            dispatch_attention(_req(**strided_kw))


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
                arch = candidate.capability.arches[0]
                ok, why = candidate.admits(_pinned(candidate, arch=arch))
                self.assertFalse(ok)
                self.assertIn("attn_bias", why)
                # Bias is the only reason: the same request without it is admitted.
                ok, why = candidate.admits(
                    _pinned(candidate, arch=arch, use_attn_bias=False)
                )
                self.assertTrue(ok, why)

    def test_explicit_spec_on_gfx942_and_gfx950_only(self):
        problem = replace(attention_problem(_req()), use_additive_bias=True)
        config = ExplicitAttention2DConfig(
            num_warps=1, block_m_per_warp=16, tile_policy="1x"
        )
        for arch in ("gfx942", "gfx950"):
            with self.subTest(arch=arch):
                spec = make_explicit_attention_2d_spec(problem, config, arch=arch)
                self.assertTrue(spec.use_additive_bias)
        with self.assertRaisesRegex(ValueError, "gfx942/gfx950-only"):
            _bias_fields(problem, "gfx1201", {})

    def test_explicit_gfx942_4warp_spec_refuses_bias(self):
        # An explicit 4-warp GQA spec with bias is rejected up front, not by
        # build_gfx942_4warp_gqa at build time. The same config without bias is
        # accepted.
        problem = attention_problem(
            _req(arch="gfx942", hdim_q=256, hdim_v=256, nhead_k=2)
        )
        config = ExplicitAttention2DConfig(
            num_warps=1,
            block_m_per_warp=32,
            tile_policy="64",
            builder_kind="gfx942_4warp_gqa",
        )
        plain = replace(problem, use_additive_bias=False)
        make_explicit_attention_2d_spec(plain, config, arch="gfx942")
        with self.assertRaisesRegex(ValueError, "4-warp GQA or 32x32 atom paths"):
            make_explicit_attention_2d_spec(problem, config, arch="gfx942")

    def test_spec_build_applies_the_shared_bias_geometry_rule(self):
        from dispatch.attention.axes import _CODEPATH_KNOBS

        problem = replace(attention_problem(_req()), use_additive_bias=True)
        wide32 = tuple(_CODEPATH_KNOBS[("gfx950", "wide32")].items())
        config = ExplicitAttention2DConfig(
            num_warps=2, block_m_per_warp=32, tile_policy="2x", knobs=wide32
        )
        # Dispatch's own refusal text, not the kernel validator's, so the
        # shared rule must be what raises.
        with self.assertRaisesRegex(
            ValueError,
            "additive bias is not emitted on the 4-warp GQA or 32x32 atom paths",
        ):
            make_explicit_attention_2d_spec(problem, config, arch="gfx950")

    def test_declared_bias_support_matches_the_kernel(self):
        # candidate._variant_features declares which geometries take a bias as a
        # fact about the kernel, but the kernel is what actually decides that
        # (the spec rejects bias on the 32x32 atom paths). test_attention.py's
        # frozen-feature test re-derives the same rule from the candidate name,
        # so it agrees with the code by construction. Build each variant's own
        # explicit spec directly (bypassing capability/admits) and check it
        # against the kernel builder's own verdict, not the name.
        from dispatch.attention.axes import _CODEPATH_KNOBS
        from dispatch.attention.gfx942_unified import GFX942_TUNING_VARIANTS
        from dispatch.attention.gfx950_unified import GFX950_TUNING_VARIANTS

        def build(problem, variant):
            # The codepath (narrow/wide32/transposed32) is fixed by these
            # knobs, not by num_warps/block_m_per_warp alone -- the same
            # (num_warps, block_m_per_warp) pair is shared by more than one
            # codepath, so without them this builds the wrong kernel.
            fixed_knobs = tuple(
                _CODEPATH_KNOBS.get((variant.arch, variant.codepath), {}).items()
            )
            if variant.path == "3d":
                config = ExplicitAttention3DConfig(
                    num_segments=variant.num_segments,
                    tile_policy=variant.tile_policy,
                    knobs=fixed_knobs,
                )
                make_explicit_attention_3d_specs(problem, config, arch=variant.arch)
            else:
                config = ExplicitAttention2DConfig(
                    num_warps=variant.num_warps,
                    block_m_per_warp=variant.block_m_per_warp,
                    tile_policy=variant.tile_policy,
                    compile_backend=variant.compile_backend,
                    builder_kind=variant.builder_kind,
                    knobs=fixed_knobs,
                )
                make_explicit_attention_2d_spec(problem, config, arch=variant.arch)

        bias_problem = replace(attention_problem(_req()), use_additive_bias=True)
        plain_problem = replace(attention_problem(_req()), use_additive_bias=False)
        for arch, variants in (
            ("gfx942", GFX942_TUNING_VARIANTS),
            ("gfx950", GFX950_TUNING_VARIANTS),
        ):
            self.assertTrue(variants)
            compared = Counter()
            for variant in variants:
                candidate = _candidate(variant.candidate_name)
                declared = "attn_bias" in candidate.capability.supports_features
                with self.subTest(candidate=candidate.name):
                    try:
                        build(plain_problem, variant)
                    except ValueError:
                        # This default shape/geometry combination isn't buildable
                        # at all (e.g. an LDS/tile-size budget miss unrelated to
                        # bias); it can't tell us anything about bias support.
                        continue
                    try:
                        build(bias_problem, variant)
                        kernel_accepts = True
                    except ValueError:
                        kernel_accepts = False
                    self.assertEqual(
                        kernel_accepts,
                        declared,
                        f"declared attn_bias={declared} but the kernel "
                        f"{'accepted' if kernel_accepts else 'refused'} a bias spec",
                    )
                    compared[variant.codepath] += 1
            # The skip above must not empty a code path: each one, including
            # the ones that refuse bias, has to be compared at least once.
            paths = {
                "gfx942": ("narrow", "wide32x8", "transposed_x8", "splitkv"),
                "gfx950": ("narrow", "transposed32", "wide32", "splitkv"),
            }[arch]
            for codepath in paths:
                with self.subTest(arch=arch, codepath=codepath):
                    self.assertGreater(compared[codepath], 0)


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
            "dtype": (_Tensor((1, 1, 1, 1024), "torch.bfloat16"), "float32"),
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

    def test_unit_key_stride_is_required_even_for_one_key(self):
        # The unified runner refuses any key stride but 1, so the binding must
        # too, including Sk == 1 where the stride of a size-1 dim is arbitrary.
        from dispatch.attention.bindings import _tuning_additive_bias

        req = _req(seqlen_q=1, seqlen_k=1, batch=2)
        problem = replace(attention_problem(req), use_additive_bias=True)
        broadcast = _Tensor((2, 32, 1, 1), "torch.float32", strides=(0, 0, 0, 0))
        with self.assertRaisesRegex(ValueError, r"contiguous along Sk"):
            _tuning_additive_bias(req, problem, {"attn_bias": broadcast})
        unit = _Tensor((2, 32, 1, 1), "torch.float32", strides=(32, 1, 1, 1))
        kwargs = _tuning_additive_bias(req, problem, {"attn_bias": unit})
        self.assertEqual(kwargs["additive_bias_head_stride"], 1)

    def test_binding_refuses_what_the_runner_refuses(self):
        # The binding runs the runner's own bias validation on the derived
        # strides, so each of these fails at bind time, not at the launch.
        too_far = 70_000_000  # 31 heads * 70M elements is past the i32 index
        cases = {
            "i32_index": (
                _Tensor((1, 32, 1, 1024), "torch.float32", strides=(0, too_far, 0, 1)),
                NotImplementedError,
                "i32 bias addressing",
            ),
            "storage": (
                _Tensor((1, 32, 1, 1024), "torch.float32", storage_elems=1024),
                ValueError,
                "only backs 1024 elements",
            ),
            "device": (
                _Tensor((1, 1, 1, 1024), "torch.float32", device="cuda:1"),
                ValueError,
                "same device as q",
            ),
        }
        for label, (bias, error, message) in cases.items():
            tensors = _paged_tensors()
            tensors["attn_bias"] = bias
            with self.subTest(case=label), self.assertRaisesRegex(
                error, r"tensors\['attn_bias'\].*" + message
            ):
                self.candidate.bound_torch(self.req, self.spec, tensors)
        self.assertEqual(self.captured, {})

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
        with self.assertRaisesRegex(
            ValueError,
            r"kernel_spec\.use_additive_bias=False disagrees with "
            r"problem\.use_additive_bias=True",
        ):
            self.candidate.bound_torch(self.req, plain_spec, tensors)
        # The other direction: a bias kernel for a request without bias.
        plain_req = _pinned(self.candidate, use_attn_bias=False)
        with self.assertRaisesRegex(
            ValueError,
            r"kernel_spec\.use_additive_bias=True disagrees with "
            r"problem\.use_additive_bias=False",
        ):
            self.candidate.bound_torch(plain_req, self.spec, _paged_tensors())
        tensors["problem"] = replace(_tuning_problem(self.req), use_additive_bias=False)
        with self.assertRaisesRegex(ValueError, "use_attn_bias=True disagrees"):
            self.candidate.bound_torch(self.req, self.spec, tensors)


if __name__ == "__main__":
    unittest.main()
