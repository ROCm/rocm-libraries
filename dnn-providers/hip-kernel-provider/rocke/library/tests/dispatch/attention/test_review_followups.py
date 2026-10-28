# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU checks for the review fixes that the existing suites did not pin."""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from benchmarks.common.attention_flops import attended_pairs
from dispatch.attention import (
    ATTENTION_EXECUTION_REGISTRY,
    ATTENTION_ROUTE_REGISTRY,
    AttentionRequest,
    attention_dispatch_result,
    attention_tuning_spec,
)
from dispatch.attention.bindings import bind_dense_attention_torch
from dispatch.attention.common import AttentionTuningSpec, _problem
from dispatch.attention.unified_rules import (
    AttentionGeometryVariant,
    _explicit_configs,
)
from kernels.common import attention_unified as au


def _req(**kw) -> AttentionRequest:
    base = dict(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=128,
        seqlen_k=128,
        hdim_q=128,
        hdim_v=128,
        arch="gfx942",
        dtype="bf16",
        kv_block_size=16,
        mask_type=1,
    )
    base.update(kw)
    return AttentionRequest(**base)


_TUNING_FIELDS = dict(
    path="2d",
    arch="gfx950",
    builder_kind="tiled",
    compile_backend="llvm",
    candidate_name="attention_gfx950_u2d_test",
    tuning_id="test@0",
    kernel_spec=object(),
)


class TestReviewFollowups(unittest.TestCase):
    def test_noncausal_sliding_window_flops_match_the_reference_mask(self):
        # sq=4, sk=8, window=3. Non-causal window is right-aligned:
        # first = max(0, qi + (sk-sq) - window + 1), last = sk-1.
        # Pairs: 3+4+5+6 = 18. Causal last is qi+(sk-sq): 3+3+3+3 = 12.
        self.assertEqual(attended_pairs(4, 8, causal=False, sliding_window=3), 18)
        self.assertEqual(attended_pairs(4, 8, causal=True, sliding_window=3), 12)

    def test_gfx950_3d_graph_replay_is_opt_in(self):
        problem = _problem(_req(arch="gfx950", seqlen_q=1, seqlen_k=4096))
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("HIPDNN_GFX950_3D_GRAPH", None)
            self.assertFalse(au._enable_3d_graph_replay(problem, "gfx950"))
            os.environ["HIPDNN_GFX950_3D_GRAPH"] = "1"
            self.assertTrue(au._enable_3d_graph_replay(problem, "gfx950"))
            os.environ["HIPDNN_GFX950_3D_GRAPH"] = "off"
            self.assertFalse(au._enable_3d_graph_replay(problem, "gfx950"))

    def test_capture_fence_is_only_used_while_capturing(self):
        calls = []

        def pipeline(_vals, _cfgs, *, stream):
            calls.append(("pipe", stream))

        prepared = SimpleNamespace(
            pipeline=pipeline, seg_config=1, red_config=SimpleNamespace(fence=False)
        )

        class _Fence:
            def __enter__(self):
                calls.append("fence")
                return self

            def __exit__(self, *_args):
                return False

        with mock.patch.object(au, "no_fence", _Fence):
            au._launch_3d_pipeline(prepared, (), (), 7, capturing=False)
            self.assertEqual(calls, [("pipe", 7)])
            calls.clear()
            au._launch_3d_pipeline(prepared, (), (), 7, capturing=True)
        self.assertEqual(calls, ["fence", ("pipe", 7)])

    def test_4warp_grid_matches_the_launch_helper(self):
        fold = _problem(_req(sliding_window=16, dtype="bf16"))
        self.assertEqual(au.gfx942_4warp_launch_grid(fold), (8, 5, 1))
        plain = _problem(_req(dtype="fp16", sliding_window=0))
        self.assertEqual(au.gfx942_4warp_launch_grid(plain), (32, 2, 1))

        candidate = next(
            c for c in ATTENTION_EXECUTION_REGISTRY.candidates() if "4warp" in c.name
        )
        pinned = replace(
            _req(sliding_window=16),
            algorithm=candidate.algorithm,
            spec_id=candidate.spec_id,
        )
        spec = candidate.select_spec(pinned)
        self.assertEqual(
            candidate.grid(spec, pinned),
            au.gfx942_4warp_launch_grid(_problem(pinned)),
        )

    def test_pinned_gfx950_dense_spec_is_its_candidate_default(self):
        req = _req(
            arch="gfx950",
            nhead_q=16,
            nhead_k=4,
            seqlen_q=2048,
            seqlen_k=2048,
            hdim_q=64,
            hdim_v=64,
        )
        spec = attention_tuning_spec(req, "gfx950_dense_grid").kernel_spec
        self.assertEqual((spec.block_m, spec.block_n), (256, 64))
        self.assertFalse(spec.persistent)

    def test_gfx942_dense_binding_rejects_paged_block_tables(self):
        req = _req(
            nhead_q=16,
            nhead_k=4,
            seqlen_q=2048,
            seqlen_k=2048,
            hdim_q=64,
            hdim_v=64,
        )
        spec = attention_tuning_spec(req, "gfx942_dense")
        binding = bind_dense_attention_torch(
            req, spec, {"q": None, "k": None, "v": None, "out": None}
        )
        with self.assertRaisesRegex(NotImplementedError, "block_tables"):
            binding.launch(block_tables=object())

    def test_opt_in_candidates_stay_out_of_auto_and_remain_pinnable(self):
        auto = _req(arch="gfx950", algorithm="auto", spec_id="auto")
        for registry in (ATTENTION_ROUTE_REGISTRY, ATTENTION_EXECUTION_REGISTRY):
            for candidate in registry.supported(auto):
                self.assertFalse(candidate.opt_in)
        tuning = next(
            c
            for c in ATTENTION_EXECUTION_REGISTRY.candidates()
            if c.name.startswith("attention_gfx950_u2d_narrow")
        )
        self.assertTrue(tuning.opt_in)
        pinned = replace(auto, algorithm=tuning.algorithm, spec_id=tuning.spec_id)
        visible = {c.name for c in ATTENTION_EXECUTION_REGISTRY.supported(pinned)}
        self.assertIn(tuning.name, visible)

    def test_dispatch_result_stores_the_opt_in_probe(self):
        auto = _req(arch="gfx950", algorithm="auto", spec_id="auto")
        candidate = next(
            c
            for c in ATTENTION_EXECUTION_REGISTRY.candidates()
            if c.name.startswith("attention_gfx950_u2d_narrow")
        )
        pinned = replace(auto, algorithm=candidate.algorithm, spec_id=candidate.spec_id)
        spec = candidate.select_spec(pinned)
        result = attention_dispatch_result(auto, candidate, spec)
        self.assertEqual(result.request.algorithm, candidate.algorithm)
        self.assertEqual(result.request.spec_id, candidate.spec_id)
        self.assertEqual(auto.algorithm, "auto")

    def test_unknown_arch_tuning_profiles_raise(self):
        problem = _problem(_req(arch="gfx950"))
        variant = AttentionGeometryVariant(
            arch="gfx1250",
            path="2d",
            codepath="narrow",
            builder_kind="tiled",
            tile_policy="4x",
        )
        with self.assertRaisesRegex(ValueError, "no explicit 2D"):
            list(_explicit_configs(problem, variant))
        with self.assertRaisesRegex(ValueError, "no explicit 3D"):
            list(_explicit_configs(problem, replace(variant, path="3d")))

    def test_explicit_path_support_is_not_skipped(self):
        problem = _problem(_req(hdim_q=7, hdim_v=7, arch="gfx950"))
        ok, _why = au._explicit_path_supported(problem, None, "2d", "gfx950")
        self.assertFalse(ok)
        ok, _why = au._explicit_path_supported(
            problem, AttentionTuningSpec(**_TUNING_FIELDS), "2d", "gfx950"
        )
        self.assertFalse(ok)
        ok, why = au._explicit_path_supported(
            problem,
            AttentionTuningSpec(**_TUNING_FIELDS, allow_unsupported=True),
            "2d",
            "gfx950",
        )
        self.assertTrue(ok)
        self.assertIn("unsupported override", why)

    def test_tuning_spec_backend_conflict_raises(self):
        spec = AttentionTuningSpec(**dict(_TUNING_FIELDS, path="3d"))
        old = au._RESOLVED_ATTENTION_ARCH
        try:
            au._RESOLVED_ATTENTION_ARCH = "gfx950"
            with self.assertRaisesRegex(ValueError, "conflicts"):
                au.run_unified_attention_torch(
                    problem=_problem(_req(arch="gfx950")),
                    q=None,
                    k=None,
                    v=None,
                    out=None,
                    cu_seqlens_q=None,
                    seqused_k=None,
                    softmax_scale=1.0,
                    block_table=None,
                    softcap=0.0,
                    backend="tiled",
                    tuning_spec=spec,
                )
        finally:
            au._RESOLVED_ATTENTION_ARCH = old


class TestFp8DenseGuards(unittest.TestCase):
    """CPU guards for the fp8 dense review fixes (scale forwarding + rejections)."""

    def _fp8_req(self, **kw):
        base = dict(
            arch="gfx950",
            nhead_q=16,
            nhead_k=4,
            seqlen_q=2048,
            seqlen_k=2048,
            hdim_q=64,
            hdim_v=64,
            use_fp8=True,
        )
        base.update(kw)
        return _req(**base)

    def _fp8_grid_spec(self):
        req = self._fp8_req()
        return req, attention_tuning_spec(req, "gfx950_dense_grid")

    def _capture_runner_call(self, *, bind_kwargs, launch_kwargs):
        """Bind an fp8 grid spec and launch, returning the dict the runner saw."""
        req, spec = self._fp8_grid_spec()
        captured = {}
        fp8 = SimpleNamespace(dtype="torch.float8_e4m3fn")
        with mock.patch(
            "dispatch.attention.bindings._dense_runner",
            return_value=lambda **call: captured.update(call),
        ):
            binding = bind_dense_attention_torch(
                req, spec, {"q": None, "k": fp8, "v": fp8, "out": None}, **bind_kwargs
            )
            binding.launch(**launch_kwargs)
        return captured

    def test_binding_launch_scale_overrides_bind_scale(self):
        # A launch kwarg may still override the bind-time scale (e.g. a sweep
        # varying scale per launch); the bind scale is the graph-capture value.
        captured = self._capture_runner_call(
            bind_kwargs={"k_scale": 1.0, "v_scale": 1.0},
            launch_kwargs={"k_scale": 0.5, "v_scale": 0.25},
        )
        self.assertEqual(captured.get("k_scale"), 0.5)
        self.assertEqual(captured.get("v_scale"), 0.25)

    def test_binding_rejects_fp8_spec_with_no_scale(self):
        # Regression: the dense binding dropped k_scale/v_scale, so every fp8
        # launch dequantized at unit scale (silently wrong). An fp8 spec with no
        # scale supplied anywhere must now raise rather than default to 1.0 -- a
        # caller wanting unit scale passes k_scale=1.0 explicitly (the combo sweep).
        with self.assertRaisesRegex(ValueError, "scale"):
            self._capture_runner_call(bind_kwargs={}, launch_kwargs={})

    def test_binding_forwards_kv_scales_at_bind_time(self):
        # The solera bench binds once (passing the scales) then launches with no
        # kwargs for graph capture, so the scales must be captured at bind time.
        captured = self._capture_runner_call(
            bind_kwargs={"k_scale": 0.5, "v_scale": 0.25}, launch_kwargs={}
        )
        self.assertEqual(captured.get("k_scale"), 0.5)
        self.assertEqual(captured.get("v_scale"), 0.25)

    def test_binding_rejects_bf16_kv_for_an_fp8_spec(self):
        req, spec = self._fp8_grid_spec()
        bf16 = SimpleNamespace(dtype="torch.bfloat16")
        with mock.patch(
            "dispatch.attention.bindings._dense_runner",
            return_value=lambda **_call: None,
        ):
            with self.assertRaisesRegex(ValueError, "fp8"):
                bind_dense_attention_torch(
                    req, spec, {"q": None, "k": bf16, "v": bf16, "out": None}
                )

    def test_supports_accepts_fp8_bf16_and_fp16_but_rejects_sinks(self):
        from kernels.gfx950.attention_dense import supports_attention_dense

        ks = self._fp8_grid_spec()[1].kernel_spec
        # bf16 and fp16 compute/output are both valid with fp8 KV.
        self.assertTrue(supports_attention_dense(ks, arch="gfx950")[0])
        self.assertTrue(
            supports_attention_dense(replace(ks, dtype="fp16"), arch="gfx950")[0]
        )
        # Sinks on dense are a separate ticket; the fp8 loader has no sink path.
        self.assertFalse(
            supports_attention_dense(replace(ks, use_sinks=True), arch="gfx950")[0]
        )

    def test_gfx942_dense_rejects_fp8_storage(self):
        from kernels.gfx942.attention_dense import supports_attention_dense

        ks = self._fp8_grid_spec()[1].kernel_spec
        ok, reason = supports_attention_dense(ks, arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("fp8", reason.lower())

    def test_fnuz_fp8_request_is_rejected_on_gfx950(self):
        with self.assertRaisesRegex(ValueError, "OCP|FNUZ"):
            attention_tuning_spec(self._fp8_req(fp8_fnuz=True), "gfx950_dense_grid")

    def test_runner_rejects_non_ocp_kv_for_an_fp8_spec(self):
        # The binding guard only covers dispatch callers; a direct call to the
        # runner must reject a bf16 or FNUZ K/V for an fp8 spec too, before any
        # launch -- otherwise the OCP-declared pointers read the wrong encoding.
        # The runner compares the dtype by name, so fakes keep this on the CPU lane.
        from kernels.gfx950.attention_dense import run_attention_dense_torch

        spec = self._fp8_grid_spec()[1].kernel_spec
        for wrong in ("torch.bfloat16", "torch.float8_e4m3fnuz"):
            kv = SimpleNamespace(dtype=wrong)
            with self.subTest(dtype=wrong):
                with self.assertRaisesRegex(ValueError, r"fp8 dense expects OCP fp8 k"):
                    run_attention_dense_torch(
                        spec=spec,
                        q=kv,
                        k=kv,
                        v=kv,
                        out=kv,
                        scale=0.125,
                        k_scale=0.5,
                        v_scale=0.5,
                    )

    def test_runner_rejects_fp8_spec_with_no_scale(self):
        # The runner defaults k_scale/v_scale to None, not 1.0: a direct caller
        # (bypassing the binding) that omits the fp8 dequant scale must raise, not
        # silently dequantize at unit scale.
        from kernels.gfx950.attention_dense import run_attention_dense_torch

        spec = self._fp8_grid_spec()[1].kernel_spec
        fp8 = SimpleNamespace(dtype="torch.float8_e4m3fn")
        with self.assertRaisesRegex(ValueError, r"requires k_scale"):
            run_attention_dense_torch(
                spec=spec, q=fp8, k=fp8, v=fp8, out=fp8, scale=0.125
            )
        with self.assertRaisesRegex(ValueError, r"requires v_scale"):
            run_attention_dense_torch(
                spec=spec, q=fp8, k=fp8, v=fp8, out=fp8, scale=0.125, k_scale=0.5
            )

    def test_spec_rejects_unknown_kv_storage_dtype(self):
        ks = self._fp8_grid_spec()[1].kernel_spec
        with self.assertRaisesRegex(ValueError, r"kv_storage_dtype must be .*fp8e5m2"):
            replace(ks, kv_storage_dtype="fp8e5m2")

    def test_spec_rejects_fp8_with_unsupported_layouts(self):
        # fp8 KV lands on the contiguous default-grid body only.
        ks = self._fp8_grid_spec()[1].kernel_spec
        for field, overrides in (
            ("persistent", {"persistent": True}),
            ("ragged", {"ragged": True}),
            ("varlen", {"varlen": True}),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ValueError, rf"fp8 KV is not yet supported with {field}"
                ):
                    replace(ks, **overrides)

    def test_spec_rejects_fp8_with_paged_kv(self):
        # Paged needs a geometry the bf16 body accepts (D128 sliding window), so
        # the only thing that can reject it is the fp8 guard.
        req = self._fp8_req(hdim_q=128, hdim_v=128)
        ks = attention_tuning_spec(req, "gfx950_dense_grid").kernel_spec
        paged = dict(paged=True, block_size=64, num_kv_blocks=32, sliding_window=128)
        replace(ks, kv_storage_dtype=None, **paged)  # bf16: valid
        with self.assertRaisesRegex(
            ValueError, r"fp8 KV is not yet supported with paged"
        ):
            replace(ks, **paged)

    def test_supports_rejects_fp8_tile_not_divisible_by_loader_width(self):
        # The fp8 loaders move 8 elements per thread, so block_n * head_size must
        # divide by num_waves * 64 * 8. bm256/bn32/D64 does not (2048 vs 4096).
        from kernels.gfx950.attention_dense import supports_attention_dense

        ks = self._fp8_grid_spec()[1].kernel_spec
        self.assertTrue(supports_attention_dense(ks, arch="gfx950")[0])
        ok, reason = supports_attention_dense(replace(ks, block_n=32), arch="gfx950")
        self.assertFalse(ok)
        self.assertIn("block_n*head_size", reason)
        self.assertIn("block_n=32", reason)

    def _bind_unified_fp8(self, **bind_kwargs):
        """Bind the unified tuning path for an fp8 problem with the paged-tensor
        validators stubbed out, so only the scale handling is exercised."""
        from dispatch.attention.bindings import bind_tuning_attention_torch

        req = self._fp8_req()
        problem = _problem(req)
        self.assertTrue(problem.use_fp8)
        spec = SimpleNamespace(
            path="2d", launch_grid=lambda _p: (1, 1, 1), launch_block=lambda: (64, 1, 1)
        )
        tensors = dict.fromkeys(
            ("q", "k", "v", "out", "cu_seqlens_q", "seqused_k", "block_table")
        )
        tensors["problem"] = problem
        with mock.patch(
            "dispatch.attention.bindings.validate_tuning_attention_contract"
        ), mock.patch(
            "dispatch.attention.bindings.validate_tuning_attention_tensors",
            return_value=1,
        ):
            return bind_tuning_attention_torch(req, spec, tensors, **bind_kwargs)

    def test_unified_binding_rejects_fp8_problem_with_no_scale(self):
        with self.assertRaisesRegex(ValueError, r"fp8 attention requires 'k_scale'"):
            self._bind_unified_fp8()
        with self.assertRaisesRegex(ValueError, r"fp8 attention requires 'v_scale'"):
            self._bind_unified_fp8(k_scale=0.5)

    def test_unified_binding_forwards_bind_time_scales(self):
        # Regression for the unified path: k_scale/v_scale given at bind time must
        # reach run_unified_attention_torch on a launch with no kwargs.
        captured = {}
        # Patched before binding: the binding imports the runner when it is built.
        with mock.patch(
            "kernels.common.attention_unified.run_unified_attention_torch",
            side_effect=lambda **call: captured.update(call),
        ):
            self._bind_unified_fp8(k_scale=0.5, v_scale=0.25).launch()
        self.assertEqual(captured.get("k_scale"), 0.5)
        self.assertEqual(captured.get("v_scale"), 0.25)
