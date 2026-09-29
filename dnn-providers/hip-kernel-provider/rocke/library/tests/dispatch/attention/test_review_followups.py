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
)
from dispatch.attention.bindings import bind_dense_attention_torch
from dispatch.attention.common import AttentionTuningSpec, _problem
from dispatch.attention.gfx942 import _dense_spec as _dense_spec_gfx942
from dispatch.attention.gfx950 import _dense_spec as _dense_spec_gfx950
from dispatch.attention.tuning_common import (
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
        old = au._RESOLVED_ATTENTION_ARCH
        try:
            au._RESOLVED_ATTENTION_ARCH = "gfx950"
            with mock.patch.dict("os.environ", {}, clear=False):
                import os

                os.environ.pop("HIPDNN_GFX950_3D_GRAPH", None)
                self.assertFalse(au._enable_3d_graph_replay(problem))
                os.environ["HIPDNN_GFX950_3D_GRAPH"] = "1"
                self.assertTrue(au._enable_3d_graph_replay(problem))
                os.environ["HIPDNN_GFX950_3D_GRAPH"] = "off"
                self.assertFalse(au._enable_3d_graph_replay(problem))
        finally:
            au._RESOLVED_ATTENTION_ARCH = old

    def test_capture_fence_is_only_used_while_capturing(self):
        calls = []

        def pipeline(_vals, _cfgs, *, stream):
            calls.append(("pipe", stream))

        prepared = SimpleNamespace(pipeline=pipeline, seg_config=1, red_config=2)

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

    def test_one_arg_gfx950_dense_spec_selects_a_variant(self):
        req = _req(
            arch="gfx950",
            nhead_q=16,
            nhead_k=4,
            seqlen_q=2048,
            seqlen_k=2048,
            hdim_q=64,
            hdim_v=64,
            algorithm="attention_dense",
            dense_persistent="off",
        )
        spec = _dense_spec_gfx950(req)
        self.assertGreater(spec.block_m, 0)

    def test_gfx942_dense_binding_rejects_paged_block_tables(self):
        req = _req(
            nhead_q=16,
            nhead_k=4,
            seqlen_q=2048,
            seqlen_k=2048,
            hdim_q=64,
            hdim_v=64,
            algorithm="attention_dense",
            dense_persistent="off",
        )
        spec = _dense_spec_gfx942(req)
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
        old = au._RESOLVED_ATTENTION_ARCH
        try:
            au._RESOLVED_ATTENTION_ARCH = "gfx950"
            ok, _why = au._explicit_path_supported(problem, None, "2d")
            self.assertFalse(ok)
            ok, _why = au._explicit_path_supported(
                problem, AttentionTuningSpec(**_TUNING_FIELDS), "2d"
            )
            self.assertFalse(ok)
            ok, why = au._explicit_path_supported(
                problem,
                AttentionTuningSpec(**_TUNING_FIELDS, allow_unsupported=True),
                "2d",
            )
            self.assertTrue(ok)
            self.assertIn("unsupported override", why)
        finally:
            au._RESOLVED_ATTENTION_ARCH = old

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
