# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Executable attention candidates own a complete launch contract."""

from __future__ import annotations

import unittest
from unittest import mock

from dispatch.attention import (
    ATTENTION_EXECUTION_REGISTRY,
    ATTENTION_ROUTE_REGISTRY,
    AttentionRequest,
    attention_execution_candidates,
)
from dispatch.attention import bindings as attention_bindings
from kernels.common.attention_dense_spec import AttentionDenseSpec
from dispatch.attention.common import AttentionTuningSpec


def _req(arch="gfx950", **kw) -> AttentionRequest:
    base = dict(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=1024,
        seqlen_k=1024,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        dtype="bf16",
        mask_type=1,
    )
    base.update(kw)
    return AttentionRequest(**base)


class TestRegistrySplit(unittest.TestCase):
    def test_routing_labels_are_not_on_the_execution_registry(self):
        exec_names = {c.name for c in attention_execution_candidates()}
        route_names = {c.name for c in ATTENTION_ROUTE_REGISTRY.candidates()}
        for name in (
            "attention_unified_2d",
            "attention_unified_3d",
            "attention_d256_decode",
            "attention_gfx942_dense_pipe",
            "attention_gfx950_d256",
        ):
            self.assertIn(name, route_names)
            self.assertNotIn(name, exec_names)

    def test_route_only_candidates_are_not_executable(self):
        for name in (
            "attention_unified_2d",
            "attention_unified_3d",
            "attention_d256_decode",
            "attention_gfx942_dense_pipe",
            "attention_gfx950_d256",
        ):
            with self.subTest(name=name):
                candidate = ATTENTION_ROUTE_REGISTRY.get(name)
                self.assertIsNone(candidate.build)
                self.assertIsNone(candidate.bind_torch)

    def test_every_execution_candidate_is_buildable_and_torch_bindable(self):
        self.assertTrue(ATTENTION_EXECUTION_REGISTRY.require_build)
        self.assertTrue(ATTENTION_EXECUTION_REGISTRY.require_torch_binding)
        for candidate in attention_execution_candidates():
            with self.subTest(candidate=candidate.name):
                self.assertIsNotNone(candidate.build)
                self.assertIsNotNone(candidate.bind_torch)

    def test_dense_select_spec_returns_the_concrete_dense_spec(self):
        candidate = ATTENTION_EXECUTION_REGISTRY.get("attention_gfx950_dense")
        req = _req(algorithm="attention_dense")
        spec = candidate.select_spec(req)
        self.assertIsInstance(spec, AttentionDenseSpec)
        self.assertNotEqual(candidate.grid(spec, req), (0, 0, 0))
        self.assertNotEqual(candidate.block(spec), (0, 0, 0))
        self.assertTrue(candidate.signature(spec))

    def test_tuning_select_spec_returns_an_attention_tuning_spec(self):
        candidate = next(
            c
            for c in attention_execution_candidates()
            if c.name.startswith("attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm")
        )
        req = _req(algorithm=candidate.algorithm, spec_id=candidate.spec_id)
        spec = candidate.select_spec(req)
        self.assertIsInstance(spec, AttentionTuningSpec)
        self.assertIn("@", spec.tuning_id)
        self.assertNotEqual(candidate.grid(spec, req), (0, 0, 0))
        built = candidate.built(spec, "gfx950")
        self.assertTrue(getattr(built, "name", None) or built)

    def test_gfx942_dense_bind_torch_omits_paged_kwargs(self):
        candidate = ATTENTION_EXECUTION_REGISTRY.get("attention_gfx942_dense")
        req = _req(arch="gfx942", dtype="fp16", algorithm="attention_dense")
        spec = candidate.select_spec(req)
        tensors = {"q": object(), "k": object(), "v": object(), "out": object()}
        captured = {}

        def fake_run(
            *,
            spec,
            q,
            k,
            v,
            out,
            scale,
            stream=0,
            arch="gfx942",
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
        ):
            captured.update(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                scale=scale,
                stream=stream,
                arch=arch,
            )
            return "ok"

        with mock.patch.object(
            attention_bindings,
            "_dense_runner",
            return_value=(fake_run, lambda _s: (1, 1, 1), lambda _s: (64, 1, 1)),
        ):
            binding = attention_bindings.bind_dense_attention_torch(req, spec, tensors)
            binding.launch()
        self.assertEqual(captured["q"], tensors["q"])
        self.assertEqual(captured["arch"], "gfx942")
        self.assertNotIn("block_tables", captured)
        self.assertNotIn("kv_lens", captured)
        self.assertNotIn("sinks", captured)


if __name__ == "__main__":
    unittest.main()
