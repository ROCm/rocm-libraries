# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Contracts for explicit gfx942/gfx950 unified-attention tuning."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

import kernels.common.attention_unified as au
from dispatch.attention.tuning_specs import (
    ExplicitAttention2DConfig,
    ExplicitAttention3DConfig,
    make_explicit_attention_2d_spec,
    make_explicit_attention_3d_specs,
)
from dispatch.attention import (
    AttentionRequest,
    attention_candidates,
    dispatch_attention,
)
from rocke import lower_kernel_to_llvm


def _problem(**kw):
    base = dict(
        total_q=1024,
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=1024,
        max_seqlen_k=1024,
        dtype="bf16",
    )
    base.update(kw)
    return au.UnifiedAttentionProblem(**base)


def _request(arch="gfx950", **kw):
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
    )
    base.update(kw)
    return AttentionRequest(**base)


class TestExplicitAttentionBuilders(unittest.TestCase):
    def test_2d_builder_does_not_call_policy_helpers(self):
        with (
            mock.patch.object(
                au, "_select_2d_tile_size", side_effect=AssertionError("policy")
            ),
            mock.patch.object(
                au, "_select_2d_num_warps", side_effect=AssertionError("policy")
            ),
            mock.patch.object(
                au, "_enable_combo_2d", side_effect=AssertionError("policy")
            ),
        ):
            spec = make_explicit_attention_2d_spec(
                _problem(),
                ExplicitAttention2DConfig(
                    num_warps=2,
                    block_m_per_warp=16,
                    tile_policy="4x",
                ),
                arch="gfx950",
            )
        self.assertEqual(spec.num_warps, 2)
        self.assertEqual(spec.tile_size_eff, 64)

    def test_3d_builder_does_not_call_segment_policy(self):
        with mock.patch.object(
            au, "_num_segments", side_effect=AssertionError("policy")
        ):
            segment, reduce = make_explicit_attention_3d_specs(
                _problem(max_seqlen_q=1, total_q=1),
                ExplicitAttention3DConfig(num_segments=32),
                arch="gfx950",
            )
        self.assertEqual(segment.num_segments, 32)
        self.assertEqual(reduce.num_segments, 32)

    def test_invalid_geometry_is_rejected_not_fixed(self):
        with self.assertRaises(ValueError):
            make_explicit_attention_2d_spec(
                _problem(block_size=32),
                ExplicitAttention2DConfig(
                    num_warps=4,
                    block_m_per_warp=32,
                    tile_policy="1x",
                    knobs=(("use_k_single_buffer", True),),
                ),
                arch="gfx950",
            )

    def test_explicit_fp8_encoding_must_match_architecture(self):
        config = ExplicitAttention2DConfig(
            num_warps=2,
            block_m_per_warp=16,
            tile_policy="4x",
        )
        with self.assertRaisesRegex(ValueError, "requires FNUZ"):
            make_explicit_attention_2d_spec(
                _problem(use_fp8=True, fp8_fnuz=False),
                config,
                arch="gfx942",
            )
        with self.assertRaisesRegex(ValueError, "requires OCP"):
            make_explicit_attention_2d_spec(
                _problem(use_fp8=True, fp8_fnuz=True),
                config,
                arch="gfx950",
            )
        gfx942, _reduce = make_explicit_attention_3d_specs(
            _problem(
                total_q=1,
                max_seqlen_q=1,
                max_seqlen_k=4096,
                use_fp8=True,
                fp8_fnuz=True,
            ),
            ExplicitAttention3DConfig(num_segments=32),
            arch="gfx942",
        )
        gfx950 = make_explicit_attention_2d_spec(
            _problem(use_fp8=True, fp8_fnuz=False),
            config,
            arch="gfx950",
        )
        self.assertEqual(gfx942.kv_storage_dtype, "fp8e4m3")
        self.assertEqual(gfx950.kv_storage_dtype, "fp8e4m3")
        self.assertIn("fnuz", gfx942.kernel_name())
        self.assertNotIn("fnuz", gfx950.kernel_name())

    def test_representative_ir_builds_for_both_arches_and_paths(self):
        cases = (
            (
                "gfx950",
                "attention_gfx950_u2d_transposed32_nw2_mw32_t4xb_llvm",
                _request(),
            ),
            (
                "gfx950",
                "attention_gfx950_u3d_splitkv_seg64_t1xb",
                _request(seqlen_q=1, seqlen_k=4096),
            ),
            (
                "gfx942",
                "attention_gfx942_u2d_transposed_x8_nw2_mw32_t4xb_llvm",
                _request("gfx942", dtype="fp16"),
            ),
            (
                "gfx942",
                "attention_gfx942_u3d_splitkv_seg64_t1xb",
                _request("gfx942", seqlen_q=1, seqlen_k=4096),
            ),
        )
        candidates = attention_candidates()
        for arch, prefix, req in cases:
            with self.subTest(arch=arch, prefix=prefix):
                candidate = next(c for c in candidates if c.name.startswith(prefix))
                tuned_req = replace(
                    req, algorithm=candidate.algorithm, spec_id=candidate.spec_id
                )
                spec = candidate.select_spec(tuned_req)
                built = candidate.built(spec, arch)
                kernels = built if isinstance(built, tuple) else (built,)
                self.assertTrue(kernels)
                self.assertTrue(all(k.name for k in kernels))
                for kernel in kernels:
                    llvm = lower_kernel_to_llvm(kernel, arch=arch)
                    self.assertIn("define", llvm)
                    self.assertIn(kernel.name, llvm)


class TestAttentionTuningRegistry(unittest.TestCase):
    def test_tuning_candidates_are_arch_specific_and_opt_in(self):
        from dispatch.attention.gfx942_tuning import GFX942_TUNING_VARIANTS
        from dispatch.attention.gfx950_tuning import GFX950_TUNING_VARIANTS

        tuning = [c for c in attention_candidates() if c.algorithm == "unified_tuning"]
        self.assertEqual(
            len(tuning), len(GFX942_TUNING_VARIANTS) + len(GFX950_TUNING_VARIANTS)
        )
        for candidate in tuning:
            self.assertEqual(len(candidate.capability.arches), 1)
            arch = candidate.capability.arches[0]
            self.assertFalse(candidate.admits(_request(arch))[0])

    def test_auto_dispatch_is_unchanged(self):
        from dispatch.attention import (
            ATTENTION_EXECUTION_REGISTRY,
            ATTENTION_ROUTE_REGISTRY,
        )

        cases = (
            ("gfx950", 1, 4096, 128, "bf16", "attention_unified_3d"),
            ("gfx950", 1024, 1024, 128, "bf16", "attention_unified_2d"),
            ("gfx942", 1, 4096, 128, "bf16", "attention_unified_3d"),
            ("gfx942", 1024, 1024, 128, "fp16", "attention_gfx942_dense_pipe"),
        )
        old = au._RESOLVED_ATTENTION_ARCH
        try:
            for arch, sq, sk, hdim, dtype, expected in cases:
                with self.subTest(arch=arch, sq=sq, sk=sk, dtype=dtype):
                    au._RESOLVED_ATTENTION_ARCH = arch
                    req = _request(
                        arch,
                        seqlen_q=sq,
                        seqlen_k=sk,
                        hdim_q=hdim,
                        hdim_v=hdim,
                        dtype=dtype,
                    )
                    result = dispatch_attention(req)
                    self.assertEqual(result.candidate.name, expected)
                    self.assertFalse(result.candidate.opt_in)
                    for registry in (
                        ATTENTION_ROUTE_REGISTRY,
                        ATTENTION_EXECUTION_REGISTRY,
                    ):
                        for candidate in registry.supported(req):
                            self.assertFalse(candidate.opt_in)
                            self.assertNotIn("unified_tuning", candidate.name)
        finally:
            au._RESOLVED_ATTENTION_ARCH = old

    def test_explicit_spec_id_selects_only_that_geometry_candidate(self):
        req = replace(
            _request(seqlen_q=1, seqlen_k=4096),
            algorithm="unified_tuning",
            spec_id="gfx950_u3d_splitkv_seg64_t1xb",
        )
        result = dispatch_attention(req)
        self.assertEqual(
            result.candidate.name,
            "attention_gfx950_u3d_splitkv_seg64_t1xb",
        )
        self.assertEqual(result.spec.kernel_spec.num_segments, 64)

    def _specs_for(self, prefix, **req_kw):
        candidate = next(c for c in attention_candidates() if c.name.startswith(prefix))
        req = replace(
            _request(**req_kw),
            algorithm=candidate.algorithm,
            spec_id=candidate.spec_id,
        )
        return candidate.sweep_space(req)

    def test_narrow_codepath_registers_the_sched_barrier_lever(self):
        """The fence is emitted only in the narrow QK loop, so that is the
        codepath that has to offer it -- with its mask, which changes codegen."""
        specs = self._specs_for("attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm")
        masks = {
            s.kernel_spec.sched_barrier_mask
            for s in specs
            if s.kernel_spec.use_sched_barrier
        }
        self.assertEqual(masks, {0, 0x008, 0x108})

    def test_transposed_codepath_registers_q_reread(self):
        specs = self._specs_for("attention_gfx950_u2d_transposed32_nw2_mw32_t4xb_llvm")
        self.assertTrue(any(s.kernel_spec.use_q_reread for s in specs))
        # Re-read needs a surviving Q_lds; direct-register Q never stages one.
        self.assertFalse(
            any(
                s.kernel_spec.use_q_reread and s.kernel_spec.use_q_direct_reg
                for s in specs
            )
        )

    def test_no_registered_spec_pairs_the_fence_with_the_interleave_hint(self):
        """The 2D emitter rejects that pair, and it does so at build time rather
        than in ``__post_init__`` -- so an offered spec would survive selection
        and only fail once a sweep tried to build it."""
        for prefix in (
            "attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm",
            "attention_gfx950_u2d_transposed32_nw2_mw32_t4xb_llvm",
        ):
            with self.subTest(candidate=prefix):
                for spec in self._specs_for(prefix):
                    ks = spec.kernel_spec
                    self.assertFalse(
                        ks.use_sched_barrier and ks.use_softmax_mfma_interleave
                    )

    def test_candidate_sweep_expands_valid_unique_specs(self):
        candidate = next(
            c
            for c in attention_candidates()
            if c.name.startswith("attention_gfx950_u2d_transposed32_nw2_mw32_t4xb_llvm")
        )
        req = replace(
            _request(),
            algorithm=candidate.algorithm,
            spec_id=candidate.spec_id,
        )
        specs = candidate.sweep_space(req)
        self.assertGreater(len(specs), 1)
        self.assertEqual(len(specs), len({repr(s) for s in specs}))
        self.assertTrue(all(s.path == "2d" for s in specs))


if __name__ == "__main__":
    unittest.main()
