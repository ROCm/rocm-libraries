# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The executable gfx950 bf16 D256 prefill geometry.

CPU-only. ``attention_gfx950_d256`` routes the cohort but returns the deferred
``AttentionSpec``; ``attention_gfx950_u2d_d256_nw2_mw32_t64_llvm`` is the opt-in tuning
geometry that carries the same codegen constellation as a concrete kernel spec, so it can
be enumerated, packaged and launched. Each test states the invariant it pins.
"""

from __future__ import annotations

import dataclasses
import unittest

import kernels.common.attention_unified as au
from dispatch.attention import (
    ATTENTION_EXECUTION_REGISTRY,
    AttentionRequest,
    AttentionTuningSpec,
    _problem,
    attention_candidates,
    attention_execution_candidates,
    dispatch_attention,
)
from dispatch.attention.unified_rules import TUNING_ALGORITHM
from rocke import lower_kernel_to_llvm

SPEC_ID = "gfx950_u2d_d256_nw2_mw32_t64_llvm"


def _req(**kw) -> AttentionRequest:
    base = dict(
        batch=2,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=4096,
        seqlen_k=4096,
        hdim_q=256,
        hdim_v=256,
        arch="gfx950",
        dtype="bf16",
        mask_type=1,
        algorithm=TUNING_ALGORITHM,
        spec_id=SPEC_ID,
    )
    base.update(kw)
    return AttentionRequest(**base)


def _candidate(candidates):
    (found,) = [c for c in candidates if c.spec_id == SPEC_ID]
    return found


class TestRegistration(unittest.TestCase):
    def test_registered_on_route_and_execution_as_opt_in(self):
        """An execution candidate must also be routable by pin, and never auto-picked."""
        for candidates in (attention_candidates(), attention_execution_candidates()):
            candidate = _candidate(candidates)
            self.assertEqual(candidate.algorithm, TUNING_ALGORITHM)
            self.assertTrue(candidate.opt_in)
            self.assertFalse(
                candidate.admits(_req(algorithm="auto", spec_id="auto"))[0]
            )

    def test_auto_dispatch_still_returns_the_deferred_d256_spec(self):
        """The route's D256 candidate is unchanged: the new one must not displace it."""
        result = dispatch_attention(_req(algorithm="auto", spec_id="auto"))
        self.assertEqual(result.candidate.spec_id, "gfx950_d256")
        self.assertTrue(result.spec.tiled_overrides)

    def test_executable_candidate_has_a_build_and_a_torch_binding(self):
        """The execution registry refuses a candidate without either."""
        self.assertIn(
            SPEC_ID, [c.spec_id for c in ATTENTION_EXECUTION_REGISTRY.candidates()]
        )


class TestKernelSpecMatchesProduction(unittest.TestCase):
    def test_kernel_spec_equals_the_route_builders_spec(self):
        """The constellation is one source of truth: a drift between the executable
        geometry and the kernel the production route builds would ship another kernel.
        """
        for block_size in (16, 32, 64):
            for mask_type in (1, 2):
                with self.subTest(block_size=block_size, mask_type=mask_type):
                    req = _req(kv_block_size=block_size, mask_type=mask_type)
                    spec = dispatch_attention(req).spec
                    self.assertIsInstance(spec, AttentionTuningSpec)
                    produced = au._tiled_spec_from_problem(_problem(req), "gfx950")
                    self.assertEqual(spec.kernel_spec, produced)

    def test_default_spec_is_the_production_waves_per_eu(self):
        """Production builds D256 at waves_per_eu=2; the default must too."""
        spec = dispatch_attention(_req()).spec
        self.assertEqual(spec.kernel_spec.waves_per_eu, 2)
        self.assertEqual(
            (
                spec.kernel_spec.tile_size,
                spec.kernel_spec.num_warps,
                spec.kernel_spec.block_m_per_warp,
            ),
            (64, 2, 32),
        )

    def test_the_two_causal_masks_are_distinct_kernels(self):
        """Top-left and bottom-right are different kernel bodies, not one spec."""
        top_left = dispatch_attention(_req(mask_type=1)).spec.kernel_spec
        bottom_right = dispatch_attention(_req(mask_type=2)).spec.kernel_spec
        self.assertTrue(top_left.causal_top_left)
        self.assertFalse(bottom_right.causal_top_left)
        self.assertEqual(top_left.right_bound, bottom_right.right_bound)
        self.assertNotEqual(top_left, bottom_right)

    def test_every_constellation_knob_reaches_the_spec(self):
        """A knob the space silently dropped would build the generic transposed kernel."""
        spec = dispatch_attention(_req()).spec.kernel_spec
        for name, value in au._d256_gfx950_spec_overrides().items():
            with self.subTest(knob=name):
                self.assertEqual(getattr(spec, name), value)

    def test_production_sweep_is_one_kernel_in_three_occupancy_hints(self):
        """The only production axis is waves_per_eu; the constellation is fixed."""
        candidate = _candidate(attention_candidates())
        specs = list(candidate.sweep_space(_req()))
        self.assertEqual(
            sorted(str(s.kernel_spec.waves_per_eu) for s in specs), ["2", "4", "None"]
        )
        stripped = {
            dataclasses.replace(s.kernel_spec, waves_per_eu=None) for s in specs
        }
        self.assertEqual(len(stripped), 1)


class TestAdmission(unittest.TestCase):
    def _admits(self, **kw) -> bool:
        return _candidate(attention_candidates()).admits(_req(**kw))[0]

    def test_admits_the_cohort(self):
        self.assertTrue(self._admits())
        self.assertTrue(self._admits(mask_type=2))

    def test_declines_fp16(self):
        """ATT-34: fp16 D256 is not served by this kernel."""
        self.assertFalse(self._admits(dtype="fp16"))

    def test_declines_other_head_sizes(self):
        self.assertFalse(self._admits(hdim_q=128, hdim_v=128))

    def test_declines_non_gfx950(self):
        self.assertFalse(self._admits(arch="gfx942"))

    def test_declines_decode(self):
        """The cohort is prefill: a single-token query is the 3D decode path."""
        self.assertFalse(self._admits(batch=1, seqlen_q=1, seqlen_k=8192))

    def test_declines_sliding_window(self):
        self.assertFalse(self._admits(sliding_window=256))

    def test_declines_no_mask(self):
        """The route's D256 candidate serves causal masks only; so does this one."""
        self.assertFalse(self._admits(mask_type=0))


class TestLaunchContract(unittest.TestCase):
    def test_block_is_two_waves(self):
        spec = dispatch_attention(_req()).spec
        self.assertEqual(spec.launch_block(), (128, 1, 1))

    def test_grid_covers_every_query_block(self):
        req = _req(batch=3, seqlen_q=1024, seqlen_k=2048)
        spec = dispatch_attention(req).spec
        problem = _problem(req)
        block_m = spec.kernel_spec.num_warps * spec.kernel_spec.block_m_per_warp
        block_q = block_m // problem.num_queries_per_kv
        self.assertEqual(
            spec.launch_grid(problem),
            (8, problem.total_q // block_q + problem.num_seqs, 1),
        )

    def test_builds_to_llvm(self):
        """The spec is buildable and lowers: the whole point of making it executable."""
        spec = dispatch_attention(_req(batch=1, seqlen_q=1024, seqlen_k=1024)).spec
        kernel = spec.build("gfx950")
        self.assertIn("define", lower_kernel_to_llvm(kernel, arch="gfx950"))


if __name__ == "__main__":
    unittest.main()
