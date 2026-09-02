# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection + support + arch-gating tests for the conv_wgrad dispatcher family."""

from __future__ import annotations

import unittest

from rocke.dispatch.families.conv_wgrad import (
    CONV_WGRAD_REGISTRY,
    ConvWgradRequest,
    ConvWgradWorkspaceSpec,
    compute_wgrad_workspace_spec,
    dispatch_conv_wgrad,
    query_wgrad_support,
    wgrad_stage2_grid,
)
from rocke.instances.common.conv_implicit_gemm_wgrad_two_stage import (
    build_implicit_gemm_conv_wgrad_two_stage,
    wgrad_two_stage_workspace_nbytes,
)


def _wgrad(arch, **kw):
    base = dict(N=8, C=64, K=64, Hi=56, Wi=56, Y=3, X=3, arch=arch)
    base.update(kw)
    return ConvWgradRequest(**base)


class TestConvWgradWorkspaceSpec(unittest.TestCase):
    def test_preferred_bytes_formula(self):
        # K=64 → wg_M=64; Y=3 X=3 C=64 → wg_N=576; split_k=8
        req = _wgrad("gfx950", K=64, C=64, Y=3, X=3)
        spec = compute_wgrad_workspace_spec(req, split_k=8)
        self.assertEqual(spec.preferred_bytes, 8 * 64 * 576 * 4)
        self.assertEqual(spec.preferred_bytes, 1_179_648)

    def test_workspace_fits_within_default_cap(self):
        req = _wgrad("gfx950", K=64, C=64, Y=3, X=3)
        spec = compute_wgrad_workspace_spec(req, split_k=1)
        self.assertTrue(spec.workspace_fits)
        self.assertEqual(spec.fallback_reason, "")

    def test_workspace_fallback_when_cap_exceeded(self):
        req = _wgrad("gfx950", K=64, C=64, Y=3, X=3)
        spec = compute_wgrad_workspace_spec(req, split_k=8, hard_cap=1024)
        self.assertFalse(spec.workspace_fits)
        self.assertEqual(spec.fallback_split_k, 1)
        self.assertEqual(spec.minimum_bytes, 0)
        self.assertGreater(len(spec.fallback_reason), 0)

    def test_query_wgrad_support_split_k1_not_deterministic(self):
        # split_k=1 bypasses the two-stage path entirely; must report False.
        req = _wgrad("gfx950")
        info = query_wgrad_support(req, split_k=1)
        self.assertFalse(info["supports_deterministic"])

    def test_query_wgrad_support_split_k_gt1_fits(self):
        req = _wgrad("gfx950")
        info = query_wgrad_support(req, split_k=8)
        self.assertTrue(info["supports_deterministic"])

    def test_query_wgrad_support_fallback(self):
        req = _wgrad("gfx950")
        info = query_wgrad_support(req, split_k=8, hard_cap=1024)
        self.assertFalse(info["supports_deterministic"])


class TestConvWgradDispatch(unittest.TestCase):
    def test_dispatch_selects_gfx950(self):
        result = dispatch_conv_wgrad(_wgrad("gfx950"))
        self.assertGreater(len(result.grid), 0)
        self.assertGreater(len(result.block), 0)
        self.assertGreaterEqual(result.spec.split_k, 1)

    def test_dispatch_selects_gfx942(self):
        result = dispatch_conv_wgrad(_wgrad("gfx942"))
        self.assertGreater(len(result.grid), 0)
        self.assertGreater(len(result.block), 0)
        self.assertGreaterEqual(result.spec.split_k, 1)

    def test_dispatch_rejects_unknown_arch(self):
        with self.assertRaises(ValueError):
            dispatch_conv_wgrad(_wgrad("gfx000"))

    def test_dispatch_rejects_degenerate_output(self):
        with self.assertRaises(ValueError):
            dispatch_conv_wgrad(_wgrad("gfx950", Hi=3, Wi=3, Y=5, X=5))

    def test_dispatch_rejects_zero_dim(self):
        with self.assertRaises(ValueError):
            dispatch_conv_wgrad(_wgrad("gfx950", N=0))

    def test_dispatch_result_grid_shape(self):
        result = dispatch_conv_wgrad(_wgrad("gfx950"))
        self.assertEqual(len(result.grid), 3)
        self.assertEqual(result.grid[2], result.spec.split_k)

    def test_wgrad_stage2_grid_is_3_tuple_z_is_1(self):
        result = dispatch_conv_wgrad(_wgrad("gfx950"))
        s2 = wgrad_stage2_grid(result.spec)
        self.assertEqual(len(s2), 3)
        self.assertEqual(s2[2], 1)

    def test_two_stage_true_when_split_k_gt_1(self):
        # Tiny spatial dims: grid is small, heuristic picks split_k > 1.
        req = _wgrad("gfx950", N=1, Hi=4, Wi=4, K=64, C=64, Y=3, X=3)
        result = dispatch_conv_wgrad(req)
        self.assertGreater(
            result.spec.split_k, 1, "expected split_k > 1 for small problem"
        )
        self.assertTrue(result.spec.two_stage)

    def test_two_stage_false_when_split_k_1(self):
        # Large K/C fill the tile grid; heuristic picks split_k=1.
        req = _wgrad("gfx950", N=2, Hi=4, Wi=4, K=512, C=512, Y=3, X=3)
        result = dispatch_conv_wgrad(req)
        self.assertEqual(
            result.spec.split_k, 1, "expected split_k=1 for large-tile problem"
        )
        self.assertFalse(result.spec.two_stage)

    def test_unique_candidate_names(self):
        names = [c.name for c in CONV_WGRAD_REGISTRY.candidates()]
        self.assertEqual(len(names), len(set(names)))


class TestConvWgradTwoStagePipelineBuilder(unittest.TestCase):
    def _spec_split_k_gt1(self):
        """Dispatch a spec guaranteed to have split_k > 1."""
        req = _wgrad("gfx950", N=1, Hi=4, Wi=4, K=64, C=64, Y=3, X=3)
        return dispatch_conv_wgrad(req).spec

    def test_raises_when_split_k_le_1(self):
        # Large-tile problem where heuristic picks split_k=1.
        req = _wgrad("gfx950", N=2, Hi=4, Wi=4, K=512, C=512, Y=3, X=3)
        spec = dispatch_conv_wgrad(req).spec
        self.assertEqual(spec.split_k, 1)
        with self.assertRaises(ValueError):
            build_implicit_gemm_conv_wgrad_two_stage(spec, arch="gfx950")

    def test_workspace_nbytes_positive(self):
        spec = self._spec_split_k_gt1()
        self.assertGreater(spec.split_k, 1)
        ws_nbytes = wgrad_two_stage_workspace_nbytes(spec)
        self.assertGreater(ws_nbytes, 0)
        expected = spec.split_k * spec.wg_M * spec.wg_N * 4
        self.assertEqual(ws_nbytes, expected)

    def test_cdna_candidates_reject_rdna(self):
        with self.assertRaises(ValueError):
            dispatch_conv_wgrad(_wgrad("gfx1151"))

    def test_dims_contains_wg_keys(self):
        req = _wgrad("gfx950")
        d = req.dims()
        self.assertIn("wg_M", d)
        self.assertIn("wg_N", d)
        self.assertIn("wg_K", d)


if __name__ == "__main__":
    unittest.main()
