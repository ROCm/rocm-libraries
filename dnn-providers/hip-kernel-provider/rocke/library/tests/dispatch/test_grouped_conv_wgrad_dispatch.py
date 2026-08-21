# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection + support + grid tests for grouped/merged wgrad dispatch.

CPU-only (no GPU / no comgr): asserts that the grouped-convolution dispatcher
admits grouped and group-merged (NumGroupsToMerge) backward-weight requests, and
that the launch grid it derives matches the kernel's block_id_z contract --

    grid = (ceil(wg_N / tile_n), ceil(wg_M / tile_m), ceil(G / Gm) * split_k)

with the merged per-workgroup dims wg_M = Gm*kpg, wg_N = spatial * Gm*cpg. This is
the same grid the GPU correctness test (platform tests
``test_conv_wgrad_correctness.py``) launches and validates numerically, so a match
here proves the dispatch path launches a correct grid. Also checks that invalid
merge factors are rejected loudly (the kernel no longer raises on groups>1, so the
dispatcher must be the gate).
"""

from __future__ import annotations

import math
import unittest

from dispatch.grouped_convolution import (
    ConvGroupedRequest,
    conv_grouped_candidates,
    dispatch_conv_grouped,
)


def _wgrad(arch="gfx942", **kw):
    base = dict(
        N=2,
        C=64,
        K=64,
        Hi=14,
        Wi=14,
        Y=3,
        X=3,
        pad_h=1,
        pad_w=1,
        arch=arch,
        direction="wgrad",
        # force default epilogue (vec_size_c=1) so grouped isn't rejected for
        # cshuffle; grouped wgrad supports only the direct-store epilogue.
        vec_size_c=1,
    )
    base.update(kw)
    return ConvGroupedRequest(**base)


def _expected_grid(req, spec):
    p_groups = int(req.G)
    gm = spec.num_groups_to_merge
    kpg = req.K // p_groups
    cpg = req.C // p_groups
    spatial = req.Y * req.X  # 2D shapes in this test
    wg_M = gm * kpg
    wg_N = spatial * gm * cpg
    gx = math.ceil(wg_N / spec.tile_n)
    gy = math.ceil(wg_M / spec.tile_m)
    gz = math.ceil(p_groups / gm) * spec.split_k
    return (gx, gy, gz)


class TestGroupedWgradDispatch(unittest.TestCase):

    # ---- admittance + grid ---------------------------------------------------

    def test_grouped_admitted_grid_per_group(self):
        # groups=4, Gm=1: grid-per-group, one group-batch per z index.
        for arch in ("gfx942", "gfx950"):
            r = dispatch_conv_grouped(_wgrad(arch, G=4))
            self.assertEqual(r.spec.direction, "wgrad")
            self.assertEqual(r.spec.epilogue, "default")
            self.assertEqual(r.spec.split_k, 1, "grouped wgrad must use split_k=1")
            self.assertEqual(r.spec.num_groups_to_merge, 1)
            self.assertEqual(r.grid[2], 4, "z must be one index per group")
            self.assertEqual(r.grid, _expected_grid(r.request, r.spec))

    def test_merged_admitted_grid(self):
        # groups=4 Gm=2 and groups=32 Gm=4: z = ceil(G/Gm) group-batches.
        cases = [
            dict(G=4, num_groups_to_merge=2, C=64, K=64, exp_z=2),
            dict(G=32, num_groups_to_merge=4, C=256, K=256, exp_z=8),
        ]
        for arch in ("gfx942", "gfx950"):
            for kw in cases:
                exp_z = kw.pop("exp_z")
                r = dispatch_conv_grouped(_wgrad(arch, **kw))
                kw["exp_z"] = exp_z  # restore for the next arch iteration
                self.assertEqual(r.spec.num_groups_to_merge, kw["num_groups_to_merge"])
                self.assertEqual(r.spec.split_k, 1)
                self.assertEqual(r.grid[2], exp_z)
                self.assertEqual(r.grid, _expected_grid(r.request, r.spec))

    def test_ungrouped_grid_unchanged(self):
        # groups=1: Gm stays 1 (auto split_k). With one group-batch the grid must
        # reduce to the pre-grouped (gx, gy, split_k) form: gx/gy from the DENSE
        # dims (wg_M=K, wg_N=spatial*C) and z the auto-resolved split_k (>=1).
        req = _wgrad("gfx942", G=1, vec_size_c=None)
        r = dispatch_conv_grouped(req)
        self.assertEqual(r.spec.num_groups_to_merge, 1)
        gx = math.ceil(req.Y * req.X * req.C / r.spec.tile_n)
        gy = math.ceil(req.K / r.spec.tile_m)
        self.assertEqual(r.grid[0], gx)
        self.assertEqual(r.grid[1], gy)
        self.assertGreaterEqual(r.grid[2], 1)

    # ---- rejection of invalid merge factors ----------------------------------

    def test_rejects_non_power_of_two_merge(self):
        # Gm=3 is not a power of two. Use G=8 (divides C=64) so the request itself
        # is valid and the merge-factor validator is what fires.
        with self.assertRaises(ValueError):
            dispatch_conv_grouped(_wgrad("gfx942", G=8, num_groups_to_merge=3))

    def test_rejects_indivisible_merge(self):
        # groups=4 not divisible by Gm=8.
        with self.assertRaises(ValueError):
            dispatch_conv_grouped(_wgrad("gfx942", G=4, num_groups_to_merge=8))

    def test_rejects_merge_without_groups(self):
        # Gm>1 requires groups>1.
        with self.assertRaises(ValueError):
            dispatch_conv_grouped(_wgrad("gfx942", G=1, num_groups_to_merge=2))

    def test_candidate_admits_flags(self):
        # candidate-level admittance mirrors the dispatch result.
        cands = {c.name: c for c in conv_grouped_candidates("wgrad")}
        self.assertTrue(any("gfx942" in n for n in cands))
        for c in cands.values():
            ok_bad, why = c.admits(
                _wgrad(c.capability.arches[0], G=8, num_groups_to_merge=3)
            )
            self.assertFalse(ok_bad)
            self.assertIn("power of two", why)


if __name__ == "__main__":
    unittest.main()
