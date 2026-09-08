# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection + support + grid tests for grouped wgrad dispatch.

CPU-only (no GPU / no comgr): asserts that the grouped-convolution dispatcher
admits grouped backward-weight requests, and that the launch grid it derives
matches the kernel's block_id_z contract --

    grid = (ceil(wg_N / tile_n), ceil(wg_M / tile_m), groups * split_k)

with the per-group dims wg_M = kpg, wg_N = spatial * cpg. This is the same grid
the GPU correctness test (platform tests ``test_conv_wgrad_correctness.py``)
launches and validates numerically, so a match here proves the dispatch path
launches a correct grid.
"""

from __future__ import annotations

import math
import unittest

from dispatch.grouped_convolution import (
    ConvGroupedRequest,
    _problem,
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
    # Mirror dispatch._wgrad_grid: per-group tiling on x/y, and z = groups *
    # split_k with the group riding block_id_z alongside the K-slice. split_k
    # == -1 is the auto sentinel; resolve it via the same CK formula the grid
    # uses so this stays an independent re-derivation of the wiring.
    p = _problem(req)
    spatial = (p.Z if p.is_3d else 1) * p.Y * p.X
    kpg = p.K // p.groups
    cpg = p.C // p.groups
    wg_M = kpg
    wg_N = spatial * cpg
    gx = math.ceil(wg_N / spec.tile_n)
    gy = math.ceil(wg_M / spec.tile_m)
    split_k = spec.split_k
    if split_k == -1:
        from rocke.helpers.split_k import select_split_k_wgrad

        split_k = select_split_k_wgrad(
            wg_M=wg_M,
            wg_N=wg_N,
            wg_K=p.N * p.Ho * p.Wo * (p.Do if p.is_3d else 1),
            tile_m=spec.tile_m,
            tile_n=spec.tile_n,
            tile_k=spec.tile_k,
            arch=spec.arch,
        ).split_k
    return (gx, gy, p.groups * split_k)


class TestGroupedWgradDispatch(unittest.TestCase):

    # ---- admittance + grid ---------------------------------------------------

    def test_grouped_admitted_grid_per_group(self):
        # groups=4: grid-per-group. The group rides block_id_z alongside the
        # K-slice, so z = groups*split_k (split_k auto-resolved, >= 1).
        for arch in ("gfx942", "gfx950"):
            r = dispatch_conv_grouped(_wgrad(arch, G=4))
            self.assertEqual(r.spec.direction, "wgrad")
            self.assertEqual(r.spec.epilogue, "default")
            self.assertEqual(r.grid[2] % 4, 0, "z must be a multiple of groups")
            self.assertGreaterEqual(r.grid[2] // 4, 1, "split_k >= 1 per group")
            self.assertEqual(r.grid, _expected_grid(r.request, r.spec))

    def test_grouped_cshuffle_admitted(self):
        # A grouped request whose vec derives a cshuffle epilogue is admitted:
        # grouping is orthogonal to the epilogue (the staged store threads the
        # per-group k_out fold).
        for arch in ("gfx942", "gfx950"):
            r = dispatch_conv_grouped(_wgrad(arch, G=4, vec_size_c=8))
            self.assertEqual(r.spec.direction, "wgrad")
            self.assertEqual(r.spec.epilogue, "cshuffle")
            self.assertEqual(r.grid, _expected_grid(r.request, r.spec))

    def test_gfx1250_grouped_admitted_wmma(self):
        # gfx1250 (wave32 WMMA 16x16x32): grouped grid-per-group, split_k forced
        # to 1 (WMMA has no split_k), direct-store epilogue.
        r = dispatch_conv_grouped(_wgrad("gfx1250", G=4))
        self.assertEqual(r.spec.direction, "wgrad")
        self.assertEqual(r.spec.epilogue, "default")
        self.assertEqual(r.spec.split_k, 1, "WMMA wgrad must use split_k=1")
        self.assertEqual(r.grid[2], 4, "z must be one index per group")
        self.assertEqual(r.grid, _expected_grid(r.request, r.spec))

    def test_ungrouped_grid_unchanged(self):
        # groups=1: the grid reduces to the pre-grouped (gx, gy, split_k) form:
        # gx/gy from the DENSE dims (wg_M=K, wg_N=spatial*C) and z the
        # auto-resolved split_k (>=1).
        req = _wgrad("gfx942", G=1, vec_size_c=None)
        r = dispatch_conv_grouped(req)
        gx = math.ceil(req.Y * req.X * req.C / r.spec.tile_n)
        gy = math.ceil(req.K / r.spec.tile_m)
        self.assertEqual(r.grid[0], gx)
        self.assertEqual(r.grid[1], gy)
        self.assertGreaterEqual(r.grid[2], 1)

    # ---- candidate admittance ------------------------------------------------

    def test_candidate_admits_grouped(self):
        # candidate-level admittance mirrors the dispatch result for a valid
        # grouped request.
        cands = {c.name: c for c in conv_grouped_candidates("wgrad")}
        self.assertTrue(any("gfx942" in n for n in cands))
        c = next(c for n, c in cands.items() if "gfx942" in n)
        ok, why = c.admits(_wgrad("gfx942", G=4))
        self.assertTrue(ok, why)


class TestGroupedConvDirectionSurface(unittest.TestCase):
    """The grouped-conv directions handled here are reachable from one module.

    ``dispatch.grouped_convolution`` covers forward and backward-weight (wgrad);
    both share a single ``ConvGroupedRequest`` import surface.
    """

    def test_each_direction_returns_candidates(self):
        for direction in ("fwd", "wgrad"):
            self.assertGreater(len(conv_grouped_candidates(direction)), 0, direction)


if __name__ == "__main__":
    unittest.main()
