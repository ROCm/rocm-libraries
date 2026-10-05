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
    _block,
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
            groups=p.groups,
            block_size=_block(spec)[0],
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

    ``dispatch.grouped_convolution`` covers forward, backward-weight (wgrad) and
    backward-data (dgrad); all three share a single ``ConvGroupedRequest``
    import surface.
    """

    def test_each_direction_returns_candidates(self):
        for direction in ("fwd", "wgrad", "dgrad"):
            self.assertGreater(len(conv_grouped_candidates(direction)), 0, direction)


class TestTwoStageSelection(unittest.TestCase):
    """two_stage is chosen by atomic availability, not by a caller flag."""

    def test_odd_wg_N_takes_the_scratch_path(self):
        # wg_N = Y*X*cpg odd => the packed <2 x bf16> atomic pair is not
        # dword-aligned, so split-K can only go through the f32 scratch.
        r = dispatch_conv_grouped(_wgrad("gfx950", C=3, K=24, Y=3, X=3, dtype="bf16"))
        ws = r.spec.to_wgrad_spec(_problem(r.request))
        if ws.split_k > 1:
            self.assertTrue(
                ws.two_stage,
                "odd wg_N with a 16-bit dW must resolve to the two-stage path",
            )

    def test_even_wg_N_stays_on_the_packed_atomic(self):
        # wg_N even => the packed atomic can address dW directly and no
        # scratch/second launch is needed.
        r = dispatch_conv_grouped(_wgrad("gfx950", C=64, K=64, Y=3, X=3, dtype="bf16"))
        ws = r.spec.to_wgrad_spec(_problem(r.request))
        self.assertFalse(
            ws.two_stage, "an even wg_N should reach split-K via packed atomics"
        )

    def test_split_k_1_needs_no_two_stage(self):
        r = dispatch_conv_grouped(_wgrad("gfx1250"))
        ws = r.spec.to_wgrad_spec(_problem(r.request))
        self.assertEqual(ws.split_k, 1, "gfx1250 always uses split_k=1")
        self.assertFalse(ws.two_stage, "split_k=1 needs no two_stage")


class TestTwoStageGridShape(unittest.TestCase):
    """Stage 1 and Stage 2 grid shapes for the two-stage deterministic path."""

    def test_stage1_grid_z_is_groups_times_split_k(self):
        # Stage 1 grid z encodes both group and split-K slice:
        #   z = groups * split_k
        for arch in ("gfx942", "gfx950"):
            r = dispatch_conv_grouped(_wgrad(arch, G=4))
            groups = r.request.G
            split_k = r.grid[2] // groups
            self.assertEqual(r.grid[2], groups * split_k)

    def test_stage2_grid_z_is_groups(self):
        # Stage 2 (workspace-reduce) uses grid z = groups: block_id_z is the
        # group index, one CTA per group covering wg_M x wg_N output elements.
        from kernels.common.conv_wgrad_workspace_reduce import (
            WgradReduceSpec,
            wgrad_reduce_grid,
        )

        for arch in ("gfx942", "gfx950"):
            r = dispatch_conv_grouped(_wgrad(arch, G=4))
            ws = r.spec.to_wgrad_spec(_problem(r.request))
            s2_spec = WgradReduceSpec(
                problem=ws.problem,
                dtype_d=ws.data.dtype_d,
                groups=r.request.G,
            )
            grid = wgrad_reduce_grid(s2_spec)
            self.assertEqual(grid[2], r.request.G, "Stage 2 grid z must equal groups")


def _dgrad(arch="gfx950", **kw):
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
        direction="dgrad",
    )
    base.update(kw)
    return ConvGroupedRequest(**base)


class TestGroupedDgradDispatch(unittest.TestCase):
    """Selection + grid + K-outer policy for the gfx950 dgrad candidate.

    The grid contract differs from wgrad's: dgrad's M-tile count is not a closed
    form over the problem dims, because stride > 1 splits the convolution into
    ``y_tilde * x_tilde`` sub-GEMMs of differing sizes. The x extent is the
    cumulative tile count of the last sub-GEMM, and the group rides ``blockIdx.y``
    rather than sharing z with the K-slice.
    """

    def _select(self, req):
        # These tests pin the igemm candidate's own contract. Grouped stride-1
        # requests are also admitted by the higher-priority direct-MFMA
        # candidate (covered by TestGroupedDirectDgradDispatch), so look the
        # igemm one up by name rather than expecting it to be the only taker.
        cands = [
            c
            for c in conv_grouped_candidates("dgrad")
            if c.name == "implicit_gemm_conv_dgrad" and c.admits(req)[0]
        ]
        self.assertEqual(len(cands), 1, f"igemm dgrad must admit {req}: {cands}")
        return cands[0], cands[0].select_spec(req)

    def test_admitted_and_grid_matches_sub_gemm_tiling(self):
        req = _dgrad()
        cand, spec = self._select(req)
        p = _problem(req)
        # Independent re-derivation: stride 1 is a single sub-GEMM, so the flat
        # tile count is just the M/N tiling of that one GEMM.
        gemm_m = p.N * p.Hi * p.Wi
        expected_x = math.ceil(gemm_m / spec.tile_m) * math.ceil(
            (p.C // p.groups) / spec.tile_n
        )
        self.assertEqual(cand.grid(spec, req), (expected_x, 1, 1))

    def test_strided_grid_uses_tilde_decomposition(self):
        # stride 2 gives y_tilde = x_tilde = 2: four sub-GEMMs over a quarter of
        # the rows each. The flat tile count must therefore differ from the
        # stride-1 count rather than reusing a single-GEMM formula.
        strided = _dgrad(stride_h=2, stride_w=2)
        cand, spec = self._select(strided)
        gx_strided = cand.grid(spec, strided)[0]

        plain = _dgrad()
        cand1, spec1 = self._select(plain)
        gx_plain = cand1.grid(spec1, plain)[0]

        self.assertNotEqual(gx_strided, gx_plain)
        self.assertGreater(gx_strided, 0)

    def test_group_rides_block_id_y(self):
        req = _dgrad(C=64, K=64, G=4)
        cand, spec = self._select(req)
        self.assertEqual(cand.grid(spec, req)[1], 4)

    def test_k_outer_selected_on_even_channel_run(self):
        _cand, spec = self._select(_dgrad())
        self.assertTrue(spec.lds_k_outer)
        self.assertEqual(spec.direction, "dgrad")

    def test_k_outer_declined_on_odd_channel_run(self):
        # cpg = 48 / 16 = 3. The B load width collapses to 1, axis_b is already
        # "col", and there is no transpose-on-store left to remove -- K-outer
        # would only add the read-side cost. The predicate must decline.
        _cand, spec = self._select(_dgrad(C=48, G=16))
        self.assertFalse(spec.lds_k_outer)

    def test_dgrad_candidate_rejects_other_directions(self):
        cand = conv_grouped_candidates("dgrad")[0]
        for direction in ("fwd", "wgrad"):
            ok, why = cand.admits(_dgrad(direction=direction))
            self.assertFalse(ok, direction)
            self.assertIn("dgrad", why)

    def test_epilogue_follows_store_vector_width(self):
        # dX's last dim is C, so a wide store vector needs cshuffle's LDS
        # staging; the direct-store 'default' path writes scalars. Pinning
        # 'default' unconditionally is silently valid -- vector_size_c is left
        # unset, so the validator rule never fires -- and costs store bandwidth
        # on every non-grouped shape. Derive it instead.
        _cand, wide = self._select(_dgrad(C=128, K=128, Hi=32, Wi=32))
        self.assertEqual(wide.epilogue, "cshuffle")

        # cpg = 3: no legal width > 1, so the scalar direct store is correct.
        _cand, narrow = self._select(_dgrad(C=48, G=16))
        self.assertEqual(narrow.epilogue, "default")

    def test_vec_size_c_uses_the_dgrad_formula(self):
        # Each direction has its own default_vector_sizes and they are not
        # interchangeable: dgrad's takes the per-group runs (cpg, kpg), so a
        # fallthrough to the forward formula sizes off the wrong extent once
        # groups > 1.
        from dispatch.grouped_convolution import _vec_size_c
        from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec

        req = _dgrad(C=48, G=16)
        p = _problem(req)
        _va, _vb, expected = DgradConvSpec.default_vector_sizes(
            p.cpg, p.kpg, req.dtype.lower()
        )
        self.assertEqual(_vec_size_c(req), expected)

    def test_spec_round_trips_to_instance_spec(self):
        req = _dgrad()
        _cand, spec = self._select(req)
        inst = spec.to_dgrad_spec(_problem(req))
        # The dispatcher's K-outer decision must survive into the instance spec;
        # a spec that silently reverts to M-outer would still run and still be
        # correct, so nothing else would catch it.
        self.assertEqual(inst.lds_k_outer, spec.lds_k_outer)
        self.assertEqual(inst.tile_m, spec.tile_m)
        self.assertEqual(inst.warp_tile_n, spec.warp_tile_mn)
        inst.validate()


class TestGfx950DgradTileTable(unittest.TestCase):
    """Shape-keyed tile selection of the gfx950 dgrad candidate."""

    def _tile(self, **kw):
        req = _dgrad(dtype="bf16", **kw)
        r = dispatch_conv_grouped(req)
        s = r.spec
        inst = s.to_dgrad_spec(_problem(req))
        ok, why = _dgrad_valid(inst)
        self.assertTrue(ok, why)
        return (s.tile_m, s.tile_n, s.tile_k, s.warp_m, s.warp_n, s.warp_tile_mn), inst

    def test_large_grid_takes_the_128x128_tile(self):
        # N=8 C=640 K=768 32x32: 64 M tiles x 5 N tiles = 320 workgroups.
        tile, inst = self._tile(N=8, C=640, K=768, Hi=32, Wi=32)
        self.assertEqual(tile, (128, 128, 64, 2, 2, 16))
        self.assertTrue(inst.uses_tap_outer_k)

    def test_one_workgroup_per_cu_keeps_the_64x64_tile(self):
        # N=8 C=512 K=768 32x32: 64 M tiles x 4 N tiles = 256 workgroups,
        # one per CU: below the 128x128 floor.
        tile, _ = self._tile(N=8, C=512, K=768, Hi=32, Wi=32)
        self.assertEqual(tile, (64, 64, 64, 2, 2, 32))

    def test_small_grid_keeps_the_64x64_tile(self):
        # N=4 C=1024 16x16: the 128x128 tile would leave 64 workgroups.
        tile, _ = self._tile(N=4, C=1024, K=1024, Hi=16, Wi=16)
        self.assertEqual(tile, (64, 64, 64, 2, 2, 32))

    def test_few_channels_keep_the_64x64_tile(self):
        # cpg = 64 < 128: a 128-wide N tile would be half empty.
        tile, _ = self._tile(N=16, C=64, K=192, Hi=40, Wi=40)
        self.assertEqual(tile, (64, 64, 64, 2, 2, 32))

    def test_kpg_odd_multiple_of_32_keeps_the_default_tile(self):
        # kpg % 64 == 32: tile_k 64 straddles taps (flat folded loop), but
        # tile_k 32 only won with cache-resident operands, so the default is
        # kept for every such problem -- few or many groups, short or long
        # reductions, small and large grids.
        for kw in (
            {"N": 1, "C": 64, "K": 96, "Hi": 13, "Wi": 17, "G": 1, "Y": 3},
            {"N": 1, "C": 9600, "K": 9600, "Hi": 8, "Wi": 8, "G": 300, "Y": 7},
            {"N": 4, "C": 8192, "K": 4096, "Hi": 7, "Wi": 7, "G": 128, "Y": 3},
            {"N": 8, "C": 512, "K": 768, "Hi": 7, "Wi": 7, "G": 8, "Y": 7},
            {"N": 1, "C": 9600, "K": 28800, "Hi": 20, "Wi": 20, "G": 300, "Y": 7},
            {"N": 8, "C": 12800, "K": 19200, "Hi": 9, "Wi": 9, "G": 200, "Y": 3},
            {"N": 4, "C": 1024, "K": 2560, "Hi": 8, "Wi": 8, "G": 16, "Y": 1},
        ):
            with self.subTest(**kw):
                y = kw.pop("Y")
                tile, inst = self._tile(Y=y, X=y, pad_h=y // 2, pad_w=y // 2, **kw)
                self.assertEqual(tile, (64, 64, 64, 2, 2, 32))
                if y > 1:
                    self.assertFalse(inst.uses_tap_outer_k)
                    self.assertTrue(inst.folds_sub_gemm_record)

    def test_small_pointwise_keeps_the_default_tile(self):
        # Ungrouped 1x1 below the 128x128 grid bound: the unchanged 64x64
        # default (a wider tile halves an already small grid), and the
        # runtime record (no fold, so no constant trip count to unroll).
        for kw in (
            {"N": 1, "C": 256, "K": 256, "Hi": 14, "Wi": 14},
            {"N": 1, "C": 512, "K": 512, "Hi": 7, "Wi": 7},
            {"N": 8, "C": 512, "K": 1024, "Hi": 28, "Wi": 28},
            {"N": 2, "C": 64, "K": 128, "Hi": 28, "Wi": 28},
        ):
            with self.subTest(**kw):
                tile, inst = self._tile(Y=1, X=1, pad_h=0, pad_w=0, **kw)
                self.assertEqual(tile, (64, 64, 64, 2, 2, 32))
                self.assertFalse(inst.folds_sub_gemm_record)
                self.assertFalse(inst.uses_tap_outer_k)

    def test_large_pointwise_takes_the_128x128_tile(self):
        # 392 M tiles x 2 N tiles: the grid still fills the device.
        tile, inst = self._tile(
            N=64, C=256, K=512, Hi=28, Wi=28, Y=1, X=1, pad_h=0, pad_w=0
        )
        self.assertEqual(tile, (128, 128, 64, 2, 2, 16))
        self.assertFalse(inst.folds_sub_gemm_record)
        # Three tile_k steps (kpg 129..192) is the shortest K loop that takes it.
        tile, _ = self._tile(
            N=8, C=256, K=160, Hi=56, Wi=56, Y=1, X=1, pad_h=0, pad_w=0
        )
        self.assertEqual(tile, (128, 128, 64, 2, 2, 16))

    def test_short_k_pointwise_keeps_the_64x64_tile_on_large_grids(self):
        # Ungrouped 1x1 with a K loop of one or two tile_k steps (kpg <= 128):
        # the 64x64 tile even when the 128x128 grid is far above the floor
        # (784-3136 workgroups here).
        for kw in (
            {"N": 8, "C": 512, "K": 64, "Hi": 56, "Wi": 56},
            {"N": 8, "C": 1024, "K": 64, "Hi": 56, "Wi": 56},
            {"N": 8, "C": 256, "K": 32, "Hi": 56, "Wi": 56},
            {"N": 8, "C": 512, "K": 128, "Hi": 56, "Wi": 56},
            {"N": 32, "C": 512, "K": 96, "Hi": 56, "Wi": 56},
        ):
            with self.subTest(**kw):
                tile, inst = self._tile(Y=1, X=1, pad_h=0, pad_w=0, **kw)
                self.assertEqual(tile, (64, 64, 64, 2, 2, 32))
                self.assertFalse(inst.folds_sub_gemm_record)

    def test_wide_c_pointwise_needs_eight_workgroups_per_cu(self):
        # Ungrouped 1x1 with cpg > 768 and a K loop of at most eight tile_k
        # steps (kpg <= 512) keeps the 64x64 tile below 2048 128x128
        # workgroups (8 per CU on gfx950).
        for kw in (
            {"N": 64, "C": 1024, "K": 256, "Hi": 14, "Wi": 14},  # 784
            {"N": 128, "C": 1024, "K": 256, "Hi": 14, "Wi": 14},  # 1568
            {"N": 128, "C": 2048, "K": 512, "Hi": 7, "Wi": 7},  # 784
            {"N": 8, "C": 896, "K": 256, "Hi": 28, "Wi": 28},  # 343
        ):
            with self.subTest(**kw):
                tile, _ = self._tile(Y=1, X=1, pad_h=0, pad_w=0, **kw)
                self.assertEqual(tile, (64, 64, 64, 2, 2, 32))
        # From 8 workgroups per CU up, with a longer K loop, or at cpg <= 768
        # the 1.25-per-CU floor alone applies.
        for kw in (
            {"N": 64, "C": 1024, "K": 256, "Hi": 28, "Wi": 28},  # 3136
            {"N": 32, "C": 2048, "K": 512, "Hi": 28, "Wi": 28},  # 3136
            {"N": 64, "C": 1024, "K": 1024, "Hi": 14, "Wi": 14},  # 16 steps
            {"N": 64, "C": 768, "K": 256, "Hi": 14, "Wi": 14},  # cpg 768
        ):
            with self.subTest(**kw):
                tile, _ = self._tile(Y=1, X=1, pad_h=0, pad_w=0, **kw)
                self.assertEqual(tile, (128, 128, 64, 2, 2, 16))

    def test_grouped_pointwise_folds_the_record(self):
        # Grouped 1x1 has no divide-free fast path, so it keeps the fold.
        tile, inst = self._tile(
            N=8, C=256, K=256, Hi=28, Wi=28, Y=1, X=1, pad_h=0, pad_w=0, G=2
        )
        self.assertEqual(tile, (64, 64, 64, 2, 2, 32))
        self.assertTrue(inst.folds_sub_gemm_record)

    def test_strided_keeps_the_default_tile(self):
        tile, inst = self._tile(
            N=16, C=256, K=256, Hi=56, Wi=56, stride_h=2, stride_w=2
        )
        self.assertEqual(tile, (64, 64, 64, 2, 2, 32))
        self.assertFalse(inst.folds_sub_gemm_record)


def _dgrad_valid(inst):
    from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

    return is_valid_dgrad_spec(inst, "gfx950")


class TestGroupedDirectDgradDispatch(unittest.TestCase):
    """Selection, launch plan and grid for the gfx950 direct-MFMA dgrad.

    Grouped stride-1 dgrad with cpg/kpg multiples of 4 up to 32 and filters up
    to 7x7, outside the measured igemm-win corners, must route to the
    direct-MFMA candidate: one kernel that reads W with flipped,
    channel-swapped addressing (the batched 4x4x4 kernel for cpg = kpg = 4),
    or the pre-pass pipeline past the fused form's register budget;
    everything else must keep the igemm candidate.
    """

    _DIRECT = "direct_mfma_conv_dgrad"
    _IGEMM = "implicit_gemm_conv_dgrad"

    @staticmethod
    def _grouped(**kw):
        base = {
            "N": 8,
            "C": 128,
            "K": 128,
            "Hi": 14,
            "Wi": 14,
            "G": 32,
            "dtype": "bf16",
        }
        base.update(kw)
        return _dgrad(**base)

    def _pick(self, req):
        r = dispatch_conv_grouped(req)
        return r.candidate.name, r

    # ---- routing: in-region requests take the direct pipeline -------------

    def test_target_shapes_route_to_direct_with_table_knobs(self):
        # (N, C, K, H, W, G) -> (variant, block_q, block_groups, block_h,
        #                         fold_k32, rule, waves_per_eu)
        cases = {
            (128, 128, 128, 56, 56, 32): ("4c", 4, 16, 0, False, "cpg_kpg_4", 0),
            (128, 512, 512, 14, 14, 32): ("generic", 16, 2, 0, False, "chan_16_31", 4),
            (128, 512, 512, 14, 14, 16): ("generic", 16, 1, 0, True, "chan_ge_32", 0),
        }
        for (n, c, k, h, w, g), want in cases.items():
            with self.subTest(shape=(n, c, k, h, w, g)):
                name, r = self._pick(self._grouped(N=n, C=c, K=k, Hi=h, Wi=w, G=g))
                self.assertEqual(name, self._DIRECT)
                s = r.spec
                self.assertEqual(
                    (
                        s.variant,
                        s.block_q,
                        s.block_groups,
                        s.block_h,
                        s.fold_k32,
                        s.rule_id,
                        s.waves_per_eu,
                    ),
                    want,
                )
                # Single kernel: weights transformed in the prologue, staged
                # through LDS with transpose reads on gfx950.
                self.assertEqual((s.fused_weights, s.weights_lds), (True, True))
                self.assertEqual(
                    (s.waves_q, s.waves_k, s.runtime_k_loop), (1, 1, False)
                )

    def test_both_candidates_admit_and_direct_outranks(self):
        req = self._grouped()
        names = [c.name for c in conv_grouped_candidates("dgrad") if c.admits(req)[0]]
        self.assertEqual(names, [self._DIRECT, self._IGEMM])

    # ---- routing: out-of-region requests keep igemm ------------------------

    def test_out_of_region_requests_keep_igemm(self):
        cases = {
            "dense": {"C": 64, "K": 64, "G": 1},
            "stride2": {"stride_h": 2, "stride_w": 2},
            "dilation2": {"dilation_h": 2, "dilation_w": 2, "pad_h": 2, "pad_w": 2},
            "cpg64": {"C": 2048, "K": 2048, "G": 32},
            "cpg6_not_vec4": {"C": 192, "K": 128, "G": 32},
            "kpg6_not_vec4": {"C": 128, "K": 192, "G": 32},
            "nonsquare_filter": {"Y": 3, "X": 1, "pad_w": 0},
            "asym_pad": {"pad_h": 1, "pad_w": 0},
            "pad_beyond_filter": {"pad_h": 3, "pad_w": 3},
            "pad0_not_same": {"pad_h": 0, "pad_w": 0},
            "depthwise": {"C": 64, "K": 64, "G": 64},
            "filter9x9": {"N": 128, "Y": 9, "X": 9, "pad_h": 4, "pad_w": 4},
            "filter11x11": {"N": 128, "Y": 11, "X": 11, "pad_h": 5, "pad_w": 5},
        }
        for label, kw in cases.items():
            with self.subTest(case=label):
                req = self._grouped(**kw)
                direct = next(
                    c
                    for c in conv_grouped_candidates("dgrad")
                    if c.name == self._DIRECT
                )
                ok, why = direct.admits(req)
                self.assertFalse(ok, f"{label}: direct must decline ({why})")
                if label != "depthwise":
                    name, _ = self._pick(req)
                    self.assertEqual(name, self._IGEMM, label)

    # ---- measured policy: corners where igemm wins keep igemm --------------

    @staticmethod
    def _req_from(shape):
        n, c, k, h, w, y, g, dt = shape
        return _dgrad(
            N=n,
            C=c,
            K=k,
            Hi=h,
            Wi=w,
            Y=y,
            X=y,
            pad_h=y // 2,
            pad_w=y // 2,
            G=g,
            dtype=dt,
        )

    def test_policy_declines_where_igemm_measured_faster(self):
        # Same-session measurements of the selected direct spec against the
        # igemm candidate put each of these well on the igemm side.
        # (N, C, K, H, W, Y, G, dtype)
        from dispatch.grouped_convolution import _DIRECT_DGRAD_POLICY_PREFIX

        cases = {
            "tiny_2x2_cpg32_G2": (16384, 64, 64, 2, 2, 3, 2, "bf16"),
            "tiny_3x3_cpg32_G2": (8192, 64, 64, 3, 3, 3, 2, "bf16"),
            "tiny_4x4_cpg32_G8": (1024, 256, 256, 4, 4, 3, 8, "bf16"),
            "tiny_4x4_cpg16_G2_fp16": (4096, 32, 32, 4, 4, 3, 2, "fp16"),
            "tiny_4x4_1x1_cpg32_G4": (2048, 128, 128, 4, 4, 1, 4, "bf16"),
            "W3_column_cpg8_kpg4": (68, 64, 32, 31, 3, 3, 8, "bf16"),
            "W5_cpg32_G4": (2048, 128, 128, 5, 5, 3, 4, "bf16"),
            "prepass_7x7_cpg24_kpg24_5x5img": (581, 288, 288, 5, 5, 7, 12, "bf16"),
            "prepass_7x7_cpg32_kpg24": (105, 128, 96, 22, 18, 7, 4, "bf16"),
            "prepass_7x7_cpg32_kpg4_W24": (210, 64, 8, 20, 24, 7, 2, "bf16"),
            "prepass_7x7_cpg32_kpg4_W24_G4": (105, 128, 16, 20, 24, 7, 4, "bf16"),
            "prepass_7x7_cpg32_kpg8_14x14": (384, 128, 32, 14, 14, 7, 4, "bf16"),
            "1x1_cpg32_kpg32_G8": (64, 256, 256, 28, 28, 1, 8, "bf16"),
            "1x1_cpg8_kpg32_G16": (128, 128, 512, 14, 14, 1, 16, "bf16"),
            "5x5_cpg24_kpg4_8x8_lowfill": (440, 192, 32, 8, 8, 5, 8, "bf16"),
            # Pre-pass form whose weight transpose is not amortized (many
            # groups, 7x7 or 5x5 weights, 10x10 images, few images), or
            # whose main kernel loses outright (8x8, partial K atom).
            "prepass_unamortized_G200_N1": (1, 6400, 3200, 10, 10, 7, 200, "fp16"),
            "prepass_unamortized_G300_N1": (1, 9600, 9600, 10, 10, 7, 300, "fp16"),
            "prepass_unamortized_G300_N4": (4, 9600, 4800, 10, 10, 7, 300, "fp16"),
            "prepass_unamortized_G200_bf16": (1, 5600, 3200, 10, 10, 7, 200, "bf16"),
            "prepass_unamortized_5x5_G300": (4, 9600, 9600, 10, 10, 5, 300, "fp16"),
            "prepass_main_loses_8x8_kpg20": (42, 1980, 1980, 8, 8, 7, 99, "bf16"),
            # Narrow reduction (kpg 8 under cpg 20: half-filled K atoms) on a
            # short, narrow image.
            "prepass_main_loses_narrow_5x8": (153, 5060, 2024, 5, 8, 7, 253, "bf16"),
        }
        direct = next(
            c for c in conv_grouped_candidates("dgrad") if c.name == self._DIRECT
        )
        for label, shape in cases.items():
            with self.subTest(case=label):
                req = self._req_from(shape)
                ok, why = direct.admits(req)
                self.assertFalse(ok, label)
                self.assertIn(_DIRECT_DGRAD_POLICY_PREFIX, why, label)
                self.assertEqual(self._pick(req)[0], self._IGEMM, label)

    def test_prepass_decline_names_the_cost_model(self):
        # The pre-pass decline quotes the predicted cost ratio, and the ratio
        # falls as the same weights are amortized over more images.
        from dispatch.grouped_convolution import (
            _DIRECT_DGRAD_PREPASS_MAX_COST_RATIO,
            _direct_dgrad_policy_errors,
            _direct_dgrad_prepass_cost_ratio,
            _direct_dgrad_problem,
            _select_direct_dgrad_spec,
        )

        req = self._req_from((1, 6400, 3200, 10, 10, 7, 200, "fp16"))
        spec = _select_direct_dgrad_spec(req)
        self.assertFalse(spec.fused_weights)
        errors = _direct_dgrad_policy_errors(req, spec)
        self.assertEqual(len(errors), 1)
        self.assertIn("not amortized", errors[0])
        ratios = []
        for n in (1, 4, 16):
            r = self._req_from((n, 6400, 3200, 16, 16, 7, 200, "fp16"))
            ratios.append(
                _direct_dgrad_prepass_cost_ratio(
                    _direct_dgrad_problem(r), _select_direct_dgrad_spec(r)
                )
            )
        self.assertEqual(ratios, sorted(ratios, reverse=True))
        self.assertGreater(ratios[0], _DIRECT_DGRAD_PREPASS_MAX_COST_RATIO)
        self.assertLess(ratios[-1], _DIRECT_DGRAD_PREPASS_MAX_COST_RATIO)

    def test_policy_admits_where_direct_measured_faster(self):
        # Measured direct wins over the igemm candidate, across the classes
        # the policy separates (fused single kernel, 4c row, pre-pass fallback
        # with full K atoms, 1x1 narrow reductions, few groups).
        cases = {
            "S1_cpg4_G32": (128, 128, 128, 56, 56, 3, 32, "bf16"),
            "S2_cpg16_G32": (128, 512, 512, 14, 14, 3, 32, "bf16"),
            "S4_cpg32_G16": (128, 512, 512, 14, 14, 3, 16, "bf16"),
            "prepass_7x7_cpg32_kpg32": (16, 256, 256, 28, 28, 7, 8, "fp16"),
            "5x5_cpg32_kpg32_28x28": (32, 128, 128, 28, 28, 5, 4, "bf16"),
            "prepass_7x7_cpg4_kpg32_G12": (193, 48, 384, 20, 20, 7, 12, "fp16"),
            "prepass_7x7_cpg28_kpg32_G64": (8, 1792, 2048, 16, 16, 7, 64, "bf16"),
            "prepass_7x7_cpg32_kpg16_G200_16x16": (
                4,
                6400,
                3200,
                16,
                16,
                7,
                200,
                "fp16",
            ),
            "3x3_cpg32_kpg8_G2_16x16": (128, 64, 16, 16, 16, 3, 2, "bf16"),
            "3x3_cpg32_kpg8_G7_12x12": (96, 224, 56, 12, 12, 3, 7, "bf16"),
            "1x1_cpg32_kpg8_G2": (144, 64, 16, 14, 14, 1, 2, "bf16"),
            "1x1_cpg24_kpg24_G2": (141, 48, 48, 56, 56, 1, 2, "fp16"),
            "3x3_cpg32_kpg8_21x21_G32": (8, 1024, 256, 21, 21, 3, 32, "bf16"),
            "7x7_cpg16_kpg16_G2": (128, 32, 32, 14, 14, 7, 2, "bf16"),
        }
        for label, shape in cases.items():
            with self.subTest(case=label):
                self.assertEqual(
                    self._pick(self._req_from(shape))[0], self._DIRECT, label
                )

    def test_4c_row_needs_its_grid_floor(self):
        from dispatch.grouped_convolution import _DIRECT_DGRAD_4C_MIN_GRID

        # S1: (56/4) * (32/16) * 128 workgroups -> the batched 4x4x4 kernel.
        _name, r = self._pick(self._grouped(N=128, Hi=56, Wi=56))
        s = r.spec
        self.assertEqual((s.variant, s.block_q, s.block_groups), ("4c", 4, 16))
        self.assertTrue(s.fused_weights)
        self.assertEqual(r.block, (64, 1, 1))
        self.assertEqual(r.grid, (14, 2, 128))
        # One image: 28 workgroups, below the floor -> the generic kernel.
        name, r = self._pick(self._grouped(N=1, Hi=56, Wi=56))
        self.assertEqual(name, self._DIRECT)
        self.assertEqual(r.spec.variant, "generic")
        self.assertLess(14 * 2 * 1, _DIRECT_DGRAD_4C_MIN_GRID)
        # 4c needs groups % 16 == 0 and a 1x1/3x3 filter.
        _name, r = self._pick(self._grouped(N=256, C=32, K=32, G=8))
        self.assertEqual(r.spec.variant, "generic")

    def test_vec_size_c_is_ignored_and_reported(self):
        r = dispatch_conv_grouped(self._grouped(vec_size_c=8))
        self.assertEqual(r.candidate.name, self._DIRECT)
        self.assertTrue(any("vec_size_c=8 ignored" in e for e in r.explanation))

    def test_other_arches_never_see_direct(self):
        direct = next(
            c for c in conv_grouped_candidates("dgrad") if c.name == self._DIRECT
        )
        for arch in ("gfx942", "gfx1250"):
            ok, why = direct.admits(self._grouped(arch=arch))
            self.assertFalse(ok)
            self.assertIn("capability", why)

    # ---- spatial policy -----------------------------------------------------

    def test_block_h_policy(self):
        # Small grid (few wave columns) -> tile H; tiny H -> never tile;
        # large filter -> tile; very tall image -> tile even when the grid is big.
        def bh(**kw):
            return self._pick(self._grouped(**kw))[1].spec.block_h

        # wave columns = ceil(W/16) * G * N; target 3072 (2048 when H <= 16).
        self.assertEqual(bh(N=1, C=128, K=128, Hi=56, Wi=56), 4)  # 128 * 7 rows
        self.assertEqual(bh(N=4, C=128, K=128, Hi=56, Wi=56), 8)  # 512 * 7
        self.assertEqual(bh(N=128, C=128, K=128, Hi=56, Wi=56), 0)  # 16384
        self.assertEqual(bh(N=32, C=512, K=512, Hi=14, Wi=14), 8)  # 1024 * 2
        self.assertEqual(bh(N=16, C=512, K=512, Hi=14, Wi=14), 4)  # 512 * 2
        self.assertEqual(bh(N=64, C=512, K=512, Hi=14, Wi=14), 0)  # 2048
        self.assertEqual(bh(N=128, Hi=7, Wi=7), 0)
        self.assertEqual(bh(N=128, Hi=8, Wi=8), 0)
        self.assertEqual(bh(N=128, Hi=14, Wi=14, Y=5, X=5, pad_h=2, pad_w=2), 4)
        # (cpg 8: the cpg = kpg = 4 shapes above 300 workgroups take the 4c
        # row, which streams whole columns.)
        self.assertEqual(bh(N=128, C=256, K=256, Hi=96, Wi=96), 8)

    def test_short_image_block_h_policy(self):
        # H <= 16: whole image unless tiling buys enough extra waves to pay
        # for the near-empty last tile and the re-loaded halo rows.
        # (N, C, K, H, W, G) -> block_h
        cases = {
            "10x10_G8_cpg32_kpg8": ((128, 256, 64, 10, 10, 8), 0),
            "9x9_G4_cpg32": ((200, 128, 128, 9, 9, 4), 0),
            "9x9_G8_cpg16": ((200, 128, 128, 9, 9, 8), 0),
            "10x10_G4_cpg32_N256": ((256, 128, 128, 10, 10, 4), 0),
            "9x9_G4_cpg32_kpg16": ((200, 128, 64, 9, 9, 4), 0),
            "12x12_G8_cpg32_N128": ((128, 256, 256, 12, 12, 8), 0),
            "16x16_G4_cpg32_kpg8": ((128, 128, 32, 16, 16, 4), 8),
            "14x14_G2_cpg32_N128": ((128, 64, 64, 14, 14, 2), 4),
            "14x14_G4_cpg16_kpg8": ((128, 64, 32, 14, 14, 4), 4),
            "16x16_G2_cpg32_N128": ((128, 64, 64, 16, 16, 2), 4),
        }
        for label, ((n, c, k, h, w, g), want) in cases.items():
            with self.subTest(case=label):
                name, r = self._pick(self._grouped(N=n, C=c, K=k, Hi=h, Wi=w, G=g))
                self.assertEqual(name, self._DIRECT, label)
                self.assertEqual(r.spec.block_h, want, label)

    def test_block_q_policy(self):
        # 32-wide strips need H tiling, no extra W padding, enough channels
        # per wave (>= 16, or >= 8 with a 5x5/7x7) and >= 768 waves after
        # halving the strip count.
        def bq(**kw):
            return self._pick(self._grouped(**kw))[1].spec.block_q

        self.assertEqual(bq(N=16, C=1024, K=128, Hi=28, Wi=28), 32)  # cpg 32
        self.assertEqual(bq(N=128, C=512, K=512, Hi=14, Wi=14), 16)  # untiled
        self.assertEqual(bq(N=8, C=256, K=256, Hi=28, Wi=28), 16)  # cpg 8
        self.assertEqual(
            bq(N=8, C=256, K=256, Hi=28, Wi=28, Y=7, X=7, pad_h=3, pad_w=3), 32
        )
        self.assertEqual(
            bq(N=32, C=512, K=512, Hi=40, Wi=40), 16
        )  # pads 40 to 64, not 48
        # G=3: one 32-wide strip per row tile leaves 3 * 32 * 7 = 672 waves.
        self.assertEqual(bq(N=32, C=72, K=72, Hi=28, Wi=28, G=3), 16)

    def test_big_filter_halves_block_groups(self):
        def bg(**kw):
            return self._pick(self._grouped(**kw))[1].spec.block_groups

        big = {"Y": 5, "X": 5, "pad_h": 2, "pad_w": 2}
        self.assertEqual(bg(N=32, C=256, K=256, Hi=28, Wi=28), 4)  # cpg 8
        self.assertEqual(bg(N=32, C=256, K=256, Hi=28, Wi=28, **big), 2)
        self.assertEqual(bg(N=32, C=512, K=512, Hi=28, Wi=28, **big), 1)  # cpg 16

    def test_block_groups_divides_groups(self):
        # kpg <= 8 wants block_groups=4; G=2 forces it down to 2.
        # N=128 keeps both inside the admitted region.
        _name, r = self._pick(self._grouped(N=128, C=8, K=8, G=2))
        self.assertEqual(r.spec.block_groups, 2)
        _name, r = self._pick(self._grouped(N=128, C=24, K=24, G=3))
        self.assertEqual(r.spec.block_groups, 1)

    # ---- launch plan / grid contract ---------------------------------------

    def test_grid_and_block_match_the_launch_plan(self):
        for kw in (
            {},
            {"N": 1, "Hi": 56, "Wi": 56},
            {"N": 32, "C": 512, "K": 512, "G": 16},
        ):
            with self.subTest(kw=kw):
                req = self._grouped(**kw)
                _name, r = self._pick(req)
                plan = r.spec.launch_plan(req)
                self.assertEqual(r.grid, plan.main.grid)
                self.assertEqual(r.block, plan.main.block)
                p = _problem(req)
                # Independent re-derivation of the main grid.
                h_tiles = math.ceil(p.Hi / r.spec.block_h) if r.spec.block_h else 1
                self.assertEqual(
                    r.grid,
                    (
                        math.ceil(p.Wo / r.spec.block_q),
                        p.groups // r.spec.block_groups,
                        p.N * h_tiles,
                    ),
                )
                self.assertEqual(r.block, (r.spec.block_groups * 64, 1, 1))

    def test_plan_fused_path_is_one_kernel_without_workspace(self):
        req = self._grouped()
        r = dispatch_conv_grouped(req)
        plan = r.spec.launch_plan(req)
        self.assertEqual([s.role for s in plan.stages], ["main"])
        self.assertEqual(plan.workspace_bytes, 0)
        main = plan.main
        self.assertEqual((main.a, main.b, main.d), ("dY", "W", "dX"))
        # The main spec is the transposed problem: channels swapped.
        p = _problem(req)
        fp = main.spec.problem
        self.assertEqual((fp.cpg, fp.kpg), (p.kpg, p.cpg))
        self.assertTrue(main.spec.dgrad_fused_weights)
        self.assertTrue(any("pipeline=main" in e for e in r.explanation))

    def test_plan_falls_back_to_the_pre_pass_over_the_register_budget(self):
        # 7x7 with cpg = kpg = 32: the fused weight fragments exceed their
        # register budget, so the transpose pre-pass pipeline runs instead.
        req = self._req_from((16, 256, 256, 28, 28, 7, 8, "fp16"))
        r = dispatch_conv_grouped(req)
        self.assertEqual(r.candidate.name, self._DIRECT)
        self.assertFalse(r.spec.fused_weights)
        plan = r.spec.launch_plan(req)
        self.assertEqual([s.role for s in plan.stages], ["transpose", "main"])
        p = _problem(req)
        self.assertEqual(plan.workspace_bytes, p.C * p.Y * p.X * p.kpg * 2)
        self.assertEqual(plan.main.b, "ws_wt")
        self.assertTrue(any("pipeline=transpose+main" in e for e in r.explanation))

    def test_kernel_name_separates_knobs(self):
        from dataclasses import replace

        spec = dispatch_conv_grouped(self._grouped()).spec
        names = {
            spec.kernel_name(),
            replace(spec, block_h=spec.block_h + 8).kernel_name(),
            replace(spec, block_groups=1).kernel_name(),
            replace(spec, fold_k32=True).kernel_name(),
            replace(spec, runtime_k_loop=True).kernel_name(),
            replace(spec, fused_weights=False, weights_lds=False).kernel_name(),
            replace(spec, weights_lds=False).kernel_name(),
            replace(spec, waves_per_eu=spec.waves_per_eu + 2).kernel_name(),
            replace(spec, variant="4c").kernel_name(),
        }
        self.assertEqual(len(names), 9)

    def test_rule_table_has_catch_all_last(self):
        from dispatch.grouped_convolution import GFX950_DIRECT_DGRAD_RULES

        self.assertTrue(GFX950_DIRECT_DGRAD_RULES[-1].applies(4, 28))
        for rule in GFX950_DIRECT_DGRAD_RULES:
            self.assertIn(rule.block_groups, (1, 2, 4, 8, 16))


class TestGfx1250WgradKOuterReachable(unittest.TestCase):
    """The gfx1250 wgrad candidate must actually enable the K-outer layout.

    ``WgradConvSpec.default_lds_k_outer`` returns True for every fp16/bf16
    gfx1250 wgrad request (wave32, 16x16 atom edge), but the candidate used to
    never ask -- so the headline transpose-read path was unreachable through
    library dispatch and was exercised only by the sweep driver and the
    direct-build tests.
    """

    def _spec(self, dtype="fp16"):
        return dispatch_conv_grouped(_wgrad("gfx1250", G=4, dtype=dtype)).spec

    def test_dispatch_spec_enables_k_outer(self):
        for dtype in ("fp16", "bf16"):
            self.assertTrue(
                self._spec(dtype).lds_k_outer,
                f"gfx1250 wgrad dispatch must enable lds_k_outer for {dtype}",
            )

    def test_decision_survives_into_the_instance_spec(self):
        r = dispatch_conv_grouped(_wgrad("gfx1250", G=4))
        inst = r.spec.to_wgrad_spec(_problem(r.request))
        self.assertTrue(inst.lds_k_outer)
        inst.validate()

    def test_agrees_with_the_selection_policy(self):
        # Dispatch must not hand-roll the gate; it must match the one policy
        # function the sweep driver also calls.
        from rocke.core.arch import ArchTarget
        from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

        spec = self._spec()
        self.assertEqual(
            spec.lds_k_outer,
            WgradConvSpec.default_lds_k_outer(
                arch="gfx1250",
                dtype_a="fp16",
                dtype_b="fp16",
                warp_tile_m=spec.warp_tile_mn,
                warp_tile_n=spec.warp_tile_mn,
                wave_size=ArchTarget.from_gfx("gfx1250").wave_size,
            ),
        )


class TestGroupedSpecKernelNameDistinguishesBody(unittest.TestCase):
    """Dispatch kernel names must separate specs that emit different bodies.

    This is the layer whose names key the host-side compile cache, so two specs
    that lower differently sharing one name is a cache-collision bug, not a
    cosmetic one.
    """

    def test_k_outer_changes_the_name(self):
        from dispatch.grouped_convolution import ConvGroupedSpec

        base = dispatch_conv_grouped(_wgrad("gfx950", G=4)).spec
        from dataclasses import replace

        on = replace(base, lds_k_outer=True)
        off = replace(base, lds_k_outer=False)
        self.assertNotEqual(
            on.kernel_name(),
            off.kernel_name(),
            "lds_k_outer changes the LDS tile shape and operand fetch",
        )
        self.assertIn("kouter", on.kernel_name())
        assert isinstance(base, ConvGroupedSpec)

    def test_ws_replicas_changes_the_instance_name(self):
        from dataclasses import replace as _replace

        r = dispatch_conv_grouped(_wgrad("gfx950", C=3, K=24, Y=3, X=3, dtype="bf16"))
        ws = r.spec.to_wgrad_spec(_problem(r.request))
        self.assertNotEqual(
            ws.kernel_name(),
            _replace(ws, ws_replicas=ws.ws_replicas + 1).kernel_name(),
            "ws_replicas changes the scratch addressing, so it must reach the "
            "name the compile cache keys on",
        )


if __name__ == "__main__":
    unittest.main()
