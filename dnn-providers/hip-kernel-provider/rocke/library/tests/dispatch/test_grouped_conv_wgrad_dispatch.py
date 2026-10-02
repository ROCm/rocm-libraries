# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection + support + grid tests for grouped wgrad dispatch.

CPU-only (no GPU / no comgr): asserts that the grouped-convolution dispatcher
admits grouped backward-weight requests, and that the launch grid it derives
matches the kernel's block_id_z contract --

    grid = (ceil(wg_N / tile_n), ceil(wg_M / tile_m), (groups / Gm) * split_k)

with the per-tile dims wg_M = kpg * Gm, wg_N = spatial * cpg * Gm. ``Gm`` is the
merged-group degree: it folds Gm consecutive groups into one tile, so it
multiplies both GEMM extents and divides the group axis of the grid. It is 1
for everything except gfx950 depthwise, where dispatch picks it -- see
``TestWgradMergeDegree``. This is the same grid the GPU correctness test (platform
tests ``test_conv_wgrad_correctness.py``) launches and validates numerically, so
a match here proves the dispatch path launches a correct grid.
"""

from __future__ import annotations

import math
import unittest

from dispatch.grouped_convolution import (
    ConvGroupedRequest,
    _block,
    _problem,
    # Deliberately the alias dispatch itself calls, not a fresh import from
    # kernels: the test asserts on exactly the predicate dispatch consults.
    _wgrad_atomic_epilogue_available,
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
    # Mirror dispatch._wgrad_grid: per-tile tiling on x/y, and z = (groups/Gm)
    # * split_k with the merged group riding block_id_z alongside the K-slice.
    # split_k == -1 is the auto sentinel; resolve it via the same CK formula
    # the grid uses so this stays an independent re-derivation of the wiring.
    p = _problem(req)
    spatial = (p.Z if p.is_3d else 1) * p.Y * p.X
    gm = max(1, getattr(spec, "group_merge", 1))
    # Merging folds Gm groups into one tile, so it scales both GEMM extents
    # and shrinks the group axis by the same factor.
    wg_M = (p.K // p.groups) * gm
    wg_N = spatial * (p.C // p.groups) * gm
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
            groups=p.groups // gm,
            block_size=_block(spec)[0],
        ).split_k
    return (gx, gy, (p.groups // gm) * split_k)


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

    def test_merged_split_k_never_takes_the_packed_atomic(self):
        # Merging makes two_stage a *correctness* requirement rather than a
        # performance choice, and it overrides atomic availability. A merged
        # tile computes a Gm x Gm block of group *pairs* and wants only the
        # diagonal, which the packed-atomic epilogue has no way to mask off.
        # Taking it anyway would accumulate an off-diagonal pair's partial sum
        # into a live dW element -- wrong gradients, and nothing raises.
        #
        # Today the `gm > 1` clause in _resolve_wgrad_split_k is *subsumed*, and
        # this test deliberately does not pretend otherwise. Dispatch emits only
        # fp16/bf16, and merging is depthwise-only, so cpg == 1 forces a
        # store-vector width of 1 and wgrad_atomic_epilogue_available already
        # returns False on every shape that can merge -- `not atomic_ok` carries
        # the invariant unaided. The clause is a belt kept for the case that
        # stops being true (an fp32 dW, or merging extended past depthwise),
        # where it becomes the only thing standing between a merged tile and a
        # silently wrong gradient. What is asserted below is the invariant
        # itself, which holds either way; `atomic_ok` is asserted False so that
        # if the subsumption ever lifts, this test says so out loud instead of
        # quietly changing meaning.
        seen_merged = 0
        for G, Y, X in ((256, 3, 3), (128, 3, 3), (512, 1, 3), (64, 5, 5)):
            for dtype in ("fp16", "bf16"):
                r = dispatch_conv_grouped(
                    _wgrad(
                        "gfx950", C=G, K=G, G=G, Y=Y, X=X,
                        pad_h=Y // 2, pad_w=X // 2, dtype=dtype,
                    )
                )
                p = _problem(r.request)
                ws = r.spec.to_wgrad_spec(p)
                if ws.group_merge <= 1 or ws.split_k <= 1:
                    continue
                seen_merged += 1
                where = f"G={G} {Y}x{X} {dtype} gm={ws.group_merge} split_k={ws.split_k}"
                self.assertTrue(
                    ws.two_stage, f"{where}: merged split-K must be two-stage"
                )
                atomic_ok, _ = _wgrad_atomic_epilogue_available(p, dtype, None)
                self.assertFalse(
                    atomic_ok,
                    f"{where}: the packed atomic became available on a mergeable "
                    f"shape -- the `gm > 1` clause in _resolve_wgrad_split_k is no "
                    f"longer subsumed and is now the sole guard; re-read it",
                )
        self.assertGreater(seen_merged, 0, "no merged split-K case was exercised")


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
        cands = [c for c in conv_grouped_candidates("dgrad") if c.admits(req)[0]]
        self.assertEqual(len(cands), 1, f"expected exactly one candidate: {cands}")
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


class TestWgradMergeDegree(unittest.TestCase):
    """gfx950 depthwise wgrad must pick a ``Gm`` the kernel accepts.

    Depthwise wgrad is the degenerate wgrad GEMM: wg_M = kpg = 1 and
    wg_N = Y*X*cpg = Y*X, so the free axis is one element per group and every
    load is scalar. Merging Gm groups into one tile is what restores a
    vectorisable run, and the degree is bounded by ``spatial * Gm <= tile_n``.

    The tile is *not* a free variable: the selector keeps the shipped tile and
    chooses only the degree, so these tests assert ``tile_n == 64`` everywhere
    alongside the degree.

    The sharp edge: at ``split_k == -1`` the gate's split-K clause is vacuous,
    so ``support()`` validates little more than that tile bound. A degree the
    gate refuses does not merely merge badly -- it makes ``support()`` reject,
    and depthwise wgrad stops dispatching at all. Hence
    :meth:`test_every_choice_is_admissible`, the load-bearing one here.
    """

    def _dw(self, G, Y, X, arch="gfx950", **kw):
        # Depthwise is C == K == groups (cpg == kpg == 1).
        return _wgrad(
            arch, C=G, K=G, G=G, Y=Y, X=X, pad_h=Y // 2, pad_w=X // 2, **kw
        )

    # ---- admissibility -------------------------------------------------------

    def test_every_choice_is_admissible(self):
        from kernels.common.conv_implicit_gemm_wgrad import (
            wgrad_group_merge_available,
        )

        n = 0
        for G in (8, 32, 96, 240, 672, 2048):
            # No 1x1: grouped pointwise wgrad is refused upstream of dispatch.
            for Y, X in ((1, 3), (3, 3), (5, 5), (7, 7), (11, 11), (3, 5)):
                for dtype in ("fp16", "bf16"):
                    r = dispatch_conv_grouped(self._dw(G, Y, X, dtype=dtype))
                    inst = r.spec.to_wgrad_spec(_problem(r.request))
                    ok, why = wgrad_group_merge_available(inst, arch="gfx950")
                    self.assertTrue(
                        ok, f"G={G} {Y}x{X} {dtype} -> tile_n={r.spec.tile_n} "
                            f"gm={r.spec.group_merge}: {why}"
                    )
                    inst.validate()
                    n += 1
        self.assertGreater(n, 0)

    def test_merging_shrinks_the_group_axis_of_the_grid(self):
        r = dispatch_conv_grouped(self._dw(256, 3, 3))
        self.assertGreater(r.spec.group_merge, 1, "3x3 depthwise must merge")
        self.assertEqual(r.grid, _expected_grid(r.request, r.spec))
        self.assertEqual(r.grid[2] % (256 // r.spec.group_merge), 0)

    # ---- blast radius --------------------------------------------------------

    def test_non_depthwise_is_untouched(self):
        # cpg/kpg > 1 shapes never merge: the kernel gate forbids it.
        for C, K, G in ((64, 64, 4), (64, 64, 1), (256, 128, 8), (64, 128, 32)):
            r = dispatch_conv_grouped(_wgrad("gfx950", C=C, K=K, G=G))
            self.assertEqual(r.spec.group_merge, 1, f"C={C} K={K} G={G}")
            self.assertEqual(r.spec.tile_n, 64, f"C={C} K={K} G={G}")

    def test_other_arches_are_untouched(self):
        # Merging is a gfx950 candidate decision; nothing else started merging.
        for arch in ("gfx942", "gfx1250"):
            r = dispatch_conv_grouped(self._dw(256, 3, 3, arch=arch))
            self.assertEqual(r.spec.group_merge, 1, arch)

    def test_a_single_group_never_merges(self):
        # C == K == G == 1 looks depthwise by the cpg/kpg test but has nothing
        # to merge. (The tile is not asserted: this is also the scalar-B case,
        # which pins its own narrower geometry.)
        r = dispatch_conv_grouped(_wgrad("gfx950", C=1, K=1, G=1))
        self.assertEqual(r.spec.group_merge, 1)

    # ---- the degree table ----------------------------------------------------

    def test_degree_table(self):
        # Group count is a power of two well above every ladder rung, so these
        # cases isolate the tile bound (spatial * Gm <= 64) from divisibility.
        # The degree falls as the filter grows and reaches 1 once a single pair
        # of groups no longer fits the tile -- which is also the point where
        # merging stops being offered at all.
        from dispatch.grouped_convolution import _wgrad_merge_degree

        cases = {
            (1, 3): 16,  # spatial 3: 3*16 = 48 <= 64, 3*32 > 64
            (1, 7): 8,  # spatial 7: 7*8 = 56 <= 64
            (3, 3): 4,  # spatial 9: 9*4 = 36 <= 64, 9*8 > 64
            (3, 5): 4,  # spatial 15: 15*4 = 60 <= 64
            (5, 5): 2,  # spatial 25: 25*2 = 50 <= 64
            (7, 7): 1,  # spatial 49: 49*2 > 64, nothing merges
            (11, 11): 1,  # spatial 121
            (13, 13): 1,  # spatial 169
        }
        for (Y, X), want in cases.items():
            req = self._dw(2048, Y, X)
            self.assertEqual(_wgrad_merge_degree(req), want, f"{Y}x{X}")
            spec = dispatch_conv_grouped(req).spec
            self.assertEqual(spec.group_merge, want, f"{Y}x{X}")
            # The tile is never spent on merging -- that was fitted and lost.
            self.assertEqual(spec.tile_n, 64, f"{Y}x{X}")

    def test_degree_is_ladder_capped_not_only_tile_capped(self):
        # A 1x1 filter would fit 64 groups in the 64-wide tile and still have
        # room; the degree stops at 64 because that is the last rung the kernel
        # offers. (Checked on the selector directly: grouped pointwise wgrad is
        # refused upstream, so this shape never reaches dispatch.)
        from dispatch.grouped_convolution import _wgrad_merge_degree

        self.assertEqual(_wgrad_merge_degree(self._dw(2048, 1, 1)), 64)

    def test_degree_respects_group_divisibility(self):
        # 1x3 leaves room for 16 under the tile bound, so below that the only
        # thing left to cap the degree is what divides the group count.
        from dispatch.grouped_convolution import _wgrad_merge_degree

        for G, want in ((2048, 16), (96, 16), (24, 8), (12, 4), (6, 2), (3, 1)):
            self.assertEqual(_wgrad_merge_degree(self._dw(G, 1, 3)), want, f"G={G}")

    def test_unmergeable_shapes_dispatch_unmerged(self):
        # A filter too large for two groups to share a tile, and a group count
        # no ladder rung divides, both fall back to the shipped unmerged spec.
        for req in (self._dw(2048, 13, 13), self._dw(3, 3, 3)):
            spec = dispatch_conv_grouped(req).spec
            self.assertEqual(spec.group_merge, 1)
            self.assertEqual(spec.tile_n, 64)

    # ---- the name the compile cache keys on ----------------------------------

    def test_merge_degree_reaches_the_kernel_name(self):
        from dataclasses import replace

        base = dispatch_conv_grouped(self._dw(2048, 3, 3)).spec
        self.assertIn(f"gm{base.group_merge}", base.kernel_name())
        self.assertNotEqual(
            base.kernel_name(),
            replace(base, group_merge=1).kernel_name(),
            "group_merge is a different GEMM -- it must not share a name",
        )


if __name__ == "__main__":
    unittest.main()
