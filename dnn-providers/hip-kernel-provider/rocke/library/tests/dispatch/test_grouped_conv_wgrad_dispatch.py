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


class TestLaunchContractMatchesKernel(unittest.TestCase):
    """The dispatcher's signature and launch values must describe the kernel
    ``to_wgrad_spec`` builds and the grid ``_wgrad_grid`` launches.

    Both go through the split-K resolver, so the raw ``spec.split_k`` (-1 on
    the auto path) and the spec's (absent) two-stage flag must not leak into
    the kernargs: kernargs pack positionally, and a mismatch launches.
    """

    def _check(self, req):
        from dispatch.grouped_convolution import launch_values_for
        from kernels.common.conv_args import ConvArgs

        r = dispatch_conv_grouped(req)
        p = _problem(r.request)
        ws = r.spec.to_wgrad_spec(p)
        abi = ConvArgs.from_problem(
            p,
            direction="wgrad",
            tile_m=ws.tile_m,
            tile_n=ws.tile_n,
            tile_k=ws.tile_k,
        ).arg_names(two_stage=ws.two_stage)
        self.assertEqual([a["name"] for a in r.signature], [n for n, _ in abi])

        ws_kw = dict(ws_ptr=0x9000, ws_bytes=64) if ws.two_stage else {}
        values = launch_values_for(
            r.request,
            r.spec,
            A_ptr=0x1000,
            B_ptr=0x2000,
            D_ptr=0x3000,
            A_bytes=1,
            B_bytes=1,
            D_bytes=1,
            **ws_kw,
        )
        self.assertEqual(values["ks_count"], ws.split_k)
        self.assertEqual(r.grid[2], p.groups * ws.split_k)
        return ws

    def test_auto_split_k(self):
        self._check(_wgrad("gfx942", G=4))

    def test_odd_wg_N_two_stage(self):
        # Odd wg_N with a 16-bit dW cannot use the packed atomic, so split-K
        # resolves to the two-stage path and its scratch pair joins the ABI.
        ws = self._check(_wgrad("gfx950", C=3, K=24, Y=3, X=3, dtype="bf16"))
        if ws.split_k > 1:
            self.assertTrue(ws.two_stage)


if __name__ == "__main__":
    unittest.main()
