# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the dgrad ``pp_boxes`` 8 planner (kernels/conv/conv_dgrad_box_plan):
every dX pixel in exactly one record, each record's tap box holds its pixels' taps,
the launch within the CU budget, deterministic, and a planned makespan no longer than
that of the ``pp_boxes`` 7 packing."""

from __future__ import annotations

import unittest
import warnings

import kernels.conv.conv_dgrad_box_plan as P
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec, pp_box_sub_gemms

# Only the MILP classes need scipy (>= 1.9: scipy.optimize.milp); the greedy
# twin and the fallback are tested everywhere.
_needs_milp = unittest.skipUnless(P.HAVE_MILP, "scipy.optimize.milp is not available")


def _p3(D=18, H=10, W=10, C=1024, K=1024):
    return ConvProblem(
        N=1, Di=D, Hi=H, Wi=W, C=C, K=K, Z=3, Y=3, X=3, sD=1, pD=0, dD=1, pH=0, pW=0
    )


def _need(c, n_out, filt):
    return {f for f in range(filt) if 0 <= c - f <= n_out - 1}


def _blocks_makespan(p, sgs, tile_k, cus=228):
    import heapq

    kt = -(-p.kpg // tile_k)
    h = [(0, i) for i in range(cus)]
    mk = 0
    for sg in sgs:
        span = sg.y_dot_slice * sg.x_dot_slice * sg.gemm_k_padded * kt
        S = max(1, sg.b_x_stride)
        per = -(-span // S)
        for i in range(sg.block_end - sg.block_start):
            ln = max(0, min(per, span - (i % S) * per))
            t0, c = heapq.heappop(h)
            heapq.heappush(h, (t0 + ln, c))
            mk = max(mk, t0 + ln)
    return mk


@_needs_milp
class TestPlannedBoxes(unittest.TestCase):
    def _check(self, p, tile_m, tile_n, tile_k, split, cus):
        sgs = pp_box_sub_gemms(
            p, tile_m, tile_n, 8, tile_k=tile_k, max_split=split, cus=cus
        )
        seen = []
        nt = -(-p.cpg // tile_n)
        start = 0
        for sg in sgs:
            self.assertIsNotNone(sg.row_list)
            self.assertEqual(sg.gemm_m, len(sg.row_list))
            self.assertEqual(sg.block_start, start)
            self.assertEqual(
                sg.block_end - sg.block_start,
                -(-sg.gemm_m // tile_m) * nt * sg.b_x_stride,
            )
            self.assertLessEqual(sg.b_x_stride, split)
            start = sg.block_end
            zlo, zs = sg.b_y_stride, sg.gemm_k_padded
            ylo, ys, xlo, xs = (
                sg.b_y_offset,
                sg.y_dot_slice,
                sg.b_x_offset,
                sg.x_dot_slice,
            )
            for pix in sg.row_list:
                d, r = divmod(pix, p.Hi * p.Wi)
                h, w = divmod(r, p.Wi)
                self.assertTrue(_need(d, p.Do, p.Z) <= set(range(zlo, zlo + zs)))
                self.assertTrue(_need(h, p.Ho, p.Y) <= set(range(ylo, ylo + ys)))
                self.assertTrue(_need(w, p.Wo, p.X) <= set(range(xlo, xlo + xs)))
                seen.append(pix)
        self.assertEqual(sorted(seen), list(range(p.Di * p.Hi * p.Wi)))
        self.assertLessEqual(start, cus)
        return sgs

    def test_s10(self):
        p = _p3()
        sgs = self._check(p, 128, 256, 64, 8, 228)
        old = pp_box_sub_gemms(p, 128, 256, 7, tile_k=64, max_split=8, cus=228)
        self.assertLess(_blocks_makespan(p, sgs, 64), _blocks_makespan(p, old, 64))
        # deterministic
        again = pp_box_sub_gemms(p, 128, 256, 8, tile_k=64, max_split=8, cus=228)
        self.assertEqual([s.row_list for s in sgs], [s.row_list for s in again])

    def test_s7_like(self):
        p = _p3(D=34, H=18, W=18, C=512, K=512)
        sgs = self._check(p, 256, 256, 32, 3, 228)
        old = pp_box_sub_gemms(p, 256, 256, 77, tile_k=32, max_split=3, cus=228)
        self.assertLessEqual(_blocks_makespan(p, sgs, 32), _blocks_makespan(p, old, 32))

    def test_unsplit(self):
        p = _p3(D=10, H=6, W=6, C=256, K=128)
        sgs = self._check(p, 128, 256, 64, 1, 228)
        self.assertTrue(all(sg.b_x_stride == 1 for sg in sgs))

    def test_label_and_validity(self):
        from kernels.common.conv_implicit_gemm import ConvDataSpec
        from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

        spec = DgradConvSpec(
            problem=_p3(C=256, K=128),
            data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
            tile_m=128,
            tile_n=256,
            tile_k=64,
            warp_m=2,
            warp_n=4,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            lds_k_outer=False,
            tap_outer_k=True,
            tapk_prefetch=1,
            b_kpack=4,
            lds_xor=True,
            lds_k_pad=0,
            tapk_pingpong=4,
            tapk_pp_sched=2,
            pp_boxes=8,
            split_k=8,
        )
        ok, why = is_valid_dgrad_spec(spec, "gfx942")
        self.assertTrue(ok, why)
        self.assertIn("pbl", spec.kernel_name())  # the pp_boxes 7 binary


def _check_plan(tc, P, p, tm, ms, plan):
    """Every pixel in exactly one box holding its taps; tight tile counts."""
    cl = P.pixel_classes(p)
    owner = {pix: c for c, v in cl.items() for pix in v}
    seen = []
    for box, tiles, split, rows in plan:
        tc.assertTrue(rows)
        tc.assertEqual(tiles, -(-len(rows) // tm))
        tc.assertLessEqual(split, ms)
        for r in rows:
            tc.assertTrue(P._contains(box, owner[r]))
        seen += rows
    tc.assertEqual(sorted(seen), sorted(owner))


class TestGreedySolver(unittest.TestCase):
    """The pure-Python twin (asked for explicitly) yields valid plans."""

    def test_greedy_valid(self):
        for p, tm, tk, ms, cus in (
            (_p3(), 128, 64, 8, 228),
            (_p3(D=34, H=18, W=18, C=512, K=512), 256, 32, 3, 456),
        ):
            _T, plan = P.plan_boxes(p, tm, 256, tk, ms, cus, solver="greedy")
            _check_plan(self, P, p, tm, ms, plan)

    def test_unknown_solver(self):
        with self.assertRaises(ValueError):
            P.plan_boxes(_p3(), 128, 256, 64, 8, 228, solver="lp")


class TestPlanEnvironment(unittest.TestCase):
    """The plan is a function of the problem, tile, budget and solver only."""

    def test_no_milp_falls_back_to_pp7(self):
        # Without scipy's MILP no plan is made (never the greedy one): the
        # launch runs the pp_boxes 7 records, and a warning says so.
        old = P.HAVE_MILP
        try:
            P.HAVE_MILP = False
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.assertIsNone(P.plan_boxes(_p3(), 128, 256, 64, 8, 228))
            self.assertTrue(any("scipy" in str(x.message) for x in w))
            p = _p3(D=10, H=6, W=6, C=256, K=128)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = pp_box_sub_gemms(p, 128, 256, 8, tile_k=64, max_split=4, cus=228)
            want = pp_box_sub_gemms(p, 128, 256, 7, tile_k=64, max_split=4, cus=228)
            self.assertEqual(
                [(s.row_list, s.b_x_stride, s.block_end) for s in got],
                [(s.row_list, s.b_x_stride, s.block_end) for s in want],
            )
        finally:
            P.HAVE_MILP = old

    def test_module_reads_no_environment(self):
        import inspect

        src = inspect.getsource(P)
        self.assertNotIn("os.environ", src)
        self.assertNotIn("getenv", src)

    def test_tight_drops_slack(self):
        b1, b2 = (0, 2, 0, 2, 0, 2), (0, 1, 0, 2, 0, 2)
        c = (0, 1, 0, 1, 0, 1)
        plan = [(b1, 3, [(c, 130)]), (b2, 1, [(c, 0)])]
        self.assertEqual(P._tight(plan, 128), [(b1, 2, [(c, 130)])])

    @_needs_milp
    def test_milp_plans_are_tight(self):
        p = _p3(D=10, H=6, W=6, C=256, K=128)
        _T, plan = P.plan_boxes(p, 128, 256, 64, 4, 228)
        _check_plan(self, P, p, 128, 4, plan)

    @_needs_milp
    def test_node_budget_bounds_the_plan(self):
        # The MILPs of one plan share PLAN_NODE_BUDGET: a spent budget keeps
        # the best plan so far (still a valid cover), deterministically.
        p = _p3(D=10, H=6, W=6, C=256, K=128)
        old = P.PLAN_NODE_BUDGET
        try:
            P.PLAN_NODE_BUDGET = 1
            P._plan_cached.cache_clear()
            a = P.plan_boxes(p, 128, 256, 64, 4, 228)
            P._plan_cached.cache_clear()
            b = P.plan_boxes(p, 128, 256, 64, 4, 228)
            self.assertEqual(a, b)
            if a is not None:
                _check_plan(self, P, p, 128, 4, a[1])
        finally:
            P.PLAN_NODE_BUDGET = old
            P._plan_cached.cache_clear()
