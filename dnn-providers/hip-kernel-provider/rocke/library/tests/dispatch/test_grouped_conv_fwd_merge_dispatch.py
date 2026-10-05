# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection + grid tests for the merged-groups forward conv candidate.

CPU-only (no GPU / no comgr). Three things are asserted, and they are three
different bugs:

1. **Routing** -- a depthwise gfx950 request reaches
   ``implicit_gemm_conv_dw_merged`` and everything else falls through to the
   plain fwd candidate. Priority is an ascending sort in
   ``CandidateRegistry.candidates()``, so "wins" means a *smaller* number; a
   candidate registered on the wrong side of 10 admits perfectly and is never
   selected, which looks like a no-op rather than a bug.
2. **Host/device grid agreement** -- ``_fwd_grid`` and the kernel's own
   ``implicit_gemm_conv_grid`` must return the same tuple. Under merge a
   workgroup owns ``Gm`` conv groups, so a grid taken off the *true* problem
   launches ``Gm x`` redundant CTAs in z while covering only 1 of ``Gm``
   columns in x. Nothing crashes; the output is simply wrong in most of its
   channels.
3. **Buildability** -- the spec dispatch hands out must pass the emitter's own
   ``is_valid_spec``. Two copies of a gate is how the wgrad family came to have
   a dispatcher that admitted specs the builder then rejected.

The numerics live in ``TestConvFwdGroupMergeNumerics``
(``test_conv_fwd_correctness.py``); nothing here proves the redundant MACs
cancel.
"""

from __future__ import annotations

import unittest

from dispatch.grouped_convolution import (
    ConvGroupedRequest,
    _fwd_grid,
    _pick_group_merge,
    _problem,
    dispatch_conv_grouped,
)
from kernels.common.conv_implicit_gemm import implicit_gemm_conv_grid, is_valid_spec

_MERGED = "implicit_gemm_conv_dw_merged"
_PLAIN = "implicit_gemm_conv"


def _req(**kw) -> ConvGroupedRequest:
    base = dict(
        N=1,
        C=144,
        K=144,
        Hi=56,
        Wi=56,
        Y=3,
        X=3,
        G=144,
        pad_h=1,
        pad_w=1,
        dtype="bf16",
        arch="gfx950",
        direction="fwd",
    )
    base.update(kw)
    return ConvGroupedRequest(**base)


class TestGroupedConvFwdMergeDispatch(unittest.TestCase):
    def test_depthwise_selects_the_merged_candidate(self):
        res = dispatch_conv_grouped(_req())
        self.assertEqual(res.kernel_id.candidate, _MERGED)
        self.assertGreater(res.spec.group_merge, 1)
        # Forced, not derived from the true C/K: merging widens the store past
        # 1, and is_valid_spec rejects the direct epilogue once it does.
        self.assertEqual(res.spec.epilogue, "cshuffle")

    def test_merge_degree_divides_groups_and_fits_the_tile(self):
        for groups in (64, 128, 144, 192, 256, 384, 512, 768, 960, 1024, 1536):
            with self.subTest(groups=groups):
                res = dispatch_conv_grouped(_req(C=groups, K=groups, G=groups))
                gm = res.spec.group_merge
                self.assertEqual(groups % gm, 0, f"G={groups} Gm={gm}")
                self.assertLessEqual(gm, res.spec.tile_n)

    def test_fallthrough_when_merging_does_not_apply(self):
        cases = {
            # No power of two divides 3, so there is no degree to pick.
            "G=3": dict(C=3, K=3, G=3),
            # cpg/kpg > 1: the group index is not the only free axis, and the
            # diagonal argument does not hold.
            "grouped": dict(C=64, K=64, G=4, Hi=14, Wi=14),
            "dense": dict(C=64, K=64, G=1, Hi=14, Wi=14),
            # Deliberate deferral, not a corner case: the flat pointwise path
            # builds ``valid`` from scratch and has no slot for the diagonal.
            "pointwise": dict(
                C=64, K=64, G=64, Y=1, X=1, Hi=14, Wi=14, pad_h=0, pad_w=0
            ),
        }
        for tag, kw in cases.items():
            with self.subTest(case=tag):
                res = dispatch_conv_grouped(_req(**kw))
                self.assertEqual(res.kernel_id.candidate, _PLAIN)
                self.assertEqual(res.spec.group_merge, 1)

    def test_non_gfx950_does_not_reach_the_merged_candidate(self):
        # The gate is MFMA/wave64 and the tile is gfx950's 32x32x16 atom.
        res = dispatch_conv_grouped(_req(arch="gfx942"))
        self.assertNotEqual(res.kernel_id.candidate, _MERGED)
        self.assertEqual(res.spec.group_merge, 1)

    def test_host_grid_matches_the_kernels_own_grid(self):
        # The highest-severity failure mode on this path, and the one that
        # produces a plausible-looking wrong answer instead of a crash.
        for kw in (
            dict(),
            dict(N=2, C=64, K=64, G=64, Hi=14, Wi=14),
            dict(C=192, K=192, G=192, Y=7, X=7, pad_h=3, pad_w=3, dtype="fp16"),
            dict(C=3, K=3, G=3),  # unmerged fallthrough
        ):
            with self.subTest(**kw):
                req = _req(**kw)
                spec = dispatch_conv_grouped(req).spec
                fwd = spec.to_fwd_spec(_problem(req))
                self.assertEqual(_fwd_grid(spec, req), implicit_gemm_conv_grid(fwd))

    def test_grid_z_shrinks_by_the_merge_degree(self):
        # Stated independently of _fwd_grid's formula so a change to both at
        # once still has to face the invariant.
        req = _req()
        spec = dispatch_conv_grouped(req).spec
        _gx, _gy, gz = _fwd_grid(spec, req)
        self.assertEqual(gz * spec.group_merge, int(req.G))

    def test_dispatched_spec_passes_the_emitter_gate(self):
        for kw in (dict(), dict(N=2, C=64, K=64, G=64, Hi=14, Wi=14)):
            with self.subTest(**kw):
                req = _req(**kw)
                fwd = dispatch_conv_grouped(req).spec.to_fwd_spec(_problem(req))
                ok, why = is_valid_spec(fwd, arch=req.arch)
                self.assertTrue(ok, why)

    def test_kernel_names_distinguish_merged_from_plain(self):
        # Both layers key a cache on this string. Untagged, the merged and
        # unmerged builds of one shape collide on a single symbol.
        merged = dispatch_conv_grouped(_req()).spec.kernel_name()
        plain = dispatch_conv_grouped(_req(C=3, K=3, G=3)).spec.kernel_name()
        self.assertIn("gm", merged)
        self.assertNotIn("gm", plain)
        self.assertNotEqual(merged, plain)

    def test_degree_policy_lands_close_to_the_measured_optimum(self):
        """The policy is a cost model, so the gate is distance-to-best, not a pin.

        Each entry carries that shape's *measured* per-degree curve, normalised
        to its own best, taken at the 64x64 tile dispatch launches -- the policy
        is only ever asked about that tile, and an earlier version of this test
        pinned degrees from a tile-swept run, which is a different question and
        gave different answers on 3 of its 5 shapes.

        Asserting ``curve[pick] >= 0.93`` rather than ``pick == oracle`` is the
        point: the model's constants are empirical and a refit will move some
        picks, but no refit may move one onto a degree that measures badly. Six
        of these nine are exact today; the three that are not sit on curves whose
        top is flat enough that the difference is small.
        """
        # kwargs -> {degree: measured throughput / that shape's best}
        measured = (
            (
                dict(
                    N=128, C=512, K=512, G=512, Hi=14, Wi=14, Y=7, X=7, pad_h=3, pad_w=3
                ),
                {1: 0.164, 2: 0.349, 4: 0.622, 8: 0.924, 16: 1.0, 32: 0.853, 64: 0.669},
            ),
            (
                dict(N=128, C=960, K=960, G=960, Hi=24, Wi=24, dtype="fp16"),
                {1: 0.056, 2: 0.191, 4: 0.423, 8: 0.620, 16: 0.848, 32: 1.0, 64: 0.948},
            ),
            (
                dict(N=42, C=256, K=256, G=256, Hi=60, Wi=80),
                {1: 0.053, 2: 0.161, 4: 0.347, 8: 0.571, 16: 0.887, 32: 0.936, 64: 1.0},
            ),
            # M = 49: one tile of work per group however the degree is chosen,
            # so the whole curve is within 4% from 1 to 16 and only the
            # footprint brake at 32/64 is real.
            (
                dict(
                    N=1, C=1536, K=1536, G=1536, Hi=7, Wi=7, Y=7, X=7, pad_h=3, pad_w=3
                ),
                {1: 0.963, 2: 0.998, 4: 1.0, 8: 0.993, 16: 0.978, 32: 0.658, 64: 0.336},
            ),
            # A filter this large has no K-padding left to recover, so the brake
            # is the only live mechanism and the turnover comes early.
            (
                dict(
                    N=1,
                    C=256,
                    K=256,
                    G=256,
                    Hi=28,
                    Wi=28,
                    Y=31,
                    X=31,
                    pad_h=15,
                    pad_w=15,
                ),
                {1: 0.368, 2: 0.604, 4: 0.972, 8: 1.0, 16: 0.584, 32: 0.261, 64: 0.112},
            ),
            (
                dict(
                    N=128, C=128, K=128, G=128, Hi=56, Wi=56, Y=7, X=7, pad_h=3, pad_w=3
                ),
                {1: 0.134, 2: 0.279, 4: 0.568, 8: 0.850, 16: 1.0, 32: 0.975, 64: 0.827},
            ),
            (
                dict(N=24, C=64, K=64, G=64, Hi=64, Wi=64, stride_h=2, stride_w=2),
                {1: 0.214, 2: 0.775, 4: 0.978, 8: 1.0, 16: 0.975, 32: 0.973, 64: 0.966},
            ),
            (
                dict(N=8, C=1152, K=1152, G=1152, Hi=7, Wi=7),
                {1: 0.591, 2: 0.981, 4: 1.0, 8: 0.971, 16: 0.966, 32: 0.972, 64: 0.942},
            ),
        )
        for kw, curve in measured:
            with self.subTest(**kw):
                gm = _pick_group_merge(_req(**kw), 64, 64)
                self.assertIn(gm, curve, f"picked an unmeasured degree {gm}")
                self.assertGreaterEqual(
                    curve[gm],
                    0.93,
                    f"Gm={gm} measures {curve[gm]:.3f} of this shape's best "
                    f"(oracle {max(curve, key=curve.get)}); curve={curve}",
                )

    def test_known_residuals_stay_bounded(self):
        """Two documented corners the cost model under-merges, kept visible.

        Both are asserted at the ratio they currently reach rather than deleted
        from the suite: a refit that fixes one shows up as a bound that wants
        tightening, not as a silent improvement nothing records, and a refit
        that makes one *worse* fails here instead of only moving a geomean.
        """
        residuals = (
            # N=8 on a 7x7 map at 5x5 is 392 rows of M -- six CTAs per group --
            # and the curve is flat-to-falling from 2 through 8 before jumping
            # at 16, which no monotone footprint brake reproduces. Picks 8.
            (
                dict(N=8, C=576, K=576, G=576, Hi=7, Wi=7, Y=5, X=5, pad_h=2, pad_w=2),
                {1: 0.548, 2: 0.597, 4: 0.591, 8: 0.591, 16: 1.0, 32: 0.576, 64: 0.556},
                0.55,
            ),
            # The largest launch in the corpus: 3x3 stride 2 on a 264x264 map.
            # The CTA-count term keeps paying all the way to 64 but the brake
            # turns the model over at 32.
            (
                dict(N=32, C=192, K=192, G=192, Hi=264, Wi=264, stride_h=2, stride_w=2),
                {1: 0.045, 2: 0.110, 4: 0.227, 8: 0.394, 16: 0.625, 32: 0.894, 64: 1.0},
                0.88,
            ),
        )
        for kw, curve, floor in residuals:
            with self.subTest(**kw):
                gm = _pick_group_merge(_req(**kw), 64, 64)
                self.assertIn(gm, curve)
                self.assertGreaterEqual(curve[gm], floor, f"Gm={gm} curve={curve}")

    def test_degree_policy_respects_divisibility_and_the_tile(self):
        # Plenty of M, so only divisibility and the N tile are left to bind.
        big = dict(N=64, Hi=56, Wi=56)
        self.assertEqual(
            _pick_group_merge(_req(C=1024, K=1024, G=1024, **big), 64, 64), 32
        )
        self.assertEqual(_pick_group_merge(_req(C=12, K=12, G=12, **big), 64, 64), 4)
        self.assertEqual(_pick_group_merge(_req(C=6, K=6, G=6, **big), 64, 64), 2)
        self.assertEqual(_pick_group_merge(_req(C=3, K=3, G=3, **big), 64, 64), 1)
        # tile_n binds when it is the smallest cap.
        self.assertEqual(
            _pick_group_merge(_req(C=1024, K=1024, G=1024, **big), 64, 4), 4
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
