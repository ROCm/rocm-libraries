# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dispatch of depthwise (cpg = kpg = 1) dgrad to the windowed direct kernel.

Host-only: which candidate wins, which requests fall back, and the invariants
of the knob heuristic (block_w, waves, channel packing, dot2, H split).
"""

from __future__ import annotations

import math
import unittest

from dispatch.grouped_convolution import (
    ConvDepthwiseDgradSpec,
    ConvGroupedRequest,
    conv_grouped_candidates,
    dispatch_conv_grouped,
)
from kernels.common.conv_direct_grouped import (
    DirectDepthwiseDgradWindowedSpec,
    is_valid_depthwise_dgrad_win_spec,
)

_DW = "direct_depthwise_dgrad_win"
_IGEMM = "implicit_gemm_conv_dgrad"


def _dw(C=512, Hi=14, Wi=14, Y=7, X=7, pad=3, N=128, dtype="bf16", **kw):
    base = {
        "N": N,
        "C": C,
        "K": C,
        "G": C,
        "Hi": Hi,
        "Wi": Wi,
        "Y": Y,
        "X": X,
        "pad_h": pad,
        "pad_w": pad,
        "dtype": dtype,
        "arch": "gfx950",
        "direction": "dgrad",
    }
    base.update(kw)
    return ConvGroupedRequest(**base)


# Depthwise layers of common mobile / ConvNeXt-style networks plus odd sizes.
_COHORT = (
    _dw(),
    _dw(Hi=16, Wi=16),
    _dw(C=1536, Hi=12, Wi=12, Y=3, X=3, pad=1, dtype="fp16"),
    _dw(N=32, C=128, Hi=56, Wi=56),
    _dw(N=32, C=256, Hi=28, Wi=28),
    _dw(C=768, Hi=7, Wi=7),
    _dw(N=64, C=240, Hi=28, Wi=28, Y=5, X=5, pad=2, dtype="fp16"),
    _dw(N=64, C=144, Hi=56, Wi=56, Y=3, X=3, pad=1, dtype="fp16"),
    _dw(N=32, C=32, Hi=112, Wi=112, Y=3, X=3, pad=1, dtype="fp16"),
    _dw(N=1, C=66, Hi=13, Wi=17, Y=5, X=5, pad=2),
    _dw(N=3, C=130, Hi=15, Wi=7, Y=3, X=3, pad=0, dtype="fp16"),
)


class TestDepthwiseDgradDispatch(unittest.TestCase):
    def test_depthwise_requests_take_the_windowed_kernel(self):
        for req in _COHORT:
            with self.subTest(req=req):
                res = dispatch_conv_grouped(req)
                self.assertEqual(res.candidate.name, _DW)
                self.assertIsInstance(res.spec, ConvDepthwiseDgradSpec)
                inst = res.spec.instance
                self.assertEqual(res.grid, inst.grid())
                self.assertEqual(res.block, (inst.threads_per_block, 1, 1))
                self.assertTrue(is_valid_depthwise_dgrad_win_spec(inst, req.arch)[0])
                self.assertEqual(res.spec.kernel_name(), inst.kernel_name())
                # A non-sentinel field round-trips into the instance problem.
                self.assertEqual(inst.problem.PAD, req.pad_h)
                self.assertEqual(inst.problem.dtype, req.dtype)

    def test_target_shape_picks(self):
        s6 = dispatch_conv_grouped(_dw()).spec.instance
        self.assertEqual(
            (s6.block_w, s6.block_waves, s6.ch_per_lane, s6.block_h, s6.dot2),
            (14, 4, 1, 0, True),
        )
        s8 = dispatch_conv_grouped(_dw(Hi=16, Wi=16)).spec.instance
        self.assertEqual((s8.block_w, s8.dot2, s8.h_tiles), (16, True, 1))
        s10 = dispatch_conv_grouped(_COHORT[2]).spec.instance
        self.assertEqual(
            (s10.block_w, s10.ch_per_lane, s10.dot2, s10.h_tiles), (12, 2, False, 1)
        )

    def test_heuristic_invariants(self):
        for req in _COHORT:
            inst: DirectDepthwiseDgradWindowedSpec = dispatch_conv_grouped(
                req
            ).spec.instance
            with self.subTest(req=req):
                W = req.Wi
                if W <= 16:
                    self.assertEqual(inst.block_w, W)
                else:
                    # Tiles of about 8 columns; never more tiles than needed.
                    self.assertLessEqual(inst.block_w, 8)
                    tiles = math.ceil(W / inst.block_w)
                    self.assertEqual(tiles, math.ceil(W / 8))
                self.assertEqual(inst.dot2, req.X >= 5)
                if inst.dot2:
                    self.assertEqual(inst.ch_per_lane, 1)
                self.assertLessEqual(inst.block_waves, 4)
                # Never more lanes than channels by a whole wave.
                self.assertLess(
                    inst.block_waves * 64 * inst.ch_per_lane,
                    req.C + 64 * inst.ch_per_lane,
                )
                if inst.h_tiles > 1:
                    self.assertGreaterEqual(inst.rows_per_block, 2 * req.Y)
                self.assertLessEqual(inst.unrolled_fmas(), 1 << 14)

    def test_block_w_divides_w_when_possible(self):
        for W, bw in ((28, 7), (56, 8), (112, 8), (14, 14), (12, 12), (7, 7)):
            inst = dispatch_conv_grouped(
                _dw(N=8, C=64, Hi=8, Wi=W, Y=3, X=3, pad=1)
            ).spec.instance
            self.assertEqual(inst.block_w, bw, W)
            self.assertEqual(W % inst.block_w, 0, W)

    def test_small_grid_splits_h(self):
        inst = dispatch_conv_grouped(_dw(N=32, C=128, Hi=56, Wi=56)).spec.instance
        self.assertGreater(inst.h_tiles, 1)
        inst = dispatch_conv_grouped(_dw()).spec.instance
        self.assertEqual(inst.h_tiles, 1)

    def test_large_filter_balances_rows_and_block_w(self):
        # Past one filter height of rows the unroll budget shrinks the larger
        # of rows and block_w, so neither collapses to 1 while the other is
        # still wide (each block would re-read a whole filter-height halo).
        for req in (
            _dw(N=1, C=128, Hi=64, Wi=64, Y=31, X=31, pad=15, dtype="fp16"),
            _dw(N=1, C=64, Hi=64, Wi=64, Y=33, X=33, pad=16, dtype="fp16"),
            _dw(N=32, C=512, Hi=28, Wi=28, Y=31, X=31, pad=15),
        ):
            inst = dispatch_conv_grouped(req).spec.instance
            with self.subTest(req=req):
                self.assertLessEqual(inst.unrolled_fmas(), 1 << 14)
                self.assertGreater(inst.rows_per_block, 1)
                bw, rows = inst.block_w, inst.rows_per_block
                self.assertLessEqual(max(bw, rows), 2 * min(bw, rows))
                self.assertTrue(is_valid_depthwise_dgrad_win_spec(inst, req.arch)[0])

    def test_fallback_and_rejections(self):
        # Requests the windowed kernel declines: no other candidate covers
        # depthwise dgrad, so these raise with the windowed reason in the text.
        for kw, needle in (
            ({"stride_h": 2, "stride_w": 2}, "stride 1"),
            ({"dilation_h": 2, "dilation_w": 2}, "dilation"),
            ({"pad_w": 2}, "pad_h == pad_w"),
            ({"arch": "gfx942"}, "gfx942"),
            ({"N": 4096, "Hi": 112, "Wi": 112, "C": 64}, "sentinel"),
        ):
            with self.subTest(kw=kw):
                with self.assertRaises(ValueError) as cm:
                    dispatch_conv_grouped(_dw(**kw))
                self.assertIn(needle, str(cm.exception))
        # Grouped (cpg > 1) dgrad is not this candidate's (the grouped direct
        # or the igemm candidate takes it, depending on the shape).
        res = dispatch_conv_grouped(_dw(C=512, G=128, K=512))
        self.assertNotEqual(res.candidate.name, _DW)
        cand = next(c for c in conv_grouped_candidates("dgrad") if c.name == _DW)
        self.assertFalse(cand.admits(_dw(C=512, G=128, K=512))[0])

    def test_candidate_rejects_other_directions(self):
        cand = next(c for c in conv_grouped_candidates("dgrad") if c.name == _DW)
        for direction in ("fwd", "wgrad"):
            ok, why = cand.admits(_dw(direction=direction))
            self.assertFalse(ok)
            self.assertIn("dgrad", why)

    def test_candidate_build_matches_selection(self):
        req = _dw(N=2, C=64, Hi=9, Wi=13, Y=5, X=5, pad=2)
        res = dispatch_conv_grouped(req)
        kernel = res.candidate.built(res.spec, req.arch)
        self.assertEqual(kernel.name, res.spec.kernel_name())


if __name__ == "__main__":
    unittest.main()
