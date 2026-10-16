# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dispatch of depthwise (cpg = kpg = 1) dgrad to the direct kernels.

Host-only: which candidate wins (the windowed kernel at stride 1, the
row-streaming kernel at stride 2 and for what the windowed one declines),
which requests have no path, and the knob rules -- every pick a knob dict
of the AOT cache (the cached tiles, waves and MFMA tile heights): the
Toeplitz MFMA form's admission box (7x7 'same' filters, C a multiple of 64,
7-16 rows, 7-8 / 13-16 / 19-112 columns, size caps), pinned at its edges, its
fold / waves / tile-height rule, and the VALU form's dot2, channel-pair,
waves and tile rules.
"""

from __future__ import annotations

import unittest

from dispatch import grouped_conv_dgrad_direct as ddp
from dispatch.grouped_convolution import (
    ConvDirectDgradSpec,
    ConvGroupedRequest,
    _direct_problem,
    conv_grouped_candidates,
    dispatch_conv_grouped,
)
from kernels.common.conv_direct_grouped import (
    DW_DGRAD_MFMA_CH,
    direct_launch_geometry,
    is_valid_direct_dgrad_spec,
)

_WIN = "direct_depthwise_dgrad_win"
_STREAM = "direct_depthwise_dgrad_stream"


def _dw(C=256, Hi=13, Wi=13, Y=7, X=7, pad=3, N=32, dtype="bf16", **kw):
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
    _dw(Hi=15, Wi=15),
    _dw(N=16, C=768, Hi=12, Wi=12, Y=3, X=3, pad=1, dtype="fp16"),
    _dw(N=32, C=128, Hi=50, Wi=50),
    _dw(N=32, C=256, Hi=28, Wi=28),
    _dw(C=768, Hi=7, Wi=7),
    _dw(N=64, C=240, Hi=28, Wi=28, Y=5, X=5, pad=2, dtype="fp16"),
    _dw(N=64, C=144, Hi=50, Wi=50, Y=3, X=3, pad=1, dtype="fp16"),
    _dw(N=32, C=32, Hi=112, Wi=112, Y=3, X=3, pad=1, dtype="fp16"),
    _dw(N=1, C=66, Hi=13, Wi=17, Y=5, X=5, pad=2),
    _dw(N=3, C=130, Hi=15, Wi=7, Y=3, X=3, pad=0, dtype="fp16"),
    _dw(N=4, C=96, Hi=20, Wi=20, Y=11, X=11, pad=5),
)


def _knobs(req):
    r = dispatch_conv_grouped(req)
    return r, r.spec.knob_dict()


class TestDepthwiseDgradDispatch(unittest.TestCase):
    def test_depthwise_requests_take_the_windowed_kernel(self):
        for req in _COHORT:
            with self.subTest(req=req):
                r = dispatch_conv_grouped(req)
                self.assertEqual(r.candidate.name, _WIN)
                self.assertIsInstance(r.spec, ConvDirectDgradSpec)
                self.assertEqual(r.spec.variant, ddp.VARIANT_DW_WIN)
                self.assertEqual(r.spec.abi_direction, "dgrad")
                p = _direct_problem(req)
                inst = r.spec.instance(p)
                self.assertTrue(
                    is_valid_direct_dgrad_spec(r.spec.variant, inst, "gfx950")[0]
                )
                self.assertEqual((r.grid, r.block), direct_launch_geometry(inst))
                self.assertEqual((r.spec.KH, r.spec.PAD), (req.Y, req.pad_h))

    def test_strided_depthwise_takes_the_streaming_kernel(self):
        for req in (
            _dw(N=8, C=64, Hi=28, Wi=28, Y=5, X=5, pad=2, stride_h=2, stride_w=2),
            _dw(N=8, C=192, Hi=50, Wi=50, Y=3, X=3, pad=1, stride_h=2, stride_w=2),
            _dw(N=2, C=32, Hi=15, Wi=15, Y=7, X=7, pad=0, stride_h=2, stride_w=2),
        ):
            with self.subTest(req=req):
                r, knobs = _knobs(req)
                self.assertEqual(r.candidate.name, _STREAM)
                self.assertEqual(r.spec.variant, ddp.VARIANT_DW_STREAM)
                self.assertIn(knobs["block_w"], ddp.DW_STREAM_BLOCK_W)
                self.assertIn(knobs["block_waves"], ddp.DW_STREAM_WAVES)
                self.assertLessEqual(knobs["block_waves"] * 64, max(64, req.C))

    def test_stride1_streams_what_the_windowed_kernel_declines(self):
        win = next(c for c in conv_grouped_candidates("dgrad") if c.name == _WIN)
        stream = next(c for c in conv_grouped_candidates("dgrad") if c.name == _STREAM)
        req = _dw()
        self.assertTrue(win.admits(req)[0] and stream.admits(req)[0])
        self.assertEqual(dispatch_conv_grouped(req).candidate.name, _WIN)
        self.assertLess(win.priority, stream.priority)

    def test_mfma_admission_box_edges(self):
        inside = (
            {},
            {"Hi": 16, "Wi": 16},
            {"N": 1, "C": 64, "Hi": 7, "Wi": 7},
            {"N": 256, "C": 64, "Hi": 7, "Wi": 8},
            {"N": 256, "C": 2048, "Hi": 7, "Wi": 13},
            {"N": 1, "C": 2048, "Hi": 16, "Wi": 19},
            {"N": 64, "C": 1024, "Hi": 12, "Wi": 27, "dtype": "fp16"},
            {"N": 32, "C": 192, "Hi": 9, "Wi": 55},
            {"N": 4, "C": 64, "Hi": 16, "Wi": 112},
        )
        for kw in inside:
            req = _dw(**kw)
            with self.subTest(kw=kw):
                self.assertTrue(ddp._dw_mfma_admits(_direct_problem(req), req.arch))
                self.assertTrue(_knobs(req)[1].get("mfma"))
        outside = (
            {"Hi": 13, "Wi": 6},
            {"Hi": 13, "Wi": 9},
            {"Hi": 13, "Wi": 12},
            {"Hi": 13, "Wi": 17},
            {"N": 4, "C": 64, "Hi": 16, "Wi": 113},
            {"Hi": 6, "Wi": 13},
            {"Hi": 17, "Wi": 13},
            {"N": 257, "C": 64},
            {"C": 2112},
            {"C": 96},
            {"N": 256, "C": 2048, "Hi": 16, "Wi": 33},
            {"Y": 5, "X": 5, "pad": 2},
            {"Y": 3, "X": 3, "pad": 1},
            {"Y": 9, "X": 9, "pad": 4},
            {"pad": 2},
        )
        for kw in outside:
            req = _dw(**kw)
            with self.subTest(kw=kw):
                self.assertFalse(ddp._dw_mfma_admits(_direct_problem(req), req.arch))
                knobs = _knobs(req)[1]
                self.assertFalse(knobs.get("mfma", False))

    def test_mfma_knob_rule(self):
        def pick(**kw):
            req = _dw(**kw)
            return ddp._dw_mfma_knobs(_direct_problem(req), req.arch)

        # w_fold: 4 images per tile up to 8 columns, 2 up to 16, 1 up to 32.
        for W, fold in ((7, 4), (8, 4), (13, 2), (16, 2), (19, 1), (32, 1)):
            self.assertEqual(pick(Hi=8, Wi=W)["w_fold"], fold, W)
        # Whole images up to 16 rows run on the smallest cached tile height
        # covering them; small grids drop to 4 waves, then to shorter tiles.
        for H, bh in ((7, 7), (8, 8), (9, 12), (13, 14), (15, 16)):
            with self.subTest(H=H):
                k = pick(N=96, C=1024, Hi=H, Wi=H)
                self.assertEqual((k["block_waves"], k["block_h"]), (8, bh))
        small = pick(N=2, C=256, Hi=13, Wi=13)
        self.assertEqual((small["block_waves"], small["block_h"]), (4, 4))
        for knobs in (small, pick(N=96, C=1024, Hi=9, Wi=9)):
            self.assertIn(knobs["block_h"], ddp.DW_WIN_MFMA_BLOCK_H)
            self.assertIn(knobs["block_waves"], ddp.DW_WIN_MFMA_WAVES)
            self.assertEqual(knobs["prefetch_rows"], ddp.DW_WIN_MFMA_PREFETCH)

    def test_valu_knob_rules(self):
        for req in _COHORT:
            _, knobs = _knobs(req)
            if knobs.get("mfma"):
                self.assertEqual((req.Y, req.X), (7, 7))
                self.assertEqual(req.C % DW_DGRAD_MFMA_CH, 0)
                continue
            with self.subTest(req=req):
                self.assertIn((knobs["block_w"], knobs["block_h"]), ddp.DW_WIN_TILES)
                self.assertEqual(knobs["dot2"], req.X >= 5)
                pairs = req.X == 3 and req.C % 2 == 0 and req.C >= 128
                self.assertEqual(knobs["ch_per_lane"], 2 if pairs else 1)
                if pairs:
                    self.assertEqual(knobs["block_waves"], ddp.DW_WIN_PAIR_WAVES)
                else:
                    self.assertIn(knobs["block_waves"], ddp.DW_WIN_WAVES)
                fmas = (
                    knobs["block_h"] * req.Y * req.X * knobs["block_w"]
                    * knobs["ch_per_lane"]
                )  # fmt: skip
                self.assertLessEqual(fmas, ddp.DW_WIN_UNROLL_BUDGET)

    def test_small_maps_stay_whole(self):
        # A late-stage map with a full grid keeps a whole-image cached tile.
        _, knobs = _knobs(_dw(N=256, C=768, Hi=12, Wi=12, Y=3, X=3, pad=1))
        self.assertEqual((knobs["block_w"], knobs["block_h"]), (12, 12))

    def test_no_path(self):
        # No cached filter (even, larger than 11, non-square), asymmetric
        # padding, dilation, gfx942: no gfx950 depthwise kernel.
        for kw, needle in (
            ({"Y": 13, "X": 13, "pad": 6}, "square filters"),
            ({"Y": 3, "X": 5, "pad": 1}, "square filters"),
            ({"pad_w": 2}, "pad_h == pad_w"),
            ({"dilation_h": 2, "dilation_w": 2}, "dilation"),
            ({"arch": "gfx942"}, "gfx942"),
        ):
            with self.subTest(kw=kw):
                with self.assertRaises(ValueError) as cm:
                    dispatch_conv_grouped(_dw(**kw))
                self.assertIn(needle, str(cm.exception))

    def test_grouped_requests_are_not_depthwise(self):
        req = _dw(C=384, G=96, K=384)
        for c in conv_grouped_candidates("dgrad"):
            if c.name in (_WIN, _STREAM):
                self.assertFalse(c.admits(req)[0])

    def test_candidates_reject_other_directions(self):
        for name in (_WIN, _STREAM):
            cand = next(c for c in conv_grouped_candidates("dgrad") if c.name == name)
            for direction in ("fwd", "wgrad"):
                ok, why = cand.admits(_dw(direction=direction))
                self.assertFalse(ok)
                self.assertIn("dgrad", why)


if __name__ == "__main__":
    unittest.main()
