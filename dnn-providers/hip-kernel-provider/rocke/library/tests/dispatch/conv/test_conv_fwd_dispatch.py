# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection + support + arch-gating tests for the grouped conv fwd dispatcher."""

from __future__ import annotations

import unittest

from dispatch.grouped_convolution import (
    ConvGroupedRequest,
    conv_grouped_candidates,
    dispatch_conv_grouped,
)


def _fwd(arch, **kw):
    base = dict(
        N=4,
        C=64,
        K=64,
        Hi=56,
        Wi=56,
        Y=3,
        X=3,
        pad_h=1,
        pad_w=1,
        arch=arch,
        direction="fwd",
    )
    base.update(kw)
    return ConvGroupedRequest(**base)


class TestConvFwdDispatch(unittest.TestCase):

    # ---- candidate registry -----------------------------------------------

    def test_fwd_candidates_nonempty(self):
        candidates = conv_grouped_candidates("fwd")
        self.assertGreater(len(candidates), 0)

    def test_fwd_candidates_distinct_names(self):
        names = [c.name for c in conv_grouped_candidates("fwd")]
        self.assertEqual(len(names), len(set(names)))

    # ---- request validation ------------------------------------------------

    def test_rejects_unknown_arch(self):
        with self.assertRaises(ValueError):
            dispatch_conv_grouped(_fwd("gfx000"))

    def test_rejects_unsupported_dtype(self):
        with self.assertRaises(ValueError):
            dispatch_conv_grouped(_fwd("gfx950", dtype="fp8"))

    def test_gfx1250_wgrad_dispatches(self):
        # gfx1250 wgrad candidate added alongside fwd.
        r = dispatch_conv_grouped(_fwd("gfx1250", direction="wgrad"))
        self.assertIsNotNone(r.candidate)

    # ---- gfx950 selection --------------------------------------------------

    def test_gfx950_fp16_dispatches(self):
        r = dispatch_conv_grouped(_fwd("gfx950"))
        self.assertIsNotNone(r.candidate)
        self.assertIsNotNone(r.spec)

    def test_gfx950_bf16_dispatches(self):
        r = dispatch_conv_grouped(_fwd("gfx950", dtype="bf16"))
        self.assertIsNotNone(r.candidate)

    def test_gfx950_result_has_grid(self):
        r = dispatch_conv_grouped(_fwd("gfx950"))
        self.assertIsNotNone(r.grid)
        self.assertGreater(r.grid[0], 0)

    # ---- gfx942 selection --------------------------------------------------

    def test_gfx942_fp16_dispatches(self):
        r = dispatch_conv_grouped(_fwd("gfx942"))
        self.assertIsNotNone(r.candidate)

    def test_gfx942_result_has_spec(self):
        r = dispatch_conv_grouped(_fwd("gfx942"))
        self.assertIsNotNone(r.spec)

    # ---- gfx1250 selection -------------------------------------------------

    def test_gfx1250_dispatches(self):
        r = dispatch_conv_grouped(_fwd("gfx1250"))
        self.assertIsNotNone(r.candidate)

    # ---- result contract ---------------------------------------------------

    def test_result_has_kernel_id(self):
        r = dispatch_conv_grouped(_fwd("gfx950"))
        self.assertIsNotNone(r.kernel_id)

    def test_result_has_spec(self):
        r = dispatch_conv_grouped(_fwd("gfx950"))
        self.assertIsNotNone(r.spec)


if __name__ == "__main__":
    unittest.main()
