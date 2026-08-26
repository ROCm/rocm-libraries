# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection + support + arch-gating tests for the grouped conv wgrad dispatcher."""

from __future__ import annotations

import unittest

from dispatch.grouped_convolution import (
    ConvGroupedRequest,
    conv_grouped_candidates,
    dispatch_conv_grouped,
)


def _wgrad(arch, **kw):
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
        direction="wgrad",
    )
    base.update(kw)
    return ConvGroupedRequest(**base)


class TestConvWgradDispatch(unittest.TestCase):

    # ---- candidate registry -----------------------------------------------

    def test_wgrad_candidates_nonempty(self):
        candidates = conv_grouped_candidates("wgrad")
        self.assertGreater(len(candidates), 0)

    def test_wgrad_candidates_distinct_names(self):
        names = [c.name for c in conv_grouped_candidates("wgrad")]
        self.assertEqual(len(names), len(set(names)))

    # ---- request validation ------------------------------------------------

    def test_rejects_unknown_arch(self):
        with self.assertRaises((ValueError, KeyError, RuntimeError)):
            dispatch_conv_grouped(_wgrad("gfx000"))

    def test_rejects_unsupported_dtype(self):
        with self.assertRaises((ValueError, KeyError, RuntimeError)):
            dispatch_conv_grouped(_wgrad("gfx950", dtype="fp8"))

    # ---- gfx950 selection --------------------------------------------------

    def test_gfx950_fp16_dispatches(self):
        r = dispatch_conv_grouped(_wgrad("gfx950"))
        self.assertIsNotNone(r.candidate)
        self.assertIsNotNone(r.spec)

    def test_gfx950_bf16_dispatches(self):
        r = dispatch_conv_grouped(_wgrad("gfx950", dtype="bf16"))
        self.assertIsNotNone(r.candidate)

    def test_gfx950_result_has_grid(self):
        r = dispatch_conv_grouped(_wgrad("gfx950"))
        self.assertIsNotNone(r.grid)
        self.assertGreater(r.grid[0], 0)

    def test_gfx950_split_k_sentinel(self):
        # split_k=-1 is the auto sentinel: resolved at launch, not dispatch time.
        r = dispatch_conv_grouped(_wgrad("gfx950"))
        self.assertIsNotNone(r.spec.split_k)

    # ---- gfx942 selection --------------------------------------------------

    def test_gfx942_fp16_dispatches(self):
        r = dispatch_conv_grouped(_wgrad("gfx942"))
        self.assertIsNotNone(r.candidate)

    def test_gfx942_split_k_sentinel(self):
        r = dispatch_conv_grouped(_wgrad("gfx942"))
        self.assertIsNotNone(r.spec.split_k)

    # ---- result contract ---------------------------------------------------

    def test_result_has_kernel_id(self):
        r = dispatch_conv_grouped(_wgrad("gfx950"))
        self.assertIsNotNone(r.kernel_id)

    def test_result_has_spec(self):
        r = dispatch_conv_grouped(_wgrad("gfx950"))
        self.assertIsNotNone(r.spec)


if __name__ == "__main__":
    unittest.main()
