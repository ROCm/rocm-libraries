# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""identity_supports_problem refuses the problems an x-halo dgrad binary
computes wrongly (no GPU): filters its tap masks or its y-ring were not built
for, and taps whose tap-in-base dY resource would start past dY."""

from __future__ import annotations

import dataclasses
import unittest

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.conv.kernel_identity import KernelIdentity, identity_supports_problem

_RING = (
    "dgrad-implicit_gemm-bf16-3d-t256x64x32-w4x1-a16x16x16-v8x8x8-mem-cshuffle-grp-f0x3-"
    "s1x1-d1x1-wpe2-xcd6-lxor-tapk-tpf3-xh-bkp2-ilv4-xhx-mch2-lda1-lsb-sp1-bw-bpad4-zt1-"
    "ak4-xo6-xdf4-xfa1-xep1-elb-nag-mb11-xsm-xyr130-xel-xtr-msg1024-wave64-gfx942-llvm22"
)


def _p(**kw):
    base = dict(
        N=1,
        Di=6,
        Hi=32,
        Wi=130,
        C=64,
        K=64,
        Z=3,
        Y=3,
        X=3,
        sD=1,
        pD=0,
        dD=1,
        pH=0,
        pW=0,
    )
    base.update(kw)
    return ConvProblem(**base)


class TestXHaloFilters(unittest.TestCase):
    def setUp(self):
        self.ring = KernelIdentity.from_label(_RING)
        self.fast_a = dataclasses.replace(self.ring, xh_yring=0)

    def test_ring_serves_its_filter_height(self):
        self.assertTrue(identity_supports_problem(self.ring, _p())[0])
        for kw in (dict(Y=1), dict(Y=5), dict(Z=5)):
            with self.subTest(kw=kw):
                self.assertFalse(identity_supports_problem(self.ring, _p(**kw))[0])

    def test_tap_masks_serve_up_to_3(self):
        self.assertTrue(identity_supports_problem(self.fast_a, _p())[0])
        self.assertTrue(identity_supports_problem(self.fast_a, _p(Y=1))[0])
        self.assertFalse(identity_supports_problem(self.fast_a, _p(Z=5))[0])
        dlds = dataclasses.replace(
            self.fast_a, xh_fast_a=0, xh_tap_rsrc=False, xh_dlds=4
        )
        self.assertFalse(identity_supports_problem(dlds, _p(Z=4))[0])

    def test_tap_base_inside_dy(self):
        # N=1 with Do <= pD (or Ho <= pH): a tap offset passes the end of dY.
        xtr = dataclasses.replace(self.fast_a, m_band=0, xcd_remap=0, tapk_ztrim=0)
        ok = _p(Di=16, Hi=16, Wi=16, pD=1, pH=1, pW=0)
        self.assertTrue(
            identity_supports_problem(xtr, ok)[0], identity_supports_problem(xtr, ok)
        )
        for kw in (dict(Di=1, pD=1), dict(Di=1, Hi=1, pD=0, pH=1)):
            with self.subTest(kw=kw):
                p = _p(**dict(dict(Hi=16, Wi=64, pW=0), **kw))
                self.assertFalse(identity_supports_problem(xtr, p)[0])


if __name__ == "__main__":
    unittest.main()
