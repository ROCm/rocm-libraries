# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""identity_supports_problem refuses the fwd launches a binary's kernargs
cannot honour (no GPU): it asks the same predicate as
is_valid_spec_for_problem (fwd_launch_problem_reason)."""

from __future__ import annotations

import dataclasses
import unittest

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_implicit_gemm import fwd_launch_problem_reason
from kernels.conv.kernel_identity import KernelIdentity, identity_supports_problem

# The shipped gfx942 fwd y-ring pin (epi_quad_xpose 4, xh_yring 14).
_XYR = (
    "fwd-implicit_gemm-bf16-3d-t256x64x32-w4x1-a16x16x16-v8x8x8-mem-cshuffle-grp-"
    "xcd6-ukd-rpf1-lxor-tapf-kprd-uxh8-prio1-lxrk-xws-eqx4-xsm-fxd1040-xyr14-"
    "wave64-gfx942-llvm22"
)


def _p(**kw):
    base = dict(
        N=1,
        Di=130,
        Hi=66,
        Wi=66,
        C=64,
        K=128,
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


class TestFwdLaunchRules(unittest.TestCase):
    def setUp(self):
        self.ident = KernelIdentity.from_label(_XYR)

    def test_table_problem_is_admitted(self):
        self.assertEqual(identity_supports_problem(self.ident, _p()), (True, "ok"))

    def test_refusals_match_the_predicate(self):
        for kw in (dict(C=48), dict(dH=2)):
            with self.subTest(kw=kw):
                p = _p(**kw)
                why = fwd_launch_problem_reason(self.ident, p)
                self.assertIsNotNone(why)
                self.assertEqual(identity_supports_problem(self.ident, p), (False, why))

    def test_stream_k_serves_one_group(self):
        sk = dataclasses.replace(
            self.ident, xh_yring=0, epi_quad_xpose=0, split_k=2, split_streamk=True
        )
        p = _p(C=128, K=128, groups=2)
        why = fwd_launch_problem_reason(sk, p)
        self.assertIsNotNone(why)
        ok, reason = identity_supports_problem(sk, p)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
