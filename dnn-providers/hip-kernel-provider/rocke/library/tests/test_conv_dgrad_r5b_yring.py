# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the round-5b opt-in dgrad x-halo y-ring (DgradConvSpec.xh_yring): validation, geometry,
names, identity label / problem support, AOT knob parsing, IR build."""

from __future__ import annotations

import dataclasses
import unittest

from kernels.common.conv_implicit_gemm_dgrad import (
    build_implicit_gemm_conv_dgrad,
    is_valid_dgrad_spec,
    xh_yring_geometry,
)
from kernels.conv import aot_jobs
from kernels.conv.kernel_identity import KernelIdentity, identity_supports_problem

from tests.test_conv_dgrad_r5_knobs import _ARCH, _problem, _xh


def _ring(wi, **kw):
    base = dict(xh_yring=wi, xh_tap_rsrc=True, xh_epi_lin=True, epi_lds_barriers=True,
                tapk_ztrim=1)
    base.update(kw)
    return _xh(**base)


class TestYRing(unittest.TestCase):
    def test_default_off(self):
        self.assertNotIn("_xyr", _xh().kernel_name())
        self.assertIsNone(xh_yring_geometry(_xh()))

    def test_geometry(self):
        self.assertEqual(xh_yring_geometry(_ring(130)), (130, 260, 128))
        self.assertEqual(xh_yring_geometry(_ring(66)), (66, 264, 64))
        self.assertIsNone(xh_yring_geometry(_ring(2)))     # no new main rows
        self.assertIsNone(xh_yring_geometry(_ring(100)))   # 98 rows: not whole loader passes
        self.assertIsNone(xh_yring_geometry(_ring(390)))   # main rows > tile_m
        self.assertEqual(xh_yring_geometry(_ring(2066)), (66, 264, 64))  # full reload every 2nd y
        self.assertIsNone(xh_yring_geometry(_ring(1066)))  # d = 1 is no ring

    def test_valid_build_and_named(self):
        for s in (_ring(130), _ring(66), _ring(2066), _ring(66, xh_defer=1, no_agpr=True),
                  _ring(130, xh_tap_rsrc=False, xh_fast_a=0),
                  _ring(130, no_agpr=True)):
            with self.subTest(name=s.kernel_name()):
                ok, why = is_valid_dgrad_spec(s, _ARCH)
                self.assertTrue(ok, why)
                self.assertIn(f"_xyr{s.xh_yring}", s.kernel_name())
                build_implicit_gemm_conv_dgrad(s, arch=_ARCH)

    def test_rejections(self):
        for s in (_ring(130, tapk_ztrim=2), _ring(130, tapk_ztrim=0),
                  _ring(130, tapk_prefetch=1, xh_order=2, xh_defer=0),
                  _ring(130, xh_fast_a=2), _ring(130, split_k=2), _ring(2)):
            with self.subTest(name=s.kernel_name()):
                self.assertFalse(is_valid_dgrad_spec(s, _ARCH)[0])

    def test_identity_and_support(self):
        sets = aot_jobs._parse_dgrad_xh_knobs("xyr=130,xtr=1,ak4=1")
        cfg = aot_jobs._dgrad_xh_knob_cfg(sets[1], tpf=3, grouped=False, epilogue="cshuffle")
        self.assertEqual(cfg["xh_yring"], 130)
        self.assertIsNone(
            aot_jobs._dgrad_xh_knob_cfg(sets[1], tpf=1, grouped=False, epilogue="cshuffle"))
        ident = KernelIdentity.from_label(
            "dgrad-implicit_gemm-bf16-3d-t256x64x32-w4x1-a16x16x16-v8x8x8-mem-cshuffle-grp-f0x3-"
            "s1x1-d1x1-wpe2-xcd6-lxor-tapk-tpf3-xh-bkp2-ilv4-xhx-mch2-lda1-lsb-sp1-bw-bpad4-zt1-"
            "ak4-xo6-xdf4-xfa1-xep1-elb-nag-mb11-xsm-xyr130-xel-xtr-msg1024-wave64-gfx942-llvm22")
        self.assertIn("xyr130", ident.label())
        self.assertEqual(KernelIdentity.from_label(ident.label()).xh_yring, 130)
        base = dataclasses.replace(ident, xh_yring=0)
        p130 = _problem(Wi=130, Hi=130, C=64, K=64, groups=1)
        p66 = _problem(Wi=66, Hi=66, C=64, K=64, groups=1)
        self.assertFalse(identity_supports_problem(ident, p66)[0])
        ok130 = identity_supports_problem(ident, p130)
        okbase = identity_supports_problem(base, p130)
        self.assertEqual(ok130[0], okbase[0], (ok130, okbase))


if __name__ == "__main__":
    unittest.main()
