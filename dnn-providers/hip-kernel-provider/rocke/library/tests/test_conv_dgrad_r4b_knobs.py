# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the opt-in dgrad x-halo knobs xh_stride_m, no_agpr, xh_epi 3 and tapk_ztrim:
validation, names, identity labels, AOT knob parsing, IR build."""

from __future__ import annotations

import unittest

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_implicit_gemm import ConvDataSpec
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    build_implicit_gemm_conv_dgrad,
    is_valid_dgrad_spec,
)
from kernels.conv import aot_jobs

_ARCH = "gfx942"


def _problem(**kw):
    base = dict(
        N=1, Di=10, Hi=34, Wi=34, C=64, K=64, Z=3, Y=3, X=3, sD=1, pD=0, dD=1, pH=0, pW=0
    )
    base.update(kw)
    return ConvProblem(**base)


def _xh(problem=None, **kw):
    base = dict(
        problem=problem if problem is not None else _problem(),
        data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        tile_m=256, tile_n=64, tile_k=32, warp_m=4, warp_n=1,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=8,
        pipeline="mem", epilogue="cshuffle", lds_k_outer=False,
        tap_outer_k=True, tapk_prefetch=3, tapk_xhalo=True, xhalo_exact=True,
        b_kpack=4, lds_xor=True, lds_k_pad=0, xh_a_k4=True, xh_order=6,
        cshuffle_m_chunks=2, tapk_lds_ahead=1, xh_epi=1, xh_fast_a=1,
    )
    base.update(kw)
    return DgradConvSpec(**base)


_A16 = dict(warp_tile_m=16, warp_tile_n=16, warp_tile_k=16)


class TestR4bKnobs(unittest.TestCase):
    def test_default_off(self):
        name = _xh().kernel_name()
        self.assertNotIn("_xsm", name)
        self.assertNotIn("_nag", name)

    def test_valid_build_and_named(self):
        cases = [
            (_xh(xh_stride_m=True), "_xsm"),
            (_xh(xh_stride_m=True, cshuffle_m_chunks=1), "_xsm"),
            (_xh(xh_stride_m=True, xh_defer=4), "_xsm"),
            (_xh(xh_stride_m=True, **_A16), "_xsm"),
            (_xh(xh_stride_m=True, cshuffle_m_chunks=4, **_A16), "_xsm"),
            (_xh(xh_stride_m=True, tapk_lds_ahead=0, xh_defer=2, xh_order=2, **_A16), "_xsm"),
            (_xh(xh_stride_m=True, tile_n=128, tile_k=16, warp_m=2, warp_n=2,
                 tapk_prefetch=1, xh_order=2, **_A16), "_xsm"),
            (_xh(no_agpr=True), "_nag"),
            (_xh(xh_epi=3, **_A16), "_xep3"),
            (_xh(xh_epi=3, xh_stride_m=True, **_A16), "_xep3"),
        ]
        for s, tag in cases:
            with self.subTest(name=s.kernel_name()):
                ok, why = is_valid_dgrad_spec(s, _ARCH)
                self.assertTrue(ok, why)
                self.assertIn(tag, s.kernel_name())
                build_implicit_gemm_conv_dgrad(s, arch=_ARCH)

    def test_rejections(self):
        bad = [
            _xh(xh_stride_m=True, xh_epi=0),
            _xh(xh_stride_m=True, xh_epi=2),
            _xh(xh_stride_m=True, xh_a_k4=False),
            _xh(xh_stride_m=True, tapk_lds_ahead=0, xh_defer=0),
            _xh(xh_stride_m=True, xh_persist=1),
            _xh(xh_epi=3),  # 32x32 atoms
            _xh(xh_epi=3, tile_n=16, **_A16),  # one MFMA along N
        ]
        for s in bad:
            with self.subTest(name=s.kernel_name()):
                self.assertFalse(is_valid_dgrad_spec(s, _ARCH)[0])

    def test_ztrim_xhalo(self):
        for zt in (1, 2):
            for kw in (dict(), dict(xh_stride_m=True, **_A16), dict(tapk_prefetch=1, xh_order=2)):
                s = _xh(tapk_ztrim=zt, **kw)
                with self.subTest(name=s.kernel_name()):
                    ok, why = is_valid_dgrad_spec(s, _ARCH)
                    self.assertTrue(ok, why)
                    self.assertIn(f"_zt{zt}", s.kernel_name())
                    build_implicit_gemm_conv_dgrad(s, arch=_ARCH)
        bad = [
            _xh(tapk_ztrim=3),
            _xh(tapk_ztrim=1, xh_persist=1),
        ]
        for s in bad:
            with self.subTest(name=s.kernel_name()):
                self.assertFalse(is_valid_dgrad_spec(s, _ARCH)[0])
        sets = aot_jobs._parse_dgrad_xh_knobs("zt=2,ak4=1,lxor=1")
        cfg = aot_jobs._dgrad_xh_knob_cfg(sets[1], tpf=3, grouped=False, epilogue="cshuffle")
        self.assertEqual(cfg["tapk_ztrim"], 2)

    def test_aot_keys(self):
        sets = aot_jobs._parse_dgrad_xh_knobs("xsm=1,nag=1,xep=1,ak4=1,lxor=1")
        cfg = aot_jobs._dgrad_xh_knob_cfg(sets[1], tpf=3, grouped=False, epilogue="cshuffle")
        self.assertTrue(cfg["xh_stride_m"])
        self.assertTrue(cfg["no_agpr"])


if __name__ == "__main__":
    unittest.main()
