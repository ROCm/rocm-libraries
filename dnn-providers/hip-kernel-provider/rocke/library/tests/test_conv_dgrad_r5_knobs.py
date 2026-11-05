# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the round-5 opt-in dgrad x-halo knobs xh_dlds (direct-to-LDS A loads), xh_epi_lin,
epi_lds_barriers and xh_tap_rsrc: validation, names, identity labels, AOT knob parsing, IR build."""

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
from kernels.conv.kernel_identity import KernelIdentity

_ARCH = "gfx942"


def _problem(**kw):
    base = dict(
        N=1, Di=10, Hi=34, Wi=34, C=64, K=64, Z=3, Y=3, X=3, sD=1, pD=0, dD=1, pH=0, pW=0
    )
    base.update(kw)
    return ConvProblem(**base)


def _xh(problem=None, **kw):
    """The s1/s4 x-halo winner's shape: t256x64x32 w4x1 a16 tpf3 xsm bw bpad4 bkp2."""
    base = dict(
        problem=problem if problem is not None else _problem(),
        data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        tile_m=256, tile_n=64, tile_k=32, warp_m=4, warp_n=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
        pipeline="mem", epilogue="cshuffle", lds_k_outer=False,
        tap_outer_k=True, tapk_prefetch=3, tapk_xhalo=True, xhalo_exact=True,
        b_kpack=2, lds_xor=True, lds_k_pad=0, xh_a_k4=True, xh_order=6, xh_defer=4,
        cshuffle_m_chunks=2, tapk_lds_ahead=1, xh_epi=1, xh_fast_a=1,
        xh_b_wide=True, xh_b_kpad=4, tapk_ilv=4, tapk_setprio=1, tapk_lds_barrier=True,
        xh_stride_m=True,
    )
    base.update(kw)
    return DgradConvSpec(**base)


_NEW_TAGS = ("_xdl", "_xel", "_elb", "_xtr")


class TestR5Knobs(unittest.TestCase):
    def test_default_off(self):
        name = _xh().kernel_name()
        for t in _NEW_TAGS:
            self.assertNotIn(t, name)
        self.assertTrue(is_valid_dgrad_spec(_xh(), _ARCH)[0])

    def test_valid_build_and_named(self):
        cases = [
            (_xh(xh_epi_lin=True), "_xel"),
            (_xh(epi_lds_barriers=True), "_elb"),
            (_xh(xh_tap_rsrc=True), "_xtr"),
            (_xh(xh_tap_rsrc=True, xh_epi_lin=True, epi_lds_barriers=True, tapk_ztrim=2), "_xtr"),
            (_xh(xh_tap_rsrc=True, tile_n=128, tile_k=16, tapk_prefetch=1, xh_order=2,
                 xh_defer=0, b_kpack=4), "_xtr"),
            (_xh(xh_dlds=1, xh_defer=0, xh_order=2), "_xdl1"),
            (_xh(xh_dlds=4, xh_defer=0, xh_order=2), "_xdl4"),
            (_xh(xh_dlds=5, xh_defer=3, xh_order=2), "_xdl5"),
            (_xh(xh_dlds=6, xh_defer=3, xh_order=2), "_xdl6"),
        ]
        for s, tag in cases:
            with self.subTest(name=s.kernel_name()):
                ok, why = is_valid_dgrad_spec(s, _ARCH)
                self.assertTrue(ok, why)
                self.assertIn(tag, s.kernel_name())
                build_implicit_gemm_conv_dgrad(s, arch=_ARCH)

    def test_rejections(self):
        bad = [
            _xh(xh_epi_lin=True, xh_epi=0),
            _xh(epi_lds_barriers=True, tapk_xhalo=False, xhalo_exact=False, xh_a_k4=False,
                xh_b_wide=False, xh_b_kpad=0, xh_stride_m=False, xh_epi=0, xh_fast_a=0,
                xh_order=0, xh_defer=0),
            _xh(xh_tap_rsrc=True, xh_fast_a=0),
            _xh(xh_tap_rsrc=True, xh_b_wide=False),
            _xh(xh_dlds=1),  # xh_defer 4 / xh_order 4
            _xh(xh_dlds=6, xh_defer=3, xh_order=2, warp_m=2, warp_n=2),
            _xh(xh_dlds=7, xh_defer=0, xh_order=2),
        ]
        for s in bad:
            with self.subTest(name=s.kernel_name()):
                self.assertFalse(is_valid_dgrad_spec(s, _ARCH)[0])

    def test_identity_fields(self):
        fields = KernelIdentity.__dataclass_fields__
        for f in ("xh_dlds", "xh_epi_lin", "xh_tap_rsrc", "epi_lds_barriers"):
            self.assertIn(f, fields)

    def test_aot_keys(self):
        sets = aot_jobs._parse_dgrad_xh_knobs("xel=1,elb=1,xtr=1,xdl=6,ak4=1,lxor=1")
        cfg = aot_jobs._dgrad_xh_knob_cfg(sets[1], tpf=3, grouped=False, epilogue="cshuffle")
        self.assertTrue(cfg["xh_epi_lin"])
        self.assertTrue(cfg["epi_lds_barriers"])
        self.assertTrue(cfg["xh_tap_rsrc"])
        self.assertEqual(cfg["xh_dlds"], 6)


def _pp(**kw):
    """A dense ping-pong (tapk_pp_epi) dgrad spec, s3's family (t256x256x32 w4x2)."""
    base = dict(
        problem=ConvProblem(N=1, Hi=18, Wi=18, C=256, K=128, Y=3, X=3, pH=0, pW=0),
        data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        tile_m=256, tile_n=256, tile_k=32, warp_m=4, warp_n=2,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
        pipeline="mem", epilogue="cshuffle", lds_k_outer=False,
        tap_outer_k=True, tapk_prefetch=1, b_kpack=4, lds_xor=True, lds_k_pad=0,
        tapk_pingpong=4, tapk_pp_sched=5, tapk_pp_epi=2, cshuffle_m_chunks=2,
    )
    base.update(kw)
    return DgradConvSpec(**base)


class TestR5PpEpilogueKnobs(unittest.TestCase):
    def test_pp_dense_epilogue(self):
        for kw in (dict(xh_epi_lin=True), dict(epi_lds_barriers=True),
                   dict(xh_epi_lin=True, epi_lds_barriers=True),
                   dict(xh_epi_lin=True, epi_lds_barriers=True, tapk_pp_epi=1)):
            s = _pp(**kw)
            with self.subTest(name=s.kernel_name()):
                ok, why = is_valid_dgrad_spec(s, _ARCH)
                self.assertTrue(ok, why)
                build_implicit_gemm_conv_dgrad(s, arch=_ARCH)
        self.assertIn("_xel", _pp(xh_epi_lin=True).kernel_name())
        self.assertNotIn("_xel", _pp().kernel_name())

    def test_pp_rejections(self):
        for kw in (dict(xh_epi_lin=True, tapk_pp_epi=0),
                   dict(epi_lds_barriers=True, tapk_pingpong=0, tapk_pp_sched=0, tapk_pp_epi=0),
                   dict(xh_epi_lin=True, split_k=3)):
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                self.assertFalse(is_valid_dgrad_spec(_pp(**kw), _ARCH)[0])

    def test_pp_aot_keys(self):
        sets = aot_jobs._parse_dgrad_pp_knobs("bpp=4,pps=5,ppe=2,xel=1,elb=1,@-t256x256x32-")
        cfg = aot_jobs._dgrad_pp_knob_cfg(sets[1], epilogue="cshuffle")
        self.assertTrue(cfg["xh_epi_lin"])
        self.assertTrue(cfg["epi_lds_barriers"])


if __name__ == "__main__":
    unittest.main()
