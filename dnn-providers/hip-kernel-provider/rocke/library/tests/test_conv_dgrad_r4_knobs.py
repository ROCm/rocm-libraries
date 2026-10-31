# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the opt-in dgrad knobs tapk_pp_persist, tapk_pp_xhalo,
tapk_ztrim, xh_fast_a and the partial-pass ping-pong dY tile
(pp_a_partial): validation, names, identity labels, IR build, launch grid."""

from __future__ import annotations

import unittest

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_implicit_gemm import ConvDataSpec
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    build_implicit_gemm_conv_dgrad,
    dgrad_persistent_grid,
    is_valid_dgrad_spec,
    pp_a_partial,
)

_ARCH = "gfx942"


def _problem(**kw):
    base = dict(
        N=1, Di=10, Hi=10, Wi=10, C=256, K=128, Z=3, Y=3, X=3, sD=1, pD=0, dD=1, pH=0, pW=0
    )
    base.update(kw)
    return ConvProblem(**base)


def _pp(problem=None, **kw):
    base = dict(
        problem=problem if problem is not None else _problem(),
        data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        tile_m=256, tile_n=128, tile_k=64, warp_m=4, warp_n=2,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
        pipeline="mem", epilogue="cshuffle", lds_k_outer=False,
        tap_outer_k=True, tapk_prefetch=1, b_kpack=4,
        lds_xor=True, lds_k_pad=0, vector_size_a=8,
        tapk_pingpong=4, tapk_pp_sched=2,
    )
    base.update(kw)
    return DgradConvSpec(**base)


def _xh(problem=None, **kw):
    base = dict(
        problem=problem if problem is not None else _problem(),
        data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        tile_m=256, tile_n=64, tile_k=32, warp_m=4, warp_n=1,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=8,
        pipeline="mem", epilogue="cshuffle", lds_k_outer=False,
        tap_outer_k=True, tapk_prefetch=3, tapk_xhalo=True, xhalo_exact=True,
        b_kpack=4, lds_xor=True, lds_k_pad=0, xh_a_k4=True, xh_order=6,
        cshuffle_m_chunks=2,
    )
    base.update(kw)
    return DgradConvSpec(**base)


class TestR4Knobs(unittest.TestCase):
    def test_default_off_names(self):
        for s in (_pp(), _xh()):
            name = s.kernel_name()
            for tag in ("_ppp", "_ppx", "_zt", "_xfa", "_pbs"):
                self.assertNotIn(tag, name)
        self.assertFalse(pp_a_partial(_pp()))

    def test_valid_build_and_named(self):
        cases = [
            (_pp(tapk_pp_persist=1), "_ppp1"),
            (_pp(tapk_pp_persist=2), "_ppp2"),
            (_pp(tapk_pp_xhalo=1), "_ppx1"),
            (_pp(tapk_pp_xhalo=2), "_ppx2"),
            (_pp(tapk_pp_xhalo=3, tapk_pp_epi=2), "_ppx3"),
            (_pp(tapk_pp_xhalo=4, tapk_pp_epi=2), "_ppx4"),
            (_pp(tapk_pp_xhalo=4, tile_m=512, tile_k=32, tapk_pp_epi=2, cshuffle_m_chunks=2), "_ppx4"),
            (_pp(tapk_ztrim=1), "_zt1"),
            (_pp(tapk_pp_bslab=1), "_pbs1"),
            (_pp(tapk_pp_bslab=2, tile_m=256, tile_n=256, tile_k=32, cshuffle_m_chunks=2), "_pbs2"),
            (_pp(tapk_pp_bslab=3, tile_m=256, tile_n=256, tile_k=32, cshuffle_m_chunks=2), "_pbs3"),
            (_pp(tapk_pp_bslab=1, tile_m=224, tile_n=256, tile_k=32, warp_m=2, warp_n=4,
                 cshuffle_m_chunks=7), "_pbs1"),
            (_pp(tapk_pp_xhalo=1, tapk_ztrim=1), "_zt1"),
            (_xh(xh_fast_a=1), "_xfa1"),
            (_xh(xh_fast_a=2, xh_epi=1), "_xfa2"),
            (_xh(xh_epi=1), "_xep1"),
            (_xh(xh_epi=2, xh_fast_a=1), "_xep2"),
            (_xh(xh_persist=2, xh_epi=1, xh_fast_a=1), "_xhp2"),
            (_xh(xh_epi=1, xh_fast_a=1, warp_tile_m=16, warp_tile_n=16, warp_tile_k=16), "_xep1"),
        ]
        for s, tag in cases:
            with self.subTest(tag=tag):
                ok, why = is_valid_dgrad_spec(s, _ARCH)
                self.assertTrue(ok, why)
                self.assertIn(tag, s.kernel_name())
                build_implicit_gemm_conv_dgrad(s, arch=_ARCH)

    def test_partial_dy_tile(self):
        kw = dict(tile_m=224, tile_n=256, tile_k=32, warp_m=2, warp_n=4, cshuffle_m_chunks=7)
        s = _pp(**kw)
        self.assertTrue(pp_a_partial(s))
        ok, why = is_valid_dgrad_spec(s, _ARCH)
        self.assertTrue(ok, why)
        build_implicit_gemm_conv_dgrad(s, arch=_ARCH)
        # Without the ping-pong the generic loader still refuses the split.
        s0 = _pp(**kw, tapk_pingpong=0, tapk_pp_sched=0, lds_xor=False, lds_k_pad=None)
        self.assertFalse(pp_a_partial(s0))
        self.assertFalse(is_valid_dgrad_spec(s0, _ARCH)[0])

    def test_rejections(self):
        p2d = ConvProblem(N=1, Hi=10, Wi=10, C=256, K=128, Y=3, X=3, pH=0, pW=0)
        bad = [
            _pp(tapk_pp_persist=3),
            _pp(tapk_pp_xhalo=5),
            _pp(tapk_pp_bslab=4),
            _pp(tapk_pp_bslab=1, b_kpack=2),
            _pp(tapk_pp_bslab=1, tapk_pp_xhalo=1),
            _pp(tapk_pp_xhalo=4, problem=_problem(dW=2)),
            _pp(tapk_pp_xhalo=4, warp_tile_m=32, warp_tile_n=32, warp_tile_k=8),
            _pp(tapk_pp_persist=2, tapk_pp_epi=2),
            _pp(tapk_pp_xhalo=1, tapk_pp_sched=1),
            _pp(tapk_pp_xhalo=1, problem=_problem(pW=1)),
            _pp(tapk_ztrim=1, problem=p2d),
            _pp(tapk_ztrim=1, tapk_pp_persist=1),
            _xh(xh_fast_a=1, xhalo_exact=False),
            _xh(xh_fast_a=3),
            _xh(xh_epi=1, epilogue="default"),
            _xh(xh_epi=3),
            _xh(xh_persist=3),
        ]
        for s in bad:
            with self.subTest(name=s.kernel_name()):
                self.assertFalse(is_valid_dgrad_spec(s, _ARCH)[0])

    def test_persistent_grid(self):
        self.assertEqual(dgrad_persistent_grid(8517, 0, 228), 8517)
        self.assertEqual(dgrad_persistent_grid(8517, 1, 228), 228)
        self.assertEqual(dgrad_persistent_grid(100, 1, 228), 100)
        self.assertEqual(dgrad_persistent_grid(8517, 1, None), 8517)


class TestR4Identity(unittest.TestCase):
    def test_labels_and_aot_keys(self):
        from kernels.conv.aot_jobs import _dgrad_pp_knob_cfg, _parse_dgrad_pp_knobs
        from kernels.conv.kernel_identity import KernelIdentity

        sets = _parse_dgrad_pp_knobs("bpp=4,pps=2,ppp=1,ppx=1,zt=1,@-t256")
        cfg = _dgrad_pp_knob_cfg(sets[1], epilogue="cshuffle")
        self.assertEqual(
            (cfg["tapk_pp_persist"], cfg["tapk_pp_xhalo"], cfg["tapk_ztrim"]), (1, 1, 1)
        )
        base = KernelIdentity(
            arch=_ARCH, direction="dgrad", algorithm="implicit_gemm",
            dtype_a="bf16", dtype_b="bf16", dtype_d="bf16",
            tile_m=256, tile_n=128, tile_k=64, warp_m=4, warp_n=2,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            pipeline="mem", epilogue="cshuffle", wave_size=64,
            vector_size_a=8, vector_size_b=8, vector_size_c=8,
            tap_outer_k=True, tapk_prefetch=1, b_kpack=4,
        )
        for field, tag in (
            ("tapk_pp_persist", "ppp1"),
            ("tapk_pp_xhalo", "ppx1"),
            ("tapk_ztrim", "zt1"),
            ("xh_fast_a", "xfa1"),
            ("xh_epi", "xep1"),
        ):
            with self.subTest(field=field):
                on = KernelIdentity(**{**base.to_dict(), field: 1})
                self.assertIn(f"-{tag}", on.label())
                self.assertNotEqual(on.stable_hash(), base.stable_hash())
                self.assertNotIn(tag, base.label())


if __name__ == "__main__":
    unittest.main()
