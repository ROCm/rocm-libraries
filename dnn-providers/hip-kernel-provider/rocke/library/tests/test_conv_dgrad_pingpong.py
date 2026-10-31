# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the dgrad block ping-pong loop knobs (DgradConvSpec.tapk_pingpong,
tapk_pp_sched, tapk_pp_epi): validation, names, identity labels, IR build."""

from __future__ import annotations

import unittest

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_implicit_gemm import ConvDataSpec
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    build_implicit_gemm_conv_dgrad,
    is_valid_dgrad_spec,
)

_ARCH = "gfx942"


def _problem(**kw):
    base = dict(N=1, Hi=18, Wi=18, C=256, K=128, Y=3, X=3, pH=0, pW=0)
    base.update(kw)
    return ConvProblem(**base)


def _spec(problem=None, **kw):
    base = dict(
        problem=problem if problem is not None else _problem(),
        data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        tile_m=128, tile_n=256, tile_k=64, warp_m=2, warp_n=4,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
        pipeline="mem", epilogue="cshuffle", lds_k_outer=False,
        tap_outer_k=True, tapk_prefetch=1, b_kpack=4,
        lds_xor=True, lds_k_pad=0,
        tapk_pingpong=4, tapk_pp_sched=2,
    )
    base.update(kw)
    return DgradConvSpec(**base)


class TestPingpongSpec(unittest.TestCase):
    def test_default_off(self):
        s = _spec(tapk_pingpong=0, tapk_pp_sched=0, lds_xor=False, lds_k_pad=None)
        self.assertNotIn("bpp", s.kernel_name())
        self.assertNotIn("pps", s.kernel_name())

    def test_valid_and_named(self):
        for bpp, pps in ((1, 0), (2, 0), (3, 0), (3, 2), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (5, 2)):
            with self.subTest(bpp=bpp, pps=pps):
                s = _spec(tapk_pingpong=bpp, tapk_pp_sched=pps)
                ok, why = is_valid_dgrad_spec(s, _ARCH)
                self.assertTrue(ok, why)
                self.assertIn(f"_bpp{bpp}", s.kernel_name())
                build_implicit_gemm_conv_dgrad(s, arch=_ARCH)

    def test_two_k_atom_tile(self):
        # t256x256x32: 8 waves of 128x64, one b128 per operand row feeds both
        # compute phases.
        s = _spec(tile_m=256, tile_n=256, tile_k=32, cshuffle_m_chunks=4)
        ok, why = is_valid_dgrad_spec(s, _ARCH)
        self.assertTrue(ok, why)
        build_implicit_gemm_conv_dgrad(s, arch=_ARCH)

    def test_split_k_and_epilogue_knob(self):
        ok, why = is_valid_dgrad_spec(_spec(split_k=3), _ARCH)
        self.assertTrue(ok, why)
        build_implicit_gemm_conv_dgrad(_spec(split_k=3), arch=_ARCH)
        ok, why = is_valid_dgrad_spec(_spec(tapk_pp_epi=1, cshuffle_m_chunks=2), _ARCH)
        self.assertTrue(ok, why)
        build_implicit_gemm_conv_dgrad(_spec(tapk_pp_epi=1, cshuffle_m_chunks=2), arch=_ARCH)
        for kw in (dict(), dict(tile_m=256, tile_n=256, tile_k=32, cshuffle_m_chunks=4, tapk_pp_sched=5)):
            s = _spec(tapk_pp_epi=2, **kw)
            ok, why = is_valid_dgrad_spec(s, _ARCH)
            self.assertTrue(ok, why)
            self.assertIn("_ppe2", s.kernel_name())
            build_implicit_gemm_conv_dgrad(s, arch=_ARCH)

    def test_rejected(self):
        cases = (
            dict(tapk_xhalo=True),               # x-halo loop
            dict(warp_m=2, warp_n=2),            # 4 waves
            dict(tapk_pp_sched=0),               # bpp4 needs the two-ahead loop
            dict(problem=_problem(Y=5, X=5)),    # tap mask: <= 3 per axis
            dict(tapk_pingpong=6),
            dict(tapk_pingpong=3, b_kpack=0),
            dict(tapk_pp_epi=1, epilogue="default"),
            dict(tapk_pp_epi=1, split_k=2),
            dict(tapk_pp_epi=2, split_k=2),
            dict(tapk_pp_epi=3),
            dict(tapk_pingpong=0, tapk_pp_sched=2),
            dict(tapk_pp_sched=8),
        )
        for kw in cases:
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                ok, _why = is_valid_dgrad_spec(_spec(**kw), _ARCH)
                self.assertFalse(ok)


class TestPingpongIdentity(unittest.TestCase):
    def _ident(self, **kw):
        from benchmarks.common.kernel_cache import KernelIdentity

        base = dict(
            arch=_ARCH, direction="dgrad", algorithm="implicit_gemm",
            dtype_a="bf16", dtype_b="bf16", dtype_d="bf16",
            tile_m=128, tile_n=256, tile_k=64, warp_m=2, warp_n=4,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            pipeline="mem", epilogue="cshuffle", wave_size=64,
            vector_size_a=8, vector_size_b=8, vector_size_c=8,
            lds_k_outer=False, grouped=True, is_3d=True,
            stride_h=1, stride_w=1, dilation_h=1, dilation_w=1, max_sub_gemms=1024,
            tap_outer_k=True, tapk_prefetch=1, b_kpack=4,
        )
        base.update(kw)
        return KernelIdentity(**base)

    def test_label_round_trip_and_hash(self):
        off = self._ident()
        on = self._ident(tapk_pingpong=4, tapk_pp_sched=2, tapk_pp_epi=1)
        self.assertNotEqual(off.stable_hash(), on.stable_hash())
        for tag in ("bpp4", "pps2", "ppe1"):
            self.assertIn(tag, on.label())
        self.assertEqual(type(on).from_label(on.label()), on)

    def test_supports_filters_up_to_3(self):
        from kernels.conv.kernel_identity import identity_supports_problem

        ident = self._ident(tapk_pingpong=4, tapk_pp_sched=2)
        p3 = ConvProblem(N=1, Di=10, Hi=10, Wi=10, C=256, K=128, Z=3, Y=3, X=3, sD=1, pD=0, dD=1)
        p5 = ConvProblem(
            N=1, Di=10, Hi=10, Wi=10, C=256, K=128, Z=5, Y=5, X=5, sD=1, pD=2, dD=1, pH=2, pW=2
        )
        self.assertTrue(identity_supports_problem(ident, p3)[0])
        self.assertFalse(identity_supports_problem(ident, p5)[0])


if __name__ == "__main__":
    unittest.main()
