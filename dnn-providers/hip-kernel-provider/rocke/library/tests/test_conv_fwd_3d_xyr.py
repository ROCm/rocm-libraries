"""Forward x-halo input-row ring over the y taps (``xh_yring``, tag ``xyr<n>``).

GPU: against the fp32 reference -- ring workgroups (Wo 128 with 2 rows per
tile, Wo 64 with 4), problems that fall back (rows per tile mismatched, a
plane that is not whole tiles, padding), K not a multiple of the tile, grouped,
two images; every mode (+10 early reload, +20 early ring row). CPU:
rejections, the tag and the ROCKE_AOT_FWD_XYR axis.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import pytest

from tests.test_conv_fwd_3d_kloop_knobs import _SKIP, _TOL, _problem, _run

_BASE = dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True, xh_stride_m=True,
             lds_kpack_read=True, xh_wait_smem=True, lds_xor_rowkey=True, epi_quad_xpose=4,
             rpf_setprio=1, uk_tap_fast=True)
_TILE = (256, 64, 32, 4, 1, 16)


@pytest.mark.gpu
@unittest.skipIf(_SKIP, _SKIP)
class TestFwd3dXHaloYRing(unittest.TestCase):
    def test_xh_yring(self):
        nopad = dict(pH=0, pW=0, pD=0)
        probs = [
            _problem(C=64, Hi=10, Wi=66, Di=4, **nopad),             # Wo 64: 4 rows per tile
            _problem(C=128, K=72, Hi=6, Wi=130, Di=5, **nopad),      # Wo 128: 2 rows, K tail
            _problem(C=64, Hi=7, Wi=66, Di=4, **nopad),              # plane 5 x 64: falls back
            _problem(C=64, Hi=8, Wi=66, Di=4),                       # padded: fallback
            _problem(C=256, K=128, groups=2, Hi=6, Wi=66, Di=4, **nopad),
            _problem(N=2, C=64, Hi=6, Wi=66, Di=4, **nopad),         # two images
        ]
        for knobs in (
            dict(xh_yring=2, fxh_dma=1008, uk_xhalo=4), dict(xh_yring=12, fxh_dma=1024, uk_xhalo=4),
            dict(xh_yring=22, fxh_dma=1036, uk_xhalo=4, no_agpr=True),
            dict(xh_yring=4, fxh_dma=1036, uk_xhalo=8), dict(xh_yring=14, fxh_dma=1036, uk_xhalo=8),
            dict(xh_yring=24, fxh_dma=1024, uk_xhalo=8),
        ):
            for i, prob in enumerate(probs):
                with self.subTest(prob=i, **knobs):
                    err = _run(prob, tile=_TILE, **_BASE, **knobs)
                    self.assertLess(err, _TOL, f"xyr {knobs}: rel err {err:.3e}")


class TestFwdXHaloYRingSpec(unittest.TestCase):
    """CPU: rejections, the tag, and the AOT axis (adds variants only)."""

    def _spec(self, **kw):
        from kernels.common.conv_implicit_gemm import ConvDataSpec, ImplicitGemmConvSpec

        base = dict(problem=_problem(C=64, Hi=10, Wi=66, Di=4, pH=0, pW=0, pD=0),
                    data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
                    tile_m=256, tile_n=64, tile_k=32, warp_m=4, warp_n=1,
                    warp_tile_m=16, warp_tile_n=16, warp_tile_k=16, pipeline="mem",
                    epilogue="cshuffle", uk_xhalo=8, fxh_dma=1036)
        base.update(_BASE)
        base.update(kw)
        return ImplicitGemmConvSpec(**base)

    def test_rejections_and_tag(self):
        from kernels.common.conv_implicit_gemm import _fwd_kloop_conflict

        for ok in (4, 14, 24, 2, 12, 22):
            self.assertFalse(_fwd_kloop_conflict(self._spec(xh_yring=ok)), ok)
        self.assertIn("xyr14", self._spec(xh_yring=14).kernel_name())
        self.assertNotIn("xyr", self._spec().kernel_name())
        for bad in (dict(xh_yring=3), dict(xh_yring=34), dict(xh_yring=4, fxh_dma=20),
                    dict(xh_yring=4, uk_tap_fast=False), dict(xh_yring=4, uk_chunk_group=2)):
            self.assertTrue(_fwd_kloop_conflict(self._spec(**bad)), bad)

    def test_aot_axis_adds_only(self):
        from kernels.conv import aot_jobs as A

        loops = [dict(uk_xhalo=4, xh_stride_m=True, fxh_dma=1036),
                 dict(uk_xhalo=8, xh_stride_m=True, fxh_dma=2016, uk_tap_fast=True, uk_chunk_group=2),
                 dict(uk_xhalo=4, xh_stride_m=True, fxh_dma=20), dict(reg_prefetch=1)]
        with mock.patch.dict(os.environ, {"ROCKE_AOT_FWD_XYR": ""}):
            self.assertEqual(A._fwd_xyr_k_loops(list(loops)), loops)
        with mock.patch.dict(os.environ, {"ROCKE_AOT_FWD_XYR": "12,14"}):
            out = A._fwd_xyr_k_loops(list(loops))
        self.assertEqual(out[:4], loops)
        self.assertEqual(len(out), 8)
        for kl in out[4:]:
            self.assertIn(kl["xh_yring"], (12, 14))
            self.assertTrue(kl["uk_tap_fast"])
            self.assertNotIn("uk_chunk_group", kl)
            self.assertGreaterEqual(kl["fxh_dma"], 1000)
