"""Forward x-halo loop with direct-to-LDS staging (``fxh_dma``, tag ``fxd<n>``).

GPU: against the fp32 reference -- x-halo workgroups (Wo 64 / 128: the
wave-private variants run), workgroups that fall back (Wo not a multiple of
the wave's M extent for >= 1000, range exceeded, padding), K not a multiple of
the tile, grouped, two images, tap-fastest order. CPU: rejections and the
ROCKE_AOT_FWD_FXD axis.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import pytest

from tests.test_conv_fwd_3d_kloop_knobs import _SKIP, _TOL, _problem, _run

_BASE = dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True, xh_stride_m=True,
             lds_kpack_read=True, xh_wait_smem=True, lds_xor_rowkey=True, epi_quad_xpose=4,
             uk_xhalo=8, rpf_setprio=1)
_TILE = (256, 64, 32, 4, 1, 16)


@pytest.mark.gpu
@unittest.skipIf(_SKIP, _SKIP)
class TestFwd3dXHaloDma(unittest.TestCase):
    def test_fxh_dma(self):
        nopad = dict(pH=0, pW=0, pD=0)
        probs = [
            _problem(C=64, Hi=10, Wi=66, Di=4, **nopad),             # Wo 64: wave-private ok
            _problem(C=128, K=72, Hi=6, Wi=130, Di=5, **nopad),      # Wo 128, K tail
            _problem(C=64, Hi=10, Wi=34, Di=5, **nopad),             # Wo 32: >= 1000 falls back
            _problem(C=64, Hi=8, Wi=66, Di=4),                       # padded: fallback
            _problem(C=256, K=128, groups=2, Hi=6, Wi=66, Di=4, **nopad),
            _problem(N=2, C=64, Hi=5, Wi=66, Di=4, **nopad),         # images straddled
        ]
        for knobs in (
            dict(fxh_dma=1), dict(fxh_dma=4), dict(fxh_dma=20), dict(fxh_dma=1036),
            dict(fxh_dma=2016), dict(fxh_dma=1036, no_agpr=True),
            dict(fxh_dma=1040, uk_tap_fast=True, uk_chunk_group=2),
        ):
            for i, prob in enumerate(probs):
                with self.subTest(prob=i, **knobs):
                    err = _run(prob, tile=_TILE, **_BASE, **knobs)
                    self.assertLess(err, _TOL, f"fxd {knobs}: rel err {err:.3e}")


class TestFwdXHaloDmaSpec(unittest.TestCase):
    """CPU: rejections, the tag, and the AOT axis (adds variants only)."""

    def _spec(self, **kw):
        from kernels.common.conv_implicit_gemm import ConvDataSpec, ImplicitGemmConvSpec

        base = dict(problem=_problem(C=64, Hi=10, Wi=66, Di=4, pH=0, pW=0, pD=0),
                    data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
                    tile_m=256, tile_n=64, tile_k=32, warp_m=4, warp_n=1,
                    warp_tile_m=16, warp_tile_n=16, warp_tile_k=16, pipeline="mem",
                    epilogue="cshuffle")
        base.update(_BASE)
        base.update(kw)
        return ImplicitGemmConvSpec(**base)

    def test_rejections_and_tag(self):
        from kernels.common.conv_implicit_gemm import _fwd_kloop_conflict

        self.assertFalse(_fwd_kloop_conflict(self._spec(fxh_dma=1036)))
        self.assertIn("fxd1036", self._spec(fxh_dma=1036).kernel_name())
        for bad in (dict(fxh_dma=20, defer_mfma=2), dict(fxh_dma=20, vmem_interleave=2),
                    dict(fxh_dma=20, xh_stride_m=False), dict(fxh_dma=-1),
                    dict(fxh_dma=1036, xh_wait_smem=False)):
            self.assertTrue(_fwd_kloop_conflict(self._spec(**bad)), bad)

    def test_aot_axis_adds_only(self):
        from kernels.conv import aot_jobs as A

        loops = [dict(uk_xhalo=4, xh_stride_m=True, lds_kpack_read=True, defer_mfma=2,
                      vmem_interleave=2, epi_quad_xpose=4, xh_wait_smem=True, lds_xor_rowkey=True),
                 dict(uk_xhalo=4, xh_stride_m=True, lds_kpack_read=True, epi_quad_xpose=4,
                      xh_wait_smem=True, lds_xor_rowkey=True, lds_ahead=True),   # lah twin: skipped
                 dict(uk_xhalo=4, epi_quad_xpose=4), dict(reg_prefetch=1)]
        with mock.patch.dict(os.environ, {"ROCKE_AOT_FWD_FXD": ""}):
            self.assertEqual(A._fwd_fxd_k_loops(list(loops)), loops)
        with mock.patch.dict(os.environ, {"ROCKE_AOT_FWD_FXD": "20,1036"}):
            out = A._fwd_fxd_k_loops(list(loops))
        self.assertEqual(out[:4], loops)
        self.assertEqual(sorted(kl["fxh_dma"] for kl in out[4:]), [20, 1036])
        for kl in out[4:]:
            self.assertEqual((kl["defer_mfma"], kl["vmem_interleave"]), (0, 0))


@pytest.mark.gpu
@unittest.skipIf(_SKIP, _SKIP)
class TestFwd3dMBandSplitTail(unittest.TestCase):
    """m_band with split_tail: the band order over the full tiles' whole M rows,
    the split tail in launch order (18x18 planes: 5 M tiles of 64 per plane)."""

    def test_m_band_split_tail(self):
        tile = (64, 64, 64, 2, 2, 32)
        prob = _problem(Hi=18, Wi=18, Di=10, pH=1, pW=1, pD=1, K=128)  # 51 M x 2 N tiles
        for mb in (1, 2):
            for full in (0, 40, 61, 102):
                with self.subTest(mb=mb, full=full):
                    err = _run(prob, tile=tile, split_k=2, degree=3, split_tail=True,
                               full_tiles=full, uniform_k_decode=True, reg_prefetch=1,
                               xcd_remap=6, lds_xor=True, m_band=mb)
                    self.assertLess(err, _TOL, f"mb{mb} full{full}: rel err {err:.3e}")
