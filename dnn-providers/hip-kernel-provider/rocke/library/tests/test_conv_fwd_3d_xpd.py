"""Forward x-halo DMA loop on padded problems (``xh_pad``, tag ``xpd``).

GPU: against the fp32 reference -- the wave-private DMA loop (fxh_dma >= 1000)
and its input-row ring (xh_yring) on problems padded on every axis, on H/W
only, on D only and unpadded, with a K tail, groups, two images and a problem
whose wave ranges cross an output row (falls back to the plain loop). The
tolerance is tight enough that one non-zero halo row in a tile fails. CPU:
rejections, the tag, the label round trip and the ROCKE_AOT_FWD_XPD axis.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import pytest

from tests.test_conv_fwd_3d_kloop_knobs import _SKIP, _problem, _run

_BASE = dict(
    uniform_k_decode=True,
    reg_prefetch=1,
    lds_xor=True,
    xh_stride_m=True,
    lds_kpack_read=True,
    xh_wait_smem=True,
    lds_xor_rowkey=True,
    epi_quad_xpose=4,
    rpf_setprio=1,
    xh_pad=True,
)
_TILE = (256, 64, 32, 4, 1, 16)
# bf16 output of a 27*C-term sum: a missed zero halo row moves the result by
# a whole tap's contribution, far above the rounding error.
_TOL = 1e-2


@pytest.mark.gpu
@unittest.skipIf(_SKIP, _SKIP)
class TestFwd3dXHaloPad(unittest.TestCase):
    def _check(self, probs, knob_sets):
        for knobs in knob_sets:
            for i, prob in enumerate(probs):
                with self.subTest(prob=i, **knobs):
                    err = _run(prob, tile=_TILE, **_BASE, **knobs)
                    self.assertLess(
                        err, _TOL, f"xpd {knobs} prob {i}: rel err {err:.3e}"
                    )

    def test_xh_pad_dma(self):
        probs = [
            _problem(C=64, Hi=6, Wi=64, Di=4),  # pad (1,1,1), Wo 64
            _problem(C=64, Hi=5, Wi=128, Di=3),  # pad (1,1,1), Wo 128
            _problem(C=64, Hi=6, Wi=64, Di=6, pD=0),  # pad (0,1,1)
            _problem(C=64, Hi=6, Wi=66, Di=4, pH=0, pW=0),  # pad (1,0,0)
            _problem(C=64, Hi=6, Wi=66, Di=4, pH=0, pW=0, pD=0),  # unpadded
            _problem(C=128, K=72, Hi=4, Wi=64, Di=3),  # K tail
            _problem(C=256, K=128, groups=2, Hi=4, Wi=64, Di=3),  # grouped
            _problem(N=2, C=64, Hi=4, Wi=64, Di=3),  # two images
            _problem(C=64, Hi=6, Wi=40, Di=3),  # Wo 40: falls back
        ]
        self._check(
            probs,
            (
                dict(uk_xhalo=4, fxh_dma=1036),
                dict(uk_xhalo=8, fxh_dma=2016),
                dict(uk_xhalo=8, fxh_dma=1036, no_agpr=True),
            ),
        )

    def test_xh_pad_yring(self):
        probs = [
            _problem(C=64, Hi=8, Wi=64, Di=4),  # Wo 64: 4 rows per tile
            _problem(C=128, K=72, Hi=6, Wi=128, Di=3),  # Wo 128: 2 rows, K tail
            _problem(C=64, Hi=8, Wi=64, Di=6, pD=0),  # pad (0,1,1)
            _problem(C=64, Hi=6, Wi=64, Di=4),  # plane 6 x 64: falls back
            _problem(C=256, K=128, groups=2, Hi=4, Wi=64, Di=3),
            _problem(N=2, C=64, Hi=4, Wi=64, Di=3),
        ]
        self._check(
            probs,
            (
                dict(xh_yring=4, fxh_dma=1036, uk_xhalo=8, uk_tap_fast=True),
                dict(xh_yring=14, fxh_dma=1036, uk_xhalo=8, uk_tap_fast=True),
                dict(xh_yring=24, fxh_dma=1024, uk_xhalo=8, uk_tap_fast=True),
                dict(xh_yring=12, fxh_dma=1024, uk_xhalo=4, uk_tap_fast=True),
            ),
        )


class TestFwdXHaloPadSpec(unittest.TestCase):
    """CPU: rejections, the tag, the label and the AOT axis (adds variants only)."""

    def _spec(self, **kw):
        from kernels.common.conv_implicit_gemm import ConvDataSpec, ImplicitGemmConvSpec

        base = dict(
            problem=_problem(C=64, Hi=10, Wi=64, Di=4),
            data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
            tile_m=256,
            tile_n=64,
            tile_k=32,
            warp_m=4,
            warp_n=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            uk_xhalo=8,
            fxh_dma=1036,
        )
        base.update(_BASE)
        base.update(kw)
        return ImplicitGemmConvSpec(**base)

    def test_rejections_and_tag(self):
        from kernels.common.conv_implicit_gemm import _fwd_kloop_conflict

        self.assertFalse(_fwd_kloop_conflict(self._spec()))
        self.assertFalse(_fwd_kloop_conflict(self._spec(xh_yring=14, uk_tap_fast=True)))
        self.assertIn("xpd", self._spec().kernel_name())
        self.assertNotIn("xpd", self._spec(xh_pad=False).kernel_name())
        for bad in (dict(fxh_dma=36), dict(fxh_dma=0)):
            self.assertTrue(_fwd_kloop_conflict(self._spec(**bad)), bad)

    def test_label_round_trip(self):
        from kernels.conv.kernel_identity import KernelIdentity

        lab = (
            "fwd-implicit_gemm-bf16-3d-t256x64x32-w4x1-a16x16x16-v8x8x8-mem-cshuffle-grp-xcd6-ukd-rpf1-"
            "lxor-kprd-uxh4-prio2-lxrk-xws-eqx4-nag-mb32-xsm-fxd1036-xpd-wave64-gfx942-llvm22"
        )
        ident = KernelIdentity.from_label(lab)
        import dataclasses

        self.assertTrue(ident.xh_pad)
        self.assertEqual(ident.label(), lab)
        base = dataclasses.replace(ident, xh_pad=False)
        self.assertNotIn("xpd", base.label())
        self.assertNotEqual(base.stable_hash(), ident.stable_hash())

    def test_aot_axis_adds_only(self):
        from kernels.conv import aot_jobs as A

        loops = [
            dict(uk_xhalo=4, xh_stride_m=True, fxh_dma=1036),
            dict(
                uk_xhalo=8,
                xh_stride_m=True,
                fxh_dma=1036,
                xh_yring=14,
                uk_tap_fast=True,
            ),
            dict(uk_xhalo=4, xh_stride_m=True, fxh_dma=20),
            dict(reg_prefetch=1),
        ]
        with mock.patch.dict(os.environ, {"ROCKE_AOT_FWD_XPD": ""}):
            self.assertEqual(A._fwd_xpd_k_loops(list(loops)), loops)
        with mock.patch.dict(os.environ, {"ROCKE_AOT_FWD_XPD": "1"}):
            out = A._fwd_xpd_k_loops(list(loops))
        self.assertEqual(out[:4], loops)
        self.assertEqual(len(out), 6)
        for kl in out[4:]:
            self.assertTrue(kl["xh_pad"])
            self.assertGreaterEqual(kl["fxh_dma"], 1000)
