# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Forward 3-D K-loop knobs of the uniform_k_decode / reg_prefetch=1 loop:

* ``lds_kpack_read`` -- k-packed 16-byte LDS fragment reads (permuted MFMA k
  order, 128-byte lds_xor key);
* ``uk_chunk_group`` -- channel chunks grouped inside the tap-fastest order;
* ``rpf_lds_barrier`` -- LDS-only barriers in the reg_prefetch=1 loop;
* ``rpf_fetch_late`` -- next tile's global reads behind the first LDS reads.

GPU: each against the fp32 reference (padding on every axis, several channel
chunks so the grouped order wraps). CPU: rejections, name tags, AOT env axes.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import pytest

from tests.test_conv_fwd_3d_kloop_knobs import _SKIP, _TOL, _problem, _run


@pytest.mark.gpu
@unittest.skipIf(_SKIP, _SKIP)
class TestFwd3dLdsKnobs(unittest.TestCase):
    def _check(self, problem, **kw):
        err = _run(problem, **kw)
        self.assertLess(err, _TOL, f"{kw}: rel err {err:.3e}")

    def test_lds_knobs_reg_prefetch(self):
        base = dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True)
        for tile in ((64, 64, 32, 2, 2, 16), (64, 64, 64, 2, 2, 16), (64, 64, 64, 2, 2, 32)):
            for knobs in (
                dict(lds_kpack_read=True),
                dict(lds_kpack_read=True, rpf_lds_barrier=True),
                dict(lds_kpack_read=True, rpf_fetch_late=True, lds_ahead=True),
                dict(lds_kpack_read=True, defer_mfma=1, rpf_fetch_late=True),
                dict(uk_tap_fast=True, uk_chunk_group=2, lds_kpack_read=True),
                dict(uk_tap_fast=True, uk_chunk_group=2, rpf_lds_barrier=True),
            ):
                if knobs.get("defer_mfma") and tile[2] == 32 and tile[5] == 16:
                    continue  # 2 kk steps: defer 1 is fine, kept for tile_k 64 only
                with self.subTest(tile=tile, **knobs):
                    # C=128: 4 (tile_k 32) / 2 (tile_k 64) chunks per tap.
                    self._check(_problem(C=128), tile=tile, **base, **knobs)
                    self._check(_problem(C=256, pH=0, pW=0, pD=0), tile=tile, **base, **knobs)


@pytest.mark.gpu
@unittest.skipIf(_SKIP, _SKIP)
class TestFwd3dXHalo(unittest.TestCase):
    """uk_xhalo against the fp32 reference: workgroups whose staged range fits
    (pad 0), workgroups that straddle a depth plane or exceed the range (they
    fall back to the plain loop), and a padded problem (every workgroup falls
    back)."""

    def _check(self, problem, **kw):
        err = _run(problem, **kw)
        self.assertLess(err, _TOL, f"{kw}: rel err {err:.3e}")

    def test_xhalo(self):
        base = dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True)
        # Ho = Wo = 8: a 64-pixel tile is one depth plane, its staged range
        # 7*10 + 7 + 3 = 80 pixels: uk_xhalo 20 fits, 12 does not (fallback).
        # Ho = Wo = 10 (100-pixel planes): tile 0 fits, tile 1 straddles a
        # plane and falls back. Padded: every workgroup falls back.
        nopad = dict(pH=0, pW=0, pD=0)
        probs = [
            _problem(C=64, Hi=10, Wi=10, Di=6, **nopad),
            _problem(C=128, Hi=12, Wi=12, Di=5, **nopad),
            _problem(C=64, Hi=10, Wi=10, Di=6),
            # Wo = 32: rows end on a 32-row wave extent (xh_dpp's condition).
            _problem(C=64, Hi=6, Wi=34, Di=4, **nopad),
            # Grouped: the x-halo range is offset by the group's channels.
            _problem(C=256, K=128, groups=2, Hi=10, Wi=10, Di=6, **nopad),
            # Batch 2 with 50-pixel planes: tiles straddle planes and images.
            _problem(N=2, C=64, Hi=7, Wi=12, Di=5, **nopad),
        ]
        for tile in ((64, 64, 32, 2, 2, 16), (64, 64, 64, 2, 2, 32)):
            for knobs in (
                dict(uk_xhalo=20),
                dict(uk_xhalo=12),
                dict(uk_xhalo=20, lds_kpack_read=True, rpf_lds_barrier=True),
                dict(uk_xhalo=20, uk_tap_fast=True, lds_ahead=True),
                dict(uk_xhalo=20, uk_tap_fast=True, uk_chunk_group=2, lds_kpack_read=True),
                dict(uk_xhalo=20, xh_dpp=True),
                dict(uk_xhalo=20, xh_dpp=True, lds_kpack_read=True, lds_ahead=True),
                dict(uk_xhalo=20, defer_mfma=1, lds_kpack_read=True, lds_ahead=True),
                dict(uk_xhalo=20, rpf_setprio=1, rpf_lds_barrier=True),
                dict(uk_xhalo=20, defer_mfma=2, rpf_setprio=1, lds_kpack_read=True,
                     uk_tap_fast=True, uk_chunk_group=2),
                # r3: partial waits (xws), interleaved loads, row-keyed swizzle.
                dict(uk_xhalo=20, defer_mfma=2, lds_kpack_read=True, xh_wait_smem=True,
                     vmem_interleave=2, lds_xor_rowkey=True),
            ):
                if knobs.get("lds_xor_rowkey") and tile[2] != 32:
                    continue
                for i, prob in enumerate(probs):
                    with self.subTest(tile=tile, prob=i, **knobs):
                        self._check(prob, tile=tile, **base, **knobs)


@pytest.mark.gpu
@unittest.skipIf(_SKIP, _SKIP)
class TestFwd3dQuadXposeEpilogue(unittest.TestCase):
    """epi_quad_xpose (register-only store of the 16x16 atom) against the fp32
    reference: the x-halo loop and the plain uniform_k_decode loop (B channel
    permutation of modes 2/3 at both B loaders), partial M tiles, K not a
    multiple of the tile (the last N tile's 8-channel runs straddle K), and
    grouped."""

    def _check(self, problem, **kw):
        err = _run(problem, **kw)
        self.assertLess(err, _TOL, f"{kw}: rel err {err:.3e}")

    def test_quad_xpose(self):
        base = dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True)
        nopad = dict(pH=0, pW=0, pD=0)
        probs = [
            _problem(C=64, Hi=10, Wi=10, Di=6, **nopad),
            _problem(C=64, K=72, Hi=10, Wi=10, Di=6, **nopad),
            _problem(C=128, K=40, Hi=7, Wi=9, Di=5),
            _problem(C=256, K=128, groups=2, Hi=10, Wi=10, Di=6, **nopad),
        ]
        tile = (64, 64, 32, 2, 2, 16)
        for mode in (1, 2, 3, 4):
            for knobs in (
                dict(uk_xhalo=20, lds_kpack_read=True, xh_wait_smem=True, defer_mfma=2,
                     vmem_interleave=2, lds_xor_rowkey=True),
                dict(lds_kpack_read=True),
            ):
                for i, prob in enumerate(probs):
                    with self.subTest(mode=mode, prob=i, **knobs):
                        self._check(prob, tile=tile, epi_quad_xpose=mode, **base, **knobs)


@pytest.mark.gpu
@unittest.skipIf(_SKIP, _SKIP)
class TestFwd3dXStrideM(unittest.TestCase):
    """xh_stride_m (M-strided wave rows of the x-halo loop, own A LDS layout,
    strided eqx store map) against the fp32 reference: x-halo workgroups,
    workgroups that fall back (range exceeded, Wo % 4 != 0, padding) with
    the default store map, mixed in one launch, grouped, K not a multiple of
    the tile, and the 4-fragment (64-row) waves of the production tile."""

    def test_xh_stride_m(self):
        base = dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True, xh_stride_m=True)
        nopad = dict(pH=0, pW=0, pD=0)
        probs = [
            _problem(C=64, Hi=10, Wi=10, Di=6, **nopad),           # Wo 8
            _problem(C=64, K=72, Hi=14, Wi=14, Di=5, **nopad),     # Wo 12, tiles straddle planes
            _problem(C=64, Hi=8, Wi=11, Di=5, **nopad),            # Wo 9: fallback
            _problem(C=128, K=40, Hi=7, Wi=9, Di=5),               # padded: fallback
            _problem(C=256, K=128, groups=2, Hi=10, Wi=10, Di=6, **nopad),
            _problem(N=2, C=64, Hi=7, Wi=10, Di=5, **nopad),       # Wo 8, planes/images straddled
        ]
        xk = dict(lds_kpack_read=True, xh_wait_smem=True, defer_mfma=2,
                  vmem_interleave=2, lds_xor_rowkey=True)
        for tile, xh in (((64, 64, 32, 2, 2, 16), 20), ((256, 64, 32, 4, 1, 16), 8)):
            for knobs in (
                dict(uk_xhalo=xh, epi_quad_xpose=4, **xk),
                dict(uk_xhalo=xh, epi_quad_xpose=4, lds_ahead=True, uk_tap_fast=True, **xk),
                dict(uk_xhalo=xh, epi_quad_xpose=3, lds_kpack_read=True),
                dict(uk_xhalo=xh, epi_quad_xpose=4),
            ):
                for i, prob in enumerate(probs):
                    with self.subTest(tile=tile, prob=i, **knobs):
                        err = _run(prob, tile=tile, **base, **knobs)
                        self.assertLess(err, _TOL, f"xsm {knobs}: rel err {err:.3e}")


@pytest.mark.gpu
@unittest.skipIf(_SKIP, _SKIP)
class TestFwd3dMBand(unittest.TestCase):
    """m_band (output-plane band tile order after xcd_remap) against the fp32
    reference: bands applied (Ho*Wo = 256 -> 4 M tiles per plane, mb 2 and 4),
    and the launch-order fallback (mb does not divide the plane's tiles,
    tile_m does not divide Ho*Wo)."""

    def test_m_band(self):
        base = dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True, xcd_remap=6)
        nopad = dict(pH=0, pW=0, pD=0)
        probs = [
            _problem(C=64, Hi=18, Wi=18, Di=7, **nopad),
            _problem(C=64, K=72, Hi=10, Wi=10, Di=6, **nopad),
            _problem(C=128, K=40, Hi=7, Wi=9, Di=5),
        ]
        tile = (64, 64, 32, 2, 2, 16)
        for mb in (2, 4):
            for knobs in (
                dict(uk_xhalo=20, lds_kpack_read=True, xh_wait_smem=True, defer_mfma=2,
                     vmem_interleave=2, lds_xor_rowkey=True, epi_quad_xpose=4),
                dict(lds_kpack_read=True),
            ):
                for i, prob in enumerate(probs):
                    with self.subTest(mb=mb, prob=i, **knobs):
                        err = _run(prob, tile=tile, m_band=mb, **base, **knobs)
                        self.assertLess(err, _TOL, f"mb={mb} {knobs}: rel err {err:.3e}")


class TestFwdLdsKnobSpec(unittest.TestCase):
    """CPU: rejections, name tags and the opt-in AOT env axes."""

    def _spec(self, **kw):
        from kernels.common.conv_implicit_gemm import ImplicitGemmConvSpec

        base = dict(
            problem=_problem(C=128), tile_m=64, tile_n=64, tile_k=32, warp_m=2,
            warp_n=2, warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            pipeline="compv4", uniform_k_decode=True, reg_prefetch=1, lds_xor=True,
        )
        base.update(kw)
        return ImplicitGemmConvSpec(**base)

    def test_rejections(self):
        from kernels.common.conv_implicit_gemm import _fwd_kloop_conflict

        ok = self._spec(lds_kpack_read=True, rpf_lds_barrier=True, rpf_fetch_late=True,
                        uk_tap_fast=True, uk_chunk_group=2)
        self.assertEqual(_fwd_kloop_conflict(ok), "")
        for kw in (
            dict(uk_chunk_group=2),  # needs uk_tap_fast
            dict(uk_tap_fast=True, uk_chunk_group=3),
            dict(rpf_lds_barrier=True, reg_prefetch=2),
            dict(rpf_lds_barrier=True, defer_mfma=1),
            dict(rpf_fetch_late=True, vmem_interleave=2),
            dict(lds_kpack_read=True, a_direct=True),
            dict(lds_kpack_read=True, tile_k=16, lds_xor=False),  # one kk step
        ):
            with self.subTest(**kw):
                self.assertNotEqual(_fwd_kloop_conflict(self._spec(**kw)), "")

    def test_quad_xpose_rules(self):
        from kernels.common.conv_implicit_gemm import _fwd_kloop_conflict

        self.assertEqual(_fwd_kloop_conflict(self._spec(epilogue="cshuffle", epi_quad_xpose=3)), "")
        self.assertIn("eqx3", self._spec(epilogue="cshuffle", epi_quad_xpose=3).kernel_name())
        for kw in (
            dict(epilogue="default", epi_quad_xpose=1),  # replaces the cshuffle store
            dict(epilogue="cshuffle", epi_quad_xpose=1, split_k=2),
            dict(epilogue="cshuffle", epi_quad_xpose=1, warp_tile_m=32, warp_tile_n=32,
                 warp_tile_k=8),  # 16x16 atom only
            dict(epilogue="cshuffle", epi_quad_xpose=3, tile_n=32, warp_n=2),  # odd N atoms
            dict(epilogue="cshuffle", epi_quad_xpose=5),
        ):
            with self.subTest(**kw):
                self.assertNotEqual(_fwd_kloop_conflict(self._spec(**kw)), "")

    def test_names(self):
        a = self._spec()
        b = self._spec(lds_kpack_read=True, rpf_lds_barrier=True, rpf_fetch_late=True,
                       uk_tap_fast=True, uk_chunk_group=2)
        for tag in ("kprd", "lbar", "flate", "_cg"):
            self.assertNotIn(tag, a.kernel_name())
        for tag in ("_tapf", "_cg2", "_kprd", "_lbar", "_flate"):
            self.assertIn(tag, b.kernel_name())

    def test_label_roundtrip(self):
        from kernels.conv import kernel_label as kl

        lab = ("fwd-implicit_gemm-bf16-3d-t256x64x32-w4x1-a16x16x16-v8x8x1-compv3-default-grp-"
               "xcd6-ukd-rpf1-lxor-tapf-kprd-cg2-lbar-flate-wave64-gfx942-llvm22")
        i = kl.decode(lab)
        self.assertTrue(i.lds_kpack_read and i.rpf_lds_barrier and i.rpf_fetch_late)
        self.assertEqual(i.uk_chunk_group, 2)
        self.assertEqual(kl.encode(i), lab)

    def test_xhalo_rejections(self):
        from kernels.common.conv_implicit_gemm import _fwd_kloop_conflict

        self.assertEqual(_fwd_kloop_conflict(self._spec(uk_xhalo=8, xh_dpp=True)), "")
        self.assertEqual(_fwd_kloop_conflict(self._spec(uk_xhalo=8, defer_mfma=1)), "")
        self.assertEqual(_fwd_kloop_conflict(self._spec(uk_xhalo=8, rpf_setprio=1)), "")
        # r3: the x-halo loop places its loads with vmem_interleave too.
        self.assertEqual(_fwd_kloop_conflict(self._spec(uk_xhalo=8, vmem_interleave=2)), "")
        self.assertEqual(_fwd_kloop_conflict(self._spec(uk_xhalo=8, xh_wait_smem=True)), "")
        # opt-fwds: setprio also wraps the rpf1 loop's deferred MFMA phase.
        self.assertEqual(_fwd_kloop_conflict(self._spec(rpf_setprio=1, defer_mfma=1)), "")
        for kw in (
            dict(xh_dpp=True),  # needs uk_xhalo
            dict(uk_xhalo=6),
            dict(uk_xhalo=8, lds_xor=False),
            dict(uk_xhalo=8, rpf_fetch_late=True),
            dict(xh_wait_smem=True),  # needs uk_xhalo
            dict(rpf_setprio=4),
        ):
            with self.subTest(**kw):
                self.assertNotEqual(_fwd_kloop_conflict(self._spec(**kw)), "")

    def test_r2_scope(self):
        from kernels.conv import aot_jobs

        kl = dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True, uk_xhalo=8, xcd_remap=6)
        plain = dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True)
        with mock.patch.dict(os.environ, {"ROCKE_AOT_FWD_R2_SCOPE": "256x64x32w4x1a16,128x128x64"}):
            self.assertTrue(aot_jobs._fwd_r2_in_scope(kl, 256, 64, 32, 4, 1, 16))
            self.assertFalse(aot_jobs._fwd_r2_in_scope(kl, 256, 64, 32, 2, 2, 16))
            self.assertTrue(aot_jobs._fwd_r2_in_scope(kl, 128, 128, 64, 2, 2, 32))
            self.assertTrue(aot_jobs._fwd_r2_in_scope(plain, 64, 64, 32, 2, 2, 16))
        with mock.patch.dict(os.environ, {"ROCKE_AOT_FWD_R2_SCOPE": ""}):
            self.assertTrue(aot_jobs._fwd_r2_in_scope(kl, 64, 64, 32, 2, 2, 16))

    def test_aot_env_axes(self):
        from kernels.conv import aot_jobs

        off = {k: "" for k in ("ROCKE_AOT_FWD_SCHED", "ROCKE_AOT_FWD_CG", "ROCKE_AOT_FWD_KPRD",
                               "ROCKE_AOT_FWD_LBAR", "ROCKE_AOT_FWD_FLATE", "ROCKE_AOT_FWD_LXOR",
                               "ROCKE_AOT_FWD_XHALO", "ROCKE_AOT_FWD_XDPP", "ROCKE_AOT_FWD_WPE",
                               "ROCKE_AOT_FWD_R2_XCD")}
        base = [dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True, uk_tap_fast=True)]
        with mock.patch.dict(os.environ, off):
            self.assertEqual(aot_jobs._fwd_sched_k_loops(list(base)), base)
        on = dict(off, ROCKE_AOT_FWD_CG="2", ROCKE_AOT_FWD_KPRD="1", ROCKE_AOT_FWD_LBAR="1",
                  ROCKE_AOT_FWD_FLATE="1")
        with mock.patch.dict(os.environ, on):
            out = aot_jobs._fwd_sched_k_loops(list(base))
            self.assertEqual(len(out), 16)  # x2 cg, x2 lbar, x2 flate, x2 kprd
            self.assertIn(dict(base[0], uk_chunk_group=2, rpf_lds_barrier=True,
                               rpf_fetch_late=True, lds_kpack_read=True), out)


if __name__ == "__main__":
    unittest.main()
