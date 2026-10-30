# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Forward 8-wave block ping-pong K loop (``rpf_pingpong`` / ``pp_pre_mfma``),
the chunked cshuffle C staging (``cshuffle_m_chunks``) and the x-halo loop's
``xh_wait_smem`` / ``vmem_interleave``.

GPU: each against the fp32 reference (padded and unpadded problems, several
channel chunks, split-K). CPU: rejections, name tags, label round trip, AOT
axes.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import pytest

from tests.test_conv_fwd_3d_kloop_knobs import _SKIP, _TOL, _problem, _run

_PP = dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True, lds_kpack_read=True)


@pytest.mark.gpu
@unittest.skipIf(_SKIP, _SKIP)
class TestFwdPingPong(unittest.TestCase):
    def _check(self, problem, **kw):
        err = _run(problem, **kw)
        self.assertLess(err, _TOL, f"{kw}: rel err {err:.3e}")

    def test_modes(self):
        for tile in ((256, 128, 64, 4, 2, 16), (256, 128, 64, 4, 2, 32), (128, 128, 64, 2, 4, 16)):
            for knobs in (
                dict(rpf_pingpong=1, rpf_setprio=1),
                dict(rpf_pingpong=3),
                dict(rpf_pingpong=4),
                dict(rpf_pingpong=4, pp_pre_mfma=2),
                dict(rpf_pingpong=5, vmem_interleave=4),
                dict(rpf_pingpong=5, vmem_interleave=4, pp_b_double=True),
            ):
                if knobs.get("pp_b_double") and tile[5] == 32:
                    continue  # 32x32x8 at tile_k 64: 4 kk pairs, 2 per half (fine), kept a16-only
                with self.subTest(tile=tile, **knobs):
                    # C=256: 4 channel chunks per tap; padded and unpadded.
                    self._check(_problem(C=256, K=128), tile=tile, **_PP, **knobs)
                    self._check(_problem(C=128, K=128, pH=0, pW=0, pD=0), tile=tile,
                                **_PP, uk_nopad=True, uk_tap_fast=True, **knobs)

    def test_split_and_chunked_c(self):
        for tile, mch in (((256, 128, 64, 4, 2, 16), 1), ((512, 128, 32, 4, 2, 32), 2)):
            with self.subTest(tile=tile, mch=mch):
                self._check(_problem(C=128, K=128, pH=0, pW=0, pD=0), tile=tile, **_PP,
                            uk_nopad=True, rpf_pingpong=4, cshuffle_m_chunks=mch)
                self._check(_problem(C=128, K=128, pH=0, pW=0, pD=0), tile=tile, degree=3,
                            **_PP, uk_nopad=True, rpf_pingpong=4, split_k=2,
                            cshuffle_m_chunks=mch)


class TestFwdPingPongSpec(unittest.TestCase):
    def _spec(self, **kw):
        from kernels.common.conv_implicit_gemm import ConvDataSpec, ImplicitGemmConvSpec

        base = dict(
            data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
            problem=_problem(C=128, K=128), tile_m=256, tile_n=128, tile_k=64, warp_m=4,
            warp_n=2, warp_tile_m=16, warp_tile_n=16, warp_tile_k=16, pipeline="mem",
            epilogue="cshuffle", **_PP,
        )
        base.update(kw)
        return ImplicitGemmConvSpec(**base)

    def test_rejections(self):
        from kernels.common.conv_implicit_gemm import _fwd_kloop_conflict

        for kw in (
            dict(rpf_pingpong=4),
            dict(rpf_pingpong=4, pp_pre_mfma=2, rpf_setprio=1),
            dict(rpf_pingpong=5, vmem_interleave=4),
            dict(rpf_pingpong=3, lds_kpack_read=False),
            dict(cshuffle_m_chunks=2),
        ):
            with self.subTest(**kw):
                self.assertEqual(_fwd_kloop_conflict(self._spec(**kw)), "")
        for kw in (
            dict(rpf_pingpong=6),
            dict(rpf_pingpong=4, warp_m=2),  # 4 waves
            dict(rpf_pingpong=4, lds_ahead=True),
            dict(rpf_pingpong=4, defer_mfma=1),
            dict(rpf_pingpong=4, vmem_interleave=2),  # pp5 only
            dict(rpf_pingpong=5),  # needs vmem_interleave
            dict(rpf_pingpong=4, lds_xor=False),
            dict(rpf_pingpong=4, reg_prefetch=2),
            dict(rpf_pingpong=4, pp_pre_mfma=33),
            dict(pp_pre_mfma=1),
            dict(cshuffle_m_chunks=3),
            dict(cshuffle_m_chunks=2, epilogue="default"),
            dict(xh_wait_smem=True),
            dict(pp_b_double=True),  # needs rpf_pingpong
            dict(lds_xor_rowkey=True),  # tile_k 32 only
        ):
            with self.subTest(**kw):
                self.assertNotEqual(_fwd_kloop_conflict(self._spec(**kw)), "")

    def test_names(self):
        a = self._spec()
        b = self._spec(rpf_pingpong=5, vmem_interleave=4, pp_pre_mfma=1, cshuffle_m_chunks=2)
        for tag in ("_pp", "_ppm", "_mch", "_xws"):
            self.assertNotIn(tag, a.kernel_name())
        for tag in ("_pp5", "_ppm1", "_mch2", "_vmi4"):
            self.assertIn(tag, b.kernel_name())

    def test_label_roundtrip(self):
        from kernels.conv import kernel_label as kl

        lab = ("fwd-implicit_gemm-bf16-3d-t256x128x64-w4x2-a16x16x16-v8x8x8-mem-cshuffle-grp-"
               "xcd6-ukd-rpf1-lxor-tapf-nopad-vmi4-kprd-prio1-pp5-ppm1-wave64-gfx942-llvm22")
        i = kl.decode(lab)
        self.assertEqual((i.rpf_pingpong, i.pp_pre_mfma, i.vmem_interleave), (5, 1, 4))
        self.assertEqual(kl.encode(i), lab)

    def test_default_hash_unchanged(self):
        import dataclasses

        from kernels.conv import kernel_label as kl

        lab = ("fwd-implicit_gemm-bf16-3d-t256x128x64-w4x2-a16x16x16-v8x8x8-mem-cshuffle-grp-"
               "xcd6-ukd-rpf1-lxor-tapf-nopad-kprd-wave64-gfx942-llvm22")
        i = kl.decode(lab)
        on = dataclasses.replace(i, rpf_pingpong=4)
        self.assertNotEqual(i.stable_hash(), on.stable_hash())
        self.assertEqual(
            i.stable_hash(),
            dataclasses.replace(on, rpf_pingpong=0, pp_pre_mfma=0, xh_wait_smem=False).stable_hash(),
        )

    def test_aot_axis(self):
        from kernels.conv import aot_jobs

        env = {"ROCKE_AOT_FWD_PP": "p4,p5v4,p4m1s1", "ROCKE_AOT_FWD_PP_SCOPE": "256x128x64w4x2a16"}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(
                aot_jobs._fwd_pp_variants(),
                (
                    dict(rpf_pingpong=4, lds_kpack_read=True, uk_nopad=True),
                    dict(rpf_pingpong=5, lds_kpack_read=True, uk_nopad=True, vmem_interleave=4),
                    dict(rpf_pingpong=4, lds_kpack_read=True, uk_nopad=True, pp_pre_mfma=1,
                         rpf_setprio=1),
                ),
            )
            kl = dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True, rpf_pingpong=4)
            self.assertTrue(aot_jobs._fwd_r2_in_scope(kl, 256, 128, 64, 4, 2, 16))
            self.assertFalse(aot_jobs._fwd_r2_in_scope(kl, 256, 128, 64, 4, 2, 32))
            loops = aot_jobs._fwd_pingpong_k_loops(
                [dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True),
                 dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True, lds_ahead=True)],
                "mem",
            )
            self.assertEqual(len(loops), 3)
            self.assertEqual(aot_jobs._fwd_pingpong_k_loops(loops, "compv3"), [])
        with mock.patch.dict(os.environ, {"ROCKE_AOT_FWD_PP": "p9"}):
            with self.assertRaises(ValueError):
                aot_jobs._fwd_pp_variants()
