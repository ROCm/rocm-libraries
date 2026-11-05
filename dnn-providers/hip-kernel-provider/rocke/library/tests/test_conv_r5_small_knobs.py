# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the opt-in small-grid conv knobs: dgrad tapk_pp_sched 11 / 12
and tapk_pp_bslab on the pp_boxes split sets; wgrad ws_store 2 (bf16 scratch
slabs, with the matching Stage 2 fold) and epi_store_fast: validation, names,
IR build."""

from __future__ import annotations

import unittest

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_implicit_gemm import ConvDataSpec
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    build_implicit_gemm_conv_dgrad,
    is_valid_dgrad_spec,
)
from kernels.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    build_implicit_gemm_conv_wgrad,
    is_valid_wgrad_spec,
)
from kernels.common.conv_wgrad_workspace_reduce import (
    WgradReduceSpec,
    build_conv_wgrad_workspace_reduce,
)

_ARCH = "gfx942"
_BF16 = ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16")


def _problem(**kw):
    base = dict(
        N=1, Di=10, Hi=10, Wi=10, C=256, K=128, Z=3, Y=3, X=3, sD=1, pD=0, dD=1, pH=0, pW=0
    )
    base.update(kw)
    return ConvProblem(**base)


def _pp(**kw):
    base = dict(
        problem=_problem(), data=_BF16,
        tile_m=256, tile_n=256, tile_k=32, warp_m=4, warp_n=2,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
        pipeline="mem", epilogue="cshuffle", lds_k_outer=False,
        tap_outer_k=True, tapk_prefetch=1, b_kpack=4,
        lds_xor=True, lds_k_pad=0, vector_size_a=8, xcd_remap=6,
        cshuffle_m_chunks=2, tapk_pingpong=4, tapk_pp_sched=5,
        split_k=8, split_atomic_pairs=True, pp_boxes=7,
    )
    base.update(kw)
    return DgradConvSpec(**base)


def _wg(**kw):
    base = dict(
        problem=_problem(), data=_BF16,
        tile_m=256, tile_n=256, tile_k=32, warp_m=4, warp_n=2,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
        pipeline="mem", epilogue="default", lds_k_pad=4,
        prefetch_regs=1, lds_kpack=2, sep_addr=2, plain_pp=4,
    )
    base.update(kw)
    return WgradConvSpec(**base)


class TestDgradPpSched(unittest.TestCase):
    def test_new_schedules_valid_and_named(self):
        for pps in (11, 12):
            spec = _pp(tapk_pp_sched=pps)
            ok, why = is_valid_dgrad_spec(spec, _ARCH)
            self.assertTrue(ok, why)
            self.assertIn(f"pps{pps}", spec.kernel_name())

    def test_unknown_schedules_rejected(self):
        for pps in (8, 9, 10, 13):
            ok, _ = is_valid_dgrad_spec(_pp(tapk_pp_sched=pps), _ARCH)
            self.assertFalse(ok, pps)

    def test_bslab_on_split_boxes(self):
        ok, why = is_valid_dgrad_spec(_pp(tapk_pp_bslab=3), _ARCH)
        self.assertTrue(ok, why)
        # still refused on a split grid without boxes
        ok, _ = is_valid_dgrad_spec(_pp(tapk_pp_bslab=3, pp_boxes=0, split_atomic_pairs=False), _ARCH)
        self.assertFalse(ok)

    def test_ir_builds(self):
        for kw in (dict(tapk_pp_sched=11), dict(tapk_pp_sched=12, tapk_pp_bslab=3)):
            spec = _pp(**kw)
            self.assertTrue(is_valid_dgrad_spec(spec, _ARCH)[0])
            build_implicit_gemm_conv_dgrad(spec, arch=_ARCH)


class TestDgradBoxEpiLdsBarriers(unittest.TestCase):
    """DgradConvSpec.epi_lds_barriers on the pp_boxes binaries (tag elb)."""

    def _cases(self):
        _t128 = dict(tile_m=128, tile_n=256, tile_k=64, warp_m=2, warp_n=4, cshuffle_m_chunks=1)
        return [
            dict(),  # split pixel lists, sapl, mch2
            dict(_t128, tapk_pp_sched=2, split_k=8, pp_box_order=1),  # s10 set
            dict(_t128, tapk_pp_sched=2, tapk_pp_epi=2, split_k=1, split_atomic_pairs=False,
                 pp_box_order=2),  # s6 non-split set
        ]

    def test_valid_named_and_builds(self):
        for kw in self._cases():
            spec = _pp(epi_lds_barriers=True, **kw)
            ok, why = is_valid_dgrad_spec(spec, _ARCH)
            self.assertTrue(ok, (kw, why))
            self.assertIn("elb", spec.kernel_name())
            self.assertNotIn("elb", _pp(**kw).kernel_name())
            build_implicit_gemm_conv_dgrad(spec, arch=_ARCH)

    def test_needs_a_cshuffle_pp_or_xhalo_epilogue(self):
        # r5-dlds also admits elb on the dense ping-pong cshuffle epilogue
        # (pp_boxes 0); a persistent pp loop or the default epilogue is refused.
        ok, why = is_valid_dgrad_spec(_pp(epi_lds_barriers=True, pp_boxes=0, split_k=1,
                                          split_atomic_pairs=False), _ARCH)
        self.assertTrue(ok, why)
        for kw in (dict(tapk_pp_persist=1), dict(epilogue="default", cshuffle_m_chunks=1)):
            ok, why = is_valid_dgrad_spec(_pp(epi_lds_barriers=True, pp_boxes=0, split_k=1,
                                              split_atomic_pairs=False, **kw), _ARCH)
            self.assertFalse(ok, kw)


class TestWgradSlabsAndFastStore(unittest.TestCase):
    def test_ws_store_2(self):
        spec = _wg(split_k=2, two_stage=True, ws_replicas=8, ws_store=2)
        ok, why = is_valid_wgrad_spec(spec, _ARCH)
        self.assertTrue(ok, why)
        self.assertIn("wss2", spec.kernel_name())
        build_implicit_gemm_conv_wgrad(spec, arch=_ARCH)
        ok, _ = is_valid_wgrad_spec(_wg(ws_store=2), _ARCH)  # needs two_stage
        self.assertFalse(ok)
        ok, _ = is_valid_wgrad_spec(_wg(split_k=2, two_stage=True, ws_store=3), _ARCH)
        self.assertFalse(ok)

    def test_stage2_bf16_fold(self):
        rs = WgradReduceSpec(problem=_problem(), dtype_d="bf16", ws_replicas=2, vec=4, ws_bf16=True)
        self.assertIn("wsb", rs.kernel_name())
        self.assertNotIn("wsb", WgradReduceSpec(problem=_problem(), dtype_d="bf16", vec=4).kernel_name())
        build_conv_wgrad_workspace_reduce(rs, arch=_ARCH)

    def test_epi_store_fast(self):
        spec = _wg(epi_store_fast=True)
        ok, why = is_valid_wgrad_spec(spec, _ARCH)
        self.assertTrue(ok, why)
        self.assertIn("esf", spec.kernel_name())
        build_implicit_gemm_conv_wgrad(spec, arch=_ARCH)
        for kw in (dict(split_k=2, two_stage=True), dict(epi_quad_xpose=1), dict(epilogue="cshuffle")):
            ok, _ = is_valid_wgrad_spec(_wg(epi_store_fast=True, **kw), _ARCH)
            self.assertFalse(ok, kw)

    def test_ppp_kpl_axis(self):
        """ROCKE_AOT_WGRAD_PPP_KPL doubles the plain_pp copies, ws_store twins included."""
        from unittest import mock
        import kernels.conv.aot_jobs as aj

        cfg = dict(warp_m=4, warp_n=2, tile_m=256, tile_n=256, tile_k=32, pipeline="mem",
                   split_k=2, two_stage=True, epilogue="default")
        with mock.patch.object(aj, "_WGRAD_PPP", (2,)), mock.patch.object(aj, "_WGRAD_PPP_MIN_MN", 65536), \
                mock.patch.object(aj, "_WGRAD_PPP_WSB", 1), mock.patch.object(aj, "_WGRAD_PPP_KPL", 0):
            base = aj._wgrad_kloop_variants("gfx942", cfg)
        with mock.patch.object(aj, "_WGRAD_PPP", (2,)), mock.patch.object(aj, "_WGRAD_PPP_MIN_MN", 65536), \
                mock.patch.object(aj, "_WGRAD_PPP_WSB", 1), mock.patch.object(aj, "_WGRAD_PPP_KPL", 8):
            withk = aj._wgrad_kloop_variants("gfx942", cfg)
        pp = [v for v in base if v.get("plain_pp")]
        kpl = [v for v in withk if v.get("kp_line") == 8 and v.get("plain_pp")]
        self.assertTrue(pp)
        self.assertEqual(len(kpl), len(pp))
        self.assertTrue(any(v.get("ws_store") == 2 for v in kpl))
        self.assertEqual(len(withk), len(base) + len(kpl))

    def test_defaults_unchanged(self):
        self.assertNotIn("esf", _wg().kernel_name())
        self.assertNotIn("wss", _wg(split_k=2, two_stage=True).kernel_name())


if __name__ == "__main__":
    unittest.main()
