# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the FP16 RCR GEMM dispatcher case."""

from __future__ import annotations

import unittest

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.dispatch import (
    GemmRequest,
    dispatch_gemm_fp16,
    dispatch_gemm_fp16_all,
    gemm_fp16_candidates,
    gemm_fp16_sweep_space,
    registered_gemm_fp16_combos,
)
from rocke.dispatch.gemm import build_kernel
from rocke.instances.common.gemm_skinny import SkinnyGemmSpec
from rocke.instances.common.gemm_skinny import is_valid_spec as skinny_is_valid_spec
from rocke.dispatch.gemm.common import (
    DTL_PINGPONG_ARCHES,
    DTL_PINGPONG_GEOMETRY_ARCHES,
    rilv_admits,
    rilv_pad_admits,
    rilv_persistent,
    rilv_shortk_admits,
    small_tile_shortk_tile_trait,
    dtl_pingpong_tile_trait,
)
from rocke.dispatch.gemm.support import (
    gemm_config_supported,
    request_shape_supported,
    support_query_from_universal_spec,
)
from rocke.instances.common.gemm_universal import (
    DTL_BUFFER_NUM_RECORDS,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    is_valid_spec,
    mono_data_spec,
)


class TestGemmFp16Dispatch(unittest.TestCase):
    def test_dispatch_gfx950_request(self):
        req = GemmRequest(M=4096, N=4096, K=4096, arch="gfx950")
        result = dispatch_gemm_fp16(req)
        self.assertEqual(result.kernel_id.op, "gemm")
        self.assertEqual(result.kernel_id.family, "gemm_fp16_rcr")
        self.assertEqual(result.kernel_id.arch, "gfx950")
        self.assertEqual(result.kernel_id.algorithm, "universal_gemm")
        # 4096^3 (256 tiles of 256x256): the admission cost model picks the
        # DTL ping-pong candidate; on gfx950 its aligned body is the 4-wave
        # row-interleave body.
        self.assertEqual(result.kernel_id.spec_id, "cdna_dtl_pingpong_256x256")
        self.assertEqual(result.candidate.name, "universal_gemm_fp16_cdna_dtl_pingpong")
        self.assertEqual(result.grid, (16, 16, 1))
        self.assertEqual(result.block, (256, 1, 1))
        tr = result.spec.trait
        self.assertTrue(tr.direct_to_lds and tr.dtl_prefetch and tr.lds_row_interleave)
        self.assertEqual((tr.dtl_cache_a, tr.dtl_cache_b), (0, 0))
        self.assertEqual(
            [item["name"] for item in result.signature],
            ["A", "B", "C", "M", "N", "K"],
        )
        ok, why = is_valid_spec(result.spec, arch=req.arch)
        self.assertTrue(ok, why)

    def test_small_request_keeps_cshuffle_default(self):
        # 1024^2 = 64 tiles of 128x128 (a quarter wave) at a small K: below the
        # small-tile ring's half-wave floor and the big-tile threshold rule, so
        # the 128x128 cshuffle default is selected.
        # (ring_small_any takes these sub-half-wave shapes in auto mode; the
        # cshuffle default is the decision with it off)
        from unittest import mock

        from rocke.dispatch.gemm import skinny as _SK

        from rocke.dispatch.gemm import common as _GC

        req = GemmRequest(M=1024, N=1024, K=256, arch="gfx950")
        # (ring_any is ordered ahead of it there)
        self.assertEqual(
            dispatch_gemm_fp16(req).kernel_id.spec_id, "cdna_dtl_ring_any_first"
        )
        with mock.patch.object(_GC, "MERGE_RING_ANY_FIRST", False):
            self.assertEqual(
                dispatch_gemm_fp16(req).kernel_id.spec_id, "cdna_ring_small_any"
            )
        with mock.patch.object(_SK, "RING_ANY", False):
            result = dispatch_gemm_fp16(req)
        self.assertEqual(result.kernel_id.spec_id, "cdna_cshuffle_default")
        self.assertEqual(result.grid, (8, 8, 1))
        self.assertEqual(result.block, (256, 1, 1))
        ok, why = is_valid_spec(result.spec, arch=req.arch)
        self.assertTrue(ok, why)

    def test_big_grid_and_k32_bodies(self):
        # Dispatch decisions calibrated on gfx950 (see
        # examples/gfx950/gemm_perf_square_warpspec/
        # DTL_RING_SHAPES_CASE_STUDY.md): big grids take the persistent AGPR
        # interleave body with chiplet wgm 4 in both ring candidates; K % 64 ==
        # 32 small problems the 256x128x32 AGPR interleave ring
        from rocke.dispatch import dispatch_gemm_bf16

        # (the aligned 256x256 candidate takes the row-interleave body on gfx950
        # -- see test_rilv_bodies -- where it admits the shape)
        cases = (
            ((46080, 46080, 2304), "cdna_dtl_pingpong_256x256", "rilv"),
            ((46080, 46080, 1536), "cdna_dtl_pingpong_256x256", "rilv"),
            # (the padded big grids take the row-interleave body with the vector
            # epilogue's bounds path -- see test_rilv_pad_and_shortk_bodies)
            ((38400, 30240, 3072), "cdna_dtl_ring_pad", "rilv"),
            ((30720, 20160, 7680), "cdna_dtl_ring_pad", "rilv"),
            ((24576, 23040, 12288), "cdna_dtl_pingpong_256x256", "rilv"),
            ((8192, 8192, 8192), "cdna_dtl_pingpong_256x256", "rilv"),
        )
        for fn in (dispatch_gemm_fp16, dispatch_gemm_bf16):
            dt = "fp16" if fn is dispatch_gemm_fp16 else "bf16"
            for (M, N, K), want, body in cases:
                with self.subTest(dtype=dt, shape=(M, N, K)):
                    r = fn(GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt))
                    self.assertEqual(r.kernel_id.spec_id, want)
                    tr, t = r.spec.trait, r.spec.tile
                    self.assertTrue(tr.acc_agpr)
                    self.assertEqual(tr.lds_row_interleave, body == "rilv")
                    if body == "agpr_ilv":
                        self.assertEqual((t.warp_m, t.warp_n, t.tile_k), (2, 2, 32))
                        self.assertEqual(
                            (tr.chiplet_wgm, tr.chiplet_chunk_size), (4, 64)
                        )
                        self.assertEqual(tr.persistent_ctas, 256)
                        self.assertEqual(r.grid, (256, 1, 1))
                        self.assertTrue(tr.mfma_interleave and tr.dtl_hoist)
                    ok, why = is_valid_spec(r.spec, arch="gfx950")
                    self.assertTrue(ok, why)
            r = fn(GemmRequest(M=4608, N=4608, K=288, arch="gfx950", dtype=dt))
            self.assertEqual(r.kernel_id.spec_id, "cdna_dtl_ring_small")
            t, tr = r.spec.tile, r.spec.trait
            self.assertEqual((t.tile_m, t.tile_n, t.tile_k), (256, 128, 32))
            # 4x1 waves (test_gemm_small_tile_shortk.py)
            self.assertEqual((t.warp_m, t.warp_n), (4, 1))
            self.assertTrue(tr.acc_agpr and tr.mfma_interleave)
            self.assertEqual(tr.dtl_stages, 3)
            self.assertTrue(is_valid_spec(r.spec, arch="gfx950")[0])

    def test_ring_candidates_calibrated_decisions(self):
        # Calibrated on gfx950 (see DTL_RING_SHAPES_CASE_STUDY.md): the
        # small-tile ring where the 256x256 tiles fill the last wave poorly, the
        # padded ring for ragged M / N, the aligned ring elsewhere.
        from rocke.dispatch import dispatch_gemm_bf16

        cases = (
            ((2048, 2048, 256), "cdna_dtl_ring_small"),
            ((2048, 2048, 512), "cdna_dtl_ring_small"),
            ((1792, 1792, 1792), "cdna_dtl_ring_small"),
            ((2048, 2048, 2048), "cdna_dtl_ring_small"),
            ((2304, 2304, 2304), "cdna_dtl_ring_small"),
            ((4352, 4352, 4352), "cdna_dtl_ring_small"),
            ((4608, 3840, 576), "cdna_dtl_ring_small"),
            # (the row-interleave short-K body where it admits the shape)
            ((4608, 4608, 384), "cdna_dtl_ring_small"),
            ((1536, 1440, 576), "cdna_dtl_ring_small"),
            ((3072, 1920, 576), "cdna_dtl_ring_small"),
            ((7680, 6720, 1536), "cdna_dtl_ring_pad"),
            ((15360, 13440, 1536), "cdna_dtl_ring_pad"),
            ((30720, 20160, 7680), "cdna_dtl_ring_pad"),
            ((38400, 30240, 3072), "cdna_dtl_ring_pad"),
            # exactly whole waves of 256x256 tiles: the aligned ring stays
            ((4096, 4096, 4096), "cdna_dtl_pingpong_256x256"),
            ((4096, 4096, 256), "cdna_dtl_pingpong_256x256"),
            ((8192, 8192, 8192), "cdna_dtl_pingpong_256x256"),
            ((15360, 15360, 768), "cdna_dtl_pingpong_256x256"),
        )
        for fn in (dispatch_gemm_fp16, dispatch_gemm_bf16):
            dt = "fp16" if fn is dispatch_gemm_fp16 else "bf16"
            for (M, N, K), want in cases:
                with self.subTest(dtype=dt, shape=(M, N, K)):
                    req = GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt)
                    r = fn(req)
                    if rilv_shortk_admits(req):
                        want = "cdna_dtl_pingpong_256x256"
                    self.assertEqual(r.kernel_id.spec_id, want)
                    ok, why = is_valid_spec(r.spec, arch="gfx950")
                    self.assertTrue(ok, why)
                    self.assertEqual(r.spec.trait.split_k, 1)
                    tr = r.spec.trait
                    if want == "cdna_dtl_ring_pad" and rilv_pad_admits(req):
                        # the padded row-interleave body
                        # (test_rilv_pad_and_shortk_bodies)
                        self.assertTrue(tr.pad_n and tr.dtl_tight_rsrc)
                        self.assertEqual(tr.rve_pad, 2)
                        continue
                    if want == "cdna_dtl_ring_pad":
                        self.assertEqual(r.grid, (256, 1, 1))  # persistent pool
                    r3 = small_tile_shortk_tile_trait("gfx950", M, N, K, dt)
                    if want == "cdna_dtl_ring_small" and r3 is not None:
                        # the small-tile body
                        # (test_gemm_small_tile_shortk.py)
                        self.assertEqual((r.spec.tile, tr), r3)
                        continue
                    if want != "cdna_dtl_pingpong_256x256":
                        self.assertTrue(tr.pad_m and tr.pad_n and tr.dtl_tight_rsrc)
                        self.assertTrue(tr.pad_fast_tile)
        # an aligned shape the ring's threshold rule turns down (f16 8192^2 x 64,
        # where the ring does not pay) keeps the default: the padded ring is for
        # ragged M / N only, and four whole waves are no small-tile regime
        r = dispatch_gemm_fp16(GemmRequest(M=8192, N=8192, K=64, arch="gfx950"))
        self.assertEqual(r.kernel_id.spec_id, "cdna_cshuffle_default")
        # the DTL ring candidates are gfx950-only (not calibrated on gfx942)
        for arch in ("gfx942",):
            r = dispatch_gemm_fp16(GemmRequest(M=2048, N=2048, K=2048, arch=arch))
            self.assertNotIn("ring", r.kernel_id.spec_id)

    def test_ring_and_big_grid_calibrated_decisions(self):
        # Dispatch decisions calibrated on gfx950 (every kernel full-output
        # checked; see DTL_RING_SHAPES_CASE_STUDY.md). (shape) -> (spec_id, tile (m, n, k),
        # dtl_stages, acc_agpr, persistent, chiplet_wgm, iglp_opt) per dtype.
        from rocke.dispatch import dispatch_gemm_bf16

        RS, PAD, ALN, CSH = (
            "cdna_dtl_ring_small",
            "cdna_dtl_ring_pad",
            "cdna_dtl_pingpong_256x256",
            "cdna_cshuffle_default",
        )
        S2 = (RS, (128, 128, 64), 2, False, 0, 8, None)
        S3 = (RS, (128, 128, 64), 3, False, 0, 8, None)
        K32 = (RS, (128, 128, 32), 3, False, 0, 8, None)
        K32AG = (RS, (256, 128, 32), 3, True, 0, 8, None)
        PADR = (PAD, (256, 256, 64), 2, False, 256, 8, None)
        ILVP = (PAD, (256, 256, 32), 3, True, 256, 4, None)
        ILV = (ALN, (256, 256, 32), 3, True, 256, 4, None)
        RING = (ALN, (256, 256, 64), 2, False, 0, 8, None)
        RW4 = (ALN, (256, 256, 64), 2, False, 0, 4, 1)
        CS = (CSH, None, 0, False, 0, None, None)
        # these big grids take the row-interleave bodies (aligned: the 256x256
        # candidate; ragged: the pad candidate with rve_pad)
        BA = (ALN, "rilv")
        BP = (PAD, "rilv")
        cases = {
            # small ring K floors where the cshuffle default can take the shape
            (2048, 2048, 64): (CS, S2),
            (2048, 2048, 32): (CS, CS),
            (2048, 2048, 96): (K32, K32),  # 128x128x32
            (3072, 3072, 96): (K32, K32),
            (2560, 2560, 96): (K32, K32),  # 200 256x128 tiles
            (4608, 4608, 96): (CS, CS),
            (4608, 4608, 160): (K32AG, K32AG),
            (4608, 4608, 288): (K32AG, K32AG),
            # one wave of 128x128 tiles and K >= 128: the 3-stage interleave
            (2048, 2048, 2048): (S3, S3),
            (1792, 1792, 1792): (S3, S3),
            (1536, 1440, 576): (S3, S3),
            (2048, 2048, 128): (S3, S3),
            (2304, 2304, 2304): (S2, S2),  # 324 tiles
            # ragged small K on < 1280 tiles: the small ring over the pad ring
            (8192, 8200, 64): (S2, S2),
            (8192, 8200, 512): (S2, S2),
            (7680, 6720, 256): (S2, S2),
            (8192, 8200, 1024): (BP, BP),
            (10240, 10248, 128): (PADR, PADR),  # 1640 tiles
            (12288, 12296, 64): (PADR, PADR),
            # big grids (>= 9000 256x256 tiles)
            (24576, 24576, 64): (CS, ILV),  # f16: the ring rule (K < 128)
            (24576, 24576, 128): (ILV, ILV),  # K <= 128
            (24576, 24576, 256): (RING, RING),
            (27648, 27648, 256): (RING, RING),
            (24576, 24576, 1536): (BA, BA),  # 144 MiB
            (30720, 30720, 1024): (BA, BA),  # 14400 tiles, 120 MiB
            (32768, 32768, 1024): (BA, BA),  # 16384 tiles, 128 MiB
            (35840, 35840, 640): (BA, BA),  # 19600 tiles
            (36864, 36864, 256): (RING, RING),  # >= 20000 tiles but K < 512
            (33792, 33792, 1536): (BA, BA),  # 198 MiB > 192 MiB
            (30720, 30720, 2048): (BA, BA),  # 240 MiB, within the MALL
            (36864, 36864, 1024): (BA, BA),  # >= 20000 tiles, K >= 512
            (40960, 40960, 512): (BA, BA),
            (46080, 46080, 256): (RW4, RW4),  # >= 30000 tiles
            (24576, 24576, 4096): (BA, BA),  # 403 MiB > MALL
            (36864, 36864, 2048): (BA, BA),  # 288 MiB > MALL
            (46080, 46080, 1536): (BA, BA),  # 270 MiB > MALL
            (24576, 24584, 64): (ILVP, ILVP),  # K <= 128
            (30720, 20160, 1536): (BP, BP),  # 9480 tiles
            (31744, 31752, 384): (PADR, PADR),  # K < 512
            (30720, 20160, 7680): (BP, BP),
            (28672, 28680, 2048): (BP, BP),  # 224 MiB
            (30720, 30728, 1280): (BP, BP),  # >= 12000 ragged tiles
        }
        for fn in (dispatch_gemm_fp16, dispatch_gemm_bf16):
            dt = "fp16" if fn is dispatch_gemm_fp16 else "bf16"
            for (M, N, K), want2 in cases.items():
                want = want2[0 if dt == "fp16" else 1]
                with self.subTest(dtype=dt, shape=(M, N, K)):
                    r = fn(GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt))
                    t, tr = r.spec.tile, r.spec.trait
                    self.assertEqual(r.kernel_id.spec_id, want[0])
                    self.assertTrue(is_valid_spec(r.spec, arch="gfx950")[0])
                    self.assertEqual(tr.split_k, 1)
                    if want[1] is None:
                        continue
                    if want[1] == "rilv":
                        self.assertTrue(tr.lds_row_interleave and tr.rilv_vec_epilogue)
                        continue
                    self.assertEqual((t.tile_m, t.tile_n, t.tile_k), want[1])
                    self.assertEqual(
                        (
                            tr.dtl_stages,
                            tr.acc_agpr,
                            tr.persistent_ctas,
                            tr.chiplet_wgm,
                            tr.iglp_opt,
                        ),
                        want[2:],
                    )

    def test_ring_candidates_follow_select_split_k(self):
        # the small / pad rings ask select_split_k (with the
        # default candidate's tile) instead of relying on their tile floor
        # matching its gate: with the floor lowered, a decode-like shape the
        # default would split stays with the default
        import os
        from unittest import mock

        import rocke.dispatch.gemm.common as C

        import rocke.dispatch.gemm.skinny as _SK

        req = GemmRequest(M=1024, N=1024, K=4096, arch="gfx950")
        # (ring_small_any takes it in auto mode, with ring_any ordered
        # ahead of it; the small ring's own split-K
        # gating is checked with RING_ANY off)
        self.assertIn(
            dispatch_gemm_fp16(req).kernel_id.spec_id,
            ("cdna_ring_small_any", "cdna_dtl_ring_any_first"),
        )
        _ring_any_off = mock.patch.object(_SK, "RING_ANY", False)
        _ring_any_off.start()
        self.addCleanup(_ring_any_off.stop)
        self.assertEqual(
            dispatch_gemm_fp16(req).kernel_id.spec_id, "cdna_cshuffle_default"
        )
        with mock.patch.object(C, "DTL_RING_SMALL_MIN_TILES_PER_CU", 0.1):
            small = next(
                c for c in gemm_fp16_candidates() if c.spec_id == "cdna_dtl_ring_small"
            )
            ok, why = small.admits(req)
            self.assertFalse(ok)
            self.assertIn("engages split-K", why)
            r = dispatch_gemm_fp16(req)
            self.assertEqual(r.kernel_id.spec_id, "cdna_cshuffle_default")
            self.assertGreater(r.spec.trait.split_k, 1)
        # the pad ring leaves the ragged small-K regime to the small ring
        pad = next(
            c for c in gemm_fp16_candidates() if c.spec_id == "cdna_dtl_ring_pad"
        )
        ok, why = pad.admits(GemmRequest(M=8192, N=8200, K=64, arch="gfx950"))
        self.assertFalse(ok)
        self.assertIn("ragged small-K regime", why)
        self.assertTrue(
            pad.admits(GemmRequest(M=8192, N=8200, K=1024, arch="gfx950"))[0]
        )
        # ... and poorly filled waves of 256x256 tiles (the small ring's)
        ok, why = pad.admits(GemmRequest(M=2000, N=2008, K=1024, arch="gfx950"))
        self.assertFalse(ok)
        self.assertIn("the small-tile ring's regime", why)
        # a forced split-K (operator override) keeps every ring off, as for the
        # aligned ring candidate
        with mock.patch.dict(os.environ, {"ROCKE_GEMM_SPLIT_K": "2"}):
            r = dispatch_gemm_fp16(GemmRequest(M=2048, N=2048, K=2048, arch="gfx950"))
            self.assertEqual(r.kernel_id.spec_id, "cdna_cshuffle_default")

    def test_admission_calibrated_decisions(self):
        # gfx950 dispatches the alias-scoped ring body. Shapes on both sides of
        # its calibrated boundary against the cshuffle default (which kernel was
        # faster on device, f16/bf16 full-output checked). The default "auto"
        # policy (the calibrated threshold rule) must reproduce these
        # decisions. True = the ring kernel was the faster kernel.
        cases = {
            "fp16": (
                ((2048, 2048, 2048), True),  # 64 tiles of 256^2
                ((1792, 1792, 1792), True),
                ((2304, 2304, 2304), True),
                ((4096, 4096, 4096), True),
                ((4352, 4352, 4352), True),  # 1.13 waves
                ((8192, 8192, 64), False),  # tiny K, the default wins
                ((4096, 4096, 64), False),
                ((3072, 3072, 128), True),
                ((4096, 4096, 256), True),
                ((2048, 2048, 256), False),  # few tiles, small K
                ((2048, 2048, 512), False),
                ((256, 65536, 8192), True),  # skinny M, wide N
                ((65536, 256, 8192), True),
                ((15360, 15360, 768), True),
            ),
            "bf16": (
                ((2048, 2048, 2048), True),
                ((2304, 2304, 2304), True),
                ((8192, 8192, 64), True),
                ((3072, 3072, 64), True),
                ((2560, 2560, 64), False),
                ((2048, 2048, 256), False),
                ((3072, 3072, 256), True),
                ((256, 8192, 8192), True),
            ),
        }
        from rocke.dispatch import dispatch_gemm_bf16

        # (the small-tile ring is dispatched ahead of the ring for poorly
        # filled waves -- see test_ring_candidates_calibrated_decisions
        # -- so the ring's own admission rule is checked here)
        from rocke.dispatch.gemm.bf16_rcr import gemm_bf16_candidates

        for dtype, rows in cases.items():
            cands = (
                gemm_fp16_candidates() if dtype == "fp16" else gemm_bf16_candidates()
            )
            ring = next(c for c in cands if "dtl_pingpong" in c.name)
            for (M, N, K), want_f in rows:
                with self.subTest(dtype=dtype, shape=(M, N, K)):
                    req = GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dtype)
                    self.assertEqual(ring.admits(req)[0], want_f)
                    spec = ring.select_spec(req) if want_f else None
                    if want_f:
                        self.assertEqual(spec.trait.split_k, 1)
                        self.assertEqual(spec.trait.dtl_stages, 2)
                        # aligned >= 256-tile, K >= 768 shapes take the
                        # row-interleave body, the rest the 16-wave ring
                        rilv = rilv_admits(req)
                        self.assertEqual(spec.trait.lds_row_interleave, rilv)
                        self.assertEqual(spec.trait.interwave_setprio, not rilv)

    def test_rilv_bodies(self):
        # Aligned problems with >= 256 256x256 tiles and K >= 768 take the
        # 4-wave AGPR row-interleave body with the nt vector epilogue; the
        # hb_sched schedule for long K or big grids, else the merged interleave;
        # chiplet wgm 16 / chunk 8, except persistent on every CU with chunk 32
        # for K <= RILV_PERSIST_MAX_K.
        from rocke.dispatch import dispatch_gemm_bf16

        cases = (
            ((8192, 8192, 8192), "mrg"),
            ((15360, 15360, 768), "mrg_pc"),
            ((46080, 46080, 2304), "mrg_pc"),
            ((8192, 8192, 4096), "mrg"),
            ((23040, 24576, 12288), "hb"),
            ((5120, 16128, 16384), "hb"),
            ((8192, 8192, 16384), "hb"),
            ((16384, 16384, 8192), "hb"),
            ((4096, 4096, 4096), "mrg"),  # 256 tiles: one full wave
            ((2048, 4096, 4096), "small"),  # 128 tiles: the small-tile ring
            # K below the row-interleave floor: the short-K row-interleave body
            # (see test_rilv_pad_and_shortk_bodies)
            ((8192, 8192, 512), "ring"),
            # N % 256 != 0: the padded row-interleave body
            # (test_rilv_pad_and_shortk_bodies)
            ((8192, 8320, 8192), "pad"),
        )
        for fn in (dispatch_gemm_fp16, dispatch_gemm_bf16):
            dt = "fp16" if fn is dispatch_gemm_fp16 else "bf16"
            for (M, N, K), body in cases:
                with self.subTest(dtype=dt, shape=(M, N, K)):
                    req = GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt)
                    r = fn(req)
                    tr, t = r.spec.trait, r.spec.tile
                    self.assertTrue(is_valid_spec(r.spec, arch="gfx950")[0])
                    rilv = body in ("mrg", "mrg_pc", "hb")
                    self.assertEqual(rilv_admits(req), rilv)
                    if not rilv:
                        r2 = rilv_pad_admits(req) or rilv_shortk_admits(req)
                        self.assertEqual(r2, body in ("pad", "ring"))
                        self.assertEqual(tr.lds_row_interleave, r2)
                        continue
                    self.assertEqual(r.kernel_id.spec_id, "cdna_dtl_pingpong_256x256")
                    self.assertEqual(
                        (t.tile_m, t.tile_n, t.tile_k, t.warp_m, t.warp_n),
                        (256, 256, 64, 2, 2),
                    )
                    self.assertTrue(
                        tr.lds_row_interleave
                        and tr.rilv_vec_epilogue
                        and tr.rve_store_nt
                        and tr.acc_agpr
                        and tr.dtl_hoist
                        and not tr.lds_swizzle
                    )
                    # the conflict-free 32-byte LDS pad
                    self.assertEqual(tr.rilv_pad, 32)
                    # every class runs the hb_sched body with
                    # the DTL origin in the resource base (inline soffset 0), the
                    # per-tile DTL address SALU spread over free MFMA slots, the
                    # first DTL one slot after the release and an unrolled K loop
                    self.assertTrue(
                        tr.hb_sched and not tr.ilv_merge and not tr.sia_sched
                    )
                    self.assertEqual(tr.asm_loop, "")
                    self.assertEqual(
                        (
                            tr.hb_read_every,
                            tr.dtl_soff0,
                            tr.dtl_addr_spread,
                            tr.hb_dtl_delay,
                        ),
                        (2, True, True, 1),
                    )
                    self.assertEqual(tr.kloop_unroll, 4 if K > 2304 else 2)
                    self.assertEqual(tr.hb_skew, 1)
                    # one lgkmcnt(0) per tile, DTL address steps
                    # after the tile's last fragment read
                    self.assertEqual((tr.hb_tile_lgkm0, tr.das_late), (1, True))
                    self.assertEqual(
                        rilv_persistent(M, N, K, "gfx950"), body == "mrg_pc"
                    )
                    if body != "mrg_pc":
                        self.assertEqual(
                            (tr.chiplet_wgm, tr.chiplet_chunk_size, tr.persistent_ctas),
                            (16, 8, 0),
                        )
                    else:
                        self.assertEqual(
                            (tr.chiplet_wgm, tr.chiplet_chunk_size, tr.persistent_ctas),
                            (8, 32, 256),
                        )
                        self.assertEqual(r.grid, (256, 1, 1))
        # the gfx942 dispatch never takes it (no 16-byte DTL)
        r = dispatch_gemm_fp16(GemmRequest(M=8192, N=8192, K=8192, arch="gfx942"))
        self.assertFalse(r.spec.trait.lds_row_interleave)

    def test_rilv_pad_and_shortk_bodies(self):
        # Ragged-N problems take the row-interleave body with the branch-free
        # vector epilogue bounds (rve_pad 2) instead of the pad ring / AGPR
        # interleave body;
        # aligned short-K problems with a well-filled last wave take the
        # merged-interleave row-interleave body (non-persistent wgm 16 / chunk 8)
        # instead of the small-tile ring / 16-wave ring.
        from rocke.dispatch import dispatch_gemm_bf16
        from rocke.instances.common.gemm_universal import launch_shape_error

        pad_cases = (
            # shape, (hb, persistent, pad_fast_tile)
            ((7680, 6720, 1536), (False, True, False)),
            ((15360, 13440, 1536), (False, True, True)),
            ((30720, 20160, 7680), (True, False, True)),
            ((38400, 30240, 3072), (False, True, True)),
        )
        for fn in (dispatch_gemm_fp16, dispatch_gemm_bf16):
            dt = "fp16" if fn is dispatch_gemm_fp16 else "bf16"
            for (M, N, K), (hb, pers, pft) in pad_cases:
                with self.subTest(dtype=dt, shape=(M, N, K)):
                    req = GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt)
                    self.assertTrue(rilv_pad_admits(req))
                    r = fn(req)
                    tr = r.spec.trait
                    self.assertEqual(r.kernel_id.spec_id, "cdna_dtl_ring_pad")
                    self.assertTrue(is_valid_spec(r.spec, arch="gfx950")[0])
                    self.assertTrue(tr.lds_row_interleave and tr.rilv_vec_epilogue)
                    self.assertEqual(tr.rve_pad, 2)
                    self.assertEqual((tr.pad_m, tr.pad_n), (False, True))
                    self.assertTrue(tr.dtl_tight_rsrc)
                    self.assertEqual(tr.pad_fast_tile, pft)
                    # hb_sched (persistent / short K, and with the tuned-body
                    # levers on the long-K class; test_gemm_rilv_pad_class)
                    # never the merged interleave
                    self.assertTrue(tr.hb_sched)
                    self.assertFalse(tr.sia_sched)
                    self.assertEqual(tr.dtl_addr_spread, hb or pers)
                    self.assertFalse(tr.ilv_merge)
                    self.assertEqual(tr.persistent_ctas, 256 if pers else 0)
                    self.assertIsNone(launch_shape_error(r.spec, M, N, K))
            # ragged M too: pad_m on; N % 8 != 0 or K < 768: the ring body
            req = GemmRequest(M=15000, N=13440, K=1536, arch="gfx950", dtype=dt)
            r = fn(req)
            self.assertEqual((r.spec.trait.pad_m, r.spec.trait.rve_pad), (True, 2))
            for M, N, K in ((15360, 13444, 1536), (15360, 13440, 512)):
                req = GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt)
                self.assertFalse(rilv_pad_admits(req))
                r = fn(req)
                self.assertEqual(r.spec.trait.rve_pad, 0)
                self.assertFalse(r.spec.trait.lds_row_interleave)
        small = (
            ((4608, 4608, 384), True),  # 324 tiles, fill 0.63
            ((4608, 3840, 576), False),  # 270 tiles, fill 0.53: the ring stays
            ((4096, 4096, 512), True),  # one full wave
            ((4096, 4096, 256), False),  # K below RILV_SHORTK_MIN_K
            (
                (4096, 4096, 768),
                False,
            ),  # the aligned row-interleave rule takes K >= 768
        )
        for (M, N, K), want in small:
            for dt in ("fp16", "bf16"):
                with self.subTest(dtype=dt, shape=(M, N, K)):
                    req = GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt)
                    self.assertEqual(rilv_shortk_admits(req), want)
            for fn in (dispatch_gemm_fp16, dispatch_gemm_bf16) if want else ():
                dt = "fp16" if fn is dispatch_gemm_fp16 else "bf16"
                r = fn(GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt))
                tr = r.spec.trait
                self.assertEqual(r.kernel_id.spec_id, "cdna_dtl_pingpong_256x256")
                if (M, N) == (4608, 4608):
                    # 128x192 exact tiles divide M / N and the 256x256 waves
                    # fill poorly -> the small-tile body
                    self.assertEqual(
                        (r.spec.tile.tile_m, r.spec.tile.tile_n), (128, 192)
                    )
                    self.assertEqual(tr.epilogue, "cshuffle")
                    self.assertFalse(tr.chiplet_swizzle)
                    continue
                self.assertTrue(tr.lds_row_interleave and tr.ilv_merge)
                self.assertEqual(
                    (tr.chiplet_wgm, tr.chiplet_chunk_size, tr.persistent_ctas),
                    (16, 8, 0),
                )
        # the persistence boundary (K <= RILV_PERSIST_MAX_K and more than
        # RILV_PERSIST_MIN_WAVES waves of 256x256 tiles), on both sides
        for (M, N, K), want in (
            ((4096, 8192, 2048), False),  # 512 tiles = 2 waves
            ((8192, 4096, 1024), False),
            ((6144, 6144, 1536), True),  # 576 tiles
            ((10240, 10240, 2048), True),
            ((4096, 4096, 4096), False),  # long K
        ):
            self.assertEqual(rilv_persistent(M, N, K, "gfx950"), want, (M, N, K))
        # many waves of short tiles -> the persistent body
        for fn in (dispatch_gemm_fp16, dispatch_gemm_bf16):
            dt = "fp16" if fn is dispatch_gemm_fp16 else "bf16"
            req = GemmRequest(M=8192, N=8192, K=512, arch="gfx950", dtype=dt)
            self.assertTrue(rilv_shortk_admits(req))
            tr = fn(req).spec.trait
            self.assertTrue(tr.hb_sched and tr.dtl_soff0 and tr.dtl_addr_spread)
            self.assertEqual(
                (tr.persistent_ctas, tr.kloop_unroll, tr.hb_skew), (256, 2, 1)
            )
        # gfx942: never
        req = GemmRequest(M=15360, N=13440, K=1536, arch="gfx942")
        self.assertFalse(rilv_pad_admits(req))
        self.assertFalse(
            rilv_shortk_admits(GemmRequest(M=4608, N=4608, K=384, arch="gfx942"))
        )

    def test_rilv_sweep_variants(self):
        # the dispatched candidate's sweep sees both row-interleave schedules and grids
        cand = next(c for c in gemm_fp16_candidates() if "dtl_pingpong" in c.name)
        req = GemmRequest(M=8192, N=8192, K=8192, arch="gfx950")
        sw = cand.sweep_space(req)
        rilv = [s for s in sw if s.trait.lds_row_interleave]
        self.assertTrue(any(s.trait.hb_sched for s in rilv))
        self.assertTrue(any(s.trait.ilv_merge for s in rilv))
        self.assertTrue(any(s.trait.sia_sched for s in rilv))
        for s in sw:
            self.assertTrue(is_valid_spec(s, arch="gfx950")[0])

    def test_admission_cost_model_policy(self):
        # The refitted gfx950 cost model stays selectable; it is the less
        # accurate predictor (it keeps the default at 2048^3 where the ring
        # kernel is faster), so it is not the default.
        from unittest import mock

        from rocke.dispatch.gemm.common import DTL_ADMIT_ENV

        cand = next(c for c in gemm_fp16_candidates() if "dtl_pingpong" in c.name)
        with mock.patch.dict("os.environ", {DTL_ADMIT_ENV: "cost_model"}):
            self.assertTrue(
                cand.admits(GemmRequest(M=4096, N=4096, K=4096, arch="gfx950"))[0]
            )
            self.assertFalse(
                cand.admits(GemmRequest(M=2048, N=2048, K=2048, arch="gfx950"))[0]
            )
        with mock.patch.dict("os.environ", {DTL_ADMIT_ENV: "rule"}):
            self.assertTrue(
                cand.admits(GemmRequest(M=2048, N=2048, K=2048, arch="gfx950"))[0]
            )
            # no threshold rule is calibrated for gfx942
            ok, why = cand.admits(GemmRequest(M=8192, N=8192, K=8192, arch="gfx942"))
            self.assertFalse(ok)
            self.assertIn("no ping-pong threshold rule", why)

    def test_support_applies_the_builder_gate(self):
        # One source of truth: a spec the hardware/tile predicate accepts but
        # the builder's validity gate rejects (16x16x16 atoms + lds_swizzle: the
        # 4-half XOR group splits the 8-half LDS store) is not supported.
        from dataclasses import replace

        from rocke.dispatch.gemm.support import universal_spec_supported

        spec = UniversalGemmSpec(
            name="s",
            tile=TileSpec(128, 128, 32, 2, 2, 1, 16, 16, 16),
            trait=TraitSpec(pipeline="compv4", epilogue="default", lds_swizzle=True),
            data=mono_data_spec("fp16"),
            wave_size=64,
        )
        ok, _ = gemm_config_supported(
            support_query_from_universal_spec(spec, arch="gfx950")
        )
        self.assertTrue(ok)
        ok, why = universal_spec_supported(spec, arch="gfx950")
        self.assertFalse(ok)
        self.assertIn("is_valid_spec", why)
        ok, why = universal_spec_supported(
            replace(spec, trait=replace(spec.trait, lds_swizzle=False)), arch="gfx950"
        )
        self.assertTrue(ok, why)

    def test_support_sizes_non_2_byte_operands(self):
        from dataclasses import replace as _r

        q = support_query_from_universal_spec(
            UniversalGemmSpec(
                name="q",
                tile=TileSpec(128, 128, 64, 2, 2, 1, 32, 32, 16),
                trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
                data=mono_data_spec("fp16"),
                wave_size=64,
            ),
            arch="gfx950",
        )
        from rocke.dispatch.gemm.support import _lds_bytes

        self.assertEqual(_lds_bytes(q, 163840)[1], 32768)
        q8 = _r(q, dtype_a="fp8", dtype_b="fp8", dtype_c="fp8")
        self.assertEqual(_lds_bytes(q8, 163840)[1], 16384)

    def test_admission_policy_override(self):
        from unittest import mock

        from rocke.dispatch.gemm.common import DTL_ADMIT_ENV

        cand = next(c for c in gemm_fp16_candidates() if "dtl_pingpong" in c.name)
        small = GemmRequest(M=2048, N=2048, K=256, arch="gfx950")
        mid = GemmRequest(M=2304, N=2304, K=2304, arch="gfx950")
        self.assertFalse(cand.admits(small)[0])
        self.assertTrue(cand.admits(mid)[0])
        with mock.patch.dict("os.environ", {DTL_ADMIT_ENV: "fills_device"}):
            self.assertFalse(cand.admits(mid)[0])
            self.assertTrue(
                cand.admits(GemmRequest(M=4096, N=4096, K=64, arch="gfx950"))[0]
            )
        with mock.patch.dict("os.environ", {DTL_ADMIT_ENV: "always"}):
            self.assertTrue(cand.admits(small)[0])
        with mock.patch.dict("os.environ", {DTL_ADMIT_ENV: "never"}):
            self.assertFalse(cand.admits(mid)[0])
        with mock.patch.dict("os.environ", {DTL_ADMIT_ENV: "bogus"}), self.assertRaises(
            ValueError
        ):
            cand.admits(mid)

    def test_admission_rejected_in_split_k_regime(self):
        # The default candidate engages split-K here (decode-like grid): the ping-pong candidate
        # is not calibrated there and stays out.
        cand = next(c for c in gemm_fp16_candidates() if "dtl_pingpong" in c.name)
        ok, why = cand.admits(GemmRequest(M=1024, N=1024, K=8192, arch="gfx950"))
        self.assertFalse(ok)
        self.assertIn("split-K", why)

    def test_dtl_pingpong_gfx942(self):
        # gfx942 geometry (256x256x32, 16x16x16, 1-dword DTL) with its own
        # calibrated cost model: a large square takes it, a 64-tile square
        # (not admitted on gfx942 for f16) keeps the default. With the
        # register-staged (vgpr_prefetch) candidate on (the default) gfx942
        # dispatches that one instead (test_gemm_g5_vgpr_prefetch.py); this
        # pins the ping-pong rules underneath it.
        from unittest import mock

        from rocke.dispatch.gemm import common as _common

        with mock.patch.object(_common, "G5_VGPR", False):
            self._check_dtl_pingpong_gfx942()

    def _check_dtl_pingpong_gfx942(self):
        self.assertIn("gfx942", DTL_PINGPONG_ARCHES)
        self.assertIn("gfx942", DTL_PINGPONG_GEOMETRY_ARCHES)
        req = GemmRequest(M=8192, N=8192, K=8192, arch="gfx942")
        result = dispatch_gemm_fp16(req)
        self.assertEqual(result.kernel_id.spec_id, "cdna_dtl_pingpong_256x256")
        t = result.spec.tile
        self.assertEqual((t.tile_m, t.tile_n, t.tile_k), (256, 256, 32))
        self.assertEqual((t.warp_tile_m, t.warp_tile_n, t.warp_tile_k), (16, 16, 16))
        ok, why = is_valid_spec(result.spec, arch="gfx942")
        self.assertTrue(ok, why)
        small = dispatch_gemm_fp16(GemmRequest(M=2048, N=2048, K=2048, arch="gfx942"))
        self.assertEqual(small.kernel_id.spec_id, "cdna_cshuffle_default")
        tile, trait = dtl_pingpong_tile_trait("gfx942")
        self.assertEqual(tile, t)
        self.assertEqual(trait, result.spec.trait)

    def test_dtl_operand_range_boundary(self):
        # The DTL loads address A / B with an i32 byte offset into a buffer of
        # DTL_BUFFER_NUM_RECORDS bytes; a load past it returns 0 silently. An
        # operand of exactly 2 GiB (M=65536, K=16384 f16) must not take the DTL
        # candidate; one 256-row tile less must.
        cand = next(c for c in gemm_fp16_candidates() if "dtl_pingpong" in c.name)
        cases = (
            ((65280, 512, 16384), True),  # A = 2_139_095_040 B <= 0x7FFF0000
            ((65536, 512, 16384), False),  # A = 2 GiB
            ((512, 65536, 16384), False),  # B = 2 GiB
            ((98304, 512, 16384), False),  # A = 3 GiB
        )
        for (M, N, K), dtl in cases:
            req = GemmRequest(M=M, N=N, K=K, arch="gfx950")
            with self.subTest(shape=(M, N, K)):
                ok, why = cand.admits(req)
                self.assertEqual(ok, dtl, why)
                if not dtl:
                    self.assertIn("DTL buffer range", why)
                result = dispatch_gemm_fp16(req)
                self.assertEqual(result.spec.trait.direct_to_lds, dtl)
                if dtl:
                    t = result.spec.tile
                    self.assertLessEqual(M * K * 2, DTL_BUFFER_NUM_RECORDS)
                    self.assertEqual((t.tile_m, t.tile_n), (256, 256))

    def test_i32_element_range_boundary(self):
        # Every universal path indexes operands with an i32 element offset:
        # 2**31 elements are addressable, one more 128-row tile is not.
        ok_req = GemmRequest(M=131072, N=256, K=16384, arch="gfx950")
        self.assertFalse(dispatch_gemm_fp16(ok_req).spec.trait.direct_to_lds)
        bad_req = GemmRequest(
            M=131200, N=256, K=16384, arch="gfx950", algorithm="universal_gemm"
        )
        with self.assertRaises(Exception) as cm:
            dispatch_gemm_fp16(bad_req)
        self.assertIn("i32 element offsets", str(cm.exception))
        # auto mode: the gemm_skinny fallback (64-bit per-tile bases) serves it
        r = dispatch_gemm_fp16(GemmRequest(M=131200, N=256, K=16384, arch="gfx950"))
        self.assertIn("skinny", r.candidate.name)

    def test_dispatch_wave32_arch(self):
        req = GemmRequest(M=128, N=64, K=64, arch="gfx1151")
        result = dispatch_gemm_fp16(req)
        self.assertEqual(result.kernel_id.spec_id, "rdna_wmma_default")
        self.assertEqual(result.spec.wave_size, 32)
        self.assertEqual(result.spec.trait.pipeline, "mem")
        self.assertEqual(result.spec.trait.epilogue, "default")
        ok, why = is_valid_spec(result.spec, arch=req.arch)
        self.assertTrue(ok, why)

    def test_kernel_id_is_deterministic(self):
        req = GemmRequest(M=1024, N=2048, K=512, arch="gfx950")
        a = dispatch_gemm_fp16(req).kernel_id
        b = dispatch_gemm_fp16(req).kernel_id
        self.assertEqual(a, b)
        self.assertEqual(a.cache_key, b.cache_key)

    def test_rejects_unsupported_dtype_and_layout(self):
        # dtype and layout are declared coverage, so the capability prefilter
        # rejects them before the residual predicate is consulted.
        with self.assertRaisesRegex(ValueError, r"dtype 'bf16' not in \('fp16',\)"):
            dispatch_gemm_fp16(GemmRequest(M=1, N=1, K=1, arch="gfx950", dtype="bf16"))
        # (native NN / TN / TT: RRR / CRR / CCR are supported with the matching
        # trans flags; any other layout string is not)
        with self.assertRaisesRegex(
            ValueError, "layout RRR expects trans_a=False, trans_b=False"
        ):
            dispatch_gemm_fp16(GemmRequest(M=1, N=1, K=1, arch="gfx950", layout="RRR"))
        with self.assertRaisesRegex(ValueError, "layout 'CCC' not in"):
            dispatch_gemm_fp16(GemmRequest(M=1, N=1, K=1, arch="gfx950", layout="CCC"))
        with self.assertRaisesRegex(ValueError, "algorithm"):
            dispatch_gemm_fp16(
                GemmRequest(M=1, N=1, K=1, arch="gfx950", algorithm="direct_gemm")
            )

    def test_rejects_shapes_without_required_tile_granularity(self):
        # Registered phase-1 variants have pad_m/n/k disabled, so the problem
        # shape must be an exact multiple of the selected CTA tile. This mirrors
        # tile_engine's split between config validity and problem-shape support.
        with self.assertRaisesRegex(ValueError, "M=130 is not divisible"):
            dispatch_gemm_fp16(
                GemmRequest(
                    M=130,
                    N=128,
                    K=32,
                    arch="gfx950",
                    spec_id="cdna_cshuffle_default",
                )
            )
        # no unpadded UniversalGemm body admits M=130; the no-refusal path
        # serves it with bounded padded bodies only: ring_any / the padded
        # fallback (pad_row_clamp or dtl_tight_rsrc), gemm_skinny and its
        # padded small ring, the universal fallback (kc_row_guard)
        from unittest import mock

        from rocke.dispatch.gemm import common as _gc

        req = GemmRequest(M=130, N=128, K=32, arch="gfx950")
        self.assertEqual(
            tuple(
                s
                for s in gemm_fp16_sweep_space(
                    GemmRequest(
                        M=130, N=128, K=32, arch="gfx950", algorithm="universal_gemm"
                    )
                )
                if not s.trait.pad_m
            ),
            (),
        )
        space = gemm_fp16_sweep_space(req)
        self.assertTrue(space)
        for s in space:
            if isinstance(s, SkinnyGemmSpec):
                continue
            self.assertTrue(s.trait.pad_m)
            self.assertTrue(
                s.trait.pad_row_clamp or s.trait.dtl_tight_rsrc or s.trait.kc_row_guard
            )
        names = {s.name for s in space}
        with mock.patch.object(_gc, "G1_PAD_FALLBACK", False), mock.patch.object(
            _gc, "G1_RING_ANY", False
        ):
            off = {s.name for s in gemm_fp16_sweep_space(req)}
        self.assertNotIn("universal_gemm_fp16_cdna_pad", off)
        self.assertNotIn("universal_gemm_fp16_cdna_dtl_ring_any", off)
        self.assertLessEqual(off, names)

    def test_config_support_uses_arch_mma_and_tile_knobs(self):
        result = dispatch_gemm_fp16(
            GemmRequest(M=128, N=128, K=32, arch="gfx950", algorithm="universal_gemm")
        )
        query = support_query_from_universal_spec(result.spec, arch="gfx950")
        ok, why = gemm_config_supported(query)
        self.assertTrue(ok, why)

        # Same CTA shape with an unsupported warp tile must be rejected by the
        # arch-backed MMA catalog, independent of problem shape.
        bad_query = type(query)(
            **{
                **query.__dict__,
                "warp_tile": (64, 64, 16),
            }
        )
        ok, why = gemm_config_supported(bad_query)
        self.assertFalse(ok)
        self.assertIn("unsupported", why)

    def test_request_shape_granularity_is_separate_from_config_support(self):
        result = dispatch_gemm_fp16(
            GemmRequest(M=128, N=128, K=32, arch="gfx950", algorithm="universal_gemm")
        )
        ok, why = request_shape_supported(
            GemmRequest(M=130, N=128, K=32, arch="gfx950"), result.spec
        )
        self.assertFalse(ok)
        self.assertIn("M=130", why)

    def test_registered_variants_can_be_selected_by_spec_id(self):
        cdna = dispatch_gemm_fp16(
            GemmRequest(
                M=1024,
                N=2048,
                K=512,
                arch="gfx950",
                algorithm="universal_gemm",
                spec_id="cdna_mem_64x128",
            )
        )
        self.assertEqual(cdna.candidate.name, "universal_gemm_fp16_cdna_mem")
        self.assertEqual(cdna.spec.trait.pipeline, "mem")
        self.assertEqual(cdna.spec.trait.epilogue, "default")
        self.assertEqual(cdna.grid, (16, 16, 1))

        rdna = dispatch_gemm_fp16(
            GemmRequest(
                M=128,
                N=128,
                K=64,
                arch="gfx1151",
                spec_id="rdna_wmma_32x32",
            )
        )
        self.assertEqual(rdna.candidate.name, "universal_gemm_fp16_rdna_wmma_small")
        self.assertEqual(rdna.grid, (4, 4, 1))

    def test_registry_exposes_multiple_variants(self):
        candidates = gemm_fp16_candidates()
        self.assertGreaterEqual(len(candidates), 4)
        self.assertIn("cdna_cshuffle_default", {c.spec_id for c in candidates})
        self.assertIn("rdna_wmma_default", {c.spec_id for c in candidates})

    def test_ranker_can_override_auto_priority(self):
        def prefer_mem(_request, candidates):
            return sorted(candidates, key=lambda c: c.spec_id != "cdna_mem_64x128")

        result = dispatch_gemm_fp16(
            GemmRequest(M=1024, N=2048, K=512, arch="gfx950"),
            ranker=prefer_mem,
        )
        self.assertEqual(result.kernel_id.spec_id, "cdna_mem_64x128")
        self.assertEqual(result.candidate.name, "universal_gemm_fp16_cdna_mem")

    def test_sweep_space_is_bounded_and_valid(self):
        from rocke.dispatch.gemm import common as _common

        for arch in ("gfx942", "gfx950", "gfx1151", "gfx1201"):
            req = GemmRequest(M=512, N=512, K=512, arch=arch)
            specs = gemm_fp16_sweep_space(req)
            self.assertGreaterEqual(len(specs), 1)
            # every admitting candidate's space: up to 4 universal variants per
            # registered rule body, + the register-staged cost-model plans
            # (gfx942), + the ring_any retiles (gfx950, up to 6), + the
            # gemm_skinny region / fallback specs, + the padded fallbacks
            g5 = _common.G5_VGPR_SWEEP if arch in _common.G5_VGPR_ARCHES else 0
            self.assertLessEqual(len(specs), 4 + g5 + 6 + 2 + 2)
            for spec in specs:
                if isinstance(spec, SkinnyGemmSpec):
                    ok, why = skinny_is_valid_spec(spec, arch)
                else:
                    ok, why = is_valid_spec(spec, arch=arch)
                self.assertTrue(ok, f"{arch}: {why}")

    def test_dispatch_all_enumerates_every_eligible_variant(self):
        req = GemmRequest(M=512, N=512, K=512, arch="gfx950")
        winner = dispatch_gemm_fp16(req)
        results = dispatch_gemm_fp16_all(req)
        self.assertGreater(len(results), 1)
        self.assertEqual(results[0].candidate.name, winner.candidate.name)
        self.assertEqual(len(results), len(gemm_fp16_sweep_space(req)))
        self.assertEqual(len(results), len(registered_gemm_fp16_combos(req)))

    def test_selected_kernel_lowers_to_llvm(self):
        req = GemmRequest(M=128, N=128, K=64, arch="gfx950", algorithm="universal_gemm")
        result = dispatch_gemm_fp16(req)
        kernel = build_kernel(result)
        llvm = lower_kernel_to_llvm(kernel, arch=req.arch)
        self.assertIn("define amdgpu_kernel void", llvm)
        self.assertIn("@llvm.amdgcn.mfma.f32.32x32x16.f16", llvm)


if __name__ == "__main__":
    unittest.main()
