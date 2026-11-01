# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Plan-level native vs transpose pre-pass choice (``rocke.dispatch.gemm.plan``). CPU only."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from rocke.dispatch.gemm import GemmRequest, dispatch_gemm_bf16, dispatch_gemm_fp16
from rocke.dispatch.gemm import plan as P
from rocke.dispatch.gemm import selector as S

GEMM_SELECTOR_TESTS = True


def _req(M, N, K, lay, dt="fp16"):
    return GemmRequest(
        M=M,
        N=N,
        K=K,
        arch="gfx950",
        dtype=dt,
        layout=lay,
        trans_a=lay[0] == "C",
        trans_b=lay[1] == "C",
    )


class TestPlan(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ)
        self._env.start()
        for k in (P.GEMM_PLAN_PREPASS_ENV, S.SELECTOR_ENV, S.NATIVE_SELECTOR_ENV):
            os.environ.pop(k, None)
        S.clear_cache()

    def tearDown(self):
        self._env.stop()
        S.clear_cache()

    def test_rcr_is_native(self):
        pl = P.plan_gemm_fp16(_req(4096, 4096, 4096, "RCR"))
        self.assertEqual(pl.kind, "native")
        self.assertEqual(pl.transposes, ())
        self.assertEqual(pl.workspace_bytes, 0)

    def test_transpose_steps(self):
        st = P.transpose_steps(_req(4096, 2048, 1024, "CRR"))
        self.assertEqual(
            [(t.operand, t.rows, t.cols) for t in st],
            [("A", 1024, 4096), ("B", 1024, 2048)],
        )
        self.assertEqual(
            P.transpose_steps(_req(4096, 2048, 1024, "RRR"))[0].operand, "B"
        )
        self.assertEqual(
            P.transpose_steps(_req(4096, 2048, 1024, "CCR"))[0].operand, "A"
        )
        self.assertIsNone(P.transpose_steps(_req(4000, 2048, 1024, "CCR")))
        self.assertIsNone(P.transpose_steps(_req(4096, 2048, 1000, "RRR")))
        # rows / 64 on grid.y must fit the HIP limit
        self.assertIsNone(P.transpose_steps(_req(64, 64, 65536 * 64 + 64, "RRR")))

    def test_policies(self):
        req = _req(4096, 4096, 1024, "RRR")
        with mock.patch.dict(os.environ, {P.GEMM_PLAN_PREPASS_ENV: "off"}):
            self.assertEqual(P.plan_gemm_fp16(req).kind, "native")
        with mock.patch.dict(os.environ, {P.GEMM_PLAN_PREPASS_ENV: "always"}):
            pl = P.plan_gemm_fp16(req)
            self.assertEqual(pl.kind, "prepass")
            self.assertEqual(pl.gemm.spec.data.layout, "RCR")
            self.assertEqual(pl.workspace_bytes, 2 * 4096 * 1024)
        with mock.patch.dict(os.environ, {P.GEMM_PLAN_PREPASS_ENV: "bogus"}):
            with self.assertRaises(ValueError):
                P.plan_gemm_fp16(req)

    def test_auto_predicts_and_respects_margin(self):
        req = _req(8192, 4096, 1024, "CRR", "bf16")
        pl = P.plan_gemm_bf16(req)
        self.assertIn(pl.kind, ("native", "prepass"))
        for k in ("native", "rcr_gemm", "transposes"):
            self.assertGreater(pl.predicted_us[k], 0.0)
        with mock.patch.object(P, "GEMM_PLAN_PREPASS_MARGIN", 1e9):
            self.assertEqual(P.plan_gemm_bf16(req).kind, "native")
        with mock.patch.object(P, "GEMM_PLAN_PREPASS_MARGIN", -1e9):
            self.assertEqual(P.plan_gemm_bf16(req).kind, "prepass")

    def test_zero_margin_follows_predictions(self):
        seen = set()
        for lay in ("RRR", "CRR", "CCR"):
            for M, N, K in (
                (8192, 4096, 1024),
                (1024, 8192, 256),
                (4096, 4096, 4096),
                (256, 512, 8192),
            ):
                req = _req(M, N, K, lay)
                with mock.patch.object(P, "GEMM_PLAN_PREPASS_MARGIN", 0.0):
                    pl = P.plan_gemm_fp16(req)
                pr = pl.predicted_us
                if None in (pr.get("native"), pr.get("rcr_gemm"), pr.get("transposes")):
                    # a gemm_skinny native pick has no model time: native
                    self.assertEqual(pl.kind, "native")
                    continue
                want = (
                    "prepass"
                    if pr["native"] > pr["rcr_gemm"] + pr["transposes"]
                    else "native"
                )
                self.assertEqual(pl.kind, want, (lay, M, N, K, pr))
                seen.add(pl.kind)
        self.assertTrue(seen)

    def test_off_beats_any_margin(self):
        req = _req(4096, 4096, 1024, "RRR")
        with mock.patch.object(P, "GEMM_PLAN_PREPASS_MARGIN", -1e9):
            self.assertEqual(P.plan_gemm_fp16(req).kind, "prepass")
            with mock.patch.dict(os.environ, {P.GEMM_PLAN_PREPASS_ENV: "off"}):
                self.assertEqual(P.plan_gemm_fp16(req).kind, "native")

    def test_native_plan_is_dispatch(self):
        req = _req(1000, 712, 3000, "RRR")
        pl = P.plan_gemm_fp16(req)
        self.assertEqual(pl.kind, "native")  # not tile-eligible
        self.assertEqual(pl.gemm.spec, dispatch_gemm_fp16(req).spec)

    def test_transpose_cost_monotone(self):
        a = P.transpose_cost_us(
            "gfx950", P.transpose_steps(_req(1024, 1024, 1024, "RRR"))
        )
        b = P.transpose_cost_us(
            "gfx950", P.transpose_steps(_req(1024, 8192, 1024, "RRR"))
        )
        self.assertLess(a, b)
        self.assertIsNone(P.transpose_cost_us("gfx1151", ()))


if __name__ == "__main__":
    unittest.main()


class TestTailSplitRows(unittest.TestCase):
    """``tail_split_rows``: the partial-wave tail of a body's tile grid."""

    def _spec(self, tm, tn):
        from dataclasses import replace

        s = dispatch_gemm_bf16(_req(8192, 8192, 1024, "RCR", "bf16")).spec
        return replace(s, tile=replace(s.tile, tile_m=tm, tile_n=tn))

    def test_ragged_last_wave(self):
        # 30 x 27 tiles of 256 x 256 on 256 workgroups: 3 full waves hold 28
        # whole tile rows (756 tiles), the last 2 rows are the tail
        rows = P.tail_split_rows(
            _req(7680, 6720, 1536, "RCR", "bf16"), self._spec(256, 256)
        )
        self.assertEqual(rows, 512)

    def test_head_fits_the_full_waves(self):
        for M, N, tm, tn in (
            (7680, 6720, 256, 256),
            (15360, 13440, 256, 256),
            (4608, 3840, 128, 192),
        ):
            sp = self._spec(tm, tn)
            rows = P.tail_split_rows(_req(M, N, 1024, "RCR", "bf16"), sp)
            with self.subTest(M=M, N=N):
                self.assertGreater(rows, 0)
                self.assertEqual((M - rows) % tm, 0)
                tiles_n = -(-N // tn)
                head = (M - rows) // tm * tiles_n
                waves = (-(-M // tm) * tiles_n) // 256
                self.assertLessEqual(head, waves * 256)
                # one more tile row would not fit the full waves
                self.assertGreater(head + tiles_n, waves * 256)

    def test_no_tail(self):
        # whole waves (16 x 16 tiles on 256 workgroups) and less than one wave
        self.assertEqual(
            P.tail_split_rows(_req(4096, 4096, 1024, "RCR"), self._spec(256, 256)), 0
        )
        self.assertEqual(
            P.tail_split_rows(_req(1024, 1024, 1024, "RCR"), self._spec(256, 256)), 0
        )
        self.assertEqual(
            P.tail_split_rows(
                _req(4096, 4096, 1024, "RCR"), self._spec(256, 256), cus=300
            ),
            0,
        )
