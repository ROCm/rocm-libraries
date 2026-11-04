# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the occupancy calibration tool's pure logic (no GPU, no build).

Covers the parse/match/compare path that turns a rocprofv3 CSV into a
predicted-vs-measured verdict. The ``predict``/``measure`` halves need a compiler /
GPU and are exercised elsewhere."""
from __future__ import annotations

import unittest

from benchmarks.common import calibrate_occupancy as C


def _pred(label="k", arch="gfx950", name="rocke_kernel_abc", pred_cu=8):
    return C.Prediction(
        label=label,
        arch=arch,
        kernel_name=name,
        vgpr=256,
        agpr=0,
        lds_bytes=1024,
        waves_per_wg=4,
        predicted_waves_per_cu=pred_cu,
        predicted_waves_per_simd=pred_cu // 4,
        limited_by="VGPR",
    )


class TestParseMeasuredCsv(unittest.TestCase):
    def test_averages_over_dispatches_and_filters_counter(self):
        text = "\n".join(
            [
                "Dispatch_Id,Kernel_Name,Counter_Name,Counter_Value",
                "1,foo.kd,MeanOccupancyPerCU,8.0",
                "2,foo.kd,MeanOccupancyPerCU,10.0",  # mean 9.0
                "3,foo.kd,SQ_WAVES,9999",  # different counter -> ignored
                "4,bar.kd,MeanOccupancyPerCU,4.0",
            ]
        )
        got = C.parse_measured_csv(text)
        self.assertAlmostEqual(got["foo.kd"], 9.0)
        self.assertAlmostEqual(got["bar.kd"], 4.0)

    def test_bad_values_skipped(self):
        text = "\n".join(
            [
                "Kernel_Name,Counter_Name,Counter_Value",
                "foo.kd,MeanOccupancyPerCU,",  # empty
                "foo.kd,MeanOccupancyPerCU,7.0",
            ]
        )
        self.assertAlmostEqual(C.parse_measured_csv(text)["foo.kd"], 7.0)

    def test_missing_counter_yields_empty(self):
        text = "Kernel_Name,Counter_Name,Counter_Value\nfoo.kd,SQ_WAVES,3\n"
        self.assertEqual(C.parse_measured_csv(text), {})

    def test_accumulates_across_files_not_last_wins(self):
        # A kernel present in two CSVs is averaged over every dispatch across both,
        # not overwritten by the last file.
        f1 = "Kernel_Name,Counter_Name,Counter_Value\nk.kd,MeanOccupancyPerCU,10\n"
        f2 = (
            "Kernel_Name,Counter_Name,Counter_Value\n"
            "k.kd,MeanOccupancyPerCU,4\nk.kd,MeanOccupancyPerCU,4\n"
        )
        got = C.parse_measured_csvs([f1, f2])
        self.assertAlmostEqual(got["k.kd"], 6.0)  # (10+4+4)/3, not last-file 4.0


class TestMatchMeasured(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(C._match_measured("k", {"k": 5.0}), 5.0)

    def test_unique_substring(self):
        self.assertEqual(C._match_measured("rocke_k", {"rocke_k.kd": 6.0}), 6.0)

    def test_ambiguous_substring_is_none(self):
        m = {"rocke_k.kd": 6.0, "rocke_k.kd.clone": 7.0}
        self.assertIsNone(C._match_measured("rocke_k", m))

    def test_no_match_is_none(self):
        self.assertIsNone(C._match_measured("k", {"other": 1.0}))


class TestCompare(unittest.TestCase):
    def test_match_within_tol(self):
        rows = C.compare([_pred(pred_cu=8)], {"rocke_kernel_abc": 8.3}, tol=0.5)
        self.assertEqual(rows[0].verdict, "MATCH")

    def test_model_low_when_measured_higher(self):
        # hardware allows more waves than the conservative caps predict.
        rows = C.compare([_pred(pred_cu=8)], {"rocke_kernel_abc": 10.0})
        self.assertEqual(rows[0].verdict, "MODEL_LOW")
        self.assertAlmostEqual(rows[0].delta, 2.0)

    def test_model_high_when_measured_lower(self):
        rows = C.compare([_pred(pred_cu=8)], {"rocke_kernel_abc": 5.0})
        self.assertEqual(rows[0].verdict, "MODEL_HIGH")

    def test_no_measurement(self):
        rows = C.compare([_pred()], {})
        self.assertEqual(rows[0].verdict, "NO_MEASUREMENT")
        self.assertIsNone(rows[0].delta)


if __name__ == "__main__":
    unittest.main()
