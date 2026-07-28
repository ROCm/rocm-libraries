# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only tests for the forward-conv perf-comparison runner (AICK-1752 AC4).

Covers the pure logic (throughput/speedup math, MIOpen timing parse, dtype
mapping, corpus loading). The GPU/MIOpen execution paths are exercised on
hardware via ``python -m rocke.benchmark.conv_perf_compare``.
"""

from __future__ import annotations

import unittest

from rocke.benchmark import conv_perf_compare as cpc


class TestRowResultMath(unittest.TestCase):
    def test_tflops_and_speedup(self):
        # flops chosen so 1 ms -> exactly 1 TFLOP/s (flops / (ms*1e9)).
        r = cpc.RowResult(short="s", dtype="bf16", groups=4, flops=1_000_000_000_000)
        r.rocke_ms = 1.0  # -> 1000 TFLOP/s? 1e12/(1*1e9)=1000
        r.miopen_ms = 2.0
        d = r.as_dict()
        self.assertAlmostEqual(d["rocke_tflops"], 1000.0, places=6)
        self.assertAlmostEqual(d["miopen_tflops"], 500.0, places=6)
        # speedup vs miopen = miopen_ms / rocke_ms
        self.assertAlmostEqual(d["speedup_vs_miopen"], 2.0, places=6)
        # unrun references are None, not crashes
        self.assertIsNone(d["ck_tflops"])
        self.assertIsNone(d["oldck_tflops"])
        self.assertIsNone(d["speedup_vs_ck"])

    def test_missing_or_zero_times_are_none(self):
        r = cpc.RowResult(short="s", dtype="fp16", groups=1, flops=10)
        self.assertIsNone(r.as_dict()["rocke_tflops"])  # rocke_ms is None
        r.rocke_ms = 0.0
        self.assertIsNone(r.as_dict()["rocke_tflops"])  # guard against div-by-zero


class TestMiopenParse(unittest.TestCase):
    def test_time_regex_takes_last_elapsed(self):
        sample = (
            "MIOpen Forward Conv. Algorithm: 1, Solution: 85/ConvDirectNaiveConvFwd\n"
            "GPU Kernel Time Forward Conv. Elapsed: 0.014153 ms (average)\n"
        )
        m = cpc._MIOPEN_TIME_RE.findall(sample)
        self.assertEqual(m[-1], "0.014153")

    def test_dtype_to_miopen_mapping(self):
        self.assertEqual(cpc._DTYPE_TO_MIOPEN["fp16"], "convfp16")
        self.assertEqual(cpc._DTYPE_TO_MIOPEN["bf16"], "convbfp16")


class TestCorpusLoading(unittest.TestCase):
    def test_single_cmd(self):
        args = type(
            "A", (), {"miopen_cmd": "MIOpenDriver convfp16 -n 1", "miopen_file": None}
        )()
        self.assertEqual(cpc._load_corpus(args), ["MIOpenDriver convfp16 -n 1"])

    def test_file_skips_comments_and_blanks(self):
        import tempfile
        import os

        fd, path = tempfile.mkstemp(suffix=".txt")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("# a comment\n\nMIOpenDriver convbfp16 -n 2 -g 4\n  \n")
            args = type("A", (), {"miopen_cmd": None, "miopen_file": path})()
            self.assertEqual(
                cpc._load_corpus(args), ["MIOpenDriver convbfp16 -n 2 -g 4"]
            )
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
