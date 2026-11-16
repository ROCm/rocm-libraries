# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests for gemm_perf_fp8/memory_probe.py (bandwidth and TDM feed probes).

The probes run on a gfx1250 device; here only the generated sources and the
host-side refusal to report a rate after a failed check are tested.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rocke.examples.gfx1250.gemm_perf_fp8 import memory_probe as mp

_PLACEHOLDERS = (
    "THREADS_",
    "TILE_",
    "M_",
    "N_",
    "K_",
    "LDS_READS",
    "REPEATS",
    "GAP_US",
)


class TestTdmSource(unittest.TestCase):
    def test_every_placeholder_is_substituted(self):
        src = mp.tdm_source(4096, 2048, 8192, lds_reads=True, repeats=5, gap_ms=10)
        for key in _PLACEHOLDERS:
            with self.subTest(key=key):
                self.assertIsNone(re.search(rf"\b{key}\b", src))
        self.assertIn("constexpr int M = 4096, N = 2048; constexpr long K = 8192;", src)
        self.assertIn("const int lds_reads = 1, repeats = 5, gap_us = 10000;", src)

    def test_the_geometry_is_the_case_studys(self):
        src = mp.tdm_source(4096, 4096, 65536, lds_reads=False, repeats=7, gap_ms=25)
        self.assertIn(f"#define THREADS {mp.THREADS}", src)
        self.assertIn(f"#define T {mp.TILE}", src)
        self.assertEqual((mp.TILE, mp.THREADS), (256, 512))

    def test_the_timed_launches_skip_the_check(self):
        src = mp.tdm_source(4096, 4096, 65536, lds_reads=False, repeats=7, gap_ms=25)
        self.assertIn("go(1); CHECK(hipDeviceSynchronize());", src)
        self.assertIn("for (int i = 0; i < 3; ++i) go(0);", src)

    def test_dimensions_must_be_whole_tiles(self):
        for m, n, k in ((4000, 4096, 8192), (4096, 4100, 8192), (4096, 4096, 8000)):
            with self.subTest(shape=(m, n, k)), self.assertRaises(ValueError):
                mp.tdm_source(m, n, k, lds_reads=False, repeats=1, gap_ms=0)


class TestBandwidthSource(unittest.TestCase):
    def test_sizes_and_protocol_are_substituted(self):
        src = mp.bandwidth_source([8, 8192], 3, 25)
        self.assertIn("const std::vector<size_t> mib = {8ull, 8192ull};", src)
        self.assertIn("const int repeats = 3, gap_us = 25000;", src)


class TestRefusesAfterAFailedCheck(unittest.TestCase):
    def _run_with(self, stdout: str, rc: int = 0):
        result = SimpleNamespace(stdout=stdout, stderr="", returncode=rc)
        with mock.patch.object(
            mp, "build", return_value=(Path("probe_x"), None)
        ), mock.patch.object(mp.subprocess, "run", return_value=result):
            return mp._run("src", Path("work"))

    def test_a_passing_run_returns_its_lines(self):
        lines = self._run_with("CHECK tdm OK\nTDM 256 0.4 1.6 20.0 1.004\n")
        self.assertEqual(lines[-1].split()[0], "TDM")

    def test_a_wrong_check_or_error_is_refused(self):
        for out in (
            "CHECK tdm WRONG\n",
            "CHECK copy WRONG\n",
            "ERROR hipErrorX at 3\n",
        ):
            with self.subTest(out=out), self.assertRaises(SystemExit):
                self._run_with(out)

    def test_a_nonzero_exit_is_refused(self):
        with self.assertRaises(SystemExit):
            self._run_with("CHECK tdm OK\n", rc=1)


if __name__ == "__main__":
    unittest.main()
