# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for scrub_test_log.py (run: python3 .github/scripts/tests/test_scrub_test_log.py)."""

import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scrub_test_log import Scrubber  # noqa: E402

SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "scrub_test_log.py"
)


def scrub_all(lines, **kwargs):
    s = Scrubber(**kwargs)
    return [out for out in map(s.scrub, lines) if out is not None]


class DropsDeviceLines(unittest.TestCase):
    def test_device_description_lines_are_dropped(self):
        for line in [
            "Query device success: there are 4 devices",
            "Device ID 0 : gfx1250",
            "PCI Chip ID 0x75a0",
            "with 288 GB memory",
            "Marketing\tName: x",
            "Firmware 0x7e",
            "clockRate: 2100",
            "Compute Unit:   256",
            "Max Clock 2.1",
        ]:
            self.assertIsNone(Scrubber().scrub(line), line)

    def test_other_lines_pass(self):
        self.assertEqual(
            Scrubber().scrub("[ RUN      ] Foo.Bar"), "[ RUN      ] Foo.Bar"
        )


class Redacts(unittest.TestCase):
    def test_host_product_and_pci_address(self):
        self.assertEqual(
            Scrubber().scrub("on node-1.atl.amd.com: MI455X, mi300 at 0003:01:00.0"),
            "on <redacted>: <redacted>, <redacted> at <redacted>",
        )

    def test_a_product_named_host_is_one_host(self):
        self.assertEqual(Scrubber().scrub("mi300x.amd.com"), "<redacted>")

    def test_long_runs_of_host_characters_scrub_in_linear_time(self):
        # Quadratic backtracking took about 11 s for 100k characters and would hang on
        # a megabyte-long line; linear takes well under a second.
        start = time.monotonic()
        self.assertEqual(Scrubber().scrub("a" * 1_000_000), "a" * 1_000_000)
        self.assertEqual(Scrubber().scrub("x." * 400_000 + "amd.com"), "<redacted>")
        self.assertLess(time.monotonic() - start, 5)

    def test_near_misses_are_kept(self):
        self.assertEqual(Scrubber().scrub("MI30 0000:c1:00.8"), "MI30 0000:c1:00.8")


class ProblemProgress(unittest.TestCase):
    HEADER = (
        "run,problem-progress,solution,(m,n,b,k),num-cus,time-us,validation,temp-edge"
    )

    def test_hardware_columns_blanked_counting_from_the_end(self):
        rows = scrub_all([self.HEADER, "0,1/2,sol,(64,64,1,64),256,12.5,PASSED,55"])
        self.assertEqual(rows[1], "0,1/2,sol,(64,64,1,64),,,PASSED,")

    def test_truncated_row_is_dropped(self):
        self.assertEqual(
            scrub_all([self.HEADER, "0,1/2,sol,(64,64,1,64),256"]), [self.HEADER]
        )

    def test_rows_before_any_header_and_other_csv_pass(self):
        self.assertEqual(scrub_all(["0,1/2,a,b", "x,y,z"]), ["0,1/2,a,b", "x,y,z"])

    def test_a_new_header_replaces_the_column_map(self):
        rows = scrub_all([self.HEADER, "run,problem-progress,a,time-us", "5,6/7,x,99"])
        self.assertEqual(rows[-1], "5,6/7,x,")


class Timing(unittest.TestCase):
    def test_gtest(self):
        self.assertEqual(
            scrub_all(
                ["[       OK ] A.b (12 ms)", "3 tests from A (40 ms total)"],
                timing="gtest",
            ),
            ["[       OK ] A.b", "3 tests from A"],
        )

    def test_pytest(self):
        self.assertEqual(
            Scrubber(timing="pytest").scrub(
                "5 passed in 1.5s (0:01:02) (12 seconds) (3.5 secs elapsed) _Final 2s done (4.2s)"
            ),
            "5 passed _Final done",
        )

    def test_none_keeps_timing(self):
        self.assertEqual(Scrubber().scrub("A.b (12 ms)"), "A.b (12 ms)")

    def test_versions(self):
        self.assertEqual(
            Scrubber(versions=True).scrub("clang version 19.0.0, ROCm-7.1.0"),
            "clang <ver>, ROCm <ver>",
        )


class Main(unittest.TestCase):
    def run_script(self, data, *args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args], input=data, capture_output=True, check=True
        ).stdout

    def test_log_is_untagged_and_stdout_is_tagged(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "shard0.log")
            out = self.run_script(
                b"a (1 ms)\nDevice ID 0\nmi300\n",
                "--timing",
                "gtest",
                "--log",
                log,
                "--tag",
                "shard0",
            )
            self.assertEqual(out, b"[shard0] a\n[shard0] <redacted>\n")
            with open(log, "rb") as f:
                self.assertEqual(f.read(), b"a\n<redacted>\n")

    def test_bytes_that_are_not_utf8_pass_through(self):
        self.assertEqual(
            self.run_script(b"x \xff mi300\ny\r\nlast"),
            b"x \xff <redacted>\ny\r\nlast\n",
        )


if __name__ == "__main__":
    unittest.main()
