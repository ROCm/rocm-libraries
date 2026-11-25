# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the gfx1250 hardware job's scripts: gfx1250_blas_tests.py (filters, CPU
share, report parsing, the compiler check, public XML copies). No GPU or ROCm install
needed."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import gfx1250_blas_tests as blas  # noqa: E402

GTEST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="4" time="12.5" timestamp="2026-11-24T00:00:00" name="AllTests">
  <testsuite name="Suite" tests="4" time="12.5">
    <testcase name="passes" status="run" result="completed" time="1.0" classname="Suite"/>
    <testcase name="fails" status="run" result="completed" time="2.0" classname="Suite">
      <failure message="expected 1, got 2"/>
    </testcase>
    <testcase name="skips" status="run" result="skipped" time="0" classname="Suite">
      <skipped message="needs two GPUs"/>
    </testcase>
    <testcase name="DISABLED_off" status="notrun" result="suppressed" time="0" classname="Suite"/>
  </testsuite>
</testsuites>
"""

JUNIT_XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="2" skipped="3" tests="10"
 time="99.1" timestamp="2026-11-24T00:00:00" hostname="build-host.amd.com">
 <testcase classname="a" name="b" time="1.0"/></testsuite></testsuites>
"""


def write(directory, name, text):
    path = Path(directory) / name
    path.write_text(text)
    return path


class FilterTest(unittest.TestCase):
    def test_positive_and_negative_patterns(self):
        self.assertTrue(blas.gtest_filter_selects("A.b", "A.*"))
        self.assertFalse(blas.gtest_filter_selects("A.b", "A.*-*b"))
        self.assertTrue(blas.gtest_filter_selects("A.b", "-B.*"))

    def test_empty_filter_selects_everything(self):
        self.assertTrue(blas.gtest_filter_selects("Any.test", ""))

    def test_every_large_group_is_one_of_the_large_patterns(self):
        # A group outside LARGE_MEMORY_TESTS would run in both the shards and its own process.
        for group in blas.LARGE_MEMORY_GROUPS:
            positive = group.partition("-")[0]
            self.assertTrue(
                any(positive.startswith(p.rstrip("*")) for p in blas.LARGE_MEMORY_TESTS), group
            )

    def test_parse_test_list(self):
        listing = "Suite.\n  one\n  two  # GetParam() = 3\nOther/Param.\n  three/0\n"
        self.assertEqual(
            blas.parse_test_list(listing),
            ["Suite.one", "Suite.two", "Other/Param.three/0"],
        )


class CpuShareTest(unittest.TestCase):
    def test_pinned_container_keeps_its_cpus(self):
        with mock.patch.dict(os.environ, {"GPUS_PER_HOST": "4"}):
            self.assertEqual(blas.job_cpus(cpus_visible=64, cpus_on_host=256), 64)

    def test_unpinned_container_gets_its_gpus_share(self):
        with mock.patch.dict(os.environ, {"GPUS_PER_HOST": "4"}):
            self.assertEqual(blas.job_cpus(cpus_visible=256, cpus_on_host=256), 64)

    def test_single_gpu_host_keeps_everything(self):
        with mock.patch.dict(os.environ, {"GPUS_PER_HOST": "1"}):
            self.assertEqual(blas.job_cpus(cpus_visible=256, cpus_on_host=256), 256)


class TestResultLineTest(unittest.TestCase):
    def test_matches_result_lines_without_timing(self):
        self.assertEqual(blas.TEST_RESULT.match("[       OK ] Suite.test")[1], "OK")
        self.assertEqual(
            blas.TEST_RESULT.match("[  FAILED  ] P/Suite.t/3, where GetParam() = 7")[1],
            "FAILED",
        )

    def test_ignores_counts_and_headers(self):
        for line in (
            "[  FAILED  ] 5 tests, listed below:",
            "[  PASSED  ] 10 tests.",
            "[ RUN      ] Suite.t",
        ):
            self.assertIsNone(blas.TEST_RESULT.match(line), line)


class ReportParsingTest(unittest.TestCase):
    def test_gtest_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = blas.parse_gtest_xml(write(tmp, "r.xml", GTEST_XML))
        self.assertEqual(results["run"], 3)
        self.assertEqual(results["failed"], ["Suite.fails"])
        self.assertEqual(results["skipped"], {"Suite.skips": "needs two GPUs"})

    def test_junit_xml_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            totals = blas.parse_junit_xml(write(tmp, "j.xml", JUNIT_XML))
        self.assertEqual(
            totals, {"tests": 10, "failures": 2, "errors": 1, "skipped": 3}
        )

    def test_public_xml_drops_timings_and_host_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            for text in (GTEST_XML, JUNIT_XML):
                public = Path(tmp) / "public.xml"
                blas.write_public_xml(write(tmp, "in.xml", text), public)
                attributes = {
                    key for element in ET.parse(public).iter() for key in element.attrib
                }
                self.assertFalse(attributes & set(blas.PRIVATE_XML_ATTRIBUTES))
                self.assertNotIn("amd.com", public.read_text())

    def test_public_xml_scrubs_messages(self):
        report = GTEST_XML.replace(
            'message="expected 1, got 2"',
            'message="on node7.amd.com: expected 1, got 2"',
        ).replace(
            "</testsuite>",
            "<system-out>Device ID 0 : AMD Radeon Graphics gfx1250\nresult ok</system-out></testsuite>",
        )
        with tempfile.TemporaryDirectory() as tmp:
            public = Path(tmp) / "public.xml"
            blas.write_public_xml(write(tmp, "in.xml", report), public)
            text = public.read_text()
        self.assertIn("on &lt;redacted&gt;: expected 1, got 2", text)
        self.assertIn("Device ID 0 : AMD Radeon Graphics gfx1250", text)
        self.assertIn("result ok", text)


class ReportOutcomeTest(unittest.TestCase):
    """report_hipblaslt raises for every way a run can go wrong."""

    def process(self, tmp, name="shard0"):
        process = mock.Mock(peak_memory_gib="1.0", memory_stops=[])
        process.name = name
        process.log = write(tmp, f"{name}.log", "log text\n")
        return process

    def outcome(self, exit_code, results, problems=()):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop("GITHUB_STEP_SUMMARY", None)
            with mock.patch("sys.stdout"):
                try:
                    blas.report_hipblaslt(
                        [(self.process(tmp), exit_code, results)], list(problems)
                    )
                except blas.CheckFailed as failure:
                    return str(failure)
        return None

    def test_clean_run_passes(self):
        self.assertIsNone(self.outcome(0, {"run": 3, "failed": [], "skipped": {}}))

    def test_failed_test_fails(self):
        self.assertIn(
            "1 hipBLASLt tests failed",
            self.outcome(1, {"run": 3, "failed": ["S.t"], "skipped": {}}),
        )

    def test_crash_without_results_fails(self):
        self.assertIn("wrote no results", self.outcome(134, None))

    def test_nonzero_exit_without_failures_fails(self):
        self.assertIn(
            "exited with 1", self.outcome(1, {"run": 3, "failed": [], "skipped": {}})
        )

    def test_filter_problem_fails(self):
        self.assertIn(
            "matches no test",
            self.outcome(
                0, {"run": 3, "failed": [], "skipped": {}}, ["group matches no test"]
            ),
        )


class CompilerCheckTest(unittest.TestCase):
    def test_sdk_manifest(self):
        manifest = {
            "the_rock_commit": "b" * 40,
            "submodules": [
                {"submodule_name": "rocm-libraries", "pin_sha": "c" * 40},
                {"submodule_name": "llvm-project", "pin_sha": "a" * 40},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "share/therock"
            target.mkdir(parents=True)
            (target / "therock_manifest.json").write_text(json.dumps(manifest))
            with mock.patch("sys.stdout"):
                self.assertEqual(blas.sdk_compiler_and_therock_commits(tmp), ("a" * 40, "b" * 40))

    def test_therock_submodule_pin(self):
        # A TheRock-like repo whose compiler/amd-llvm is a gitlink to a known commit.
        pinned = "1" * 40
        with tempfile.TemporaryDirectory() as tmp:

            def git(*args):
                subprocess.run(
                    ["git", "-C", tmp, *args], check=True, capture_output=True
                )

            git("init", "-q")
            git(
                "update-index",
                "--add",
                "--cacheinfo",
                f"160000,{pinned},compiler/amd-llvm",
            )
            git(
                "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "pin"
            )
            llvm, head = blas.therock_compiler_and_commit(tmp)
        self.assertEqual(llvm, pinned)
        self.assertEqual(len(head), 40)

    def test_missing_checkout_only_warns(self):
        args = mock.Mock(therock_dir="")
        with mock.patch.dict(os.environ, {"THEROCK_DIR": ""}), mock.patch(
            "builtins.print"
        ) as printed:
            self.assertEqual(blas.check_compiler(args), 0)
        self.assertIn("::warning::", printed.call_args[0][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
