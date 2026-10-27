#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Self-test for sample_probe.py's legacy-case verdicts; runs as a ctest case."""

import argparse
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

# The harness runs from an install tree, which may be read-only.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_probe  # noqa: E402

CASE = "the case"
FAILED_SUMMARY = "test cases: 1 | 1 failed\nassertions: 5 | 2 passed | 3 failed\nboom\n"
CLEAN_SUMMARY = "All tests passed (2 assertions in 1 test case)\n"
UNMATCHED = "No test cases matched '\"the case\"'\nNo tests ran\n"


def child(output: str, exit_code: int) -> list:
    """A command that prints `output` and exits with `exit_code`."""
    return [
        sys.executable,
        "-c",
        f"import sys; sys.stdout.write({output!r}); sys.exit({exit_code})",
    ]


def crashing_child() -> list:
    if os.name == "nt":
        # STATUS_ACCESS_VIOLATION, as a native crash reports it.
        body = "import os; os._exit(-1073741819)"
    else:
        body = "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"
    return [sys.executable, "-c", body]


class CaseModeTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        self.sidecar = self.root / "sidecar.json"

    def tearDown(self):
        self._dir.cleanup()

    def write_manifest(self, entry: dict) -> Path:
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"legacy": {"cases": {CASE: entry}}}))
        return manifest

    def run_case(self, entry: dict, command: list):
        args = argparse.Namespace(
            manifest=self.write_manifest(entry), case=CASE, sidecar=self.sidecar
        )
        with redirect_stdout(StringIO()):
            code = sample_probe.case_mode(args, command)
        return code, json.loads(self.sidecar.read_text(encoding="utf-8"))

    def test_three_failed_assertions_are_not_an_unmatched_spec(self):
        # Catch2 < 3.8 exits with the failed-assertion count, so this run exits 3.
        entry = {
            "tier": "XFAIL_RUN",
            "reason": "r",
            "expect": [{"outcome": "ran-with-assertion-failures", "anchor": "boom"}],
        }
        code, sidecar = self.run_case(entry, child(FAILED_SUMMARY, 3))
        self.assertEqual(code, 0)
        self.assertEqual(sidecar["outcome"], "ran-with-assertion-failures")
        self.assertEqual(sidecar["verdict"], "expected")

    def test_unmatched_spec_is_run_failed(self):
        code, sidecar = self.run_case(
            {"tier": "VACUOUS", "reason": "r"}, child(UNMATCHED, 2)
        )
        self.assertEqual(code, 1)
        self.assertEqual(sidecar["outcome"], "run-failed")
        self.assertEqual(sidecar["reason"], "case not registered")

    def test_vacuous_case_that_runs_is_skipped(self):
        entry = {"tier": "VACUOUS", "reason": "never reaches the shim"}
        code, sidecar = self.run_case(entry, child(CLEAN_SUMMARY, 0))
        self.assertEqual(code, 77)
        self.assertEqual(sidecar["verdict"], "expected")
        self.assertEqual(sidecar["reason"], "never reaches the shim")

    def test_vacuous_case_that_crashes_fails(self):
        code, sidecar = self.run_case(
            {"tier": "VACUOUS", "reason": "r"}, crashing_child()
        )
        self.assertEqual(code, 1)
        self.assertEqual(sidecar["outcome"], "crashed")
        self.assertEqual(sidecar["verdict"], "unexpected")

    def test_run_is_not_a_legacy_tier(self):
        manifest = self.write_manifest({"tier": "RUN", "reason": "r"})
        with self.assertRaises(ValueError):
            sample_probe.load_case(manifest, CASE)


if __name__ == "__main__":
    unittest.main()
