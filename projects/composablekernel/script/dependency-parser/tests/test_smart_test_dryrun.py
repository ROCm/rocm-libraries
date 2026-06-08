#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dry-run dispatch tests for smart_test.sh, focused on the always-run class.

smart_test.sh --dry-run prints the ctest commands it would run without executing
them, so these checks need no GPU/ctest. They pin the rule that the non-compiled
"always-run" class (python/try_compile tests, with no bin/ target) runs in the
selective and none modes but not as a separate command in full mode.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SMART_TEST = Path(__file__).parent.parent / "smart_test.sh"
ALWAYS_RUN_RE = r"-R ^(test_py_a|test_py_b)$"


@unittest.skipUnless(shutil.which("jq"), "smart_test.sh requires jq")
class TestAlwaysRunDryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _write(self, name, content):
        with open(os.path.join(self.tmp, name), "w") as f:
            f.write(content)

    def _run(self, mode, non_compiled=None, chunks=None, extra_env=None):
        self._write("build_mode.env", f"SMART_BUILD_MODE={mode}\n")
        if non_compiled is not None:
            self._write(
                "reachability_result.json",
                json.dumps({"non_compiled": non_compiled}),
            )
        if chunks is not None:
            self._write("tests_to_run.json", json.dumps({"regex_chunks": chunks}))
        env = {**os.environ, "DRY_RUN": "true", "BUILD_DIR": self.tmp,
               "CTEST_PARALLEL": "4", **(extra_env or {})}
        proc = subprocess.run(
            ["bash", str(SMART_TEST), "--dry-run"],
            env=env,
            capture_output=True,
            text=True,
        )
        return proc.returncode, proc.stdout

    def test_selective_runs_chunks_and_always_run_class(self):
        rc, out = self._run(
            "selective", non_compiled=["test_py_a", "test_py_b"],
            chunks=["^(test_gemm)$"],
        )
        self.assertEqual(rc, 0)
        self.assertIn("-R ^(test_gemm)$", out)   # the selected chunk
        self.assertIn(ALWAYS_RUN_RE, out)         # the always-run class

    def test_none_runs_only_always_run_class(self):
        rc, out = self._run("none", non_compiled=["test_py_a", "test_py_b"])
        self.assertEqual(rc, 0)
        self.assertIn(ALWAYS_RUN_RE, out)

    def test_full_does_not_append_always_run_class(self):
        rc, out = self._run("full", non_compiled=["test_py_a", "test_py_b"])
        self.assertEqual(rc, 0)
        # Full runs the whole suite (bare ctest); it must not add the always-run -R.
        self.assertIn("ctest --output-on-failure", out)
        self.assertNotIn(ALWAYS_RUN_RE, out)

    def test_full_excludes_separate_suites_by_default(self):
        # rocm_ck / builder are registered with ctest but built by their own
        # targets; full mode must exclude them so they don't fail as "Not Run".
        rc, out = self._run("full")
        self.assertEqual(rc, 0)
        self.assertIn("-LE ROCM_CK_|BUILDER_SMOKE", out)

    def test_full_exclusion_overridable_to_empty(self):
        rc, out = self._run("full", extra_env={"CTEST_FULL_EXCLUDE_LABELS": ""})
        self.assertEqual(rc, 0)
        self.assertNotIn("-LE", out)

    def test_none_without_report_is_graceful(self):
        rc, out = self._run("none")  # no reachability_result.json
        self.assertEqual(rc, 0)
        self.assertIn("not found", out)

    def test_none_with_empty_class_runs_nothing(self):
        rc, out = self._run("none", non_compiled=[])
        self.assertEqual(rc, 0)
        self.assertIn("Always-run class: none", out)
        self.assertNotIn("-R ^(", out)

    def test_selective_multi_chunk_runs_each_chunk(self):
        # NUM_CHUNKS > 1 exercises the for-loop path in smart_test.sh.
        rc, out = self._run(
            "selective",
            non_compiled=["test_py_a"],
            chunks=["^(test_gemm)$", "^(test_conv)$"],
        )
        self.assertEqual(rc, 0)
        self.assertIn("-R ^(test_gemm)$", out)
        self.assertIn("-R ^(test_conv)$", out)
        self.assertIn(r"-R ^(test_py_a)$", out)  # always-run class still fires

    def test_selective_zero_chunks_warns_and_runs_always_class(self):
        # regex_chunks: [] triggers the NUM_CHUNKS=0 warning guard.
        rc, out = self._run(
            "selective",
            non_compiled=["test_py_a"],
            chunks=[],
        )
        self.assertEqual(rc, 0)
        self.assertIn("no regex_chunks", out)
        self.assertIn(r"-R ^(test_py_a)$", out)  # always-run class still fires


if __name__ == "__main__":
    unittest.main()
