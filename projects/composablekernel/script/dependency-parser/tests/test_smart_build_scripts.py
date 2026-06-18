#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shell-level regression tests for the smart-build "none" path.

The empty-selection case must remain a clean no-op end-to-end: an empty selection
becomes SMART_BUILD_MODE=none, and both the build and test stages exit 0 without
invoking ninja or ctest. The one dangerous near-miss - an empty tests_to_run.txt
while in *selective* mode - must stay a loud failure, because "none" is the only
sanctioned no-op.

These tests drive the real shell scripts via subprocess with fake ninja/ctest on
PATH that fail if ever called, so a future edit can't silently turn "nothing to
test" into "skip the tests that should have run".
"""

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent
SMART_BUILD = SCRIPT_DIR / "smart_build.sh"
SMART_TEST = SCRIPT_DIR / "smart_test.sh"


def _write_stub(path, body):
    """Write an executable shell stub at ``path``."""
    path.write_text(f"#!/bin/bash\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@unittest.skipUnless(shutil.which("bash"), "bash not available")
class TestSmartBuildNonePath(unittest.TestCase):
    """The none-path skips ninja/ctest and exits 0; selective-empty fails loud."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.build_dir = self.tmp / "build"
        self.build_dir.mkdir()
        # A stub bin/ prepended to PATH so the scripts pick up our fake tools.
        self.stub_bin = self.tmp / "bin"
        self.stub_bin.mkdir()
        # Sentinel a stub touches if it is ever invoked (it should not be).
        self.sentinel = self.tmp / "tool_was_called"
        self.env = dict(os.environ)
        self.env["PATH"] = f"{self.stub_bin}{os.pathsep}{self.env['PATH']}"
        self.env["BUILD_DIR"] = str(self.build_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _stub_tool(self, name):
        """Install a fake tool that records it was called, then fails."""
        _write_stub(self.stub_bin / name, f'touch "{self.sentinel}"\nexit 1')

    def _run(self, script, extra_env=None):
        env = dict(self.env)
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(script)],
            env=env,
            capture_output=True,
            text=True,
        )

    def test_smart_test_none_exits_zero_and_skips_ctest(self):
        """mode=none -> smart_test.sh exits 0 and never runs ctest."""
        (self.build_dir / "build_mode.env").write_text("SMART_BUILD_MODE=none\n")
        self._stub_tool("ctest")

        result = self._run(SMART_TEST)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.sentinel.exists(), "ctest was invoked in none mode")
        log = (self.build_dir / "smart_test.log").read_text()
        self.assertIn("nothing to test", log)

    def test_smart_test_selective_empty_list_fails_loud(self):
        """selective mode with an empty tests_to_run.txt must fail, not no-op."""
        (self.build_dir / "build_mode.env").write_text("SMART_BUILD_MODE=selective\n")
        (self.build_dir / "tests_to_run.txt").write_text("")
        self._stub_tool("ctest")

        result = self._run(SMART_TEST)

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertFalse(self.sentinel.exists(), "ctest ran despite empty selection")
        log = (self.build_dir / "smart_test.log").read_text()
        self.assertIn("tests_to_run.txt is empty", log)

    def test_smart_build_none_writes_mode_and_skips_ninja(self):
        """build_targets.txt=none -> smart_build.sh records mode=none, no ninja."""
        # Isolate smart_build.sh's none branch from the heavy selector: run it from
        # a scriptdir holding the real script next to a stub smart_build_ci.sh that
        # just emits "none". smart_build.sh resolves its sibling ci script via
        # SCRIPT_DIR, so the sibling stub is picked up.
        scriptdir = self.tmp / "scriptdir"
        scriptdir.mkdir()
        shutil.copy(SMART_BUILD, scriptdir / "smart_build.sh")
        _write_stub(
            scriptdir / "smart_build_ci.sh",
            'echo none > "${BUILD_DIR}/build_targets.txt"\nexit 0',
        )
        self._stub_tool("ninja")

        result = self._run(scriptdir / "smart_build.sh", extra_env={"NINJA_JOBS": "1"})

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.sentinel.exists(), "ninja ran in none mode")
        mode = (self.build_dir / "build_mode.env").read_text()
        self.assertIn("SMART_BUILD_MODE=none", mode)
        log = (self.build_dir / "smart_build.log").read_text()
        self.assertIn("nothing to build", log)


if __name__ == "__main__":
    unittest.main()
