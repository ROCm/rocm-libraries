# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The suite never reaches GitHub: a `gh` that only refuses comes first on PATH, so a call no test fakes fails at once
instead of reading, or probing, under the machine's logins. A test that needs gh puts its own fake in front of it.

This module is imported first by `unittest discover` (its name sorts first), so the PATH it sets holds for the whole
run; tests/conftest.py does the same under pytest (RKMON_PYTEST=1)."""

import os
import shutil
import subprocess
import tempfile
import unittest

_NO_GH = os.environ.get("RKMON_TESTS_NO_GH") or ""
if not _NO_GH:
    _NO_GH = tempfile.mkdtemp(prefix="rkmon_tests_nogh_")
    with open(os.path.join(_NO_GH, "gh"), "w", encoding="utf-8") as _f:
        _f.write(
            "#!/bin/sh\necho 'gh: the test suite runs no real gh (tests/test_00_no_github.py)' >&2\nexit 1\n"
        )
    os.chmod(os.path.join(_NO_GH, "gh"), 0o755)
    os.environ["RKMON_TESTS_NO_GH"] = _NO_GH
    os.environ["PATH"] = _NO_GH + os.pathsep + os.environ.get("PATH", "")
    os.environ.pop("RKMON_GH", None)

    import atexit

    atexit.register(shutil.rmtree, _NO_GH, True)

# The image routes rk's GitHub writes to its write broker and names its two login folders (docker/Dockerfile and the
# entry point set RKMON_WRITE_BROKER and GH_CONFIG_DIR): the suite runs its write commands in this process against
# their fakes, with the logins each test names, as on a host. Its PR runners' launcher (RKMON_PR_RUNNER) is the image's
# too: the suite runs every job as itself.
for _name in (
    "RKMON_PR_RUNNER",
    "RKMON_WRITE_BROKER",
    "RKMON_GH_BROKER",
    "RKMON_GH_READ_DIR",
    "GH_CONFIG_DIR",
):
    os.environ.pop(_name, None)


class TestNoGitHub(unittest.TestCase):
    def test_the_gh_on_path_refuses(self):
        p = subprocess.run(["gh", "api", "user"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn("the test suite runs no real gh", p.stderr)

    def test_no_write_broker_reaches_the_tests(self):
        """In the image, the environment names the write broker; the suite runs its writes in-process instead."""
        for name in ("RKMON_WRITE_BROKER", "GH_CONFIG_DIR"):
            self.assertNotIn(name, os.environ)


if __name__ == "__main__":
    unittest.main()
