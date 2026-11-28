# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every tracked file keeps LF line endings (lessons L-228: an edit through Windows Python can write
CRLF, which makes a focused commit read as a whole-file rewrite, and a CRLF shebang or bash script does not run).
"""

import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestLineEndings(unittest.TestCase):
    def test_no_tracked_file_has_a_carriage_return(self):
        try:
            files = subprocess.run(
                ["git", "-C", str(ROOT), "ls-files"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("not a git checkout")
        bad = []
        for f in files:
            p = ROOT / f
            if p.is_file() and b"\r" in p.read_bytes():
                bad.append(f)
        self.assertEqual(
            bad, [], "CRLF in tracked files (L-228): convert with sed -i 's/\\r$//'"
        )


if __name__ == "__main__":
    unittest.main()
