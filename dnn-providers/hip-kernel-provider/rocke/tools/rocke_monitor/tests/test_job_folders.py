# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A job (rkmon_lib/jobs.py) in a state folder whose path has a space: its cmd.sh runs the job's script and records the
script's exit code in rc, as in any other folder. A temporary folder; the job runs `exit 3`.
"""

import pathlib
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import jobs  # noqa: E402
from rkmon_lib.store import Store  # noqa: E402


class TestAFolderWithASpace(unittest.TestCase):
    def test_the_script_runs_and_its_exit_code_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d) / "a state folder")
            job = jobs.start(store, "sanity", "x", "exit 3", timeout_min=5)
            for _ in range(100):
                if jobs.status(store, job) != "running":
                    break
                time.sleep(0.1)
            self.assertEqual(jobs.status(store, job), "failed(rc=3)")


if __name__ == "__main__":
    unittest.main()
