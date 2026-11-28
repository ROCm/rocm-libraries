# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The several-deployments stage's folder reads hook (tests/fakes/folder_reads), offline: in a Python process started
with the hook's folder first on PYTHONPATH, as the stage starts each rkmon command, it logs the process's start, then
each file opened and each folder listed in a watched folder, and none elsewhere; with no folder to watch it logs
nothing.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
HOOK = ROOT / "tests" / "fakes" / "folder_reads"
sys.dont_write_bytecode = True

# The process the hook watches: it opens each file argv[1:3] names, then lists each folder argv[3:5] names, by both
# of the calls that list one.
READS = (
    "import os, sys\n"
    "for p in sys.argv[1:3]:\n"
    "    open(p).close()\n"
    "for p in sys.argv[3:5]:\n"
    "    os.listdir(p)\n"
    "    list(os.scandir(p))\n"
)


class TestTheFolderReadsHook(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        top = pathlib.Path(os.path.realpath(tmp.name))
        self.watched, self.mine = top / "other", top / "mine"
        for folder in (self.watched, self.mine):
            folder.mkdir()
            (folder / "f.txt").write_text("x\n", encoding="utf-8")
        self.log = top / "reads.log"

    def reads(self, watch: str) -> list:
        """The hook's log of a process that reads a file and lists the folder, in the watched folder and in another,
        WATCH the folders it watches: its lines, read as JSON; [] when there is no log.
        """
        env = dict(
            os.environ,
            PYTHONPATH=str(HOOK),
            PYTHONDONTWRITEBYTECODE="1",
            FOLDER_READS_WATCH=watch,
            FOLDER_READS_LOG=str(self.log),
        )
        argv = [self.watched / "f.txt", self.mine / "f.txt", self.watched, self.mine]
        subprocess.run(
            [sys.executable, "-c", READS, *map(str, argv)],
            env=env,
            check=True,
            timeout=60,
        )
        if not self.log.exists():
            return []
        return [
            json.loads(x) for x in self.log.read_text(encoding="utf-8").splitlines()
        ]

    def test_it_logs_the_start_then_the_reads_in_a_watched_folder_only(self):
        lines = self.reads(str(self.watched))
        self.assertIn(str(self.watched / "f.txt"), lines[0]["started"])
        self.assertEqual(
            [(x["event"], x["path"]) for x in lines[1:]],
            [
                ("open", str(self.watched / "f.txt")),
                ("os.listdir", str(self.watched)),
                ("os.scandir", str(self.watched)),
            ],
        )
        self.assertEqual({x["pid"] for x in lines}, {lines[0]["pid"]})

    def test_with_no_folder_to_watch_it_logs_nothing(self):
        self.assertEqual(self.reads(""), [])


if __name__ == "__main__":
    unittest.main()
