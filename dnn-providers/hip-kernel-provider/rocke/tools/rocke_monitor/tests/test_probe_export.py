# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rkmon_lib/probes.py export: a commit of a tree's clone exported into a folder (`git archive`, then tar) closes the
archive's pipes, so no ResourceWarning is left. A temporary git repository."""

import gc
import pathlib
import subprocess
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import probes  # noqa: E402


def git(repo, *args) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True,
        check=True,
    )


class TestExport(unittest.TestCase):
    def test_a_commit_is_exported_and_no_pipe_is_left_open(self):
        with tempfile.TemporaryDirectory() as d:
            tree, dest = pathlib.Path(d) / "tree", pathlib.Path(d) / "dest"
            tree.mkdir()
            git(tree, "init", "-q")
            (tree / "a.txt").write_text("a\n", encoding="utf-8")
            git(tree, "add", "a.txt")
            git(tree, "commit", "-q", "--no-gpg-sign", "-m", "a")
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always", ResourceWarning)
                self.assertEqual(probes.export(tree, "HEAD", dest, []), "")
                gc.collect()
            self.assertEqual((dest / "a.txt").read_text(encoding="utf-8"), "a\n")
        self.assertEqual([str(w.message) for w in seen], [])


if __name__ == "__main__":
    unittest.main()
