# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""git plumbing for the merge plan: two PRs' merges with develop merged on develop, blob ids read from
trees, blob texts in one batch, and an object git cannot read is no clean merge."""

import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import gitops  # noqa: E402


class TestMergeTree(unittest.TestCase):
    def test_merge_tree_on_develop_blob_ids_and_batch_reads(self):
        with tempfile.TemporaryDirectory() as d:

            def g(*a):
                return subprocess.run(
                    ["git", "-C", d, *a], check=True, capture_output=True, text=True
                ).stdout.strip()

            def commit(files: dict, msg: str) -> str:
                for path, text in files.items():
                    p = pathlib.Path(d, path)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(text)
                g("add", "-A")
                g("commit", "-qm", msg)
                return g("rev-parse", "HEAD")

            def branch(name: str, at: str, files: dict) -> str:
                g("checkout", "-q", "-b", name, at)
                sha = commit(files, name)
                g("checkout", "-q", "develop")
                return sha

            g("init", "-q", "-b", "develop")
            g("config", "user.email", "t@t")
            g("config", "user.name", "t")
            base = commit({"rk/a.py": "1\n2\n3\n", "rk/b.py": "b\n"}, "base")
            a = branch("a", base, {"rk/a.py": "1a\n2\n3\n"})
            b = branch("b", base, {"rk/a.py": "1b\n2\n3\n"})
            c = branch("c", base, {"rk/b.py": "c\n"})
            dev = commit({"rk/d.py": "d\n"}, "develop moves")
            ta, conf = gitops.merge_tree(d, dev, a)
            self.assertEqual(conf, [])
            tb, _ = gitops.merge_tree(d, dev, b)
            tc, _ = gitops.merge_tree(d, dev, c)
            self.assertEqual(
                gitops.merge_tree(d, ta, tb, base=dev)[1], ["rk/a.py"]
            )  # two trees, merged on develop
            self.assertEqual(gitops.merge_tree(d, ta, tc, base=dev)[1], [])
            self.assertEqual(
                gitops.merge_conflicts(d, a, b), ["rk/a.py"]
            )  # the head-to-head check, same answer
            self.assertEqual(gitops.merge_tree(d, dev, "f" * 40), (None, None))
            # git exits 1, the conflict code, with no output for an object it cannot read: that is no clean merge
            # (`merge_conflicts` must not give []).
            self.assertIsNone(gitops.merge_conflicts(d, dev, "nosuchref"))
            blobs = gitops.tree_blobs(d, dev, ["rk/a.py", "rk/none.py"])
            self.assertEqual(blobs, {"rk/a.py": g("rev-parse", f"{dev}:rk/a.py")})
            texts = gitops.cat_blobs(
                d, [f"{ta}:rk/a.py", f"{dev}:rk/none.py", f"{dev}:rk/b.py"]
            )
            self.assertEqual(
                texts,
                {
                    f"{ta}:rk/a.py": "1a\n2\n3\n",
                    f"{dev}:rk/none.py": None,
                    f"{dev}:rk/b.py": "b\n",
                },
            )


if __name__ == "__main__":
    unittest.main()
