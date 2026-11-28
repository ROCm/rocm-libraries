# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/since_review.py: a follow-up's "Since our last review" section, base-aware: a force-pushed re-cut reads as
rebase-only or not at a glance (agent_script_overlap.md, fix 5: range-diff and the real delta by hand in 33 of 39
reviews)."""

import pathlib
import subprocess
import sys
import tempfile
import unittest

Y = "".join(f"y{i} = {i}\n" for i in range(12))
TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "since_review.py"


def g(repo, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class TestSinceReview(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        r = self.repo = pathlib.Path(self.d.name)
        g(r, "init", "-q", "-b", "develop")
        (r / "rk").mkdir()
        (r / "rk/a.py").write_text("a = 1\n")
        (r / "rk/b.py").write_text("b = 1\n")
        g(r, "add", "-A")
        g(r, "commit", "-qm", "base")
        g(r, "checkout", "-qb", "pr")
        self.commit({"rk/a.py": "a = 1\nx = 2\n"}, "pr: add x")
        self.old = self.commit({"rk/b.py": "b = 1\n" + Y}, "pr: add y")
        g(r, "checkout", "-q", "develop")
        self.commit({"rk/c.py": "c = 1\n"}, "develop moves on")
        g(r, "checkout", "-q", "pr")

    def commit(self, files: dict, msg: str) -> str:
        for path, text in files.items():
            (self.repo / path).write_text(text)
        g(self.repo, "add", "-A")
        g(self.repo, "commit", "-qm", msg)
        return g(self.repo, "rev-parse", "HEAD")

    def run_tool(self, new: str) -> list[str]:
        p = subprocess.run(
            [
                sys.executable,
                "-I",
                str(TOOL),
                "--repo",
                str(self.repo),
                "--old",
                self.old,
                "--new",
                new,
                "--rocke-path",
                "rk",
                "--base",
                "develop",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout.splitlines()

    def test_a_rebase_that_keeps_every_patch_reads_rebase_only(self):
        g(self.repo, "rebase", "-q", "develop")
        out = self.run_tool(g(self.repo, "rev-parse", "HEAD"))
        self.assertIn(
            "commits (range-diff, merges aside): 2: 2 same patch; merge commits: 0 then, 0 now",
            out,
        )
        self.assertIn(
            "the PR's own files: 2, 0 with a different patch (develop's changes under them aside)",
            out,
        )
        self.assertEqual(
            out[-1],
            "SINCE pass same patches: 2 commits the same, none changed, new or dropped, and no "
            "PR file's patch changed (a rebase only)",
        )

    def test_a_merge_of_develop_reads_merge_only(self):
        g(self.repo, "merge", "-q", "--no-edit", "develop")
        out = self.run_tool(g(self.repo, "rev-parse", "HEAD"))
        self.assertIn(
            "commits (range-diff, merges aside): 2: 2 same patch; merge commits: 0 then, 1 now",
            out,
        )
        self.assertTrue(out[-1].endswith("(a merge only)"), out[-1])

    def test_a_re_cut_says_which_commits_and_files_changed(self):
        g(self.repo, "rebase", "-q", "develop")
        g(self.repo, "reset", "-q", "--hard", "HEAD~1")
        self.commit(
            {"rk/b.py": "b = 1\n" + Y.replace("y5 = 5", "y5 = 50")}, "pr: add y"
        )  # one value changed
        new = self.commit({"rk/d.py": "d = 1\n"}, "pr: add d")
        out = self.run_tool(new)
        self.assertIn(
            "commits (range-diff, merges aside): 3: 1 same patch, 1 changed, 1 new; merge commits: 0 then, 0 "
            "now",
            out,
        )
        self.assertTrue(
            any(x.startswith("   changed: ") and x.endswith(" pr: add y") for x in out),
            out,
        )
        self.assertIn("   b.py: +12 -0 then, +12 -0 now (2 patch lines differ)", out)
        self.assertIn("   d.py: +0 -0 then, +1 -0 now (newly changed by the PR)", out)
        self.assertEqual(
            out[-1],
            "SINCE pass changed: 1 commit changed, 1 commit new; 2 PR files with a different "
            "patch: b.py, d.py",
        )

    def test_a_head_the_mirror_lacks_is_not_run(self):
        self.old = "f" * 40
        out = self.run_tool(g(self.repo, "rev-parse", "HEAD"))
        self.assertEqual(
            out,
            [
                "NOT RUN: our reviewed head ffffffffff is not in the mirror",
                "SINCE skip not run: our reviewed head not in the mirror",
            ],
        )


class TestWiring(unittest.TestCase):
    def test_a_follow_ups_checks_have_the_section_and_it_flags_no_merge_only_head(self):
        sys.path.insert(0, str(TOOL.parents[1]))
        from rkmon_lib import review

        cfg = {
            "rocke_path": "rk",
            "repo": "o/r",
            "paths": {"working_dir": "/w"},
        }
        plan = review.plan_checks(cfg, {}, ["rk/library/x.py"], {})
        plan["_dirs"] = {}
        fu = review.checks_script(
            cfg, 7, "a" * 40, "follow-up", plan, None, reviewed="b" * 40
        )
        self.assertIn(f"sec '{review.SINCE_SECTION}'", fu)
        self.assertIn(
            f"since_review.py --repo $R0 --old {'b' * 40} --new {'a' * 40} --rocke-path $K",
            fu,
        )
        self.assertLess(
            fu.index(review.SINCE_SECTION), fu.index(review.MUTANTS_SECTION)
        )
        fr = review.checks_script(cfg, 7, "a" * 40, "first-round", plan, None)
        self.assertNotIn(review.SINCE_SECTION, fr)
        self.assertIn("since", review.LEAD_SECTIONS)


if __name__ == "__main__":
    unittest.main()
