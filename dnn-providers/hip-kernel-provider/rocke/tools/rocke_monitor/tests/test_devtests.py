# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Merge-only coverage (lessons L-596): tools/devtests.py (develop's tests changed since the develop the PR's CI merged)
and the IMPORTERS_FAIL / DEVTESTS_FAIL lines, and the checks' section that runs develop's changed tests on the merge.
A temp git repository, no network."""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT))
import devtests  # noqa: E402
import importer_tests as it  # noqa: E402
from rkmon_lib import prcode, review  # noqa: E402

ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def git(repo, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ, **ENV),
    ).stdout.strip()


def write(root: pathlib.Path, files: dict) -> None:
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)


class Repo:
    """A: rocKE's module and an old test. The PR (H) changes the module; develop then adds a test that reads the old
    value (B, passes on develop, fails on the merge), changes a helper (C) and a test outside rocKE (D). M merges H
    into D."""

    def __init__(self, d: pathlib.Path):
        self.w = d / "W"
        self.w.mkdir()
        git(self.w, "init", "-q", "-b", "develop")
        write(
            self.w,
            {
                "rk/platform/python/rocke/__init__.py": "",
                "rk/platform/python/rocke/x.py": "X = 1\n",
                "rk/platform/tests/test_old.py": "def test_ok():\n    pass\n",
            },
        )
        git(self.w, "add", "-A")
        git(self.w, "commit", "-qm", "A")
        self.a = git(self.w, "rev-parse", "HEAD")
        git(self.w, "checkout", "-qb", "pr")
        write(self.w, {"rk/platform/python/rocke/x.py": "X = 2\n"})
        git(self.w, "commit", "-qam", "H")
        self.h = git(self.w, "rev-parse", "HEAD")
        git(self.w, "checkout", "-q", "develop")
        write(
            self.w,
            {
                "rk/platform/tests/test_new.py": "from rocke.x import X\n\n\ndef test_x():\n    assert X == 1\n"
            },
        )
        git(self.w, "add", "-A")
        git(self.w, "commit", "-qm", "B")
        self.b = git(self.w, "rev-parse", "HEAD")
        write(self.w, {"rk/platform/tests/helper.py": "Y = 1\n"})
        git(self.w, "add", "-A")
        git(self.w, "commit", "-qm", "C")
        write(self.w, {"other/test_elsewhere.py": "def test_e():\n    pass\n"})
        git(self.w, "add", "-A")
        git(self.w, "commit", "-qm", "D")
        self.d = git(self.w, "rev-parse", "HEAD")
        git(self.w, "checkout", "-q", "--detach", self.h)
        git(self.w, "merge", "-q", "--no-edit", self.d)


class TestSelect(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.r = Repo(pathlib.Path(self.t.name))

    def select(self, since, exclude=()):
        return devtests.select(
            str(self.r.w), "rk", self.r.w / "rk", since, self.r.d, set(exclude)
        )

    def test_develops_test_files_since_cis_merge_with_their_commit(self):
        sel = self.select(self.r.a)
        self.assertEqual(sel["chosen"], ["platform/tests/test_new.py"])
        self.assertEqual(sel["by_file"]["platform/tests/test_new.py"], self.r.b)
        self.assertEqual(sel["since"], self.r.a)
        lines = devtests.report(sel, devtests.CAP)
        self.assertIn(f"{self.r.a[:10]}..{self.r.d[:10]}", lines[0])
        self.assertIn(f"  platform/tests/test_new.py (develop {self.r.b[:10]})", lines)

    def test_none_when_ci_merged_a_develop_that_had_the_test_and_files_excluded(self):
        self.assertEqual(self.select(self.r.b)["chosen"], [])
        self.assertEqual(
            self.select(self.r.a, ["platform/tests/test_new.py"])["chosen"], []
        )

    def test_a_commit_off_develop_counts_from_its_merge_base(self):
        # The PR head is not on develop: count from its merge base with develop (A).
        self.assertEqual(self.select(self.r.h)["since"], self.r.a)

    def test_an_unreadable_commit_selects_nothing_and_says_so(self):
        sel = self.select("0" * 40)
        self.assertEqual(sel["chosen"], [])
        self.assertIn("could not be read", devtests.report(sel, 12)[0])

    def test_the_cap(self):
        for i in range(3):
            write(
                self.r.w,
                {f"rk/platform/tests/test_c{i}.py": "def test_c():\n    pass\n"},
            )
            git(self.r.w, "add", "-A")
            git(self.r.w, "commit", "-qm", f"c{i}")
        top = git(self.r.w, "rev-parse", "HEAD")
        sel = devtests.select(
            str(self.r.w), "rk", self.r.w / "rk", self.r.a, top, set(), cap=2
        )
        self.assertEqual(
            sel["chosen"], ["platform/tests/test_c2.py", "platform/tests/test_c1.py"]
        )
        self.assertEqual(
            sel["rest"], ["platform/tests/test_c0.py", "platform/tests/test_new.py"]
        )
        self.assertTrue(
            devtests.report(sel, 2)[-1].startswith("NOT RUN (over the cap, 2)")
        )


class TestFailLines(unittest.TestCase):
    def test_devtests_and_importers_fail_lines(self):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "tests_pr_failed.txt"
            f.write_text(
                "platform/tests/test_a.py::test_x\nplatform/tests/test_a.py::test_y\n"
                "RUN-ERROR(rc=4):platform/tests/test_b.py\n"
            )
            self.assertEqual(
                it.failed_files(f),
                ["platform/tests/test_a.py", "platform/tests/test_b.py"],
            )
            self.assertEqual(
                devtests.fail_lines(f, {"platform/tests/test_a.py": "c" * 40}),
                [
                    f"DEVTESTS_FAIL platform/tests/test_a.py (develop {'c' * 10})",
                    "DEVTESTS_FAIL platform/tests/test_b.py (develop unknown)",
                ],
            )
            self.assertEqual(
                it.fail_lines(
                    f,
                    {
                        "platform/tests/test_a.py": [
                            "platform/python/rocke/core/foo.py",
                            "library/kernels/common/conv.py",
                        ]
                    },
                ),
                [
                    "IMPORTERS_FAIL platform/tests/test_a.py (rocke.core.foo, kernels.common.conv)",
                    "IMPORTERS_FAIL platform/tests/test_b.py (a changed module)",
                ],
            )
            self.assertEqual(it.failed_files(pathlib.Path(d) / "none.txt"), [])

    def test_the_importer_cli_writes_the_map_and_prints_the_fail_lines(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            write(
                d / "rk",
                {
                    "platform/python/rocke/core/foo.py": "X = 1\n",
                    "platform/tests/test_a.py": "from rocke.core.foo import X\n",
                },
            )
            run = lambda *a: subprocess.run(
                [sys.executable, "-I", str(TOOLS / "importer_tests.py"), *a],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            run(
                "--tree",
                str(d / "rk"),
                "--list",
                str(d / "l.txt"),
                "--map",
                str(d / "m.json"),
                "--",
                "platform/python/rocke/core/foo.py",
            )
            self.assertEqual(
                json.loads((d / "m.json").read_text()),
                {"platform/tests/test_a.py": ["platform/python/rocke/core/foo.py"]},
            )
            (d / "f.txt").write_text("platform/tests/test_a.py::test_x\n")
            self.assertEqual(
                run("--fails", str(d / "f.txt"), "--map", str(d / "m.json")),
                "IMPORTERS_FAIL platform/tests/test_a.py (rocke.core.foo)\n",
            )
            self.assertEqual(
                run("--fails", str(d / "none.txt"), "--map", str(d / "m.json")), ""
            )


class TestChecksWiring(unittest.TestCase):
    CFG = {
        "rocke_path": "rk",
        "repo": "o/r",
        "paths": {"working_dir": "/w"},
        "hot_spots": {"RG": "rk/g.json", "DG": "rk/d.json"},
    }

    def script(self, files):
        plan = review.plan_checks(self.CFG, {"n": 7}, files, {})
        plan["_dirs"] = {}
        return review.checks_script(self.CFG, 7, "a" * 40, "first-round", plan, None)

    def section(self, script):
        return script.split(f"sec '{review.DEVTESTS_SECTION}", 1)[1].split("\nsec ", 1)[
            0
        ]

    def test_every_merge_runs_develops_changed_tests_after_the_importers(self):
        script = self.script(
            ["rk/platform/python/rocke/core/foo.py", "rk/platform/tests/test_foo.py"]
        )
        sec = self.section(script)
        self.assertIn(
            '--exclude-list "$OUT/importers/files.txt" --exclude platform/tests/test_foo.py',
            sec,
        )
        self.assertIn("pr/meta.md", sec)  # CI's develop: the PR's base.sha
        self.assertIn("merge-base", sec)  # else the merge base
        self.assertIn('devtests.py --fails "$OUT/devtests/tests_pr_failed.txt"', sec)
        self.assertIn(
            'importer_tests.py --fails "$OUT/importers/tests_pr_failed.txt"', script
        )
        self.assertIn("st devtests ", sec)
        self.assertIn("devtests", review.LEAD_SECTIONS)
        self.assertLess(
            script.index(review.IMPORTERS_SECTION),
            script.index(review.DEVTESTS_SECTION),
        )
        self.assertGreater(
            script.index(review.DEVTESTS_SECTION),
            script.index('if [ "$MERGED" = 1 ]; then'),
        )
        # A docs-only PR has the section too: develop's tests meet the PR's merge whatever it changes.
        self.assertIn(review.DEVTESTS_SECTION, self.script(["rk/docs/x.md"]))
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True
            ).returncode,
            0,
        )

    def test_the_section_runs_in_bash_and_names_the_develop_commit(self):
        if subprocess.run(
            [sys.executable, "-c", "import pytest"], capture_output=True
        ).returncode:
            self.skipTest("the section runs pytest")
        sec = (
            self.section(self.script(["rk/platform/python/rocke/x.py"]))
            .split("\n", 1)[1]
            .rstrip()
        )
        if sec.endswith("\nelse"):
            sec = sec[: -len("\nelse")]
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            r = Repo(d)
            out = d / "out"
            out.mkdir()
            (out / "head_sha").write_text(r.h + "\n")
            (out / "develop_sha").write_text(r.d + "\n")
            # The slot commands as the checks have them, in a slot of the session user's (a host's).
            pre = prcode.SHELL + textwrap.dedent(
                f"""
                D={d}; K=rk; N=7; WTN=W; W=$D/W; P=$W/$K/platform; L=$W/$K/library; M={ROOT}; WK=$D/wk
                OUT=$D/out; : > $OUT/status; BW=$D/none; BR=''
                TMPDIR=$D; prslot || exit 9
                st() {{ echo "$1 $2 $3" >> $OUT/status; }}
                base_tree() {{ return 1; }}
                {review.RESULT_FN}
                {review.U8_FN}
                """
            )
            shim = d / "bin"
            shim.mkdir()
            (shim / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
            (shim / "python").chmod(0o755)
            p = subprocess.run(
                ["bash", "-c", pre + sec],
                capture_output=True,
                text=True,
                timeout=300,
                env={
                    "PATH": f"{shim}:{pathlib.Path(sys.executable).parent}:/usr/bin:/bin"
                },
            )
            log = p.stdout + p.stderr
            self.assertIn(
                f"CI's develop: the merge base (no base.sha staged) {r.a[:10]}",
                p.stdout,
                log,
            )
            self.assertIn(
                f"DEVTESTS_FAIL platform/tests/test_new.py (develop {r.b[:10]})",
                p.stdout,
                log,
            )
            self.assertTrue(
                (out / "status").read_text().startswith("devtests fail"), log
            )
            # With the PR's base.sha staged at B (CI merged a develop that had the test), nothing is selected.
            (d / "wk/pr7/pr").mkdir(parents=True)
            (d / "wk/pr7/pr/meta.md").write_text(
                f"- Head: x @ {r.h[:10]}; base: develop @ {r.b[:10]}\n"
            )
            (out / "status").write_text("")
            p = subprocess.run(
                ["bash", "-c", pre + sec],
                capture_output=True,
                text=True,
                timeout=300,
                env={
                    "PATH": f"{shim}:{pathlib.Path(sys.executable).parent}:/usr/bin:/bin"
                },
            )
            self.assertIn(
                f"CI's develop: the PR's base commit (base.sha) {r.b[:10]}",
                p.stdout,
                p.stderr,
            )
            self.assertTrue(
                (out / "status").read_text().startswith("devtests skip no test to run")
            )


if __name__ == "__main__":
    unittest.main()
