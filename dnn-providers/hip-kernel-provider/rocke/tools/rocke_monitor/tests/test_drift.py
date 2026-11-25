# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Develop drift: which PRs get their tests run on today's develop, how a
failure fails G6 for that head and tree only, shows before any approval and keeps the PR off the approval lists, how a
finished job is read, and the job script's shape."""

import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from rkmon_lib import drift, prcode, queue, report  # noqa: E402
from rkmon_lib.store import Store, empty_state, now  # noqa: E402
import home_repository  # noqa: E402

G = "\U0001f7e2"
# daily_dir: the daily-log file of a guarded write goes to a folder of this module's own.
_DAILY = tempfile.TemporaryDirectory()


def tearDownModule():
    _DAILY.cleanup()


CFG = {
    "rocke_path": "rk",
    "repo": "ROCm/r",
    "operator": {"github_login": "me"},
    "deployment": {"phase": "queue"},
    "paths": {
        "working_dir": "/nonexistent",
        "repo": "/nonexistent",
        "daily_dir": _DAILY.name,
    },
    "merge_authority": "arm-cleared",
    "max_posts_per_hour": 6,
    "coordination_notices": False,
    "review_first_line": "x",
    "codeowner_teams": {"platform": ["plat"], "library": ["lib"]},
}
DEV, OLD = "d" * 40, "c" * 40
TREE, OLD_TREE = "e" * 40, "f" * 40


def head(n: int) -> str:
    return (str(n) * 40)[:40]


def rec(n: int, **kw) -> dict:
    r = {
        "n": n,
        "head": head(n),
        "size": 100,
        "status": "open",
        "draft": False,
        "author": "a",
        "title": f"t{n}",
        "labels": [],
        "base": "develop",
        "head_ref": f"b{n}",
        "mergeable": "MERGEABLE",
        "ci_rollup": "SUCCESS",
        "review_decision": "REVIEW_REQUIRED",
        "reviews": [],
        "files_cache": {
            "head": head(n),
            "hot": [],
            "areas": ["library"],
            "files": [f"rk/library/tests/t{n}.py"],
        },
        "ours": {
            "verdict": G,
            "head": head(n),
            "at": now(),
            "url": "https://github.com/ROCm/r/pull/1#c",
        },
        "threads": {"updated_at": None, "open": []},
        # tests last ran at review time, on an older develop
        "gates": {
            "head": head(n),
            "develop": OLD,
            "develop_tree": OLD_TREE,
            "results": {"G5": ["pass", ""], "G6": ["pass", ""], "G7": ["pass", ""]},
        },
    }
    r.update(kw)
    return r


def row(n: int, plan_rank=None, g1="pass", g4="pass", g8="pass") -> dict:
    return {
        "n": n,
        "plan_rank": plan_rank,
        "tier": 2,
        "size": 100,
        "gates": {"G1": [g1, ""], "G4": [g4, ""], "G8": [g8, ""]},
    }


def state(*recs) -> dict:
    st = empty_state()
    st["develop"]["sha"], st["develop"]["rocke_tree"] = DEV, TREE
    st["prs"] = {str(r["n"]): r for r in recs}
    return st


class TestTargets(unittest.TestCase):
    def test_clean_prs_whose_tests_ran_on_an_older_rocke_tree_in_plan_order(self):
        """#1111: our review was clean, it waited for approval, and its tests last ran on a develop before #1234."""
        fresh = rec(
            4,
            gates={
                "head": head(4),
                "develop": OLD,
                "develop_tree": TREE,
                "results": {},
            },
        )  # same tree
        st = state(rec(1), rec(2), rec(3), fresh, rec(5), rec(6))
        rows = [
            row(1, plan_rank=5),
            row(2, plan_rank=1),
            row(3, g1="fail"),
            row(4),
            row(5, g4="fail"),
            row(6, g8="pending"),
        ]
        self.assertEqual([r["n"] for r in drift.targets(CFG, st, rows)], [2, 1])

    def test_a_drift_run_on_this_tree_or_a_new_head_decides(self):
        done = rec(
            1, drift={"head": head(1), "develop": OLD, "tree": TREE, "status": "pass"}
        )
        moved = rec(
            2, drift={"head": "9" * 40, "develop": DEV, "tree": TREE, "status": "pass"}
        )  # an older head's run
        st = state(done, moved)
        self.assertEqual(
            [r["n"] for r in drift.targets(CFG, st, [row(1), row(2)])], [2]
        )

    def test_a_frozen_queue_does_not_stop_the_check(self):
        """G8 fails every PR while develop is red, and the drift check still starts on the first tick with it."""
        st = state(rec(1), rec(2))
        frozen = row(1)
        frozen["gates"]["G8"] = ["fail", "queue frozen: develop red (INC-2026-01-01-9)"]
        stacked = row(2)
        stacked["gates"]["G8"] = ["fail", "stacked on #5"]
        self.assertEqual(
            [r["n"] for r in drift.targets(CFG, st, [frozen, stacked])], [1]
        )

    def test_drafts_and_closed_prs_are_skipped(self):
        st = state(rec(1, draft=True), rec(2, status="closed"))
        self.assertEqual(drift.targets(CFG, st, [row(1), row(2)]), [])


class TestGateAndBlockers(unittest.TestCase):
    def failing(self, **kw) -> dict:
        d = {
            "head": head(1),
            "develop": DEV,
            "tree": TREE,
            "status": "fail",
            "detail": "3 failing (on the merge only): rk/library/tests/t1.py::test_x",
        }
        d.update(kw)
        return rec(1, drift=d)

    def test_a_failure_fails_g6_for_its_head_and_tree_only(self):
        g = queue.gates_dynamic(self.failing(), DEV, TREE)
        self.assertEqual(g["G6"][0], queue.FAIL)
        self.assertIn("tests fail on today's develop", g["G6"][1])
        # develop moved on, but its rocKE tree did not: still failing
        self.assertEqual(
            queue.gates_dynamic(self.failing(develop=OLD), DEV, TREE)["G6"][0],
            queue.FAIL,
        )
        # a new develop with a new rocKE tree, or a new head: the old failure no longer counts
        self.assertNotEqual(
            queue.gates_dynamic(self.failing(), "b" * 40, "a" * 40)["G6"][0], queue.FAIL
        )
        self.assertNotEqual(
            queue.gates_dynamic(self.failing(head="9" * 40), DEV, TREE)["G6"][0],
            queue.FAIL,
        )
        self.assertNotEqual(
            queue.gates_dynamic(self.failing(status="pass"), DEV, TREE)["G6"][0],
            queue.FAIL,
        )

    def test_it_shows_before_any_approval_and_keeps_the_pr_off_the_approval_lists(self):
        st = state(self.failing())
        st["codeowners"] = {"teams": {"plat": ["me"], "lib": ["y"]}}
        blockers = report.merge_blockers(CFG, st, st["prs"]["1"])
        self.assertTrue(
            any(
                b["kind"] == "analysis_fail"
                and "its tests fail on today's develop" in b["what"]
                for b in blockers
            ),
            blockers,
        )
        self.assertIsNone(report._approvable(CFG, st, st["prs"]["1"]))
        ok = state(rec(1))
        ok["codeowners"] = st["codeowners"]
        self.assertFalse(
            any(
                "today's develop" in b["what"]
                for b in report.merge_blockers(CFG, ok, ok["prs"]["1"])
            )
        )


class TestCollect(unittest.TestCase):
    def test_results_are_recorded_and_failures_name_the_likely_cause(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d) / "state")
            j = pathlib.Path(d) / "job"
            (j / "pr1").mkdir(parents=True)
            (j / "pr2").mkdir()
            (j / "pr3").mkdir()
            (j / "pr1" / "result").write_text(
                "TESTS fail 3 failing (on the merge only): rk/library/tests/t1.py::a\n"
            )
            (j / "pr1" / "tests").mkdir()
            (j / "pr1" / "tests" / "tests_failed.txt").write_text(
                "library/tests/t1.py::a\nlibrary/tests/t1.py::b\n"
            )
            (j / "pr2" / "result").write_text("TESTS pass 120 passed\n")
            (j / "pr3" / "result").write_text("SKIP no test file to run\n")
            st = state(rec(1), rec(2), rec(3))
            st["drift_job"] = {
                "job": {"id": "x"},
                "dir": str(j),
                "develop": DEV,
                "tree": TREE,
                "items": [
                    {"n": 1, "head": head(1)},
                    {"n": 2, "head": head(2)},
                    {"n": 3, "head": head(3)},
                ],
            }
            with mock.patch.object(
                drift.jobs, "status", return_value="done"
            ), mock.patch.object(drift, "suspects", return_value=[1234]):
                lines = drift.collect(CFG, st, store)
            self.assertIsNone(st["drift_job"])
            d1 = st["prs"]["1"]["drift"]
            self.assertEqual(
                (d1["status"], d1["since"], d1["suspects"]), ("fail", OLD, [1234])
            )
            self.assertEqual(
                d1["failed"], ["library/tests/t1.py::a", "library/tests/t1.py::b"]
            )
            self.assertEqual(st["prs"]["2"]["drift"]["status"], "pass")
            self.assertEqual(st["prs"]["3"]["drift"]["status"], "skip")
            fail = next(x for x in lines if x.startswith("DRIFT_FAIL #1"))
            self.assertIn(f"tests last ran on develop {OLD[:10]}", fail)
            self.assertIn("#1234", fail)
            self.assertIn(f"`rk notice 1 FILE --kind drift --key {DEV[:10]}`", fail)
            self.assertTrue(
                any(x.startswith("DRIFT ok") and "#2" in x for x in lines), lines
            )
            # the failure now fails G6 and blocks before approval
            self.assertEqual(drift.fresh_fail(st, st["prs"]["1"]), d1["detail"])

    def test_a_running_job_is_left_alone(self):
        st = state(rec(1))
        st["drift_job"] = {
            "job": {"id": "x"},
            "dir": "/nonexistent",
            "develop": DEV,
            "items": [{"n": 1, "head": head(1)}],
        }
        with mock.patch.object(drift.jobs, "status", return_value="running"):
            self.assertEqual(drift.collect(CFG, st, None), [])
        self.assertIsNotNone(st["drift_job"])


class TestScript(unittest.TestCase):
    def test_each_pr_runs_its_tests_on_its_merge_with_develop_and_on_develop(self):
        s = drift.script(
            CFG,
            "/j",
            DEV,
            [
                {"n": 1, "head": head(1), "tests": ["library/tests/t1.py"]},
                {"n": 2, "head": head(2), "tests": []},
            ],
        )
        self.assertIn(
            'xt $DEV "$TD/dev" || { echo \'develop tree not extracted\' > "$J/error"; exit 1; }',
            s,
        )
        # The trees are a tree PR code reads; the job's folder is the monitor's.
        self.assertIn('TD="$D/wtdrift"', s)
        self.assertIn('rm -rf "$TD"; prtree "$TD" && mkdir "$TD/dev"', s)
        self.assertNotIn("prw", s)
        self.assertIn(f"merge-tree --write-tree $DEV {head(1)}", s)
        self.assertIn('xt $T "$TD/mrg" && {', s)
        self.assertIn(f"xt {head(1)} \"$TD/htree\" || HR=''", s)
        # Each PR's tests in a slot of their own, their results copied back into the job's folder, which the run is
        # told (its re-run commands name their environment file there).
        self.assertIn(
            "(prslot 1 || exit 3; RKMON_HEAD_ROOT=$HR RKMON_HEAD_PYTHONPATH=$HR/platform/python:$HR/library "
            f'{prcode.RESULTS_VAR}="$O/tests" prc ',
            s,
        )
        self.assertIn(
            'python3 -I -B $M/tools/pytest_vs_base.py $SO/tests "$TD/mrg/$K" "$TD/dev/$K"',
            s,
        )
        self.assertIn('> "$O/report.txt" 2>&1; prr tests "$O/tests"; prfree) || ', s)
        self.assertIn("library/tests/t1.py", s)
        self.assertIn("CONFLICT merge with develop", s)
        self.assertIn("SKIP no test file to run", s)
        self.assertTrue(s.rstrip().endswith('echo done > "$J/done"'))

    def test_the_job_s_folders_are_quoted_wherever_the_script_names_them(self):
        """The job's folder (J) and each PR's (O) are quoted wherever the script names them: a folder with a space in
        its path holds what the job writes there, and nothing is made at its first word.
        """
        with tempfile.TemporaryDirectory() as d:
            j = pathlib.Path(d) / "a b" / "job"
            s = drift.script(
                CFG,
                str(j),
                DEV,
                [{"n": 1, "head": head(1), "tests": ["library/tests/t1.py"]}],
            )
            lines = s.splitlines()
            self.assertEqual(
                [line for line in lines if re.search(r'(?<!["/])\$[JO]\b', line)], []
            )
            setup = next(line for line in lines if line.startswith("K="))
            pr = next(line for line in lines if line.startswith("O="))
            r = subprocess.run(
                ["bash", "-c", f'{setup}; mkdir -p "$J"; {pr}'],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual((j / "pr1" / "head").read_text().strip(), head(1))
            self.assertFalse((pathlib.Path(d) / "a").exists())

    def test_the_trees_and_tmp_folders_are_quoted_wherever_the_script_names_them(self):
        """The trees' folder (TD, the working folder's wtdrift), the tmp folder (TP) and the folder xt extracts into are
        quoted wherever the script names them: under a working folder with a space in its path, the job removes and
        makes its trees there, and nothing at the path's first word."""
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d)
            (
                base / "a"
            ).mkdir()  # the working folder's first word: what an unquoted `rm -rf $TD` would remove
            (base / "a" / "kept").write_text("x\n")
            cfg = dict(CFG, paths=dict(CFG["paths"], working_dir=str(base / "a b")))
            s = drift.script(
                cfg,
                str(base / "a b" / "job"),
                DEV,
                [{"n": 1, "head": head(1), "tests": ["library/tests/t1.py"]}],
            )
            lines = s.splitlines()
            self.assertEqual(
                [ln for ln in lines if re.search(r'(?<!["/:])\$(TD|TP|D)\b', ln)], []
            )
            self.assertIn('tar -x -C "$2" && [ -d "$2/$K" ]; }', s)

            def line(start):
                return next(ln for ln in lines if ln.startswith(start))

            # The opening lines as they run, with prtree a plain folder (prcode.SHELL's makes the kind PR code reads).
            r = subprocess.run(
                [
                    "bash",
                    "-c",
                    "\n".join(
                        [lines[0], 'prtree() { mkdir -p "$1"; }']
                        + [line("K="), line("mkdir -p ")]
                    ),
                ],
                cwd=d,
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue((base / "a b" / "wtdrift" / "dev").is_dir())
            self.assertTrue((base / "a b" / "tmp").is_dir())
            self.assertEqual((base / "a" / "kept").read_text(), "x\n")
            self.assertEqual(sorted(p.name for p in base.iterdir()), ["a", "a b"])

    def test_a_test_file_the_merge_lacks_is_not_passed_to_pytest(self):
        """A renamed test file that stops pytest (rc 4) is no drift failure."""
        import subprocess, tempfile

        s = drift.script(
            CFG,
            "/j",
            DEV,
            [{"n": 1, "head": head(1), "tests": ["a/kept.py", "a/gone.py"]}],
        )
        self.assertIn('[ -e "$TD/mrg/$K/$t" ]', s)
        self.assertIn("SKIP no test file in the merge", s)
        line = next(l for l in s.splitlines() if "TS=''" in l)
        loop = line[line.index("TS=''") : line.index("; if [ -z")]
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "mrg/k/a").mkdir(parents=True)
            (pathlib.Path(d) / "mrg/k/a/kept.py").write_text("")
            out = subprocess.run(
                ["bash", "-c", f"TD={d}; K=k; {loop}; echo $TS"],
                capture_output=True,
                text=True,
            )
        self.assertEqual(out.stdout.split(), ["a/kept.py"])

    def test_parse(self):
        self.assertEqual(
            drift._parse("TESTS fail 3 failing (on the merge only): x"),
            ("fail", "3 failing (on the merge only): x"),
        )
        self.assertEqual(drift._parse("TESTS pass 5 passed"), ("pass", "5 passed"))
        self.assertEqual(
            drift._parse("CONFLICT merge with develop"),
            ("conflict", "merge with develop"),
        )
        self.assertEqual(drift._parse(""), ("error", "no result"))

    def test_extract_paths(self):
        # The home repository's drift_extract; none configured: the rocKE tree alone.
        self.assertEqual(
            drift.extract_paths(dict(CFG, rocke_path="a/prov/rocke")),
            (["a/prov/rocke"], []),
        )
        whole, files_of = drift.extract_paths(
            home_repository.config(dict(CFG, rocke_path="a/prov/rocke"))
        )
        self.assertEqual(
            whole,
            [
                "a/prov/rocke",
                "a/prov/src",
                "shared/ctest",
                "projects/hipdnn/tools/IngestorGenerator/configs",
            ],
        )
        self.assertEqual(
            files_of, ["", "a/prov"]
        )  # the repository's top-level files, the provider's own
        whole, files_of = drift.extract_paths(
            dict(CFG, rocke_path="a/prov/rocke", drift_extract=["x/y", "x/*"])
        )
        self.assertEqual((whole, files_of), (["a/prov/rocke", "x/y"], ["x"]))


# The #11111 shape on a real repository: tests the PR adds (absent on develop) that need the repository around the
# rocKE tree: its root marker, the provider's src/ tree, and a folder no extraction holds.
NEW_TESTS = """from pathlib import Path

HERE = Path(__file__).resolve()


def root():
    for p in HERE.parents:
        if (p / "projects" / "hipdnn").is_dir():
            return p
    return None


def test_needs_the_repository_root():
    assert root() is not None


def test_needs_the_provider_src_tree():
    assert (HERE.parents[2] / "src" / "engines" / "packs" / "h.hpp").is_file()


def test_needs_a_folder_no_extraction_holds():
    assert root() is not None and (root() / "other" / "marker.txt").is_file()
"""
DRIFT_TEST = """import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import val  # noqa: E402


def test_value():
    assert val.VALUE == 1
"""


# The checks' venv, whose pytest runs the job's tests: $RKMON_VENV. A test reads no monitor's home, so never the
# configuration's paths.venv.
VENV = pathlib.Path(os.environ.get("RKMON_VENV") or "/nonexistent")


@unittest.skipUnless(
    (VENV / "bin" / "python").exists(), "needs the review venv's pytest"
)
class TestLikeWithLike(unittest.TestCase):
    """The job script on a real repository: develop, the merge and the head are the same extraction, and a failure on
    the head alone is never drift."""

    def test_a_test_the_extraction_cannot_serve(self):
        import subprocess

        def git(*a):
            return subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@t",
                    *a,
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        def commit(files, msg):
            for f, text in files.items():
                (repo / f).parent.mkdir(parents=True, exist_ok=True)
                (repo / f).write_text(text)
            git("add", "-A")
            git("commit", "-qm", msg)
            return git("rev-parse", "HEAD")

        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            repo = root / "repo"
            repo.mkdir()
            git("init", "-q")
            base = commit(
                {
                    "CMakeLists.txt": "project(top)\n",
                    "prov/CMakeLists.txt": "project(prov)\n",
                    "prov/rocke/lib/val.py": "VALUE = 1\n",
                    "prov/src/engines/packs/h.hpp": "// pack\n",
                    "projects/hipdnn/tools/IngestorGenerator/configs/t.json": "{}\n",
                    "projects/hipdnn/big/unrelated.txt": "x\n",
                    "other/marker.txt": "x\n",
                },
                "base",
            )
            pr = commit(
                {
                    "prov/rocke/tests/test_new.py": NEW_TESTS,
                    "prov/rocke/tests/test_val.py": DRIFT_TEST,
                },
                "pr",
            )
            git("checkout", "-q", "-b", "dev", base)
            dev = commit({"prov/rocke/lib/val.py": "VALUE = 2\n"}, "develop moves")
            cfg = home_repository.config(  # its drift_extract: the root marker, the provider's src/
                dict(
                    CFG,
                    rocke_path="prov/rocke",
                    paths={
                        "working_dir": str(root),
                        "repo": str(repo),
                        "venv": str(VENV),
                    },
                )
            )
            s = drift.script(
                cfg,
                str(root / "job"),
                dev,
                [
                    {"n": 1, "head": pr, "tests": ["tests/test_new.py"]},
                    {"n": 2, "head": pr, "tests": ["tests/test_val.py"]},
                ],
            )
            r = subprocess.run(
                ["bash", "-c", s], cwd=d, capture_output=True, text=True, timeout=600
            )
            j = root / "job"
            self.assertTrue((j / "done").exists(), r.stdout[-2000:] + r.stderr[-2000:])
            res1 = (j / "pr1" / "result").read_text().strip()
            res2 = (j / "pr2" / "result").read_text().strip()
            failed1 = (j / "pr1" / "tests" / "tests_failed.txt").read_text().split()
            head_failed1 = (
                (j / "pr1" / "tests" / "tests_head_failed.txt").read_text().split()
            )
            trees = root / "wtdrift"
            extracted = (
                sorted(str(p.relative_to(trees)) for p in trees.rglob("*"))
                if trees.exists()
                else []
            )
        # The extraction serves the root marker and the provider's src/: those tests pass on the merge.
        self.assertEqual(
            failed1,
            ["tests/test_new.py::test_needs_a_folder_no_extraction_holds"],
            res1,
        )
        # The one the extraction cannot serve fails on the head alone too: never drift.
        self.assertEqual(head_failed1, failed1)
        self.assertEqual(
            res1, "TESTS pass 2 passed, 1 failed; 1 fail on the head alone too"
        )
        self.assertEqual(drift._parse(res1)[0], "pass")
        # A test develop's change breaks on the merge still fails the drift check.
        self.assertEqual(
            res2,
            "TESTS fail 1 failing (on the merge only): tests/test_val.py::test_value",
        )
        self.assertEqual(extracted, [], "the trees are removed after the run")


class TestDismiss(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.d.name) / "state")
        self.st = state(
            rec(
                1,
                drift={
                    "head": head(1),
                    "develop": DEV,
                    "tree": TREE,
                    "status": "fail",
                    "detail": "2 failing (on the merge only): rk/x.py::a",
                    "failed": ["x.py::a"],
                },
            )
        )

    def tearDown(self):
        self.d.cleanup()

    def test_a_dismissed_failure_stops_failing_g6_and_blocking(self):
        r = self.st["prs"]["1"]
        self.assertEqual(queue.gates_dynamic(r, DEV, TREE)["G6"][0], queue.FAIL)
        out = drift.dismiss(
            self.st, self.store, 1, DEV[:10], "the test needs the repository root", "me"
        )
        self.assertIn("dismissed as a harness false positive", out)
        self.assertNotEqual(queue.gates_dynamic(r, DEV, TREE)["G6"][0], queue.FAIL)
        self.assertIsNone(drift.fresh_fail(self.st, r))
        rows = self.store.read_jsonl(drift.DISMISSED_FILE)
        self.assertEqual(
            [(x["pr"], x["head"], x["develop"], x["reason"], x["by"]) for x in rows],
            [(1, head(1), DEV, "the test needs the repository root", "me")],
        )
        self.assertIn(
            "already dismissed",
            drift.dismiss(self.st, self.store, 1, DEV[:10], "again", "me"),
        )
        self.assertEqual(len(self.store.read_jsonl(drift.DISMISSED_FILE)), 1)
        self.assertFalse(self.store.ledger())  # local: never an outward action

    def test_a_later_run_counts_again(self):
        drift.dismiss(self.st, self.store, 1, DEV[:10], "harness", "me")
        r = self.st["prs"]["1"]
        r["drift"] = {
            "head": head(1),
            "develop": "b" * 40,
            "tree": "a" * 40,
            "status": "fail",
            "detail": "x",
        }
        self.assertEqual(
            queue.gates_dynamic(r, "b" * 40, "a" * 40)["G6"][0], queue.FAIL
        )

    def test_refusals(self):
        for develop, reason, why in (
            (OLD[:10], "r", "has no drift run on develop"),
            (DEV[:4], "r", "has no drift run"),
            (DEV[:10], "  ", "--reason"),
        ):
            with self.assertRaises(ValueError) as cm:
                drift.dismiss(self.st, self.store, 1, develop, reason, "me")
            self.assertIn(why, str(cm.exception))
        self.st["prs"]["1"]["drift"]["status"] = "pass"
        with self.assertRaises(ValueError) as cm:
            drift.dismiss(self.st, self.store, 1, DEV[:10], "r", "me")
        self.assertIn("did not fail", str(cm.exception))
        with self.assertRaises(ValueError):
            drift.dismiss(self.st, self.store, 9, DEV[:10], "r", "me")
        self.assertEqual(self.store.read_jsonl(drift.DISMISSED_FILE), [])


if __name__ == "__main__":
    unittest.main()
