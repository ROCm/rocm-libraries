# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/mutation_sweep.py: the checks a PR's lines add, one mutant each that
disables it, the order and cap, the test-id reading, the report, the monitor's step that appends a finished sweep to
its PR's history, and a sweep end to end on a small git repo whose tests catch two parts of a guard and miss the third
(as #11111's `evidence` half was missed). Temp git repos, no network."""

import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "mutation_sweep.py"
sys.path.insert(0, str(TOOL.parent))
import mutation_sweep as ms  # noqa: E402

ALL = [(1, 10**6)]

SRC = textwrap.dedent(
    """\
    import sys


    def check(x, y, args):
        if x is None or y < 0:
            raise ValueError("bad")
        if not x:
            return False
        if gap := probe(x):
            return gap
        if args.flag:
            parser.error("no")
        if x == 1:
            return True
        assert y != 3, "y"
        raise RuntimeError("always")


    if __name__ == "__main__":
        raise SystemExit(check(1, 2, None))
    """
)


def g(repo, *a) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class TestChecks(unittest.TestCase):
    def test_guards_raises_and_asserts_get_one_mutant_each(self):
        got = [
            (c["line"], c["kind"], c["what"], c["repl"])
            for c in ms.python_checks("m.py", SRC, ALL)
        ]
        self.assertEqual(
            got,
            [
                (5, "guard", "x is None", "False"),
                (5, "guard", "y < 0", "False"),
                (7, "guard", "not x", "False"),
                (9, "guard", "gap := probe(x)", "False"),
                (11, "guard", "args.flag", "False"),
                (15, "assert", 'assert y != 3, "y"', "pass"),
                (16, "raise", 'raise RuntimeError("always")', "pass"),
            ],
        )
        # `if x == 1: return True` is no check; the guard's own raise is its guard's; `__main__` runs a script.
        for c in ms.python_checks("m.py", SRC, ALL):
            mutant = ms.mutate(SRC, c)
            self.assertIsNotNone(mutant)
            compile(mutant, "m.py", "exec")
        first = ms.mutate(SRC, ms.python_checks("m.py", SRC, ALL)[0])
        self.assertIn("    if False or y < 0:\n", first)

    def test_a_refusal_helper_counts_as_a_raise(self):
        """A gate conftest that refuses through its own `_stop()` (pytest.exit), in guards and excepts, raises."""
        src = textwrap.dedent(
            """\
            def _stop(msg):
                pytest.exit(msg, returncode=4)


            def gate(env):
                if env.get("A") != "1":
                    _stop("needs A")
                try:
                    probe()
                except Exception as exc:
                    _stop(f"probe failed: {exc}")
                sys.exit(0)
            """
        )
        self.assertEqual(
            [
                (c["line"], c["kind"], c["repl"])
                for c in ms.python_checks("g.py", src, ALL)
            ],
            [(6, "guard", "False"), (11, "refusal", "pass")],
        )

    def test_only_checks_on_added_lines_and_new_ones_marked(self):
        """A check the merge base's file already had (moved or touched by the PR) is a changed one."""
        got = ms.python_checks(
            "m.py", SRC, [(7, 8), (15, 15)], "    if not x:\n        return False\n"
        )
        self.assertEqual([(c["line"], c["new"]) for c in got], [(7, False), (15, True)])
        self.assertEqual(ms.python_checks("m.py", "def broken(:\n", ALL), [])

    def test_a_mutation_counts_utf8_columns(self):
        src = 'def f(s):\n    if s == "é" or s == "ü":\n        raise ValueError(s)\n'
        mutants = [ms.mutate(src, c) for c in ms.python_checks("m.py", src, ALL)]
        self.assertEqual(
            mutants,
            [
                'def f(s):\n    if False or s == "ü":\n        raise ValueError(s)\n',
                'def f(s):\n    if s == "é" or False:\n        raise ValueError(s)\n',
            ],
        )

    def test_pick_takes_new_checks_first_asserts_last_and_files_in_turn(self):
        checks = [
            {"path": p, "line": i, "new": new, "kind": kind}
            for p, i, new, kind in [
                ("a.py", 1, True, "guard"),
                ("a.py", 2, True, "assert"),
                ("a.py", 3, True, "raise"),
                ("b.py", 1, True, "guard"),
                ("a.py", 9, False, "guard"),
                ("b.py", 9, False, "guard"),
            ]
        ]
        self.assertEqual(
            [(c["path"], c["line"]) for c in ms.pick(checks, 5)],
            [("a.py", 1), ("b.py", 1), ("a.py", 3), ("a.py", 2), ("a.py", 9)],
        )

    def test_a_slow_sets_mutants_run_after_the_others(self):
        mutants = [{"set": s} for s in "ABCBA"]
        queue = ms.defer_set(ms.collections.deque([1, 2, 3, 4]), mutants, "A", 0)
        self.assertEqual(list(queue), [1, 2, 3, 0, 4])

    def test_cpp_checks_are_listed(self):
        src = (
            'void f(bool ok, int* p) {\n    if (!ok) return false;\n    throw std::runtime_error("x");\n'
            "    assert(p);\n    return false;\n    // throw nothing\n    if (!p)\n        return -1;\n}\n"
        )
        self.assertEqual(
            [c["line"] for c in ms.cpp_checks("x.cpp", src, ALL)], [2, 3, 4, 8]
        )

    def test_a_validators_false_reason_answer_and_a_refusal_macro_are_checks(self):
        """`if spec.pipeline is None: return False, "…"` in Python and `ROCKE_SPEC_FAIL(…)` under an `if` in C++ are
        checks: a follow-up sweep never reads "no new checks" for them."""
        py = textwrap.dedent(
            """\
            def is_valid_spec(spec):
                if spec.pipeline is None:
                    return False, "pipeline must be set (got None)"
                if spec.epilogue is None:
                    return (None, f"epilogue must be set (got {spec.epilogue})")
                if spec.ok:
                    return True, ""
                if spec.zero:
                    return 0, "zero is no error word"
                return True, ""
            """
        )
        got = [(c["line"], c["what"]) for c in ms.python_checks("v.py", py, ALL)]
        self.assertEqual(
            got, [(2, "spec.pipeline is None"), (4, "spec.epilogue is None")]
        )
        cpp = (
            "bool v(const spec_t* s, char* reason, int cap)\n{\n    if(s->pipeline == NULL)\n    {\n"
            '        ROCKE_SPEC_FAIL(reason, cap, "pipeline must be set (got None)");\n    }\n'
            "    ROCKE_SPEC_FAIL_LATER(reason);\n    return true;\n}\n"
            '#define ROCKE_SPEC_FAIL(r, c, m) do { snprintf(r, c, "%s", m); return false; } while(0)\n'
        )
        self.assertEqual([c["line"] for c in ms.cpp_checks("v.cpp", cpp, ALL)], [5])

    def test_a_process_left_running_in_the_copy_is_killed(self):
        """A worker that a timed-out mutant's test starts in a session of its own does not outlive the run's process
        group in the removed copy."""
        with tempfile.TemporaryDirectory() as d:
            scratch = pathlib.Path(d) / "rocke"
            scratch.mkdir()
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import time; time.sleep(120)",
                    str(scratch / "marker"),
                ],
                cwd=scratch,
                start_new_session=True,
            )
            try:
                self.assertGreaterEqual(ms.kill_strays(scratch), 1)
                self.assertEqual(p.wait(timeout=10), -9)
            finally:
                if p.poll() is None:
                    p.kill()
            self.assertEqual(ms.kill_strays(scratch), 0)

    def test_failed_ids_keep_spaces_and_drop_messages(self):
        out = (
            "FAILED a/test_x.py::test_y[a b] - AssertionError: boom\nERROR a/test_z.py - ImportError: no\n"
            "1 failed, 1 error in 0.1s\n"
        )
        self.assertEqual(
            ms.failed_ids(out), ["a/test_x.py::test_y[a b]", "a/test_z.py"]
        )


class TestReport(unittest.TestCase):
    def test_survivors_first_and_nothing_run_never_reads_clean(self):
        m = lambda line, state, **kw: dict(
            {
                "path": "p/m.py",
                "line": line,
                "kind": "guard",
                "what": "x is None",
                "repl": "False",
                "new": True,
                "state": state,
                "set": "A",
                "secs": 1.0,
                "why": "",
            },
            **kw,
        )
        res = {
            "checks": 3,
            "new_checks": 3,
            "cpp": [{"path": "c.cpp", "line": 4, "what": "throw x;"}],
            "secs": 9,
            "copy": "git archive of the head",
            "sets": {
                "A": {
                    "files": ["t/test_m.py"],
                    "deselect": [],
                    "clean": "3 passed",
                    "secs": 1.0,
                    "dropped": [],
                    "why": None,
                }
            },
            "mutants": [
                m(5, "killed"),
                m(7, "survived"),
                m(9, "NOT RUN", why="the time budget ran out"),
            ],
        }
        lines = ms.render(res, "rk", 40)
        text = "\n".join(lines)
        self.assertIn(
            "mutants: 3: 1 survived, 1 killed, 0 timeout, 1 NOT RUN", lines[0]
        )
        self.assertIn(
            "tests [A] (clean run 3 passed in 1.0 s): PYTHONPATH=platform/python:library python -B -m "
            "pytest -x -q -o addopts= t/test_m.py",
            text,
        )
        self.assertLess(
            text.index("SURVIVED: p/m.py:7 `x is None` -> `False` [A] (new)"),
            text.index("NOT RUN: p/m.py:9"),
        )
        self.assertIn("killed (1): m.py:5 [A] 1.0 s", text)
        self.assertIn("C/C++ (listed): c.cpp:4 `throw x;`", text)
        self.assertEqual(
            lines[-1],
            "MUTANTS fail 1 of 2 mutants survive (checks no test catches): m.py:7; 1 NOT RUN",
        )
        self.assertEqual(ms.verdict({"mutants": [], "checks": 0, "cpp": []})[0], "skip")
        self.assertEqual(
            ms.verdict({"mutants": [m(1, "NOT RUN", why="x")], "checks": 1, "cpp": []})[
                0
            ],
            "skip",
        )
        self.assertEqual(
            ms.verdict(
                {"mutants": [m(1, "killed"), m(2, "timeout")], "checks": 2, "cpp": []}
            ),
            ("pass", "all 2 mutants killed"),
        )


class TestChecksWiring(unittest.TestCase):
    PLAN = {
        "families": [],
        "tests": [],
        "build": False,
        "golden": False,
        "engine_wide": False,
        "outside_rocke": [],
        "libparity": False,
        "_dirs": {},
    }

    def test_every_review_runs_the_sweep_at_the_head_and_a_gate_does_not(self):
        from rkmon_lib import review

        cfg = {
            "rocke_path": "rk",
            "repo": "o/r",
            "paths": {"working_dir": "/w"},
        }
        script = review.checks_script(cfg, 7, "a" * 40, "first-round", self.PLAN, None)
        sec = script.split(f'sec "{review.MUTANTS_SECTION}"', 1)[1].split("\nsec ", 1)[
            0
        ]
        self.assertIn(
            f"nice -n 10 timeout {review.MUTANTS_BUDGET_S + review.MUTANTS_RERUN_S + 300} python -I -B $M/tools/mutation_sweep.py "
            f"--repo $R0 --head {'a' * 40} --out $SO/mutants --tree \"$D/wt$N\"",
            sec,
        )  # a first round
        self.assertIn(f"--budget {review.MUTANTS_BUDGET_S}", sec)
        self.assertIn("st mutants ", sec)
        self.assertIn(
            "MUTANTS skip no result", sec
        )  # a sweep that wrote nothing never reads clean
        # outside the merged-tree block: it runs at the head, merge or no merge
        self.assertGreater(
            script.index(review.MUTANTS_SECTION),
            script.rindex('\nif [ "$MERGED" = 1 ]; then\n'),
        )
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True
            ).returncode,
            0,
        )
        gate = review.checks_script(
            cfg, 7, "a" * 40, "first-round", self.PLAN, None, mode="gate"
        )
        self.assertNotIn("mutation_sweep.py", gate)

    def test_a_follow_up_sweeps_its_push_and_a_merge_only_round_none(self):
        from rkmon_lib import review

        cfg = {
            "rocke_path": "rk",
            "repo": "o/r",
            "paths": {"working_dir": "/w"},
        }
        fu = review.checks_script(
            cfg, 7, "a" * 40, "follow-up", self.PLAN, None, reviewed="b" * 40
        )
        self.assertIn(f"--head {'a' * 40} --since {'b' * 40} --out $SO/mutants", fu)
        mo = review.checks_script(
            cfg,
            7,
            "a" * 40,
            "follow-up",
            self.PLAN,
            None,
            reviewed="b" * 40,
            merge_only=True,
        )
        sec = mo.split(f'sec "{review.MUTANTS_SECTION}"', 1)[1].split("\nsec ", 1)[0]
        # No sweep: the earlier rounds' items still at the head, from the PR's sweep history (git and the history
        # only: the monitor runs it, no PR code).
        self.assertIn(
            f"timeout 120 python -I -B $M/tools/mutation_sweep.py --repo $R0 --head {'a' * 40} --since {'b' * 40} "
            '--out "$OUT/mutants" --rocke-path $K --base $BREF --history $CA/mutants --pr $N --carry-only',
            sec,
        )
        self.assertNotIn("prc ", sec)
        self.assertNotIn("--budget", sec)
        self.assertIn(" --history $CA/mutants --pr $N", fu)
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=mo, capture_output=True, text=True
            ).returncode,
            0,
        )

    def test_the_append_step_after_a_sweep_adds_its_mutants_to_the_history(self):
        from rkmon_lib import review

        cfg = {
            "rocke_path": "rk",
            "repo": "o/r",
            "paths": {"working_dir": "/w"},
        }
        # The monitor's step, never PR code: the sweep's results copied back, then appended to the history, which
        # nothing PR code runs writes.
        step = (
            'prr mutants "$OUT/mutants"; timeout 60 python -I -B $M/tools/mutation_sweep.py append --out '
            f"\"$OUT/mutants\" --history $CA/mutants --pr $N --head {'a' * 40} 2>&1 | head -5"
        )
        for kind, reviewed in (("first-round", None), ("follow-up", "b" * 40)):
            with self.subTest(kind=kind):
                script = review.checks_script(
                    cfg, 7, "a" * 40, kind, self.PLAN, None, reviewed=reviewed
                )
                sec = script.split(f'sec "{review.MUTANTS_SECTION}"', 1)[1].split(
                    "\nsec ", 1
                )[0]
                # once the sweep has ended, before its report and its verdict are read: what it refuses opens the
                # section
                self.assertLess(sec.index("mutation_sweep.py --repo"), sec.index(step))
                self.assertLess(sec.index(step), sec.index("grep -v '^MUTANTS '"))
                self.assertEqual(sec.count("mutation_sweep.py append"), 1)
        mo = review.checks_script(
            cfg,
            7,
            "a" * 40,
            "follow-up",
            self.PLAN,
            None,
            reviewed="b" * 40,
            merge_only=True,
        )
        self.assertNotIn(
            "mutation_sweep.py append", mo
        )  # a merge-only round sweeps nothing


class TestHistoryAppend(unittest.TestCase):
    """The monitor's step after a sweep: the finished sweep's mutants.json, read as data, appended to its PR's
    history (`append`)."""

    HEAD = "c" * 40

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.d = pathlib.Path(d.name)
        self.out = self.d / "out"
        self.out.mkdir()
        self.hist = self.d / "history"

    @staticmethod
    def mutant(**kw) -> dict:
        m = {
            "path": "platform/python/pkg/mod.py",
            "line": 3,
            "kind": "guard",
            "what": "x == 42",
            "repl": "False",
            "state": "survived",
            "why": "",
            "secs": 1.5,
            "tail": "3 passed",
        }
        return {**m, **kw}

    def write(self, res) -> None:
        (self.out / "mutants.json").write_text(json.dumps(res))

    def append(self, hist: pathlib.Path | None = None) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = ms.main(
                [
                    "append",
                    "--out",
                    str(self.out),
                    "--history",
                    str(hist or self.hist),
                    "--pr",
                    "7",
                    "--head",
                    self.HEAD,
                ]
            )
        return rc, err.getvalue()

    def records(self) -> list[dict]:
        f = self.hist / "pr7.jsonl"
        return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []

    def test_this_heads_sweep_is_appended_to_the_named_prs_history(self):
        killed = self.mutant(state="killed", line=None, why=None)
        self.write({"head": self.HEAD, "pr": 9, "mutants": [self.mutant(), killed]})
        self.assertEqual(self.append(), (0, ""))
        (rec,) = self.records()
        self.assertEqual(rec["head"], self.HEAD)
        self.assertEqual(
            rec["items"],
            [{k: m[k] for k in ms.HISTORY_FIELDS} for m in (self.mutant(), killed)],
        )  # the history's fields, nothing else of the sweep's
        self.assertEqual(
            [p.name for p in self.hist.iterdir()], ["pr7.jsonl"]
        )  # the PR the monitor names, whatever the file says
        self.write({"head": self.HEAD, "mutants": []})
        self.assertEqual(self.append(), (0, ""))
        self.assertEqual(len(self.records()), 1)  # nothing swept: nothing recorded

    def test_what_is_no_sweep_of_this_head_is_never_appended(self):
        for res in (
            {"head": "d" * 40, "mutants": [self.mutant()]},  # an earlier round's file
            {"mutants": [self.mutant()]},
            [self.mutant()],
            {"head": self.HEAD, "mutants": {"0": self.mutant()}},
            "{not json",
            "[" * 200000 + "]" * 200000,  # nested deeper than the parser goes
        ):
            with self.subTest(res=res):
                if isinstance(res, str):
                    (self.out / "mutants.json").write_text(res)
                else:
                    self.write(res)
                rc, err = self.append()
                self.assertEqual(rc, 2)
                self.assertTrue(
                    err.startswith("mutation history: nothing appended: "), err
                )
                self.assertFalse(self.hist.exists())
        (self.out / "mutants.json").unlink()  # a sweep that timed out writes none
        rc, err = self.append()
        self.assertEqual(rc, 2)
        self.assertIn("mutants.json", err)
        self.assertFalse(self.hist.exists())

    def test_a_mutant_of_another_shape_is_left_out(self):
        good = self.mutant()
        odd = [
            self.mutant(line="3"),
            self.mutant(line=True),
            self.mutant(what=None),
            self.mutant(repl=["False"]),
            self.mutant(state="fixed"),
            self.mutant(why=7),
            "x == 42",
        ]
        self.write({"head": self.HEAD, "mutants": [good, *odd]})
        self.assertEqual(
            self.append(),
            (1, "mutation history: 7 of 8 mutants left out (of another shape)\n"),
        )
        self.assertEqual(
            [r["items"] for r in self.records()],
            [[{k: good[k] for k in ms.HISTORY_FIELDS}]],
        )
        self.write({"head": self.HEAD, "mutants": odd})
        self.assertEqual(self.append()[0], 2)
        self.assertEqual(len(self.records()), 1)

    def test_no_link_is_followed_to_read_the_sweep_or_to_write_the_history(self):
        sweep = json.dumps({"head": self.HEAD, "mutants": [self.mutant()]})
        elsewhere = self.d / "elsewhere.json"
        elsewhere.write_text(sweep)
        (self.out / "mutants.json").symlink_to(elsewhere)
        rc, err = self.append()
        self.assertEqual(rc, 2)
        self.assertIn("a link", err)
        self.assertFalse(self.hist.exists())
        (self.out / "mutants.json").unlink()
        self.write(json.loads(sweep))
        # The history's file, or its folder, a link to a file or a folder of the monitor's: never written through.
        mine = self.d / "mine"
        mine.mkdir()
        (mine / "pr7.jsonl").write_text("kept\n")
        hist = self.d / "h1"
        hist.mkdir()
        (hist / "pr7.jsonl").symlink_to(mine / "pr7.jsonl")
        self.assertEqual(self.append(hist)[0], 2)
        (self.d / "h2").symlink_to(mine)
        self.assertEqual(self.append(self.d / "h2")[0], 2)
        self.assertEqual((mine / "pr7.jsonl").read_text(), "kept\n")
        # A FIFO with a reader (a process left running): no regular file, never written.
        fifo = self.d / "h3" / "pr7.jsonl"
        fifo.parent.mkdir()
        os.mkfifo(fifo)
        r = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        self.addCleanup(os.close, r)
        rc, err = self.append(fifo.parent)
        self.assertEqual(rc, 2)
        self.assertIn("not a regular file", err)
        self.assertEqual(os.read(r, 100), b"")  # nothing written to it


@unittest.skipUnless(
    subprocess.run(
        [sys.executable, "-c", "import pytest"], capture_output=True
    ).returncode
    == 0,
    "the sweep runs pytest",
)
class TestSweep(unittest.TestCase):
    """A repo (rocKE tree `rk`) whose PR adds a guard of three parts to pkg/mod.py and tests that catch two of them:
    the third survives, as #11111's `evidence` half did."""

    MOD = textwrap.dedent(
        """\
        def f(x):
            if x < 0 or x > 100 or x == 42:
                raise ValueError("out of range")
            return x
        """
    )
    TEST = textwrap.dedent(
        """\
        import pytest

        from pkg.mod import f


        def test_negative():
            with pytest.raises(ValueError):
                f(-1)


        def test_big():
            with pytest.raises(ValueError):
                f(101)


        def test_ok():
            assert f(5) == 5
        """
    )

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        r = self.repo = pathlib.Path(self.d.name) / "repo"
        (r / "rk/platform/python/pkg").mkdir(parents=True)
        (r / "rk/platform/tests").mkdir(parents=True)
        g(r, "init", "-q", "-b", "develop")
        (r / "rk/platform/python/pkg/__init__.py").write_text("")
        (r / "rk/platform/python/pkg/mod.py").write_text("def f(x):\n    return x\n")
        (r / "rk/platform/tests/test_mod.py").write_text(
            "def test_nothing():\n    pass\n"
        )
        g(r, "add", "-A")
        g(r, "commit", "-qm", "base")
        g(r, "checkout", "-qb", "pr")
        (r / "rk/platform/python/pkg/mod.py").write_text(self.MOD)
        (r / "rk/platform/tests/test_mod.py").write_text(self.TEST)
        g(r, "add", "-A")
        g(r, "commit", "-qm", "pr")
        self.head = g(r, "rev-parse", "HEAD")
        self.tmp = pathlib.Path(self.d.name) / "tmp"
        self.tmp.mkdir()

    def sweep(self, tree=None, since=None) -> dict:
        out = pathlib.Path(self.d.name) / "out"
        with mock.patch.dict(os.environ, {"TMPDIR": str(self.tmp)}):
            res = ms.sweep(
                str(self.repo),
                self.head,
                out,
                K="rk",
                base="develop",
                tree=tree,
                python=sys.executable,
                per_mutant=120,
                budget=600,
                since=since,
            )
        self.assertIsNone(res["error"])
        self.assertEqual(list(self.tmp.iterdir()), [])  # the scratch copy is gone
        self.assertEqual(
            g(self.repo, "status", "--porcelain"), ""
        )  # the repo's own tree untouched
        return res

    def test_the_part_no_test_reaches_survives(self):
        res = self.sweep()
        self.assertEqual(res["copy"], "git archive of the head")
        self.assertEqual(
            [(m["what"], m["state"]) for m in res["mutants"]],
            [("x < 0", "killed"), ("x > 100", "killed"), ("x == 42", "survived")],
        )
        self.assertEqual(res["sets"]["A"]["files"], ["platform/tests/test_mod.py"])
        self.assertTrue(res["restored"])
        self.assertEqual(ms.verdict(res)[0], "fail")
        out = pathlib.Path(self.d.name) / "out"
        self.assertTrue((out / "baseline_A.log").exists())
        # From the review's head worktree when it holds the head: the same answer.
        again = self.sweep(tree=str(self.repo))
        self.assertTrue(again["copy"].startswith("a copy of "))
        self.assertEqual(
            [m["state"] for m in again["mutants"]], ["killed", "killed", "survived"]
        )

    def push(self, files: dict, msg: str) -> str:
        for path, text in files.items():
            (self.repo / path).write_text(text)
        g(self.repo, "commit", "-qam", msg)
        return g(self.repo, "rev-parse", "HEAD")

    def test_a_follow_up_takes_only_the_checks_its_push_adds_or_changes(self):
        reviewed = self.head
        # The push changes the guard's error exit (its condition's line is the same) and adds a new guard.
        self.head = self.push(
            {
                "rk/platform/python/pkg/mod.py": self.MOD.replace(
                    '"out of range"', '"bad x"'
                )
                + "\n\ndef g(y):\n    if y is None:\n        raise TypeError('y')\n    return y\n"
            },
            "fu",
        )
        res = self.sweep(since=reviewed)
        self.assertEqual(
            res["scope"], f"the checks the push since {reviewed[:10]} adds or changes"
        )
        self.assertEqual(
            [m["what"] for m in res["mutants"]],
            ["x < 0", "x > 100", "x == 42", "y is None"],
        )
        # A push that changes no check: nothing runs, and the section says so (not NOT RUN).
        fixed = self.head
        self.head = self.push(
            {"rk/platform/tests/test_mod.py": self.TEST + "\n# one more comment\n"},
            "fu2",
        )
        res = self.sweep(since=fixed)
        self.assertTrue(ms.nothing_since(res))
        self.assertEqual(res["mutants"], [])
        lines = ms.render(res, "rk", ms.CAP)
        self.assertTrue(
            lines[0].startswith(
                f"no new checks since {fixed[:10]} (the push since our last review adds or "
                f"changes none of the PR's 4 Python checks"
            )
        )
        self.assertEqual(lines[-1], f"MUTANTS skip no new checks since {fixed[:10]}")
        self.assertNotIn("NOT RUN", "\n".join(lines))
        # Our reviewed head gone from the mirror: every check of the PR, saying why.
        res = self.sweep(since="f" * 40)
        self.assertTrue(
            res["scope"].startswith(
                "every check of the PR: our reviewed head ffffffffff is not in"
            )
        )
        self.assertEqual(len(res["mutants"]), 4)

    def test_a_follow_up_finds_the_refusals_its_push_adds_in_both_languages(self):
        """A push that adds `return False, "…"` refusals and ROCKE_SPEC_FAIL refusals has new checks in both
        languages, never "no new checks since" the last review."""
        (self.repo / "rk/platform/cpp").mkdir(parents=True)
        (self.repo / "rk/platform/cpp/v.cpp").write_text(
            "bool v(const spec_t* s)\n{\n    return true;\n}\n"
        )
        g(self.repo, "add", "-A")
        g(self.repo, "commit", "-qm", "cpp")
        reviewed = g(self.repo, "rev-parse", "HEAD")
        self.head = self.push(
            {
                "rk/platform/python/pkg/mod.py": self.MOD
                + textwrap.dedent(
                    """\


                def is_valid(spec):
                    if spec.get("pipeline") is None:
                        return False, "pipeline must be set (got None)"
                    return True, ""
                """
                ),
                "rk/platform/cpp/v.cpp": (
                    "bool v(const spec_t* s)\n{\n    if(s->pipeline == NULL)\n    {\n"
                    '        ROCKE_SPEC_FAIL(reason, cap, "pipeline must be set (got None)");\n'
                    "    }\n    return true;\n}\n"
                ),
            },
            "refuse a missing pipeline",
        )
        res = self.sweep(since=reviewed)
        self.assertEqual(
            [m["what"] for m in res["mutants"]], ['spec.get("pipeline") is None']
        )
        self.assertEqual(
            [(c["path"], c["line"]) for c in res["cpp"]], [("platform/cpp/v.cpp", 5)]
        )
        self.assertFalse(ms.nothing_since(res))
        # C/C++ refusals alone: no Python mutant, and the section names them instead of "no new checks".
        reviewed = self.head
        self.head = self.push(
            {
                "rk/platform/cpp/v.cpp": (self.repo / "rk/platform/cpp/v.cpp")
                .read_text()
                .replace(
                    "    return true;",
                    '    if(s->epilogue == NULL)\n        ROCKE_SPEC_FAIL(reason, cap, "epilogue");\n'
                    "    return true;",
                )
            },
            "refuse a missing epilogue",
        )
        res = self.sweep(since=reviewed)
        self.assertTrue(ms.nothing_since(res))
        lines = ms.render(res, "rk", ms.CAP)
        self.assertTrue(
            lines[0].startswith(
                f"no new Python checks since {reviewed[:10]}; the push since our last "
                "review adds or changes 1 C/C++ checks"
            ),
            lines,
        )
        self.assertIn(
            '  C/C++ (listed): platform/cpp/v.cpp:8 `ROCKE_SPEC_FAIL(reason, cap, "epilogue");`',
            lines,
        )
        self.assertEqual(
            lines[-1],
            f"MUTANTS skip no new Python checks since {reviewed[:10]}; 1 new C/C++ checks "
            "listed",
        )

    def cli(self, *extra: str, since: str | None = None) -> list[str]:
        p = subprocess.run(
            [
                sys.executable,
                "-I",
                str(TOOL),
                "--repo",
                str(self.repo),
                "--head",
                self.head,
                "--out",
                str(pathlib.Path(self.d.name) / "cli"),
                "--rocke-path",
                "rk",
                "--base",
                "develop",
                "--python",
                sys.executable,
                *(["--since", since] if since else []),
                *extra,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "TMPDIR": str(self.tmp)},
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout.splitlines()

    def append(self, hist: pathlib.Path) -> None:
        """The monitor's step after a sweep (`append`), as the checks run it: the cli's sweep added to PR 7's
        history."""
        p = subprocess.run(
            [
                sys.executable,
                "-I",
                str(TOOL),
                "append",
                "--out",
                str(pathlib.Path(self.d.name) / "cli"),
                "--history",
                str(hist),
                "--pr",
                "7",
                "--head",
                self.head,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual((p.returncode, p.stdout, p.stderr), (0, "", ""))

    def test_earlier_rounds_survivors_still_at_the_head_are_carried_forward(self):
        """The lane mutated every check by hand on follow-ups whose section read only "no new checks since …"
        (agent_script_overlap.md, fix 6). The sweep reads the history; the monitor's append step writes it.
        """
        hist = pathlib.Path(self.d.name) / "history"
        first = self.head
        out = self.cli("--history", str(hist), "--pr", "7")
        self.assertTrue(out[-1].startswith("MUTANTS fail 1 of 3 mutants survive"), out)
        self.assertFalse(hist.exists())  # the sweep runs PR code: it writes no history
        self.append(hist)
        recs = [json.loads(x) for x in (hist / "pr7.jsonl").read_text().splitlines()]
        self.assertEqual(
            [(r["head"], [i["state"] for i in r["items"]]) for r in recs],
            [(first, ["killed", "killed", "survived"])],
        )
        # A follow-up whose push adds no check: the earlier survivor, at its line at this head.
        self.head = self.push(
            {"rk/platform/python/pkg/mod.py": "# a comment\n" + self.MOD}, "fu1"
        )
        out = self.cli("--history", str(hist), "--pr", "7", since=first)
        self.assertIn(
            "earlier rounds' items still at this head (1; from their sweeps, not re-run here):",
            out,
        )
        self.assertIn(
            f"  SURVIVED (sweep at {first[:10]}): platform/python/pkg/mod.py:3 `x == 42` -> `False`",
            out,
        )
        self.assertEqual(
            out[-1],
            f"MUTANTS skip no new checks since {first[:10]}; 1 earlier survivor still at this head",
        )
        self.append(hist)
        self.assertEqual(
            len((hist / "pr7.jsonl").read_text().splitlines()), 1
        )  # nothing swept: nothing recorded
        # A merge-only round runs no sweep and lists them too.
        out = self.cli("--history", str(hist), "--pr", "7", "--carry-only", since=first)
        self.assertEqual(
            out[0],
            f"no new checks since {first[:10]} (merge-only since our review: the sweep does not "
            "run)",
        )
        self.assertEqual(
            out[-1],
            f"MUTANTS skip no new checks since {first[:10]} (merge-only); 1 earlier survivor still "
            "at this head",
        )
        # The push that drops the check: it is no longer carried.
        fu1 = self.head
        self.head = self.push(
            {"rk/platform/python/pkg/mod.py": self.MOD.replace(" or x == 42", "")},
            "fu2",
        )
        out = self.cli("--history", str(hist), "--pr", "7", since=fu1)
        self.assertIn(
            "(1 earlier survivors or NOT RUN items are not at this head)", out
        )
        self.assertFalse(any(x.startswith("  SURVIVED (sweep at") for x in out), out)

    def test_the_cli_writes_the_report_and_mutants_json(self):
        out = pathlib.Path(self.d.name) / "cli"
        p = subprocess.run(
            [
                sys.executable,
                "-I",
                str(TOOL),
                "--repo",
                str(self.repo),
                "--head",
                self.head,
                "--out",
                str(out),
                "--rocke-path",
                "rk",
                "--base",
                "develop",
                "--python",
                sys.executable,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "TMPDIR": str(self.tmp)},
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(
            p.stdout.splitlines()[-1].startswith("MUTANTS fail 1 of 3 mutants survive")
        )
        self.assertEqual(
            len(json.loads((out / "mutants.json").read_text())["mutants"]), 3
        )


if __name__ == "__main__":
    unittest.main()
