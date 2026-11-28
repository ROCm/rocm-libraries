# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""worker/fetch_pr.py's fetch into the monitor's clone (fetch_locked): it holds the lock of the clone its command names,
and another job's lock file makes it wait and try again, up to six times; any other failure, or the last, is raised.
fetch_pr.py is a script that reads GitHub as it starts, so its function is compiled alone from the file; git is a
fake: nothing runs, and the clone is a folder of the test's own."""

import ast
import pathlib
import subprocess
import sys
import tempfile
import time
import types
import unittest

SRC = pathlib.Path(__file__).resolve().parent.parent / "worker" / "fetch_pr.py"
# fetch_locked takes the clone's lock from the worker's steps module.
sys.path.insert(0, str(SRC.parent))

import steps  # noqa: E402


def fetch_locked():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "fetch_locked"
    )
    mod = types.ModuleType("fetch_locked_only")
    mod.__dict__.update(
        subprocess=types.SimpleNamespace(
            run=None, CalledProcessError=subprocess.CalledProcessError
        ),
        sys=types.SimpleNamespace(
            stderr=types.SimpleNamespace(write=lambda text: None)
        ),
        time=time,
    )
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), str(SRC), "exec"), mod.__dict__
    )
    return mod


class TestFetchLocked(unittest.TestCase):
    def setUp(self):
        # The monitor's clone, as fetch_pr.py names it (-C): a folder of the test's own, with a git folder for the lock.
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.clone = pathlib.Path(d.name)
        (self.clone / ".git").mkdir()
        self.fetch = ["git", "-C", str(self.clone), "fetch"]

    def run_with(self, answers):
        mod, calls, waits = fetch_locked(), [], []

        def run(cmd, capture_output=True, text=True):
            calls.append(cmd)
            rc, err = answers.pop(0)
            return types.SimpleNamespace(returncode=rc, stdout="", stderr=err)

        mod.subprocess.run = run
        return mod, calls, waits

    def test_a_lock_waits_then_the_fetch_goes_through(self):
        lock = "fatal: Unable to create '/w/repo/.git/shallow.lock': File exists."
        mod, calls, waits = self.run_with([(128, lock), (128, lock), (0, "")])
        mod.fetch_locked(self.fetch, sleep=waits.append)
        self.assertEqual((len(calls), waits), (3, [5, 10]))
        # The lock it held is the clone's.
        self.assertTrue((self.clone / ".git" / steps.CLONE_LOCK).is_file())

    def test_another_failure_and_the_last_lock_are_raised(self):
        mod, calls, waits = self.run_with([(1, "fatal: couldn't find remote ref")])
        with self.assertRaises(subprocess.CalledProcessError):
            mod.fetch_locked(self.fetch, sleep=waits.append)
        self.assertEqual((len(calls), waits), (1, []))
        mod, calls, waits = self.run_with([(128, "x.lock exists")] * 6)
        with self.assertRaises(subprocess.CalledProcessError):
            mod.fetch_locked(self.fetch, sleep=waits.append)
        self.assertEqual((len(calls), waits), (6, [5, 10, 15, 20, 25]))


def gh_only():
    """fetch_pr.py's gh() and its rule, compiled alone; subprocess is a fake."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    keep = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name == "gh")
        or (
            isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") == "_PERMANENT" for t in n.targets)
        )
    ]
    import os
    import re

    mod = types.ModuleType("gh_only")
    mod.__dict__.update(
        subprocess=types.SimpleNamespace(run=None), sys=sys, time=time, os=os, re=re
    )
    exec(
        compile(ast.Module(body=keep, type_ignores=[]), str(SRC), "exec"), mod.__dict__
    )
    return mod


class TestGhRetry(unittest.TestCase):
    def run_with(self, answers):
        mod, calls, waits = gh_only(), [], []

        def run(cmd, capture_output=True, text=True):
            calls.append(cmd)
            rc, err = answers.pop(0)
            return types.SimpleNamespace(
                returncode=rc, stdout="ok" if rc == 0 else "", stderr=err
            )

        mod.subprocess.run = run
        return mod, calls, waits

    def test_a_transient_failure_is_tried_again(self):
        mod, calls, waits = self.run_with(
            [(1, "net/http: TLS handshake timeout"), (1, "HTTP 502"), (0, "")]
        )
        self.assertEqual(mod.gh("api", "x", sleep=waits.append), "ok")
        self.assertEqual((len(calls), waits), (3, [5, 15]))

    def test_a_permanent_refusal_and_the_third_failure_stop_it(self):
        mod, calls, waits = self.run_with([(1, "gh: Not Found (HTTP 404)")])
        with self.assertRaises(SystemExit):
            mod.gh("api", "x", sleep=waits.append)
        self.assertEqual((len(calls), waits), (1, []))
        mod, calls, waits = self.run_with([(1, "timeout")] * 3)
        with self.assertRaises(SystemExit):
            mod.gh("api", "x", sleep=waits.append)
        self.assertEqual((len(calls), waits), (3, [5, 15]))
        mod, calls, waits = self.run_with([(1, "API rate limit (HTTP 429)"), (0, "")])
        self.assertEqual(mod.gh("api", "x", sleep=waits.append), "ok")


class TestMeta(unittest.TestCase):
    def test_meta_names_the_operator_from_the_config(self):
        """The review book points the worker at meta.md's `Operator:` line: the operator's login, config `me`."""
        src = SRC.read_text(encoding="utf-8")
        self.assertIn('ME = rkenv.get("RKMON_OPERATOR_LOGIN")', src)
        self.assertIn('f"- Operator: @{ME}",', src)
        self.assertIn("OURS = {ME}", src)


if __name__ == "__main__":
    unittest.main()
