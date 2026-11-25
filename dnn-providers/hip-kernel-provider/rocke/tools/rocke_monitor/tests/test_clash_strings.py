# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The clash scan's extensions (lessons L-573, recipe R3): names spelled as strings (a CTest name one PR's CMake lines
drop, an attribute one PR removes that the other patches by name), the open PRs the checks' scan did not compare
(`rk clash N --new`, the premise line), and the pair merges built (`rk clash N --build`, in a PR runner's slot). Temp
git repos, no network."""

import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "fakes"))
sys.dont_write_bytecode = True

from rkmon_lib import clashes, paths, prcode, premises  # noqa: E402
from rkmon_lib import container_layout as L  # noqa: E402
import fake_slots  # noqa: E402


def g(repo, *a) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def write(repo: pathlib.Path, files: dict) -> None:
    for path, text in files.items():
        p = repo / path
        if text is None:
            p.unlink()
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))


def pr(n: int, head: str) -> dict:
    return {
        "n": n,
        "head": head,
        "base": None,
        "draft": False,
        "title": f"PR {n}",
        "author": "a",
        "stack": set(),
    }


class Branches:
    """develop, then N and M from it (`n_files`, `m_files`), and develop one commit on."""

    def make(self, base: dict, n_files: dict, m_files: dict) -> None:
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        r = self.repo = pathlib.Path(self.d.name) / "repo"
        r.mkdir()
        g(r, "init", "-q", "-b", "develop")
        write(r, base)
        g(r, "add", "-A")
        g(r, "commit", "-qm", "base")
        root = g(r, "rev-parse", "HEAD")
        heads = {}
        for name, files in (("n", n_files), ("m", m_files)):
            g(r, "checkout", "-q", root)
            g(r, "checkout", "-qb", name)
            write(r, files)
            g(r, "add", "-A")
            g(r, "commit", "-qm", name)
            heads[name] = g(r, "rev-parse", "HEAD")
        g(r, "checkout", "-q", "develop")
        write(r, {"rk/dev.txt": "develop moved on\n"})
        g(r, "add", "-A")
        g(r, "commit", "-qm", "develop moves")
        self.develop = g(r, "rev-parse", "HEAD")
        self.prs = {1: pr(1, heads["n"]), 2: pr(2, heads["m"])}


class TestParts(unittest.TestCase):
    def test_strings_on_added_lines_and_how_they_are_used(self):
        text = textwrap.dedent(
            """\
            from unittest import mock
            import mod


            def test_x(monkeypatch):
                with mock.patch.object(mod, "_tile_bias", return_value=1):
                    pass
                monkeypatch.setattr(mod, "_tuned", 2)
                with mock.patch("pkg.mod._other"):
                    pass
                assert entry == "rocke_pytest_widgets"
                assert getattr(mod, "_kept")
                assert "x" == "x"
            """
        )
        got = clashes.py_strings(text, [(1, 20)])
        self.assertEqual(
            sorted(got),
            [
                ("_kept", 12, "patch"),
                ("_other", 9, "patch"),
                ("_tile_bias", 6, "patch"),
                ("_tuned", 8, "patch"),
                ("rocke_pytest_widgets", 11, "string"),
            ],
        )
        self.assertEqual(
            clashes.py_strings(text, [(11, 11)]),
            [("rocke_pytest_widgets", 11, "string")],
        )

    def test_the_names_a_cmake_files_removed_lines_drop(self):
        old = "add_test(NAME rocke_pytest_widgets COMMAND python -m pytest)\nadd_test(NAME rocke_keep COMMAND x)\n"
        new = "add_test(NAME rocke_pytest_gadgets COMMAND python -m pytest)\nadd_test(NAME rocke_keep COMMAND x)\n"
        got = clashes.cmake_removed("rk/CMakeLists.txt", old, new, [(1, 1)])
        self.assertEqual(
            [x["name"] for x in got], ["rocke_pytest_widgets"]
        )  # COMMAND, python, pytest: kept words
        self.assertEqual(got[0]["kind"], "CMake name")


class TestStringClashes(Branches, unittest.TestCase):
    """N renames a CTest entry and removes a private helper; M's new test spells the old entry name and patches the
    helper by name. Decoys: a name another CMake file of develop still carries, and an attribute N keeps.
    """

    def setUp(self):
        self.make(
            {
                "rk/CMakeLists.txt": (
                    "add_test(NAME rocke_pytest_widgets COMMAND python -m pytest tests)\n"
                    "add_test(NAME rocke_pytest_dupes COMMAND python -m pytest dupes)\n"
                ),
                "rk/other.cmake": "set(_known rocke_pytest_dupes)\n",
                "rk/mod.py": "def _tile_bias():\n    return 1\n\n\ndef _kept():\n    return 2\n",
            },
            {
                "rk/CMakeLists.txt": "add_test(NAME rocke_pytest_gadgets COMMAND python -m pytest tests)\n",
                "rk/mod.py": "def _assist():\n    return 1\n\n\ndef _kept():\n    return 2\n",
            },
            {
                "rk/tests/test_names.py": """\
                    from unittest import mock
                    import mod


                    def test_names():
                        assert ENTRY == "rocke_pytest_widgets"
                        assert ENTRY != "rocke_pytest_dupes"
                        with mock.patch.object(mod, "_tile_bias", return_value=3):
                            pass
                        with mock.patch.object(mod, "_kept", return_value=3):
                            pass
                    """
            },
        )

    def test_both_merge_clean_and_break_at_run_time(self):
        res = clashes.scan(str(self.repo), 1, self.prs, self.develop)
        p = res["pairs"][0]
        self.assertEqual(p["orders"]["nm"]["conflicts"], [])
        got = {x["name"]: x for x in p["fwd"]["strings"]}
        self.assertEqual(set(got), {"rocke_pytest_widgets", "_tile_bias"})
        self.assertEqual(got["rocke_pytest_widgets"]["how"], "cmake")
        self.assertEqual(got["_tile_bias"]["how"], "patch")
        self.assertEqual(
            [u[:2] for u in got["_tile_bias"]["uses"]], [["rk/tests/test_names.py", 8]]
        )
        self.assertEqual(p["fwd"]["names"], [])  # no bare use: the old scan saw nothing
        self.assertEqual(clashes.verdict(res)[0], "fail")
        text = "\n".join(clashes.render(res, "rk"))
        self.assertIn(
            "silent (a string): `rocke_pytest_widgets`: this PR drops it from its CMake lines (rk/CMakeLists.txt:1); "
            "#2 spells it at rk/tests/test_names.py:6 (merges clean); no CMake file of the merge names it",
            text,
        )
        self.assertIn("silent (a string): `_tile_bias`: this PR removes it", text)
        self.assertIn(
            "ports the string uses of `_tile_bias`, `rocke_pytest_widgets`", text
        )
        back = clashes.scan(str(self.repo), 2, self.prs, self.develop)["pairs"][0]
        self.assertEqual(
            {x["name"] for x in back["rev"]["strings"]},
            {"rocke_pytest_widgets", "_tile_bias"},
        )


class TestUnscanned(unittest.TestCase):
    def test_the_open_prs_the_checks_scan_did_not_compare(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {"paths": {"working_dir": d, "workers": f"{d}/workers"}}
            st = {
                "prs": {
                    str(k): {"status": "open", "head": "a" * 40} for k in (7, 8, 9, 10)
                }
            }
            st["prs"]["11"] = {"status": "closed", "head": "b" * 40}
            self.assertEqual(clashes.unscanned(cfg, st, 7), (None, []))
            rec = clashes.record_path(cfg, 7)
            rec.parent.mkdir(parents=True)
            rec.write_text(
                json.dumps({"at": "2026-01-01T00:00:00Z", "scanned": [8, 9]})
            )
            self.assertEqual(
                clashes.unscanned(cfg, st, 7), ("2026-01-01T00:00:00Z", [10])
            )

            class Store:
                def load(self):
                    return st

            line = premises.clash_line(cfg, Store(), 7)
            self.assertEqual(
                line,
                [
                    "clash scan: #10 opened after the checks' scan (2026-01-01T00:00:00Z): run `rk clash 7 --new` "
                    "(its silent clashes are not in the checks)"
                ],
            )
            self.assertEqual(premises.clash_line(cfg, None, 7), [])

    def test_the_scan_records_when_and_whom(self):
        with tempfile.TemporaryDirectory() as d:
            r = pathlib.Path(d) / "repo"
            r.mkdir()
            g(r, "init", "-q", "-b", "develop")
            write(r, {"rk/a.py": "x = 1\n"})
            g(r, "add", "-A")
            g(r, "commit", "-qm", "base")
            g(r, "update-ref", "refs/remotes/origin/develop", "HEAD")
            head = g(r, "rev-parse", "HEAD")
            st = {
                "prs": {
                    "7": {"status": "open", "head": head},
                    "8": {"status": "open", "head": head},
                }
            }
            res, _ = clashes.run({"paths": {"repo": str(r)}, "rocke_path": "rk"}, st, 7)
            self.assertEqual(res["scanned"], [8])
            self.assertRegex(res["at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")


class TestPrcodeRun(unittest.TestCase):
    def test_on_a_host_and_in_a_pr_runners_slot_with_the_listed_environment(self):
        """prcode.run, which the pair merges' compiles and collect go through: on a host the command as the session
        user in the folder named, with a slot folder of its own (removed after) and only what worker/pr_env.py lists of
        its environment; in the container through the launcher in a PR runner's slot (begun, run, freed), the same
        list handed on its stdin."""
        seen = []

        def fake(argv, **kw):
            seen.append((argv, kw, os.path.isdir(kw["env"]["RKMON_OUT"])))
            return subprocess.CompletedProcess(argv, 0, "out", "")

        env = {"A": "1", "PATH": "/bin", "GH_TOKEN": "t"}
        r = prcode.run(
            ["c++", "-fsyntax-only", "a.cpp"], cwd="/s", env=env, environ={}, run=fake
        )
        self.assertEqual((r.returncode, r.stdout), (0, "out"))
        argv, kw, there = seen[0]
        self.assertEqual(argv, ["c++", "-fsyntax-only", "a.cpp"])
        out = kw["env"].pop("RKMON_OUT")
        self.assertEqual(
            (kw["cwd"], kw["env"]),
            ("/s", {"PATH": "/bin", **paths.worker_module("pr_env").FIXED}),
        )
        self.assertTrue(there)  # the slot's out, there while the command runs
        self.assertFalse(os.path.exists(out))  # and gone after
        r1 = L.PR_RUNNER.name
        with tempfile.TemporaryDirectory() as d:
            fs = fake_slots.Slots(d)
            with fs.here():
                r = prcode.run(["c++", "a.cpp"], cwd="/s", env=env)
        self.assertEqual(r.returncode, 0)
        self.assertEqual(fs.calls, [(r1, "begin"), (r1, "run"), (r1, "end")])
        self.assertEqual(fs.asked[1][1], ["run", "-C", "/s", "--", "c++", "a.cpp"])
        self.assertEqual(
            fs.ran[0]["env"], {"PATH": "/bin", **paths.worker_module("pr_env").FIXED}
        )

        def slow(argv, **kw):
            raise subprocess.TimeoutExpired(argv, 3)

        self.assertEqual(
            prcode.run(["x"], env={}, environ={}, run=slow).returncode, 124
        )


@unittest.skipUnless(shutil.which("c++"), "no C++ compiler here")
class TestBuild(Branches, unittest.TestCase):
    """N renames a function in a source both change and drops a module; M's lines in the same file call the old name
    and its new test imports the module: the textual merge is clean, the build and the collect are not.
    """

    def setUp(self):
        self.make(
            {
                "rk/src/shared.cpp": "int widget(int x) { return x + 1; }\n\n\n\n\nint keep() { return 0; }\n",
                "rk/include/rk/api.h": "#pragma once\nint widget(int x);\n",
                "rk/platform/python/oldmod.py": "VALUE = 1\n",
            },
            {
                "rk/src/shared.cpp": "int gadget(int x) { return x + 1; }\n\n\n\n\nint keep() { return 0; }\n",
                "rk/platform/python/oldmod.py": None,
            },
            {
                "rk/src/shared.cpp": (
                    "int widget(int x) { return x + 1; }\n\n\n\n\nint keep() { return 0; }\n"
                    "int use() { return widget(2); }\n"
                ),
                "rk/tests/test_new.py": "import oldmod\n\n\ndef test_new():\n    assert oldmod.VALUE == 1\n",
            },
        )

    def test_the_pair_merge_does_not_compile_or_collect_and_runs_in_a_slot(self):
        res = clashes.scan(str(self.repo), 1, self.prs, self.develop)
        p = res["pairs"][0]
        self.assertEqual(p["orders"]["nm"]["conflicts"], [])
        self.assertEqual(p["tests"], ["rk/tests/test_new.py"])
        ran = []

        # This process's environment, without the launcher: a host's slot.
        here = {k: v for k, v in os.environ.items() if k != prcode.RUNNER_VAR}

        def via_prc(argv, cwd=None, env=None, timeout=600):
            ran.append(argv)
            return prcode.run(argv, cwd=cwd, env=env, timeout=timeout, environ=here)

        with tempfile.TemporaryDirectory() as d:
            cfg = {
                "paths": {
                    "repo": str(self.repo),
                    "working_dir": d,
                    "workers": f"{d}/w",
                    "tmp": f"{d}/tmp",
                },
                "rocke_path": "rk",
            }
            lines = clashes.build(cfg, res, run=via_prc, python=sys.executable)
            self.assertEqual(
                sorted(pathlib.Path(d, "tmp").iterdir()), []
            )  # the scratch folder is gone
        text = "\n".join(lines)
        self.assertIn(
            "#2 (this PR first): 1 C/C++ file compiled, 1 test file collected: FAIL",
            text,
        )
        self.assertRegex(
            text, r"ERROR rk/src/shared.cpp \(the tree's include folders\): .*widget"
        )
        if importlib.util.find_spec("pytest"):
            self.assertIn("ERROR collect 1 test file", text)
        self.assertTrue(lines[-1].startswith("BUILD fail 1 pair merge"))
        self.assertEqual(
            ran[0][:2], ["c++", "-fsyntax-only"]
        )  # every run went through prcode.run
        self.assertEqual(len(ran), 2)

    def test_nothing_to_build_reads_skip(self):
        res = {
            "n": 1,
            "pairs": [{"n": 2, "shared": ["rk/a.txt"], "tests": [], "orders": {}}],
            "tests": [],
        }
        self.assertEqual(
            clashes.build({"paths": {}, "rocke_path": "rk"}, res)[-1].split(" ", 2)[:2],
            ["BUILD", "skip"],
        )


if __name__ == "__main__":
    unittest.main()
