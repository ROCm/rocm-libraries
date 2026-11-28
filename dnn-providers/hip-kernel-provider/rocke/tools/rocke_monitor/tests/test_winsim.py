# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The Windows simulation (tools/winsim.py, tools/pytest_plugins/rkmon_winsim.py, `rk winsim`, the checks' section).

- Each simulation has a negative control: a test that passes plainly and fails under it (a lead, named with that
  simulation), and a Windows-safe test of the same kind that passes under it (no lead). A test that fails plainly is no
  lead. `os` runs only when named.
- The macro scan finds a windows.h macro name the C/C++ lines add (object-like as a word; min( and max( unless the
  file defines NOMINMAX), never in comments, strings or #undef lines; on a git repository, only the added lines.
- `rk winsim` writes the tree to a scratch folder, scans, runs the PR's changed tests, and removes the scratch; the PR
  code runs in a PR runner's slot where there are PR runners, on a tree the runners read and write none of.
- The checks run it as a lead section in a review, never in a gate; its test run goes through prc.
"""

import contextlib
import grp
import io
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "pytest_plugins"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "fakes"))
sys.dont_write_bytecode = True
import rkmon_winsim  # noqa: E402
import winsim  # noqa: E402
from rkmon_lib import container_layout as L  # noqa: E402
from rkmon_lib import lanes, review  # noqa: E402
import fake_slots  # noqa: E402


def _python() -> str | None:
    """A python with pytest: $RKMON_VENV's, else this one when it has pytest."""
    v = os.environ.get("RKMON_VENV")
    if v and pathlib.Path(v, "bin", "python").exists():
        return str(pathlib.Path(v, "bin", "python"))
    try:
        import pytest  # noqa: F401

        return sys.executable
    except ImportError:
        return None


# Each pair: one test Windows breaks, one Windows-safe test of the same kind.
TESTS = {
    "test_text.py": """
        def test_reads_utf8_without_an_encoding(tmp_path):
            p = tmp_path / "a.txt"
            p.write_bytes("caf\\u00e9 \\u2192".encode("utf-8"))
            assert p.read_text() == "caf\\u00e9 \\u2192"

        def test_reads_utf8_with_its_encoding(tmp_path):
            p = tmp_path / "a.txt"
            p.write_bytes("caf\\u00e9 \\u2192".encode("utf-8"))
            assert p.read_text(encoding="utf-8") == "caf\\u00e9 \\u2192"
        """,
    "test_crlf.py": """
        def test_text_written_reads_back_as_the_same_bytes(tmp_path):
            p = tmp_path / "a.txt"
            with open(p, "w", encoding="utf-8") as f:
                f.write("a\\nb\\n")
            assert p.read_bytes() == b"a\\nb\\n"

        def test_newline_given(tmp_path):
            p = tmp_path / "a.txt"
            with open(p, "w", encoding="utf-8", newline="\\n") as f:
                f.write("a\\nb\\n")
            assert p.read_bytes() == b"a\\nb\\n"
        """,
    "test_temp.py": """
        import tempfile

        def test_reopen_while_open():
            with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as f:
                f.write("x")
                f.flush()
                with open(f.name, encoding="utf-8") as g:
                    assert g.read() == "x"

        def test_reopen_after_close():
            import os
            f = tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False)
            f.write("x")
            f.close()
            try:
                with open(f.name, encoding="utf-8") as g:
                    assert g.read() == "x"
            finally:
                os.unlink(f.name)
        """,
    "test_osflavour.py": """
        import os

        def test_assumes_a_posix_separator():
            assert "a" + os.sep + "b" == "a/b"

        def test_joins_with_os_path():
            assert os.path.join("a", "b").endswith("b")

        def test_fails_plainly():
            assert False
        """,
}


def tree(d: pathlib.Path) -> pathlib.Path:
    k = d / "K"
    (k / "platform" / "python").mkdir(parents=True)
    t = k / "platform" / "tests"
    t.mkdir(parents=True)
    (t / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    for name, body in TESTS.items():
        (t / name).write_text(textwrap.dedent(body), encoding="utf-8")
    return k


def run_tool(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    env = {"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}
    with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(out):
        rc = winsim.main(argv)
    return rc, out.getvalue()


class TestChosen(unittest.TestCase):
    def test_the_default_set_and_os_only_when_named(self):
        self.assertEqual(rkmon_winsim.chosen(""), ("text", "crlf", "tempfile"))
        self.assertEqual(rkmon_winsim.chosen("all"), ("text", "crlf", "tempfile"))
        self.assertEqual(rkmon_winsim.chosen("os,crlf,nope"), ("crlf", "os"))
        self.assertEqual(rkmon_winsim.chosen("nope"), ())

    def test_importing_the_plugin_simulates_nothing(self):
        import builtins

        self.assertIsNot(builtins.open, rkmon_winsim.sim_open)
        self.assertEqual(os.sep, "/")


@unittest.skipUnless(_python(), "no python with pytest")
class TestRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.k = tree(pathlib.Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_each_simulation_breaks_its_test_and_spares_the_safe_one(self):
        rc, out = run_tool(
            ["run", "--tree", str(self.k), "--python", _python()]
            + [f"platform/tests/{n}" for n in TESTS]
        )
        leads = [ln for ln in out.splitlines() if ln.startswith("LEAD ")]
        self.assertEqual(
            sorted(leads),
            [
                "LEAD platform/tests/test_crlf.py::test_text_written_reads_back_as_the_same_bytes passes plainly and "
                "fails under the Windows simulation (crlf)",
                "LEAD platform/tests/test_temp.py::test_reopen_while_open passes plainly and fails under the Windows "
                "simulation (tempfile)",
                "LEAD platform/tests/test_text.py::test_reads_utf8_without_an_encoding passes plainly and fails under "
                "the Windows simulation (text)",
            ],
            out,
        )
        self.assertIn(
            "platform/tests/test_text.py: plain 2 pass, 0 fail, 0 skip; Windows simulation (text,crlf,tempfile) "
            "1 pass, 1 fail, 0 skip",
            out,
        )
        # `os` is not in the default set; a test that fails plainly is no lead.
        self.assertIn(
            "platform/tests/test_osflavour.py: plain 2 pass, 1 fail, 0 skip; Windows simulation "
            "(text,crlf,tempfile) 2 pass, 1 fail, 0 skip",
            out,
        )
        self.assertTrue(
            out.strip()
            .splitlines()[-1]
            .startswith("WINSIM fail 3 test leads, 0 macro names")
        )
        self.assertEqual(rc, 1)

    def test_os_runs_when_named(self):
        rc, out = run_tool(
            [
                "run",
                "--tree",
                str(self.k),
                "--python",
                _python(),
                "--only",
                "os",
                "platform/tests/test_osflavour.py",
            ]
        )
        self.assertIn(
            "LEAD platform/tests/test_osflavour.py::test_assumes_a_posix_separator passes plainly and fails under "
            "the Windows simulation (os)",
            out,
        )
        self.assertNotIn("test_joins_with_os_path passes", out)  # os.path stays POSIX
        self.assertNotIn("test_fails_plainly passes", out)
        self.assertEqual(rc, 1)

    def test_safe_tests_pass_and_a_missing_file_is_said(self):
        rc, out = run_tool(
            [
                "run",
                "--tree",
                str(self.k),
                "--python",
                _python(),
                "platform/tests/test_text.py::test_reads_utf8_with_its_encoding",
                "platform/tests/test_crlf.py::test_newline_given",
                "platform/tests/test_temp.py::test_reopen_after_close",
                "platform/tests/test_gone.py",
            ]
        )
        self.assertIn(
            "platform/tests/test_gone.py: not run (the tree has no such file)", out
        )
        self.assertEqual(
            out.strip().splitlines()[-1], "WINSIM pass 0 test leads, 0 macro names", out
        )
        self.assertEqual(rc, 0)

    def test_a_scan_report_is_folded_in(self):
        rep = pathlib.Path(self.tmp.name, "scan.txt")
        rep.write_text(
            "SCANNED 2\nHIT src/a.cpp:3: far is a windows.h macro (minwindef.h (empty)): int far;\n"
        )
        rc, out = run_tool(["run", "--tree", str(self.k), "--scan-report", str(rep)])
        self.assertIn(
            "windows.h macro names in the C/C++ lines the PR adds: 1 in 2 C/C++ files",
            out,
        )
        self.assertIn("LEAD src/a.cpp:3: far is a windows.h macro", out)
        self.assertTrue(
            out.strip()
            .splitlines()[-1]
            .startswith("WINSIM fail 0 test leads, 1 macro name:")
        )
        rep.write_text("SCANNED 1\n")
        self.assertEqual(
            run_tool(["run", "--tree", str(self.k), "--scan-report", str(rep)])[0], 0
        )
        rc, out = run_tool(["run", "--tree", str(self.k)])
        self.assertEqual(
            (rc, out.strip()),
            (3, "WINSIM skip nothing run or scanned; 0 test leads, 0 macro names"),
        )


class TestMacroScan(unittest.TestCase):
    def hits(self, text: str, nominmax: bool = False) -> list[str]:
        lines = list(enumerate(text.strip("\n").splitlines(), 1))
        return [
            h.split(" is a windows.h macro")[0]
            for h in winsim.macro_hits("a.cpp", lines, nominmax)
        ]

    def test_names_used_as_code(self):
        self.assertEqual(
            self.hits(
                """
                float far = 1.0f, near_plane = 0.1f;
                enum class Level { ERROR, WARN };
                auto m = std::max(a, b);
                auto k = std::numeric_limits<int>::max();
                auto ok = (std::max)(a, b);
                void f(int IN, int OUT);
                """
            ),
            [
                "a.cpp:1: far",
                "a.cpp:2: ERROR",
                "a.cpp:3: max",
                "a.cpp:4: max",
                "a.cpp:6: IN",
                "a.cpp:6: OUT",
            ],
        )

    def test_comments_strings_preprocessor_and_nominmax(self):
        self.assertEqual(
            self.hits(
                """
                // the near and far planes
                const char* s = "ERROR: far";
                #undef max
                #ifdef DELETE
                /* far
                   near */ int fine = 0;
                int nearest = minimum(1, 2);
                """
            ),
            [],
        )
        self.assertEqual(self.hits("int x = std::min(a, b);", nominmax=True), [])

    def test_added_lines_from_a_diff(self):
        diff = (
            "diff --git a/K/a.cpp b/K/a.cpp\n--- a/K/a.cpp\n+++ b/K/a.cpp\n@@ -3,0 +4,2 @@\n+int far;\n+int x;\n"
            "diff --git a/K/b.cpp b/K/b.cpp\n--- a/K/b.cpp\n+++ /dev/null\n@@ -1 +0,0 @@\n-int near;\n"
        )
        self.assertEqual(
            dict(winsim.added_lines(diff)),
            {"K/a.cpp": [(4, "int far;"), (5, "int x;")]},
        )


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def a_repo(d: pathlib.Path) -> tuple[pathlib.Path, str]:
    """A repository: `main` with K/src/old.cpp, a PR head that adds C/C++ (a name in a file that includes windows.h
    through the tree's header, one in a file that includes none, a NOMINMAX file; an old line left alone), a Python
    file using max( and a test Windows breaks."""
    r = d / "repo"
    (r / "K" / "src").mkdir(parents=True)
    (r / "K" / "platform" / "tests").mkdir(parents=True)
    (r / "K" / "src" / "old.cpp").write_text("int near = 0;\n")
    (r / "K" / "platform" / "tests" / "pytest.ini").write_text("[pytest]\n")
    git(r.parent, "init", "-q", "-b", "main", str(r))
    git(r, "add", "-A")
    git(r, "commit", "-qm", "base")
    git(r, "checkout", "-qb", "pr")
    (r / "K" / "src" / "win.h").write_text("#include <windows.h>\n")
    (r / "K" / "src" / "a.cpp").write_text(
        '#include "src/win.h"\nint x = 0;\nfloat far = 1.0f;\n'
    )
    (r / "K" / "src" / "d.cpp").write_text("int near = 1;\n")
    (r / "K" / "src" / "b.cpp").write_text(
        "#define NOMINMAX\nint y = std::max(1, 2);\n"
    )
    (r / "K" / "src" / "c.py").write_text("m = max(1, 2)\n")
    (r / "K" / "platform" / "tests" / "test_crlf.py").write_text(
        textwrap.dedent(TESTS["test_crlf.py"])
    )
    git(r, "add", "-A")
    git(r, "commit", "-qm", "pr")
    return r, git(r, "rev-parse", "HEAD")


class TestReach(unittest.TestCase):
    def test_through_the_trees_headers_and_nominmax_in_cmake(self):
        with tempfile.TemporaryDirectory() as d:
            k = pathlib.Path(d)
            (k / "inc").mkdir()
            (k / "inc" / "plat.hpp").write_text("#include <Windows.h>\n")
            (k / "inc" / "mid.hpp").write_text('#include "inc/plat.hpp"\n')
            (k / "top.cpp").write_text('#include "mid.hpp"\n')
            (k / "lone.cpp").write_text("#include <vector>\n")
            self.assertEqual(
                winsim.windows_reach(k),
                ({"inc/plat.hpp", "inc/mid.hpp", "top.cpp"}, False),
            )
            hits = [
                "K/top.cpp:2: max is a windows.h macro (x): m = std::max(a, b);",
                "K/top.cpp:3: far is a windows.h macro (x): int far;",
                "K/lone.cpp:1: far is a windows.h macro (x): int far;",
            ]
            self.assertEqual(winsim.classify(hits, k, "K"), (hits[:2], hits[2:]))
            (k / "CMakeLists.txt").write_text("add_compile_definitions(NOMINMAX)\n")
            self.assertEqual(
                winsim.classify(hits, k, "K"), ([hits[1]], [hits[0], hits[2]])
            )
            self.assertEqual(winsim.classify(hits, None, "K"), (hits, []))


class TestScanOnGit(unittest.TestCase):
    def test_only_the_added_cpp_lines(self):
        with tempfile.TemporaryDirectory() as d:
            r, head = a_repo(pathlib.Path(d))
            rc, out = run_tool(
                [
                    "scan",
                    "--repo",
                    str(r),
                    "--head",
                    head,
                    "--base",
                    "main",
                    "--rocke-path",
                    "K",
                ]
            )
            rc2, out2 = run_tool(
                ["scan", "--repo", str(r), "--head", head, "--base", "main"]
                + ["--rocke-path", "K", "--tree", str(r / "K")]
            )
        self.assertEqual((rc, rc2), (0, 0))
        far = "K/src/a.cpp:3: far is a windows.h macro (minwindef.h (empty)): float far = 1.0f;"
        near = "K/src/d.cpp:1: near is a windows.h macro (minwindef.h (empty)): int near = 1;"
        # With the tree: a.cpp sees windows.h through src/win.h, d.cpp sees none.
        self.assertEqual(
            out2.splitlines(), ["SCANNED 4", f"HIT {far}", f"UNREACHED {near}"]
        )
        self.assertEqual(
            out.splitlines(),
            [
                "SCANNED 4",
                "REACH not read (no tree): every name counts",
                f"HIT {far}",
                f"HIT {near}",
            ],
        )


class TestRkWinsim(unittest.TestCase):
    def cfg(self, d: pathlib.Path, repo: pathlib.Path) -> dict:
        return {
            "rocke_path": "K",
            "paths": {
                "working_dir": str(d / "w"),
                "repo": str(repo),
                "tmp": str(d / "tmp"),
            },
        }

    @unittest.skipUnless(_python(), "no python with pytest")
    def test_the_prs_changed_tests_and_its_cpp_in_a_scratch_tree(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            r, head = a_repo(d)
            cfg = self.cfg(d, r)
            calls = []

            def run(argv, **kw):
                calls.append((argv, kw.get("input")))
                if (
                    "run" in argv and "--tree" in argv
                ):  # the test run: on a python that has pytest
                    argv = [
                        a if a != lanes._venv_python(cfg) else _python() for a in argv
                    ]
                    with mock.patch.dict(os.environ, {"LC_ALL": "C.UTF-8"}):
                        p = subprocess.run(
                            argv,
                            capture_output=True,
                            text=True,
                            env={**kw["env"], "LC_ALL": "C.UTF-8"},
                        )
                    calls.append(("out", p.stdout))
                    return p
                return subprocess.run(argv, **kw)

            out = io.StringIO()
            with mock.patch.dict(
                os.environ, {"RKMON_PR_RUNNER": ""}
            ), contextlib.redirect_stdout(out):
                rc = lanes.winsim(cfg, [], head, "main", scan=True, run=run)
            text = out.getvalue() + "".join(c[1] for c in calls if c[0] == "out")
            self.assertIn("the test files", text)
            self.assertIn("platform/tests/test_crlf.py", text)
            self.assertIn("LEAD K/src/a.cpp:3: far is a windows.h macro", text)
            self.assertIn(
                "1 more in files that include no Windows header (no lead)", text
            )
            self.assertIn(
                "test_text_written_reads_back_as_the_same_bytes passes plainly and fails",
                text,
            )
            self.assertEqual(rc, 1)
            self.assertEqual(
                list((d / "tmp").iterdir()), []
            )  # the scratch folder is gone
            self.assertTrue(
                all(c[1] is None for c in calls if c[0] != "out")
            )  # no launcher: no stdin env

    def test_the_tests_run_in_a_pr_runners_slot(self):
        """Where there are PR runners the tests run through the launcher in a runner's slot (begun, run, freed), on a
        scratch tree of the monitor's in the temporary folder that the runners read and write none of (their group,
        mode 2750), with only what worker/pr_env.py lists of the environment; the scratch is removed after.
        """
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            r, head = a_repo(d)
            cfg = self.cfg(d, r)
            fs = fake_slots.Slots(d / "s")
            seen = {}

            def on_run(slot, cmd):
                tree = pathlib.Path(cmd[cmd.index("--tree") + 1])
                s = tree.parent.stat()
                seen.update(
                    tree=tree,
                    folder=(stat.S_IMODE(s.st_mode), s.st_gid),
                    written=(tree / "platform" / "tests" / "test_crlf.py").is_file(),
                )

            fs.on_run = on_run
            # The runners' group: this user's own here, so the scratch folder is really made theirs.
            mine = grp.getgrgid(os.getgid()).gr_name
            with fs.here(), mock.patch.object(L, "PR_GROUP", mine), mock.patch.dict(
                os.environ, {"GH_TOKEN": "not-for-pr-code"}
            ), contextlib.redirect_stdout(io.StringIO()):
                rc = lanes.winsim(
                    cfg, ["platform/tests/test_crlf.py"], head, "main", scan=False
                )
            self.assertEqual(rc, 0)
            r1 = L.PR_RUNNER.name
            self.assertEqual(fs.calls, [(r1, "begin"), (r1, "run"), (r1, "end")])
            cmd = fs.ran[0]["cmd"]
            self.assertEqual(cmd[:3], ["nice", "-n", "10"])
            self.assertNotIn("--scan-report", cmd)
            self.assertEqual(cmd[-2:], ["--", "platform/tests/test_crlf.py"])
            self.assertEqual(seen["tree"].parents[1], d / "tmp")
            self.assertEqual(seen["folder"], (0o2750, os.getgid()))
            self.assertTrue(seen["written"])
            self.assertNotIn("GH_TOKEN", fs.ran[0]["env"])
            self.assertEqual(list((d / "tmp").iterdir()), [])  # the scratch is gone


class TestChecksWiring(unittest.TestCase):
    CFG = {
        "rocke_path": "rk",
        "repo": "o/r",
        "paths": {"working_dir": "/w"},
    }

    def test_a_review_runs_it_and_a_gate_does_not(self):
        plan = review.plan_checks(
            self.CFG, {"n": 7}, ["rk/platform/tests/test_foo.py"], {}
        )
        plan["_dirs"] = {}
        script = review.checks_script(self.CFG, 7, "a" * 40, "first-round", plan, None)
        sec = script.split(f"sec '{review.WINSIM_SECTION}", 1)[1].split("\nsec ", 1)[0]
        self.assertIn(
            f"python3 -I -B $M/tools/winsim.py scan --repo $R0 --head {'a' * 40} --base $BREF --rocke-path $K "
            '--tree $W/$K > "$OUT/winsim_scan.txt"',
            sec,
        )
        self.assertIn(
            f"prc nice -n 10 timeout {lanes.WINSIM_BUDGET_S + 120} python3 -I -B $M/tools/winsim.py run --tree $W/$K "
            f"--python python --scan-report $SI/winsim_scan.txt --budget {lanes.WINSIM_BUDGET_S} -- "
            "platform/tests/test_foo.py",
            sec,
        )
        # the monitor's scan, handed to the job in the slot's inputs
        self.assertIn('> "$OUT/winsim_scan.txt" 2>&1; prin "$OUT/winsim_scan.txt"', sec)
        self.assertIn("st winsim ", sec)
        self.assertIn("winsim", review.LEAD_SECTIONS)
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True
            ).returncode,
            0,
        )
        gate = review.checks_script(
            self.CFG, 7, "a" * 40, "first-round", plan, None, mode="gate"
        )
        self.assertNotIn(review.WINSIM_SECTION, gate)


if __name__ == "__main__":
    unittest.main()
