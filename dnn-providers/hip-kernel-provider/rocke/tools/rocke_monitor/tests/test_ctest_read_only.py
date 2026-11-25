# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/ctest_read_only.py: ctest on a native build the caller may only read (a kept slot's), run from the results
folder rk pr-run gives the command, the only folder it writes."""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "ctest_read_only.py"
sys.path.insert(0, str(TOOL.parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import ctest_read_only  # noqa: E402
from rkmon_lib import prcode  # noqa: E402
from test_prcode import config_home  # noqa: E402

# ctest's summary line when both of the build's passing tests pass (its releases differ on the middle part)
PASSED = r"\n100% tests passed(, 0 tests failed)? out of 2\n"
CTEST = shutil.which("ctest") or shutil.which(
    "ctest", path=os.path.dirname(sys.executable)
)


def check(mark: str) -> str:
    """A test's command line in a CTest file: passes when it runs in the folder that holds `mark`."""
    py = ctest_read_only.cmake_quoted(sys.executable)
    return f'{py} "-c" "import os, sys\\nsys.exit(not os.path.exists(\'{mark}\'))"'


def make_build(top: pathlib.Path) -> None:
    """A native build's CTest files as CMake writes them: a test in the top folder, a subfolder with its own (each runs
    in its folder), and one that fails."""
    (top / "sub").mkdir(parents=True)
    (top / "top.mark").write_text("")
    (top / "sub" / "sub.mark").write_text("")
    (top / "CTestTestfile.cmake").write_text(
        f'add_test(top {check("top.mark")})\nsubdirs("sub")\n'
    )
    py = ctest_read_only.cmake_quoted(sys.executable)
    (top / "sub" / "CTestTestfile.cmake").write_text(
        f'add_test(sub {check("sub.mark")})\nadd_test(fails {py} "-c" "raise SystemExit(1)")\n'
    )


def files(top: pathlib.Path) -> dict:
    """Every path under `top`, with its size and modification time."""
    return {
        str(p.relative_to(top)): (p.lstat().st_size, p.lstat().st_mtime_ns)
        for p in top.rglob("*")
    }


def read_only(top: pathlib.Path) -> None:
    for p in [top, *top.rglob("*")]:
        if p.is_dir():
            p.chmod(0o555)


def writable(top: pathlib.Path) -> None:
    for p in [top, *top.rglob("*")]:
        if p.is_dir():
            p.chmod(0o755)


class TestRefusals(unittest.TestCase):
    """What it cannot run read-only it refuses, exit 2, writing nothing."""

    def test_it_refuses_what_it_cannot_run_read_only(self):
        d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        out, build = d / "out", d / "b"
        out.mkdir()
        build.mkdir()
        (build / "CTestTestfile.cmake").write_text("")
        env = dict(os.environ, RKMON_OUT=str(out))
        cases = [
            ((), env, "usage"),
            (("-R", "x"), env, "usage"),  # an option where the build goes
            ((str(build), "--test-dir", str(d)), env, "--test-dir"),
            ((str(build), f"--test-dir={d}"), env, "--test-dir"),
            (
                (str(build),),
                {k: v for k, v in env.items() if k != "RKMON_OUT"},
                "rk pr-run",
            ),
            ((str(d / "b\\z"),), env, "backslash"),
            ((str(out),), env, "no CTestTestfile.cmake"),
            ((str(build),), dict(env, PATH=str(d / "none")), "no ctest"),
        ]
        for args, run_env, why in cases:
            with self.subTest(args=args, why=why):
                r = subprocess.run(
                    [sys.executable, "-B", str(TOOL), *args],
                    cwd=d,
                    env=run_env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertIn(why, r.stderr)
                self.assertEqual(list(out.iterdir()), [])


@unittest.skipUnless(CTEST, "ctest is not available")
class TestReadOnlyBuild(unittest.TestCase):
    """A build that is read-only to the caller, as a kept slot's is to every other runner."""

    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        self.out = self.d / "out"
        self.out.mkdir()
        self.path = f"{os.path.dirname(CTEST)}{os.pathsep}{os.environ.get('PATH', '')}"

    def build(self, name: str = "build") -> pathlib.Path:
        top = self.d / name
        make_build(top)
        read_only(top)
        self.addCleanup(writable, top)
        return top

    def tool(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-B", str(TOOL), *args],
            cwd=self.d,
            env=dict(os.environ, RKMON_OUT=str(self.out), PATH=self.path),
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_the_builds_tests_run_in_their_folders_and_ctest_writes_only_in_the_out(
        self,
    ):
        build = self.build()
        before = files(build)
        r = self.tool(str(build), "-E", "^fails$")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertRegex(r.stdout, PASSED)
        self.assertEqual(files(build), before)
        self.assertEqual([p.name for p in self.out.iterdir()], [ctest_read_only.FOLDER])
        self.assertTrue(
            (self.out / "ctest" / "Testing" / "Temporary" / "LastTest.log").is_file()
        )

    def test_its_other_arguments_go_to_ctest_and_its_exit_is_ctests(self):
        r = self.tool(str(self.build()), "-R", "^fails$")
        self.assertEqual(r.returncode, 8, r.stdout + r.stderr)  # ctest's: a test failed
        self.assertIn("0% tests passed, 1 tests failed out of 1", r.stdout)

    def test_a_build_whose_path_has_quotes_dollars_and_semicolons(self):
        r = self.tool(str(self.build('b "q" ${X} $y;z')), "-E", "^fails$")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertRegex(r.stdout, PASSED)

    def test_through_rk_pr_run_on_a_host_its_out_keeps_ctests_logs(self):
        """The command the verifier and the code quality reviewer are given, on a host: rk pr-run's `--out` keeps what
        ctest wrote, and the build is as it was."""
        build = self.build()
        before = files(build)
        (self.d / "w").mkdir()
        env = dict(
            os.environ,
            RKMON_HOME=str(config_home(self.d, self.d / "w")),
            PATH=self.path,
        )
        env.pop(prcode.RUNNER_VAR, None)
        res = self.d / "res"
        r = subprocess.run(
            ["sh", str(ROOT / "rk"), "pr-run", "--out", str(res), "--"]
            + ["python3", "-B", str(TOOL), str(build), "-E", "^fails$"],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertRegex(r.stdout, PASSED)
        self.assertTrue(
            (res / "ctest" / "Testing" / "Temporary" / "LastTest.log").is_file()
        )
        self.assertEqual(files(build), before)


if __name__ == "__main__":
    unittest.main()
