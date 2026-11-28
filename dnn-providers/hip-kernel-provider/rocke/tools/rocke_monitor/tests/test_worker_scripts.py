# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The worker scripts' contracts, with stand-ins for the programs they run (cmake, ctest, the venv's python, su, pkill,
the reviewer passes): build_native.py's step lines, logs and FAILED end, its RAM disk link and its -j; run_suite.py's
out.txt and its last two lines; kill_runner.py's lines; review_agent_run.py's staging and the review offload's modes;
and worker/steps.py's shell-like helpers (tail, the conflict list, word splitting)."""

import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKER = ROOT / "worker"
sys.path.insert(0, str(WORKER))
sys.dont_write_bytecode = True

import mkwt  # noqa: E402
import steps  # noqa: E402

BASE_ENV = {k: v for k, v in os.environ.items() if not k.startswith(("RKMON_", "GIT_"))}
FAKE_CMAKE = """#!/bin/sh
echo "cmake $*"
case "$*" in *"--build"*"${FAIL_AT:-none}"*) echo "boom" >&2; exit 4 ;; esac
case "$1" in -S) mkdir -p "$4" ;; esac
exit 0
"""
FAKE_CTEST = "#!/bin/sh\necho \"ctest $*\"; echo 'line 2'; echo 'Total Tests: 9'\n"
FAKE_PY = """#!/bin/sh
case "$*" in
  *"--cmakedir"*) echo "/fake/pybind11" ;;
  *"-m pytest"*) echo "pytest $*"; echo "flavor=$ROCKE_LLVM_FLAVOR"; echo "3 passed"; exit 1 ;;
  *) echo "rocke_engine import OK: x"; exit ${IMPORT_RC:-0} ;;
esac
"""


def tool(folder: pathlib.Path, name: str, text: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text)
    (folder / name).chmod(0o755)


class TestWorkerScripts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.work = self.d / "work"
        (self.work / "wt7m").mkdir(parents=True)
        self.venv = self.d / "venv"
        for name, text in (
            ("cmake", FAKE_CMAKE),
            ("ctest", FAKE_CTEST),
            ("python", FAKE_PY),
        ):
            tool(self.venv / "bin", name, text)

    def env(self, **extra) -> dict:
        e = dict(
            BASE_ENV,
            RKMON_WORKING_DIR=str(self.work),
            RKMON_REPO_DIR=str(self.work / "repo"),
            RKMON_REPO="o/r",
            RKMON_ROCKE_PATH="p",
            RKMON_BUILD=str(self.d / "build"),
            RKMON_VENV=str(self.venv),
            RKMON_SCRATCH=str(self.d / "scratch"),
        )
        e.update(extra)
        return {k: v for k, v in e.items() if v is not None}

    def run_tool(
        self, name: str, *args, prefix=(), **env
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*prefix, sys.executable, "-B", str(WORKER / name), *args],
            capture_output=True,
            text=True,
            timeout=120,
            env=self.env(**env),
        )

    def test_build_native_steps_their_logs_and_the_import(self):
        p = self.run_tool("build_native.py", "wt7m", RKMON_JOBS="3")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(
            p.stdout,
            "-- core-configure\n-- core-build\n-- pybind-configure\n-- pybind-build\n"
            "rocke_engine import OK: x\nTotal Tests: 9\nctest -R rocke_(mma_frag_ssot|"
            "arch_target_identity|future_intrinsic_lowering|ir_serialize_roundtrip)$ "
            "--output-on-failure\nline 2\nTotal Tests: 9\n",
        )
        b = self.d / "build" / "wt7m"
        self.assertEqual(
            (b / "core-build.log").read_text(), f"cmake --build {b}/core -j3\n"
        )  # $RKMON_JOBS (the job's environment; steps.build_jobs without it)
        self.assertIn(
            "-Dpybind11_DIR=/fake/pybind11 ", (b / "pybind-configure.log").read_text()
        )

    def test_build_native_a_failing_step_ends_it(self):
        p = self.run_tool("build_native.py", "wt7m", FAIL_AT="core", RKMON_JOBS="2")
        self.assertEqual(p.returncode, 1)
        self.assertTrue(
            p.stdout.endswith(
                "-- core-build\nFAILED: core-build\ncmake --build "
                f"{self.d}/build/wt7m/core -j2\nboom\n"
            ),
            p.stdout,
        )
        p = self.run_tool("build_native.py", "wt7m", IMPORT_RC="6")
        self.assertEqual(p.returncode, 6)  # the import check's own exit code (set -e)
        p = self.run_tool("build_native.py")
        self.assertEqual(
            (p.returncode, p.stdout),
            (2, "usage: build_native.py <worktree folder name>\n"),
        )

    def test_build_native_a_pr_tree_builds_on_the_ramdisk(self):
        rd = self.d / "rd"
        p = self.run_tool("build_native.py", "wt7m", RKMON_RAMDISK=str(rd))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        link = self.d / "build" / "wt7m"
        self.assertTrue(link.is_symlink())
        self.assertEqual(os.readlink(link), f"{rd}/build/wt7m")
        self.assertTrue((rd / "build" / "wt7m" / "core-build.log").is_file())
        p = self.run_tool(
            "build_native.py", "wtbase", RKMON_RAMDISK=str(rd)
        )  # not a PR's tree: on the disk
        self.assertFalse((self.d / "build" / "wtbase").is_symlink())

    def test_run_suite_writes_out_txt_and_prints_its_end(self):
        p = self.run_tool("run_suite.py", "wt7m", "plat")
        self.assertEqual(p.returncode, 0, p.stderr)
        out = (self.d / "scratch" / "plat_wt7m" / "out.txt").read_text()
        self.assertTrue(
            out.startswith(
                f"pytest -I -B -m pytest {self.work}/wt7m/p/platform/tests "
                f"{self.work}/wt7m/p/platform/python/rocke/tests -q -p no:cacheprovider "
                "--no-header -rfE --continue-on-collection-errors\nflavor=llvm22\n"
            ),
            out,
        )
        self.assertTrue(out.endswith("3 passed\nexit=1\n"))
        self.assertEqual(p.stdout, "3 passed\nexit=1\n")

    def test_kill_runner_lines(self):
        fake = self.d / "bin"
        tool(
            fake,
            "su",
            '#!/bin/sh\necho "$1: $3" >> "$LOG"\n'
            "[ \"$3\" = 'ollama ps' ] && printf 'NAME\\nm:1 a\\nm:2 b\\n'\nexit 0\n",
        )
        tool(fake, "pkill", '#!/bin/sh\necho "pkill $*" >> "$LOG"\nexit 1\n')
        log = self.d / "log"
        p = self.run_tool(
            "kill_runner.py",
            PATH=f"{fake}:{os.environ['PATH']}",
            LOG=str(log),
            RKMON_WORKER_MODEL="m:1",
            RKMON_WORKER_USER="op",
        )
        self.assertEqual(
            (p.returncode, p.stdout),
            (0, "ollama stop ok\nno runner process\nm:1 a\nm:2 b\n"),
        )
        self.assertEqual(
            log.read_text(),
            "op: ollama stop m:1\npkill -f ollama runner|llama-server\nop: ollama ps\n",
        )

    def test_review_agent_run_stages_then_drafts_by_the_offload_mode(self):
        tools = self.d / "tools"
        tools.mkdir()
        for name in ("review_agent_run.py", "rkenv.py", "steps.py"):
            (tools / name).write_text((WORKER / name).read_text())
        (tools / "fetch_pr.py").write_text(
            "import pathlib, sys\npathlib.Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)\n"
            "print('fetched', sys.argv[1])\n"
        )
        (tools / "mkwt.py").write_text(
            "print('head=a develop=b merge-base=c')\nprint('behind/ahead: 0\\t1')\n"
        )
        (tools / "review_agent.py").write_text(
            "import os, sys\nprint('agent', os.environ['NUM_BATCH'])\n"
        )
        (tools / "offload_agent.py").write_text(
            "import sys\nprint('offload', sys.argv[2:])\nsys.exit(3)\n"
        )
        book = self.d / "book.md"
        book.write_text("the book\n")
        env = dict(
            RKMON_WORKERS=str(self.work / "workers"), RKMON_REVIEW_BOOK=str(book)
        )
        run = lambda **e: subprocess.run(
            [
                sys.executable,
                "-B",
                str(tools / "review_agent_run.py"),
                "7",
            ],  # noqa: E731
            capture_output=True,
            text=True,
            timeout=60,
            env=self.env(**env, **e),
        )
        ws = self.work / "workers" / "pr7"
        p = run()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(
            p.stdout,
            "fetched 7\nhead=a develop=b merge-base=c\nbehind/ahead: 0\t1\n"
            f"== agent run, NUM_BATCH=256 ==\nagent 256\n== review: {ws}/review_agent.md ==\n",
        )
        self.assertEqual((ws / "review_book.md").read_text(), "the book\n")
        self.assertTrue((ws / ".gh-none").is_dir())
        p = run(RKMON_OFFLOAD_MODE="shadow", NUM_BATCH="64")
        self.assertEqual(p.returncode, 0)
        self.assertTrue(
            p.stdout.endswith(
                "agent 64\n"
                f"== review: {ws}/review_agent.md ==\n"
                "== shadow offload passes (they post nothing) ==\noffload ['7', 'draft', "
                f"'quality', 'regrade', '--pr-dir', '{ws}/pr']\n(the shadow offload passes "
                "failed; the draft above stands)\n"
            ),
            p.stdout,
        )
        p = run(RKMON_OFFLOAD_MODE="on")
        self.assertEqual(p.returncode, 3)  # the offload's failure ends the job (set -e)
        self.assertTrue(
            p.stdout.endswith(
                "== offload passes (draft, quality, regrade) ==\n"
                "offload ['7', 'draft', 'quality', 'regrade']\n"
            ),
            p.stdout,
        )


class TestSteps(unittest.TestCase):
    def test_last_lines_as_tail(self):
        self.assertEqual(steps.last_lines(b"a\nb\nc\n", 2), b"b\nc\n")
        self.assertEqual(steps.last_lines(b"a\nb\nc", 2), b"b\nc")  # no newline added
        self.assertEqual(steps.last_lines(b"a\n", 5), b"a\n")
        self.assertEqual(steps.last_lines(b"", 5), b"")

    def test_the_conflict_list_as_sed(self):
        out = b"treeid\np/a.txt\np/b.txt\n\nAuto-merging p/a.txt\nCONFLICT (content)\n"
        self.assertEqual(mkwt.conflict_lines(out), b"p/a.txt\np/b.txt\n\n")
        self.assertEqual(mkwt.conflict_lines(b"treeid\n"), b"")
        self.assertEqual(mkwt.conflict_lines(b"treeid\nx"), b"x\n")

    def test_words_as_an_unquoted_expansion(self):
        self.assertEqual(steps.words("  a/b  .github\tc "), ["a/b", ".github", "c"])
        self.assertEqual(steps.words(""), [])
        with tempfile.TemporaryDirectory() as d:
            cwd = os.getcwd()
            try:
                os.chdir(d)
                pathlib.Path("x1").write_text("")
                pathlib.Path("x2").write_text("")
                self.assertEqual(
                    steps.words("x* y?"), ["x1", "x2", "y?"]
                )  # a pattern with no match stays
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
