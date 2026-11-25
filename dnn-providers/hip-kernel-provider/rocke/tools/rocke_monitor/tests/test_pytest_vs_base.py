# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/pytest_vs_base.py on real pytest runs: every file runs (no -x), and a failure develop shares is not the PR's.
A stop at a test that fails on develop too (pytest -x) would leave the PR's own suite unrun.
"""

import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOL = pathlib.Path(__file__).resolve().parent.parent / "tools" / "pytest_vs_base.py"
RUNNER = [sys.executable, "-I", "-B", str(TOOL)]
sys.path.insert(0, str(TOOL.parent.parent))
sys.path.insert(0, str(TOOL.parent.parent / "tests"))
from rkmon_lib import prcode  # noqa: E402
from test_prcode import (
    config_home,
)  # noqa: E402  (a monitor's home for the package's rk on a host)


# The bin folder of the checks' venv, whose pytest runs these tests: $RKMON_VENV's. A test reads no monitor's home, so
# never the configuration's paths.venv.
VENV = pathlib.Path(os.environ.get("RKMON_VENV") or "/nonexistent") / "bin"
# The job scripts': what the re-run commands are made of. A run here sets them itself.
RERUN = (prcode.RERUN_VAR, prcode.RESULTS_VAR)

OK = "def test_ok():\n    pass\n"
# file: (the PR's source, develop's source or None when the PR adds the file)
FILES = {
    "t/test_a.py": (
        OK + "\n\ndef test_old():\n    assert 0\n",
        OK + "\n\ndef test_old():\n    assert 0\n",
    ),
    "t/test_b.py": (
        "def test_regress():\n    assert 0\n",
        "def test_regress():\n    pass\n",
    ),
    "t/test_c.py": ("def test_new():\n    assert 0\n", None),
    "t/test_d.py": (OK, OK),
}


@unittest.skipUnless((VENV / "python").exists(), "needs the review venv's pytest")
class TestPytestVsBase(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.d.name)
        for name in ("pr", "dev", "out"):
            (self.base / name).mkdir()
        for rel, srcs in FILES.items():
            for tree, src in zip(("pr", "dev"), srcs):
                if src is not None:
                    (self.base / tree / rel).parent.mkdir(parents=True, exist_ok=True)
                    (self.base / tree / rel).write_text(src)

    def tearDown(self):
        self.d.cleanup()

    def run_tool(
        self, *files: str, base: bool = True, path: str = "", **extra_env: str
    ) -> list[str]:
        env = dict(
            os.environ, PATH=f"{path + ':' if path else ''}{VENV}:{os.environ['PATH']}"
        )
        for name in RERUN:
            env.pop(name, None)
        env.update(extra_env)
        r = subprocess.run(
            [
                *RUNNER,
                str(self.base / "out"),
                str(self.base / "pr"),
                str(self.base / "dev") if base else "-",
                "",
                "",
                *files,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        return r.stdout.strip().splitlines()

    def test_failures_develop_shares_are_not_the_prs(self):
        out = self.run_tool(*FILES)
        self.assertEqual(
            out[-1],
            "TESTS fail 2 failing (on the merge only): t/test_b.py::test_regress "
            "t/test_c.py::test_new",
            out,
        )
        self.assertIn(
            "   t/test_a.py::test_old", out
        )  # listed as failing on develop too

    def test_every_batch_keeps_its_full_log(self):
        """Every batch's log is kept, so the verifier reads an earlier batch's SUBFAILED labels."""
        names = [
            f"t/test_m{i:02d}.py" for i in range(11)
        ]  # two batches: 10 files, then 1
        for i, rel in enumerate(names):
            (self.base / "pr" / rel).write_text(
                f"def test_m{i:02d}():\n    assert 0, 'batch marker {i:02d}'\n"
            )
        self.run_tool(*names, base=False)
        log = (self.base / "out/tests_all.log").read_text()
        self.assertEqual(log.count("=== batch (rc=1): "), 2, log[:400])
        self.assertIn(
            "batch marker 00", log
        )  # the first batch's failure text survives the second batch
        self.assertIn("batch marker 10", log)

    def test_the_results_folder_is_made_when_missing(self):
        """A job writes its results into a folder of their own in its slot's out (rkmon_lib/review.py): made here."""
        out = self.base / "slot" / "tests"
        r = subprocess.run(
            [*RUNNER, str(out), str(self.base / "pr"), str(self.base / "dev"), "", ""]
            + ["t/test_d.py"],
            capture_output=True,
            text=True,
            env=dict(os.environ, PATH=f"{VENV}:{os.environ['PATH']}"),
            timeout=300,
        )
        last = r.stdout.strip().splitlines()[-1:]
        self.assertEqual(last, ["TESTS pass 1 passed"], r.stdout + r.stderr)
        self.assertTrue((out / "tests.log").is_file())

    def test_only_failures_develop_shares_pass(self):
        out = self.run_tool("t/test_a.py", "t/test_d.py")
        self.assertEqual(
            out[-1], "TESTS pass 2 passed, 1 failed; 1 fail on develop too", out
        )

    def test_without_a_develop_tree_every_failure_counts(self):
        out = self.run_tool("t/test_a.py", "t/test_d.py", base=False)
        self.assertEqual(
            out[-1],
            "TESTS fail 1 failing (not compared with develop): t/test_a.py::test_old",
            out,
        )

    def test_new_ids_in_an_old_file_spaces_and_subtests(self):
        """New tests in a file develop has: the develop re-run never stops at "not found" (so develop's own failures
        never read "on the merge only"), ids with spaces are kept whole, and SUBFAILED counts.
        """
        shared = (
            "import unittest\nimport pytest\n\n\ndef test_shared():\n    assert 0\n\n\n"
            "@pytest.mark.parametrize('x', ['k must have BSHD shape'])\ndef test_p(x):\n    assert 0\n\n\n"
            "class TestSub(unittest.TestCase):\n    def test_flags(self):\n        for f in ('a', 'b'):\n"
            "            with self.subTest(flag=f):\n                self.assertEqual(f, 'a')\n"
        )
        (self.base / "pr/t/test_e.py").write_text(
            shared + "\n\ndef test_pr_new():\n    assert 0\n"
        )
        (self.base / "dev/t/test_e.py").write_text(shared)
        out = self.run_tool("t/test_e.py")
        self.assertEqual(
            out[-1],
            "TESTS fail 1 failing (on the merge only): t/test_e.py::test_pr_new",
            out,
        )
        old = (self.base / "out/tests_old_failed.txt").read_text().splitlines()
        self.assertEqual(
            old,
            [
                "t/test_e.py::TestSub::test_flags",
                "t/test_e.py::test_p[k must have BSHD shape]",
                "t/test_e.py::test_shared",
            ],
        )

    def test_bracketed_subtest_labels(self):
        """pytest prints a positional subTest label in brackets ("SUBFAILED[b] <id>") and counts the parent
        test as passed; its subtest failures never read "tests pass"."""
        (self.base / "pr/t/test_f.py").write_text(
            "import unittest\n\n\nclass TestArms(unittest.TestCase):\n    def test_arms(self):\n"
            "        for arm in ('a', 'b'):\n            with self.subTest(arm):\n"
            "                self.assertEqual(arm, 'a')\n\n    def test_both(self):\n"
            "        for arm in ('a', 'b'):\n            with self.subTest('lbl', arm=arm):\n"
            "                self.assertEqual(arm, 'a')\n"
        )
        out = self.run_tool("t/test_f.py")
        self.assertEqual(
            out[-1],
            "TESTS fail 2 failing (on the merge only): t/test_f.py::TestArms::test_arms "
            "t/test_f.py::TestArms::test_both",
            out,
        )

    def test_a_failure_develop_skips_is_not_compared(self):
        """A test develop's leg skips (no engine binding), which fails on develop with the binding, is not compared:
        never "on the merge only"."""
        (self.base / "pr/t/test_j.py").write_text(
            "def test_binding():\n    assert 0\n\n\ndef test_mine():\n    assert 0\n"
        )
        (self.base / "dev/t/test_j.py").write_text(
            "import pytest\n\n\ndef test_binding():\n    pytest.skip('no binding')\n\n\ndef test_mine():\n    pass\n"
        )
        out = self.run_tool("t/test_j.py")
        self.assertEqual(
            out[-1],
            "TESTS fail 1 failing (on the merge only): t/test_j.py::test_mine",
            out,
        )
        self.assertIn("   t/test_j.py::test_binding", out)  # listed as not compared
        (self.base / "pr/t/test_j.py").write_text("def test_binding():\n    assert 0\n")
        out = self.run_tool("t/test_j.py")
        self.assertEqual(
            out[-1],
            "TESTS fail 1 failing (not compared: develop skips them): t/test_j.py::test_binding",
            out,
        )

    def test_hipcc_failures_of_a_host_that_cannot_compile_hip_are_skipped(self):
        """A host with hipcc but no HIP headers fails every hipcc case at `#include <hip/hip_runtime.h>`: those are
        skipped as the host's, never "tests fail"."""
        hipcc = self.base / "bin" / "hipcc"
        hipcc.parent.mkdir()
        hipcc.write_text(
            '#!/bin/sh\n[ -n "$FAKE_HIPCC_BROKEN" ] && { echo "\'hip/hip_runtime.h\' file not found" >&2; '
            'exit 1; }\nfor a in "$@"; do [ -f "$a" ] && grep -q BAD "$a" && exit 1; done\nexit 0\n'
        )
        hipcc.chmod(0o755)
        (self.base / "pr/t/test_h.py").write_text(
            "import os\nimport subprocess\nimport tempfile\n\n\ndef _compile(src):\n"
            "    with tempfile.TemporaryDirectory() as td:\n        p = os.path.join(td, 'k.hip')\n"
            "        with open(p, 'w') as f:\n            f.write(src)\n"
            "        subprocess.run([os.environ['FAKE_HIPCC'], '--offload-arch=gfx942', '--genco', '-o',\n"
            "                        os.path.join(td, 'k.co'), p], check=True)\n\n\n"
            "def test_good():\n    _compile('#include <hip/hip_runtime.h>\\n')\n\n\n"
            "def test_bad():\n    _compile('BAD\\n')\n"
        )
        out = self.run_tool("t/test_h.py", FAKE_HIPCC=str(hipcc), FAKE_HIPCC_BROKEN="1")
        self.assertEqual(
            out[-1],
            "TESTS pass (2 skipped: 2 rkmon: this host's hipcc cannot compile <hip/hip_runtime.h>) "
            "2 skipped; 2 hipcc failures skipped: this host's hipcc cannot compile HIP",
            out,
        )
        out = self.run_tool(
            "t/test_h.py", FAKE_HIPCC=str(hipcc)
        )  # hipcc works: the PR's bad source still fails
        self.assertEqual(
            out[-1],
            "TESTS fail 1 failing (on the merge only): t/test_h.py::test_bad",
            out,
        )
        # A helper that wraps hipcc's failure in a RuntimeError with its stderr.
        (self.base / "pr/t/test_w.py").write_text(
            "import os\nimport subprocess\nimport tempfile\n\n\ndef _compile(src):\n"
            "    with tempfile.TemporaryDirectory() as td:\n        p = os.path.join(td, 'k.hip')\n"
            "        with open(p, 'w') as f:\n            f.write(src)\n"
            "        r = subprocess.run(['hipcc', '--offload-arch=gfx942', '--genco', '-o', os.path.join(td, 'k.co'), p],\n"
            "                           capture_output=True, text=True)\n"
            "        if r.returncode:\n            raise RuntimeError('hipcc failed: ' + r.stderr)\n\n\n"
            "def test_wrapped_good():\n    _compile('#include <hip/hip_runtime.h>\\n')\n\n\n"
            "def test_wrapped_bad():\n    _compile('BAD\\n')\n"
        )
        out = self.run_tool(
            "t/test_w.py", path=str(hipcc.parent), FAKE_HIPCC_BROKEN="1"
        )
        self.assertEqual(
            out[-1],
            "TESTS pass (2 skipped: 2 rkmon: this host's hipcc cannot compile <hip/hip_runtime.h>) "
            "2 skipped; 2 hipcc failures skipped: this host's hipcc cannot compile HIP",
            out,
        )
        out = self.run_tool(
            "t/test_w.py", path=str(hipcc.parent)
        )  # no header message: the bad source still fails
        self.assertEqual(
            out[-1],
            "TESTS fail 1 failing (on the merge only): t/test_w.py::test_wrapped_bad",
            out,
        )

    def test_tests_that_import_a_module_this_host_lacks_are_skipped(self):
        """Tests that import torch directly on a host without torch (like CI's GPU lanes) are skipped, so they never
        fail the merge gate's G6 nor hold the PR."""
        (self.base / "pr/t/test_i.py").write_text(
            "def test_needs():\n    import rkmon_no_such_module  # noqa: F401\n\n\n"
            "def test_other():\n    import rkmon_other_missing  # noqa: F401\n"
        )
        out = self.run_tool("t/test_i.py", RKMON_HOST_ABSENT="rkmon_no_such_module")
        self.assertEqual(
            out[-1],
            "TESTS fail 1 failing (on the merge only): t/test_i.py::test_other; 1 skipped: they "
            "import a module this host lacks, without pytest.importorskip",
            out,
        )
        log = (self.base / "out/tests.log").read_text()
        self.assertIn("rkmon-import:   t/test_i.py::test_needs", log)

    # agent_script_overlap.md finding 4: tests that need cmake or PyYAML skipped, and the section still read as a pass.
    SKIPS = (
        "import shutil\n\nimport pytest\n\n\ndef test_ok():\n    pass\n\n\n"
        "@pytest.mark.skipif(shutil.which('cmake') is None or shutil.which('ctest') is None,\n"
        "                    reason='CMake and CTest required')\ndef test_link():\n    pass\n\n\n"
        "def test_yaml():\n    pytest.importorskip('yaml', reason='reading the tier lists needs PyYAML')\n\n\n"
        "def test_gpu():\n    pytest.skip('needs a gfx950 GPU')\n"
    )

    def fake_tools(self) -> tuple[str, str]:
        """A bin folder with cmake and ctest stand-ins, and a folder holding a stand-in `yaml` package (the checks'
        RKMON_PYEXTRA)."""
        tools, pyextra = self.base / "fake_bin", self.base / "pyshim"
        tools.mkdir()
        for name in ("cmake", "ctest"):
            (tools / name).write_text("#!/bin/sh\nexit 0\n")
            (tools / name).chmod(0o755)
        (pyextra / "yaml").mkdir(parents=True)
        (pyextra / "yaml" / "__init__.py").write_text("")
        return str(tools), str(pyextra)

    def test_every_skip_is_listed_with_its_reason_and_a_pass_says_so(self):
        (self.base / "pr/t/test_s.py").write_text(self.SKIPS)
        tools, pyextra = self.fake_tools()
        out = self.run_tool("t/test_s.py", path=tools, RKMON_PYEXTRA=pyextra)
        self.assertIn("SKIPPED 1:", out)
        self.assertIn("   t/test_s.py::test_gpu - needs a gfx950 GPU", out)
        self.assertEqual(
            out[-1],
            "TESTS pass (1 skipped: 1 needs a gfx950 GPU) 3 passed, 1 skipped",
            out,
        )

    @unittest.skipIf(
        importlib.util.find_spec("yaml") is not None, "this interpreter has PyYAML"
    )
    def test_a_skip_for_a_tool_ci_has_reads_not_run_never_a_pass(self):
        (self.base / "pr/t/test_s.py").write_text(self.SKIPS)
        out = self.run_tool("t/test_s.py")  # the venv has cmake, and no PyYAML
        self.assertIn(
            "NOT RUN: PyYAML missing (1 test; CI's lanes have it): t/test_s.py::test_yaml",
            out,
        )
        self.assertIn(
            "   t/test_s.py::test_yaml - reading the tier lists needs PyYAML [NOT RUN: PyYAML missing]",
            out,
        )
        self.assertEqual(
            out[-1],
            "TESTS skip NOT RUN: PyYAML missing (1 test); 2 passed, 2 skipped",
            out,
        )
        # Without cmake on PATH (python alone, from a folder of its own): both tools.
        alone = self.base / "py_only"
        alone.mkdir()
        (alone / "python").write_text(f'#!/bin/sh\nexec "{VENV}/python" "$@"\n')
        (alone / "python").chmod(0o755)
        env = dict(os.environ, PATH=f"{alone}:/usr/bin:/bin")
        if shutil.which("cmake", path=env["PATH"]):
            self.skipTest("this host has cmake outside the venv")
        r = subprocess.run(
            [
                *RUNNER,
                str(self.base / "out"),
                str(self.base / "pr"),
                "-",
                "",
                "",
                "t/test_s.py",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        self.assertEqual(
            r.stdout.splitlines()[-1],
            "TESTS skip NOT RUN: PyYAML missing (1 test), cmake missing "
            "(1 test); 1 passed, 3 skipped",
            r.stdout,
        )

    @unittest.skipIf(
        importlib.util.find_spec("yaml") is not None, "this interpreter has PyYAML"
    )
    def test_a_failure_says_what_did_not_run_too(self):
        (self.base / "pr/t/test_s.py").write_text(
            self.SKIPS + "\n\ndef test_bad():\n    assert 0\n"
        )
        tools, _ = self.fake_tools()
        out = self.run_tool("t/test_s.py", base=False, path=tools)
        self.assertEqual(
            out[-1],
            "TESTS fail 1 failing (not compared with develop): t/test_s.py::test_bad; "
            "NOT RUN: PyYAML missing (1 test)",
            out,
        )

    def test_each_file_has_the_exact_command_that_reruns_it(self):
        (self.base / "pr/t/test_s.py").write_text(self.SKIPS)
        (self.base / "pr/t/test_env.py").write_text(
            "import os\n\n\ndef test_the_runs_environment():\n    assert os.environ['ROCKE_LLVM_FLAVOR'] == 'llvm22'\n"
            "    assert 'ROCKE_STRAY' not in os.environ\n    import yaml  # noqa: F401  (RKMON_PYEXTRA's)\n"
        )
        tools, pyextra = self.fake_tools()
        out = self.run_tool(
            "t/test_s.py",
            "t/test_env.py",
            "t/test_gone.py",
            path=tools,
            RKMON_PYEXTRA=pyextra,
            ROCKE_LLVM_FLAVOR="llvm22",
        )
        cmds = [ln.strip() for ln in out if ln.startswith("   (. ")]
        env = self.base / "out/tests_env.sh"
        pr = self.base / "pr"
        self.assertEqual(
            cmds,
            [
                f"(. {env} && cd {pr} && python -B -m pytest -q -rfE -p no:cacheprovider --no-header "
                f"-o addopts= --continue-on-collection-errors -p rkmon_env --rootdir={pr} {f})"
                for f in ("t/test_s.py", "t/test_env.py")
            ],
        )  # a file the tree lacks has none
        # Run in a shell that has none of the run's environment, the command runs the file as the run did.
        clean = {
            "HOME": os.environ.get("HOME", tempfile.gettempdir()),
            "PATH": "/usr/bin:/bin",
            "ROCKE_STRAY": "1",
        }
        r = subprocess.run(
            ["bash", "-c", cmds[0]],
            capture_output=True,
            text=True,
            env=clean,
            timeout=300,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("3 passed, 1 skipped", r.stdout)
        self.assertIn(
            "rkmon-skip:\t-\tt/test_s.py::test_gpu\tneeds a gfx950 GPU", r.stdout
        )
        r = subprocess.run(
            ["bash", "-c", cmds[1]],
            capture_output=True,
            text=True,
            env=clean,
            timeout=300,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("1 passed", r.stdout)

    def test_through_rk_pr_run_each_command_runs_its_file_again_as_this_run_did(self):
        """With RKMON_RERUN_VIA (the job scripts' rk pr-run), the run's environment goes to tests_env.0 as rk pr-run
        --env reads it, and each command names it in the folder the job copies the results to (RKMON_RESULTS). Run as
        printed by a shell with none of the run's environment, the package's rk runs the file again in the tree, with
        that environment and nothing of the shell's."""
        (self.base / "pr/t/test_env.py").write_text(
            "import os\n\n\ndef test_the_runs_environment():\n"
            "    assert os.environ['ROCKE_LLVM_FLAVOR'] == 'llvm22'\n"
            "    assert 'ROCKE_STRAY' not in os.environ\n"
            "    assert os.environ['LC_ALL'] == 'C'\n"
        )
        results = self.base / "results"
        out = self.run_tool(
            "t/test_env.py",
            base=False,
            ROCKE_LLVM_FLAVOR="llvm22",
            RKMON_RERUN_VIA=prcode.RERUN_VIA,
            RKMON_RESULTS=str(results),
        )
        pr, env_file = self.base / "pr", results / "tests_env.0"
        self.assertIn(
            f"re-run one file as this run did (its environment: {env_file}; add -k or a test id after the file):",
            out,
        )
        cmds = [ln.strip() for ln in out if ln.startswith(f"   {prcode.RERUN_VIA} ")]
        self.assertEqual(
            cmds,
            [
                f"{prcode.RERUN_VIA} --env {env_file} -C {pr} -- python -B -m pytest -q -rfE -p no:cacheprovider "
                f"--no-header -o addopts= --continue-on-collection-errors -p rkmon_env --rootdir={pr} t/test_env.py"
            ],
        )
        data = (self.base / "out/tests_env.0").read_bytes()
        self.assertTrue(data.endswith(b"\0"))
        env = dict(x.split(b"=", 1) for x in data.split(b"\0") if x)
        self.assertEqual(
            (env[b"LC_ALL"], env[b"ROCKE_LLVM_FLAVOR"], env[b"PYTHONPATH"]),
            (b"C", b"llvm22", bytes(TOOL.parent / "pytest_plugins")),
        )
        self.assertFalse((self.base / "out/tests_env.sh").exists())
        shutil.copytree(self.base / "out", results)  # the job's `prr tests DEST`
        (self.base / "w").mkdir()
        clean = {
            "HOME": str(self.base),
            "PATH": f"{os.path.dirname(sys.executable)}:/usr/bin:/bin",
            "TMPDIR": tempfile.gettempdir(),
            "ROCKE_STRAY": "1",
            "RKMON_HOME": str(config_home(self.base, self.base / "w")),
        }
        r = subprocess.run(
            ["bash", "-c", cmds[0]],
            capture_output=True,
            text=True,
            env=clean,
            timeout=300,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("1 passed", r.stdout)


@unittest.skipUnless(
    shutil.which("bash"), "needs bash, whose quoting tests_env.sh follows"
)
class TestEnvFileQuoting(unittest.TestCase):
    """tests_env.sh is sourced by bash: its lines are bash's own `export -p` and `printf %q` (LC_ALL=C), byte for
    byte, for every byte value and the places a ~ or # is special."""

    VALUES = (
        [bytes([c]) for c in range(1, 256)]
        + [b"a" + bytes([c]) for c in range(1, 256)]
        + [
            b"~",
            b"a:~b",
            b"a=~",
            b"~/x:~y",
            b"#x",
            b"x#",
            b"x y",
            b"/p a:b",
            "héllo".encode(),
            b"a\nb",
            b"'q' \"d\" $x `y` \\z",
        ]
    )

    def setUp(self):
        sys.path.insert(0, str(TOOL.parent))
        self.addCleanup(sys.path.remove, str(TOOL.parent))
        import pytest_vs_base

        self.pv = pytest_vs_base

    def test_printf_q_is_bashs(self):
        script = b'export LC_ALL=C\nfor v in "$@"; do printf "%q\\n" "$v"; done\n'
        out = subprocess.run(
            [b"bash", b"-c", script, b"x", *self.VALUES],
            capture_output=True,
            check=True,
        ).stdout
        self.assertEqual(
            out.split(b"\n")[:-1], [self.pv.printf_q(v) for v in self.VALUES]
        )
        self.assertEqual(self.pv.printf_q(b""), b"''")

    def test_export_p_is_bashs(self):
        env = {b"RKMON_V%03d" % i: v for i, v in enumerate(self.VALUES)}
        env[b"PATH"] = b"/usr/bin:/bin"
        out = subprocess.run(
            [b"bash", b"-c", b"export LC_ALL=C; export -p"],
            env=env,
            capture_output=True,
            check=True,
        ).stdout
        bash = [x for x in out.split(b"\n") if x.startswith(b"declare -x RKMON_V")]
        self.assertEqual(
            bash,
            [
                self.pv.declare_x(k, v)
                for k, v in sorted(env.items())
                if k.startswith(b"RKMON_V")
            ],
        )


if __name__ == "__main__":
    unittest.main()
