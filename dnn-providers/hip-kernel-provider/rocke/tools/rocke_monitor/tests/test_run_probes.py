# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/run_probes.py: saved probes of our open findings, re-run on a follow-up's tree (lessons L-525)."""

import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
import unittest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "run_probes.py"
RUNNER = [sys.executable, "-B", str(TOOL)]
# The python of the probes' torch venv (<paths.root>/venv-torch): $RKMON_TORCH_PY. A test reads no monitor's home, so
# never the configuration's paths.root.
TORCH_PY = pathlib.Path(os.environ.get("RKMON_TORCH_PY") or "/nonexistent")


def run(probe_dir, tree):
    return subprocess.run(
        [*RUNNER, str(probe_dir), str(tree)], capture_output=True, text=True, timeout=60
    ).stdout


def echo_probe(folder: pathlib.Path, line: str) -> str:
    """A probe that echoes LINE (in shell double quotes) and exits 0, with FOLDER, the test's, written as D in it:
    run_probes cuts a summary at 200 bytes, and the line reads the same whatever the temporary folder's path is.
    """
    return f'd={shlex.quote(str(folder))}\nline="{line}"\necho "${{line//"$d"/D}}"\nexit 0\n'


class RunProbes(unittest.TestCase):
    def test_each_probe_reads_as_reproduces_fixed_or_error_with_its_last_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            (p / "probes").mkdir()
            (p / "tree").mkdir()
            (p / "probes" / "N1_still.sh").write_text(
                'echo "checking $1"\necho "10 cohorts differ"\nexit 1\n'
            )
            (p / "probes" / "N2_fixed.sh").write_text(
                'echo "0 cohorts differ"\nexit 0\n'
            )
            (p / "probes" / "N3_broken.sh").write_text(
                'echo "ImportError: no module"\nexit 3\n'
            )
            (p / "probes" / "helper.py").write_text("print('not a probe')\n")
            out = run(p / "probes", p / "tree").splitlines()
        self.assertEqual(
            out,
            [
                "N1_still: REPRODUCES: 10 cohorts differ",
                "N2_fixed: FIXED: 0 cohorts differ",
                "N3_broken: ERROR rc=3: ImportError: no module",
            ],
        )

    def test_every_probe_gets_the_environment_contract(self):
        """A probe names no machine's folder: the checks venv's python, ROCm's LLVM tools, the tree's native build and
        the repository come from the job's settings, under the contract's names."""
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            (p / "probes").mkdir()
            (p / "wt7m").mkdir()
            (p / "venv" / "bin").mkdir(parents=True)
            (p / "venv" / "bin" / "python").write_text("")
            (p / "probes" / "N1_env.sh").write_text(
                echo_probe(
                    p,
                    "$PROBE_PYTHON|$RKMON_ROCM_LLVM_BIN|$RKMON_BUILD_DIR|$RKMON_REPO|$RKMON_REPO_DIR",
                )
            )
            env = dict(
                os.environ,
                RKMON_VENV=str(p / "venv"),
                RKMON_LLVM_BIN="/opt/rocm/core-x/lib/llvm/bin",
                RKMON_BUILD=str(p / "build"),
                RKMON_REPO="o/r",
                RKMON_REPO_DIR=str(p / "repo"),
            )
            out = subprocess.run(
                [*RUNNER, str(p / "probes"), str(p / "wt7m")],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            ).stdout
            self.assertEqual(
                out.strip(),
                "N1_env: FIXED: D/venv/bin/python|/opt/rocm/core-x/lib/llvm/bin|D/build/wt7m|o/r|D/repo",
            )
            env.pop("RKMON_VENV")
            self.assertEqual(
                subprocess.run(
                    [*RUNNER, str(p / "probes"), str(p / "wt7m")],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=env,
                ).stdout.split("|")[0],
                "N1_env: FIXED: python3",
            )

    def test_a_probe_runs_clean_and_a_run_of_no_test_is_no_verdict(self):
        """A caller's GPU mask or forced LLVM flavor never reaches a probe (CLEAN); pytest's rc 5 (no test ran) reads as
        an ERROR that says so."""
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            (p / "probes").mkdir()
            (p / "tree").mkdir()
            (p / "probes" / "N1_env.sh").write_text(
                echo_probe(
                    p, "${HIP_VISIBLE_DEVICES-unset}|${ROCKE_LLVM_FLAVOR-unset}|$1"
                )
            )
            (p / "probes" / "N2_none.sh").write_text('echo "no tests ran"\nexit 5\n')
            out = subprocess.run(
                [*RUNNER, str(p / "probes"), str(p / "tree")],
                capture_output=True,
                text=True,
                timeout=60,
                env={
                    **os.environ,
                    "HIP_VISIBLE_DEVICES": "-1",
                    "ROCKE_LLVM_FLAVOR": "llvm22",
                },
            ).stdout.splitlines()
        self.assertEqual(
            out,
            [
                "N1_env: FIXED: unset|unset|D/tree",
                "N2_none: ERROR rc=5 (no test ran): no tests ran",
            ],
        )

    def test_the_detail_gives_the_exit_code_the_trees_commit_and_the_time(self):
        """The checks' re-probed section (PROBE_DETAIL): a reader trusts the line without a re-run."""
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            (p / "probes").mkdir()
            (p / "tree").mkdir()
            subprocess.run(["git", "-C", str(p / "tree"), "init", "-q"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(p / "tree"),
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@t",
                    "commit",
                    "-q",
                    "--allow-empty",
                    "-m",
                    "x",
                ],
                check=True,
            )
            sha = subprocess.run(
                ["git", "-C", str(p / "tree"), "rev-parse", "--short=10", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            (p / "probes" / "N1_still.sh").write_text(
                'echo "10 cohorts differ"\nexit 1\n'
            )
            (p / "probes" / "N2_broken.sh").write_text(
                'echo "ImportError: no module"\nexit 3\n'
            )
            out = subprocess.run(
                [*RUNNER, str(p / "probes"), str(p / "tree")],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "PROBE_DETAIL": "1"},
            ).stdout.splitlines()
        self.assertRegex(
            out[0],
            rf"^N1_still: REPRODUCES rc=1 \(tree {sha}, \d+ s\): 10 cohorts differ$",
        )
        self.assertRegex(
            out[1],
            rf"^N2_broken: ERROR rc=3 \(tree {sha}, \d+ s\): ImportError: no module$",
        )

    def test_a_probe_that_crashed_is_an_error_not_reproduces(self):
        # An uncaught Python exception exits 1 too (a probe that crashes on a missing module is no
        # REPRODUCES). A deliberate assert, or a summary that names an exception, still reads as REPRODUCES.
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            (p / "probes").mkdir()
            (p / "tree").mkdir()
            (p / "probes" / "N1_crash.sh").write_text(
                "python3 -c 'import no_such_module_rk'\n"
            )
            (p / "probes" / "N2_assert.sh").write_text(
                "python3 -c 'assert 1 == 2, \"cohorts differ\"'\n"
            )
            (p / "probes" / "N3_named.sh").write_text(
                'echo "ValueError: the refusal names tile 64"\nexit 1\n'
            )
            out = run(p / "probes", p / "tree").splitlines()
        self.assertEqual(
            out,
            [
                "N1_crash: ERROR rc=1 (crashed): ModuleNotFoundError: No module named 'no_such_module_rk'",
                "N2_assert: REPRODUCES: AssertionError: cohorts differ",
                "N3_named: REPRODUCES: ValueError: the refusal names tile 64",
            ],
        )

    def test_the_probe_gets_the_tree(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            (p / "probes").mkdir()
            (p / "tree").mkdir()
            (p / "tree" / "marker").write_text("x")
            (p / "probes" / "N1.sh").write_text(
                '[ -e "$1/marker" ] && { echo "saw the tree"; exit 1; }\nexit 2\n'
            )
            self.assertEqual(
                run(p / "probes", p / "tree").strip(), "N1: REPRODUCES: saw the tree"
            )

    def test_one_probe_by_name(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            (p / "probes").mkdir()
            (p / "tree").mkdir()
            (p / "probes" / "N1_a.sh").write_text('echo "a still"\nexit 1\n')
            (p / "probes" / "N2_b.sh").write_text('echo "b fixed"\nexit 0\n')
            out = subprocess.run(
                [*RUNNER, str(p / "probes"), str(p / "tree"), "N2_b"],
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout
            self.assertEqual(out.strip(), "N2_b: FIXED: b fixed")
            out = subprocess.run(
                [*RUNNER, str(p / "probes"), str(p / "tree"), "N9_none"],
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout
            self.assertEqual(out.strip(), "no saved probe")

    def test_a_probe_runs_without_hip_visible_devices_and_with_no_gpu_shown(self):
        """A caller's HIP_VISIBLE_DEVICES=-1 makes this host's HIP runtime abort (rc 134) once a probe imports torch;
        the runner drops it and hides the GPU with an empty CUDA_VISIBLE_DEVICES."""
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            (p / "probes").mkdir()
            (p / "tree").mkdir()
            (p / "probes" / "N1_env.sh").write_text(
                'echo "HIP=${HIP_VISIBLE_DEVICES-unset} CUDA=[${CUDA_VISIBLE_DEVICES-unset}]"\nexit 1\n'
            )
            out = subprocess.run(
                [*RUNNER, str(p / "probes"), str(p / "tree")],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "HIP_VISIBLE_DEVICES": "-1"},
            ).stdout
        self.assertEqual(out.strip(), "N1_env: REPRODUCES: HIP=unset CUDA=[]")

    @unittest.skipUnless(
        TORCH_PY.exists(), "needs the probes' torch venv's python ($RKMON_TORCH_PY)"
    )
    def test_a_probe_that_imports_torch_runs_under_the_checks_environment(self):
        """A probe that imports torch under a caller's HIP_VISIBLE_DEVICES=-1 runs on the CPU, not an abort."""
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            (p / "probes").mkdir()
            (p / "tree").mkdir()
            (p / "probes" / "N1_torch.sh").write_text(
                f"{TORCH_PY} -c 'import torch; assert not torch.cuda.is_available(); "
                'print("cpu sum", int(torch.ones(3).sum()))\' || exit 3\nexit 0\n'
            )
            out = subprocess.run(
                [*RUNNER, str(p / "probes"), str(p / "tree")],
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ, "HIP_VISIBLE_DEVICES": "-1"},
            ).stdout
        self.assertEqual(out.strip(), "N1_torch: FIXED: cpu sum 3")

    def test_no_probe_or_no_tree_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            self.assertEqual(run(p / "absent", p).strip(), "no saved probe")
            (p / "probes").mkdir()
            (p / "probes" / "N1.sh").write_text("exit 1\n")
            self.assertEqual(
                run(p / "probes", p / "no_tree").strip(),
                f"no tree to probe: {p / 'no_tree'}",
            )

    def test_a_probe_past_its_time_is_an_error_and_the_runner_exits_0(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            (p / "probes").mkdir()
            (p / "tree").mkdir()
            (p / "probes" / "N1_slow.sh").write_text(
                "echo started\nsleep 5\necho never\n"
            )
            r = subprocess.run(
                [*RUNNER, str(p / "probes"), str(p / "tree")],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "PROBE_TIMEOUT": "1"},
            )
        self.assertEqual(
            (r.returncode, r.stdout), (0, "N1_slow: ERROR rc=124: started\n")
        )

    def test_the_summary_is_the_last_line_cut_at_200_bytes(self):
        """The last line as `echo "$out" | tail -1 | cut -c1-200` gives it: 200 bytes, a lone echo option prints
        nothing, a NUL byte is dropped; the probes run in the locale's collation order.
        """
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            (p / "probes").mkdir()
            (p / "tree").mkdir()
            (p / "probes" / "N1_long.sh").write_text(
                "python3 -c \"print('a' * 250)\"\nexit 1\n"
            )
            (p / "probes" / "N2_dash.sh").write_text(
                "echo first\nprintf -- '-n\\n'\nexit 1\n"
            )
            (p / "probes" / "N3_nul.sh").write_text("printf 'a\\0b\\n'\nexit 0\n")
            (p / "probes" / "N10_order.sh").write_text("exit 0\n")
            out = run(p / "probes", p / "tree").splitlines()
        self.assertEqual(
            out,
            [
                "N10_order: FIXED: ",
                "N1_long: REPRODUCES: " + "a" * 200,
                "N2_dash: REPRODUCES: ",
                "N3_nul: FIXED: ab",
            ],
        )


if __name__ == "__main__":
    unittest.main()
