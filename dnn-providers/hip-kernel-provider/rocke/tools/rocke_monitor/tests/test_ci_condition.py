# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/ci_condition.py, the checks' CI-condition run: each changed test file run as CI's test environment runs it
(no torch, no built engines, ROCm's tools hidden: pytest_plugins/rkmon_ci.py and rkmon_norocm.py). "0 of n run" and a
failure only there are leads; a skip in a tier or contract test reads NOT RUN. The environment, the report, a run on a
small tree, and the checks wiring."""

import contextlib
import io
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import ci_condition as cc  # noqa: E402
from rkmon_lib import review  # noqa: E402


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


class TestEnvironment(unittest.TestCase):
    def test_rocm_folders_leave_path_and_the_engines_leave_pythonpath(self):
        with tempfile.TemporaryDirectory() as d:
            rocm_bin, plain = pathlib.Path(d, "rocm", "bin"), pathlib.Path(d, "bin")
            rocm_bin.mkdir(parents=True)
            plain.mkdir()
            (rocm_bin / "llvm-objdump").write_text("")
            tree = pathlib.Path(d, "K")
            (tree / "platform" / "python").mkdir(parents=True)
            env, dropped = cc.ci_env(
                tree,
                {"torch"},
                {
                    "PATH": f"{rocm_bin}:{plain}",
                    "PYTHONPATH": "/build/pybind",
                    "ROCKE_LLVM_FLAVOR": "llvm22",
                    "ROCM_PATH": str(pathlib.Path(d, "rocm")),
                    "HOME": "/h",
                },
            )
        self.assertEqual(env["PATH"], str(plain))
        self.assertEqual(dropped, [str(rocm_bin)])
        self.assertNotIn("/build/pybind", env["PYTHONPATH"])
        self.assertIn(str(tree / "platform" / "python"), env["PYTHONPATH"])
        self.assertNotIn("ROCKE_LLVM_FLAVOR", env)
        self.assertNotIn("ROCM_PATH", env)
        self.assertIn(str(pathlib.Path(d, "rocm")), env["RKMON_NOROCM_HIDE"])
        self.assertEqual(env["RKMON_CI_ABSENT"], "torch")


class TestReport(unittest.TestCase):
    def test_zero_run_failures_and_tier_skips(self):
        got = cc.parse(
            "rkmon-ci-result:\tskipped\tt/test_a.py\tcould not import 'torch'\n"
            "rkmon-ci-result:\tpassed\tt/test_b.py::test_x\t\n"
            "rkmon-ci-result:\tfailed\tt/test_b.py::test_y\t\n"
        )
        self.assertEqual((got["passed"], got["failed"]), (1, ["t/test_b.py::test_y"]))
        line, leads, tier = cc.describe(
            "t/test_a.py",
            {"passed": 0, "failed": [], "skipped": got["skipped"], "errors": []},
        )
        self.assertTrue(
            line.startswith(
                "t/test_a.py: 0 of 1 run, 1 skip (1× could not import 'torch')"
            )
        )
        self.assertEqual(
            leads,
            ["0 of 1 run in CI's condition: t/test_a.py (1× could not import 'torch')"],
        )
        self.assertFalse(tier)
        _line, leads, tier = cc.describe(
            "t/test_tier_contract.py", dict(got, errors=[])
        )
        self.assertTrue(tier)  # a skip in a tier or contract test reads NOT RUN
        self.assertIn("1 fail in CI's condition", leads[0])


@unittest.skipUnless(_python(), "the run needs a python with pytest")
class TestRun(unittest.TestCase):
    def test_a_small_tree(self):
        with tempfile.TemporaryDirectory() as d:
            tree = pathlib.Path(d, "K")
            tests = tree / "platform" / "tests"
            tests.mkdir(parents=True)
            fake_rocm = pathlib.Path(d, "rocm")
            (fake_rocm / "bin").mkdir(parents=True)
            (fake_rocm / "bin" / "llvm-objdump").write_text("#!/bin/sh\n")
            (tests / "pytest.ini").write_text("[pytest]\n")
            files = {
                "test_plain.py": "def test_x():\n    assert 1\n",
                "test_torchy.py": "import pytest\n\ntorch = pytest.importorskip('torch')\n\n\ndef test_t():\n"
                "    pass\n",
                "test_tool.py": textwrap.dedent(
                    f"""\
                    import os, shutil, pytest

                    def test_found_by_path_lookup():
                        if not shutil.which("llvm-objdump"):
                            pytest.skip("llvm-objdump not found")

                    def test_found_by_a_fixed_path():
                        assert os.path.exists("{fake_rocm}/bin/llvm-objdump")
                    """
                ),
                "test_tier_contract.py": textwrap.dedent(
                    """\
                    import pytest

                    @pytest.mark.skip(reason="later")
                    def test_c():
                        pass

                    def test_d():
                        pass
                    """
                ),
            }
            for name, text in files.items():
                (tests / name).write_text(text)
            out = io.StringIO()
            env = {
                "PATH": f"{fake_rocm / 'bin'}:{os.environ.get('PATH', '')}",
                "ROCM_PATH": str(fake_rocm),
            }
            old = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                with contextlib.redirect_stdout(out):
                    cc.main(
                        [
                            "--tree",
                            str(tree),
                            "--python",
                            _python(),
                            "--absent",
                            "torch",
                        ]
                        + [f"platform/tests/{n}" for n in files]
                        + ["platform/tests/test_gone.py"]
                    )
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        text = out.getvalue()
        self.assertIn("platform/tests/test_plain.py: 1 of 1 run", text)
        self.assertIn(
            "LEAD 0 of 1 run in CI's condition: platform/tests/test_torchy.py", text
        )
        # The tool on PATH and the fixed /opt/rocm-style path are both hidden.
        self.assertIn(
            "platform/tests/test_tool.py: 1 of 2 run, 1 skip (1× llvm-objdump not found), 1 fail",
            text,
        )
        self.assertIn("platform/tests/test_tier_contract.py: 1 of 2 run, 1 skip", text)
        last = text.strip().splitlines()[-1]
        self.assertTrue(last.startswith("CICOND fail 2 leads: 0 of 1 run"), last)
        self.assertIn("NOT RUN: platform/tests/test_tier_contract.py", last)
        self.assertNotIn("test_gone", text)  # a file the tree lacks is not run


class TestChecksWiring(unittest.TestCase):
    CFG = {
        "rocke_path": "rk",
        "repo": "o/r",
        "paths": {"working_dir": "/w"},
        "hot_spots": {"RG": "rk/g.json", "DG": "rk/d.json"},
        "ci_absent_modules": ["torch", "mpmath"],
    }

    def test_a_review_runs_it_on_the_changed_tests_and_a_gate_does_not(self):
        plan = review.plan_checks(
            self.CFG, {"n": 7}, ["rk/platform/tests/test_foo.py"], {}
        )
        plan["_dirs"] = {}
        script = review.checks_script(self.CFG, 7, "a" * 40, "first-round", plan, None)
        sec = script.split(f"sec '{review.CICOND_SECTION}", 1)[1].split("\nsec ", 1)[0]
        self.assertIn(
            "python3 -I -B $M/tools/ci_condition.py --tree $W/$K --python python --absent torch,mpmath "
            f"--budget {review.CICOND_BUDGET_S} platform/tests/test_foo.py",
            sec,
        )
        self.assertIn("st cicond ", sec)
        self.assertIn("cicond", review.LEAD_SECTIONS)
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True
            ).returncode,
            0,
        )
        gate = review.checks_script(
            self.CFG, 7, "a" * 40, "first-round", plan, None, mode="gate"
        )
        self.assertNotIn(review.CICOND_SECTION, gate)


if __name__ == "__main__":
    unittest.main()
