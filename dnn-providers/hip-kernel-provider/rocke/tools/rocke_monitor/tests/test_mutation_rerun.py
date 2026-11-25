# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The mutation sweep's survivors re-run wider (lessons L-598): against every test file importing the module, a file
failing unmutated dropped (not the set), first under CI's python, then under a python with torch, with the engine build
on the path: killed by the CI importers, `vacuous: torch-only path`, or `SURVIVED (CI importers, torch, engine)`.
A temp git repo; "torch" is a stub module only the torch python's path holds. No network.
"""

import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import mutation_sweep as ms  # noqa: E402
from rkmon_lib import review  # noqa: E402


def g(repo, *a) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


MOD = textwrap.dedent(
    """\
    def f(x):
        if x < 0 or x == 42 or x == 7 or x == 99:
            raise ValueError("out of range")
        return x
    """
)
SMALL = "from pkg.mod import f\n\n\ndef test_s():\n    assert f(1) == 1\n"


def catcher(n: int, pad: int, head: str = "") -> str:
    return (
        "import pytest\n"
        + head
        + "\nfrom pkg.mod import f\n\n\ndef test_c():\n    with pytest.raises(ValueError):\n"
        + f"        f({n})\n"
        + "# padding\n" * pad
    )


class TestRerun(unittest.TestCase):
    def setUp(self):
        if (
            not subprocess.run(
                [sys.executable, "-c", "import pytest"], capture_output=True
            ).returncode
            == 0
        ):
            self.skipTest("the sweep runs pytest")
        if (
            subprocess.run(
                [sys.executable, "-c", "import torch"], capture_output=True
            ).returncode
            == 0
        ):
            self.skipTest("this python has torch: the CI run would not be torch-less")
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        d = pathlib.Path(self.d.name)
        r = self.repo = d / "repo"
        t = r / "rk/platform/tests"
        (r / "rk/platform/python/pkg").mkdir(parents=True)
        t.mkdir(parents=True)
        g(r, "init", "-q", "-b", "develop")
        (r / "rk/platform/python/pkg/__init__.py").write_text("")
        (r / "rk/platform/python/pkg/mod.py").write_text("def f(x):\n    return x\n")
        for i in range(
            6
        ):  # the sweep's own set takes the 6 smallest importers: none catches 42, 7 or 99
            (t / f"test_s{i}.py").write_text(SMALL + "#\n" * i)
        # develop-known: fails unmutated; the wider set drops it, not the set
        (t / "test_broken.py").write_text(
            "from pkg.mod import f\n\n\ndef test_b():\n    assert False\n"
            + "# pad\n" * 30
        )
        (t / "test_torchy.py").write_text(
            catcher(7, 60, 'torch = pytest.importorskip("torch")\n')
        )
        (t / "test_wide.py").write_text(catcher(42, 90))
        (t / "test_mod.py").write_text("def test_nothing():\n    pass\n")
        g(r, "add", "-A")
        g(r, "commit", "-qm", "base")
        g(r, "checkout", "-qb", "pr")
        (r / "rk/platform/python/pkg/mod.py").write_text(MOD)
        (t / "test_mod.py").write_text(catcher(-1, 0))
        g(r, "add", "-A")
        g(r, "commit", "-qm", "pr")
        self.head = g(r, "rev-parse", "HEAD")
        stub = d / "torchstub"
        stub.mkdir()
        (stub / "torch.py").write_text("")
        self.torch_py = d / "torch_python"
        self.torch_py.write_text(
            f'#!/bin/sh\nPYTHONPATH="$PYTHONPATH:{stub}" exec "{sys.executable}" "$@"\n'
        )
        self.torch_py.chmod(0o755)
        self.engine = d / "pybind"
        self.engine.mkdir()
        self.tmp = d / "tmp"
        self.tmp.mkdir()

    def sweep(self, **kw) -> dict:
        with mock.patch.dict(os.environ, {"TMPDIR": str(self.tmp)}):
            res = ms.sweep(
                str(self.repo),
                self.head,
                pathlib.Path(self.d.name) / "out",
                K="rk",
                base="develop",
                python=sys.executable,
                per_mutant=120,
                budget=600,
                **kw,
            )
        self.assertIsNone(res["error"])
        self.assertTrue(res["restored"])
        self.assertEqual(g(self.repo, "status", "--porcelain"), "")
        return res

    def test_killed_by_the_ci_importers_vacuous_under_torch_and_survived(self):
        res = self.sweep(torch_python=str(self.torch_py), engine=str(self.engine))
        got = {m["what"]: (m["state"], m.get("rerun")) for m in res["mutants"]}
        self.assertEqual(
            got,
            {
                "x < 0": ("killed", None),
                "x == 42": ("killed", "killed by the CI importers"),
                "x == 7": ("survived", "vacuous: torch-only path"),
                "x == 99": ("survived", "SURVIVED (CI importers, torch, engine)"),
            },
        )
        ci = res["rerun"]["sets"]["platform/python/pkg/mod.py (CI importers)"]
        self.assertEqual(ci["dropped"], ["platform/tests/test_broken.py"])
        self.assertIsNone(ci["why"])
        lines = ms.render(res, "rk", ms.CAP)
        text = "\n".join(lines)
        self.assertIn("  vacuous: torch-only path: platform/python/pkg/mod.py:2", text)
        self.assertIn(
            "  SURVIVED (CI importers, torch, engine): platform/python/pkg/mod.py:2",
            text,
        )
        self.assertIn("(by the CI importers' re-run)", text)
        self.assertIn("dropped (fail unmutated): platform/tests/test_broken.py", text)
        self.assertEqual(
            ms.verdict(res),
            (
                "fail",
                "2 of 4 mutants survive (checks no test catches): mod.py:2, mod.py:2; "
                "1 only under torch (vacuous in CI)",
            ),
        )

    def test_without_a_torch_python_the_re_run_names_what_ran(self):
        res = self.sweep()
        got = {m["what"]: (m["state"], m.get("rerun")) for m in res["mutants"]}
        self.assertEqual(got["x == 42"], ("killed", "killed by the CI importers"))
        self.assertEqual(got["x == 7"], ("survived", "SURVIVED (CI importers)"))
        self.assertIn("no torch python here", "\n".join(ms.render(res, "rk", ms.CAP)))
        # No re-run at all with a budget of 0: the survivors stand as the sweep found them.
        res = self.sweep(rerun_budget=0)
        self.assertNotIn("rerun", res)
        self.assertEqual(
            sorted(m["state"] for m in res["mutants"]),
            ["killed", "survived", "survived", "survived"],
        )


class TestWiring(unittest.TestCase):
    CFG = {
        "rocke_path": "rk",
        "repo": "o/r",
        "paths": {"working_dir": "/w"},
        "hot_spots": {"RG": "rk/g.json", "DG": "rk/d.json"},
    }

    def script(self, cfg):
        plan = review.plan_checks(cfg, {"n": 7}, ["rk/platform/python/pkg/mod.py"], {})
        plan["_dirs"] = {}
        line = next(
            ln
            for ln in review.checks_script(
                cfg, 7, "a" * 40, "first-round", plan, None
            ).splitlines()
            if "mutation_sweep.py" in ln
        )
        return line

    def test_the_checks_pass_the_engine_and_the_torch_python(self):
        line = self.script(self.CFG)
        self.assertIn(
            f"--rerun-budget {review.MUTANTS_RERUN_S} --engine $SB/${{WTN}}/pybind",
            line,
        )
        self.assertIn(
            f"timeout {review.MUTANTS_BUDGET_S + review.MUTANTS_RERUN_S + 300} ", line
        )
        self.assertNotIn("--torch-python", line)
        cfg = dict(
            self.CFG,
            paths={"working_dir": "/w", "torch_python": "/opt/venv-torch/bin/python"},
        )
        self.assertIn("--torch-python /opt/venv-torch/bin/python", self.script(cfg))


if __name__ == "__main__":
    unittest.main()
