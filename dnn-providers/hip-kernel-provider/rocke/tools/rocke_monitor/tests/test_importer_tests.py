# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/importer_tests.py and the checks' section that runs it: the tests the PR does not change that import a Python
module it changes (a failure in an unchanged test shows in the checks, not only in CI's logs). Temp trees, no
network."""

import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "importer_tests.py"
sys.path.insert(0, str(TOOL.parent))
sys.path.insert(0, str(TOOL.parent.parent))
import importer_tests as it  # noqa: E402
from rkmon_lib import prcode, review  # noqa: E402

FILES = {
    "platform/python/rocke/core/foo.py": "X = 1\n",
    "platform/python/rocke/core/__init__.py": "",
    "library/kernels/common/conv.py": "Z = 1\n",
    "platform/tests/test_a.py": "from rocke.core.foo import X\n",
    "platform/tests/test_b.py": "from rocke.core import (\n    bar,\n    foo,\n)\n"
    + "# padding\n" * 20,
    "platform/tests/test_c.py": "import rocke.core.foo as f\nfrom kernels.common.conv import Z\n"
    + "# pad\n" * 40,
    "platform/tests/test_d.py": "from rocke.core.foobar import y\nimport rocke.core.foobar\n",  # another module
    "platform/tests/test_own.py": "from rocke.core.foo import X\n",  # the PR's own: it ran already
    "library/tests/test_conv_x.py": "from kernels.common.conv import Z\n",
    "library/tests/helper.py": "from rocke.core.foo import X\n",  # not a test file
}


class TestImporters(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.tree = pathlib.Path(self.d.name) / "rocke"
        for path, text in FILES.items():
            (self.tree / path).parent.mkdir(parents=True, exist_ok=True)
            (self.tree / path).write_text(text)

    def test_module_names(self):
        self.assertEqual(
            it.dotted("platform/python/rocke/core/foo.py"), ("rocke.core.foo", False)
        )
        self.assertEqual(
            it.dotted("platform/python/rocke/core/__init__.py"), ("rocke.core", True)
        )
        self.assertEqual(
            it.dotted("library/kernels/common/conv.py"), ("kernels.common.conv", False)
        )
        for path in (
            "platform/tests/test_a.py",
            "library/tests/conftest.py",
            "tools/x.py",
            "README.md",
        ):
            self.assertIsNone(it.dotted(path), path)

    def test_the_importers_most_changed_modules_first_then_the_smallest_and_the_cap(
        self,
    ):
        mods = ["platform/python/rocke/core/foo.py", "library/kernels/common/conv.py"]
        sel = it.select(self.tree, mods, {"platform/tests/test_own.py"}, cap=3)
        self.assertEqual(
            sel["chosen"],
            [
                "platform/tests/test_c.py",
                "platform/tests/test_a.py",
                "library/tests/test_conv_x.py",
            ],
        )
        self.assertEqual(sel["rest"], ["platform/tests/test_b.py"])
        self.assertEqual(sel["by_file"]["platform/tests/test_c.py"], mods)
        lines = it.report(sel, 3)
        self.assertEqual(
            lines[1],
            "test files the PR does not change that import them: 4; run 3 (cap 3), the most "
            "changed modules first, then the smallest",
        )
        self.assertEqual(
            lines[-1], "NOT RUN (over the cap, 1): platform/tests/test_b.py"
        )
        # A changed package __init__: `from rocke.core import ...` reaches it, a submodule's import does not.
        init = it.select(self.tree, ["platform/python/rocke/core/__init__.py"], set())
        self.assertEqual(init["chosen"], ["platform/tests/test_b.py"])

    def test_the_cli_writes_the_list(self):
        out = pathlib.Path(self.d.name) / "files.txt"
        p = subprocess.run(
            [
                sys.executable,
                "-I",
                str(TOOL),
                "--tree",
                str(self.tree),
                "--list",
                str(out),
                "--cap",
                "12",
                "--exclude",
                "platform/tests/test_own.py",
                "--",
                "platform/python/rocke/core/foo.py",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            out.read_text().split(),
            [
                "platform/tests/test_a.py",
                "platform/tests/test_b.py",
                "platform/tests/test_c.py",
            ],
        )
        self.assertTrue(
            p.stdout.startswith(
                "changed modules: 1 (platform/python/rocke/core/foo.py)"
            ),
            p.stdout,
        )


class TestChecksWiring(unittest.TestCase):
    CFG = {
        "rocke_path": "rk",
        "repo": "o/r",
        "paths": {"working_dir": "/w"},
        "hot_spots": {"RG": "rk/g.json", "DG": "rk/d.json"},
    }

    def test_the_plan_names_the_changed_modules_and_the_script_runs_their_importers(
        self,
    ):
        files = [
            "rk/platform/python/rocke/core/foo.py",
            "rk/platform/tests/test_foo.py",
            "rk/library/tests/conftest.py",
            "rk/library/kernels/common/conv.py",
            "rk/docs/x.md",
        ]
        plan = review.plan_checks(self.CFG, {"n": 7}, files, {})
        self.assertEqual(
            plan["modules"],
            ["library/kernels/common/conv.py", "platform/python/rocke/core/foo.py"],
        )
        plan["_dirs"] = {}
        script = review.checks_script(self.CFG, 7, "a" * 40, "first-round", plan, None)
        sec = script.split(f"sec '{review.IMPORTERS_SECTION}", 1)[1].split("\nsec ", 1)[
            0
        ]
        self.assertIn(
            f"--cap {review.IMPORTERS_CAP} --exclude platform/tests/test_foo.py -- "
            "library/kernels/common/conv.py platform/python/rocke/core/foo.py",
            sec,
        )
        self.assertIn(
            f"prc timeout {review.IMPORTERS_BUDGET_S} python3 -I -B $M/tools/pytest_vs_base.py $SO/importers "
            "$W/$K $TB",
            sec,
        )
        self.assertIn(
            'prr importers "$OUT/importers";', sec
        )  # the results copied back, as data
        self.assertIn("st importers ", sec)
        self.assertIn(
            "importers", review.LEAD_SECTIONS
        )  # a lead: it flags no merge-only head
        # on the merged tree only
        self.assertGreater(
            script.index(review.IMPORTERS_SECTION),
            script.index('if [ "$MERGED" = 1 ]; then'),
        )
        self.assertLess(
            script.index(review.IMPORTERS_SECTION),
            script.index("sec 'Build, goldens, byte identity"),
        )
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True
            ).returncode,
            0,
        )
        none = review.plan_checks(
            self.CFG, {"n": 7}, ["rk/platform/tests/test_foo.py"], {}
        )
        none["_dirs"] = {}
        self.assertNotIn(
            review.IMPORTERS_SECTION,
            review.checks_script(self.CFG, 7, "a" * 40, "first-round", none, None),
        )

    def test_the_section_runs_in_bash(self):
        """The section's lines on a tree with one importer: the list, then pytest_vs_base.py's result line."""
        if subprocess.run(
            [sys.executable, "-c", "import pytest"], capture_output=True
        ).returncode:
            self.skipTest("the section runs pytest")
        plan = review.plan_checks(
            self.CFG, {"n": 7}, ["rk/platform/python/rocke/core/foo.py"], {}
        )
        plan["_dirs"] = {}
        script = review.checks_script(self.CFG, 7, "a" * 40, "first-round", plan, None)
        sec = (
            script.split(f"sec '{review.IMPORTERS_SECTION}", 1)[1]
            .split("\nsec ", 1)[0]
            .split("\n", 1)[1]
        )
        sec = sec.rstrip()
        if sec.endswith(
            "\nelse"
        ):  # the enclosing if's else, which follows the section in the whole script
            sec = sec[: -len("\nelse")]
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            tree = d / "W/rk"
            for path, text in (
                ("platform/python/rocke/__init__.py", ""),
                ("platform/python/rocke/core/__init__.py", ""),
                ("platform/python/rocke/core/foo.py", "X = 1\n"),
                (
                    "platform/tests/test_a.py",
                    "from rocke.core.foo import X\n\n\ndef test_x():\n"
                    "    assert X == 1\n",
                ),
            ):
                (tree / path).parent.mkdir(parents=True, exist_ok=True)
                (tree / path).write_text(text)
            pre = (
                prcode.SHELL
                + "\n"
                + textwrap.dedent(
                    f"""\
                D={d}; K=rk; N=7; WTN=W; W=$D/W; P=$W/$K/platform; L=$W/$K/library; M={TOOL.parent.parent}
                OUT=$D/out; mkdir -p $OUT; : > $OUT/status; BW=$D/none; BR=''
                TMPDIR=$D; prslot || exit 9
                st() {{ echo "$1 $2 $3" >> $OUT/status; }}
                base_tree() {{ return 1; }}
                {review.RESULT_FN}
                {review.U8_FN}
                """
                )
            )
            # `python` is the checks' venv's; this interpreter stands in for it. A script that execs it, not a link: a
            # venv's python found through a link elsewhere does not find its pyvenv.cfg, so it would run the base
            # Python, which has no pytest where pytest is only in the venv (the image's checks venv).
            shim = d / "bin"
            shim.mkdir()
            (shim / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
            (shim / "python").chmod(0o755)
            p = subprocess.run(
                ["bash", "-c", pre + sec],
                capture_output=True,
                text=True,
                timeout=300,
                env={
                    "PATH": f"{shim}:{pathlib.Path(sys.executable).parent}:/usr/bin:/bin"
                },
            )
            self.assertIn(
                "platform/tests/test_a.py (imports foo.py)",
                p.stdout,
                p.stdout + p.stderr,
            )
            self.assertIn("result: PASS", p.stdout, p.stdout + p.stderr)
            self.assertTrue(
                (d / "out/status").read_text().startswith("importers pass"), p.stdout
            )


if __name__ == "__main__":
    unittest.main()
