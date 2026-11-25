# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/pytest_plugins/rkmon_skips.py: a skipped test is never silent, and a skip for a tool CI's lanes have (cmake,
PyYAML) that the run lacks reads NOT RUN, never a pass (agent_script_overlap.md finding 4: `requires_cmake` and PyYAML
tests skipped and the section still read as a pass, so the verifier and the code quality reviewer re-ran them by hand).
"""

import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PLUGINS = pathlib.Path(__file__).resolve().parents[1] / "tools" / "pytest_plugins"
sys.path.insert(0, str(PLUGINS))
import rkmon_skips as sk  # noqa: E402

CMAKE_REASON = "CMake and CTest required"
YAML_REASON = "could not import 'yaml': No module named 'yaml'"


class TestLine(unittest.TestCase):
    def test_a_skip_names_a_missing_tool_only_when_the_run_lacks_it(self):
        self.assertEqual(
            sk.line("t/test_a.py::test_x", CMAKE_REASON, {"cmake"}),
            f"rkmon-skip:\tcmake\tt/test_a.py::test_x\t{CMAKE_REASON}",
        )
        self.assertEqual(
            sk.line("t/test_a.py::test_x", CMAKE_REASON, set()),
            f"rkmon-skip:\t-\tt/test_a.py::test_x\t{CMAKE_REASON}",
        )
        self.assertEqual(
            sk.line("t/test_b.py::test_y", YAML_REASON, {"cmake", "PyYAML"}).split(
                "\t"
            )[1],
            "PyYAML",
        )
        # A reason that names no missing tool is an ordinary skip, whatever else is missing.
        self.assertEqual(
            sk.line(
                "t/test_c.py::test_z", "needs a gfx950 GPU", {"cmake", "PyYAML"}
            ).split("\t")[1],
            "-",
        )

    def test_ids_and_reasons_are_one_line_each(self):
        self.assertEqual(
            sk.line("t/test_a.py::test_x[a\tb]", "two\n  lines", set()),
            "rkmon-skip:\t-\tt/test_a.py::test_x[a b]\ttwo lines",
        )

    def test_missing_tools_reads_the_runs_environment(self):
        """cmake counts only with ctest beside it on PATH; PyYAML only when `yaml` imports."""
        with tempfile.TemporaryDirectory() as d:
            for name in ("cmake", "ctest"):
                p = pathlib.Path(d, name)
                p.write_text("#!/bin/sh\n", encoding="utf-8")
                p.chmod(0o755)
            empty = pathlib.Path(d, "empty")
            empty.mkdir()
            spec = mock.patch.object(sk.importlib.util, "find_spec")
            with mock.patch.dict(os.environ, {"PATH": d}), spec as find:
                find.return_value = object()
                self.assertEqual(sk.missing_tools(), set())
                find.return_value = None
                self.assertEqual(sk.missing_tools(), {"PyYAML"})
            pathlib.Path(d, "ctest").unlink()
            with mock.patch.dict(os.environ, {"PATH": d}):
                self.assertIn("cmake", sk.missing_tools())
            with mock.patch.dict(os.environ, {"PATH": str(empty)}):
                self.assertIn("cmake", sk.missing_tools())


class TestReports(unittest.TestCase):
    TEXT = "\n".join(
        [
            "1 passed, 3 skipped in 0.1s",
            sk.line("t/test_a.py::test_link", CMAKE_REASON, {"cmake"}),
            sk.line("t/test_a.py::test_gpu", "needs a gfx950 GPU", {"cmake"}),
            "rkmon: 2 hipcc failures skipped",
            sk.line("t/test_b.py::test_gpu", "needs a gfx950 GPU", {"cmake"}),
            sk.line("t/test_a.py::test_gpu", "needs a gfx950 GPU", {"cmake"}),
        ]
    )  # the same skip twice

    def test_parse_reads_each_skip_once(self):
        self.assertEqual(
            sk.parse(self.TEXT),
            [
                ("cmake", "t/test_a.py::test_link", CMAKE_REASON),
                ("", "t/test_a.py::test_gpu", "needs a gfx950 GPU"),
                ("", "t/test_b.py::test_gpu", "needs a gfx950 GPU"),
            ],
        )

    def test_each_skip_is_listed_with_its_reason_and_a_missing_tool_reads_not_run(self):
        skips = sk.parse(self.TEXT)
        self.assertEqual(sk.not_run_text(skips), "NOT RUN: cmake missing (1 test)")
        self.assertEqual(
            sk.summary(skips), f"3 skipped: 2 needs a gfx950 GPU; 1 {CMAKE_REASON}"
        )
        self.assertEqual(
            sk.report(skips),
            [
                "NOT RUN: cmake missing (1 test; CI's lanes have it): t/test_a.py::test_link",
                "SKIPPED 3:",
                f"   t/test_a.py::test_link - {CMAKE_REASON} [NOT RUN: cmake missing]",
                "   t/test_a.py::test_gpu - needs a gfx950 GPU",
                "   t/test_b.py::test_gpu - needs a gfx950 GPU",
            ],
        )
        self.assertEqual(
            (sk.summary([]), sk.not_run_text([]), sk.report([])), ("", "", [])
        )

    def test_over_the_listed_count_the_skips_are_grouped_by_reason(self):
        skips = [
            ("", f"t/test_g.py::test_{i:02d}", "needs a GPU") for i in range(sk.LISTED)
        ]
        skips += [("", "t/test_h.py::test_t", "torch not importable")]
        out = sk.report(skips, "/out/skipped.tsv")
        self.assertEqual(
            out,
            [
                f"SKIPPED {sk.LISTED + 1}, by reason (every test and its reason: /out/skipped.tsv):",
                f"   {sk.LISTED} x needs a GPU: t/test_g.py::test_00 t/test_g.py::test_01 "
                f"t/test_g.py::test_02 (+{sk.LISTED - 3} more)",
                "   1 x torch not importable: t/test_h.py::test_t",
            ],
        )
        many = [
            ("", f"t/test_m.py::test_{i}", f"reason {i}") for i in range(sk.GROUPS + 2)
        ] + skips
        self.assertEqual(
            sk.report(many)[-1], "   (4 more reasons, 4 tests)"
        )  # GROUPS + 4 reasons in all
        self.assertTrue(sk.summary(many).endswith("; …"))

    def test_the_script_prints_the_result_and_the_report(self):
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "skipped.tsv"
            f.write_text(self.TEXT + "\n")
            run = lambda *a: subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(PLUGINS / "rkmon_skips.py"),
                    *a,
                    str(f),
                ],  # noqa
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines()
            self.assertEqual(
                run("result"),
                [
                    "3",
                    "NOT RUN: cmake missing (1 test)",
                    f"3 skipped: 2 needs a gfx950 GPU; 1 {CMAKE_REASON}",
                ],
            )
            self.assertEqual(run("report")[1], "SKIPPED 3:")
            f.write_text("")
            self.assertEqual(run("result"), ["0", "", ""])


if __name__ == "__main__":
    unittest.main()
