#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the smart-build selection validator (validate_selection.py)."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent src/ directory to path for imports
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))
MAIN_PY = Path(__file__).parent.parent / "main.py"


class TestLoaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _write(self, name, content):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_load_ninja_targets_parses_name_before_colon(self):
        from validate_selection import load_ninja_targets

        path = self._write(
            "targets.txt",
            "bin/test_gemm: phony\n"
            "bin/test_conv: CXX_EXECUTABLE_LINKER__test_conv\n"
            "lib/foo.cpp.o: CXX_COMPILER__foo\n"
            "\n",  # blank line ignored
        )
        targets = load_ninja_targets(path)
        self.assertEqual(
            targets, {"bin/test_gemm", "bin/test_conv", "lib/foo.cpp.o"}
        )

    def test_load_selected_executables_prefers_executables_key(self):
        from validate_selection import load_selected_executables

        path = self._write(
            "sel.json",
            json.dumps(
                {"executables": ["bin/a", "bin/b"], "tests_to_run": ["bin/a"]}
            ),
        )
        self.assertEqual(load_selected_executables(path), ["bin/a", "bin/b"])

    def test_load_selected_executables_falls_back_to_tests_to_run(self):
        from validate_selection import load_selected_executables

        path = self._write("sel.json", json.dumps({"tests_to_run": ["bin/x"]}))
        self.assertEqual(load_selected_executables(path), ["bin/x"])

    def test_load_ctest_tests_parses_variable_spacing(self):
        from validate_selection import load_ctest_tests

        path = self._write(
            "ctest.txt",
            "Test   #1: test_gemm\n"
            "Test  #10: test_conv\n"
            "Test #100: test_pool\n"
            "Total Tests: 3\n",
        )
        self.assertEqual(
            load_ctest_tests(path), {"test_gemm", "test_conv", "test_pool"}
        )


class TestValidate(unittest.TestCase):
    def test_all_valid_passes(self):
        from validate_selection import validate

        result = validate(["bin/a", "bin/b"], {"bin/a", "bin/b", "bin/c"})
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["n_invalid_targets"], 0)
        self.assertEqual(result["invalid_targets"], [])

    def test_one_invalid_fails_and_lists_it(self):
        from validate_selection import validate

        result = validate(["bin/a", "bin/bogus"], {"bin/a"})
        self.assertEqual(result["verdict"], "fail")
        self.assertEqual(result["n_invalid_targets"], 1)
        self.assertEqual(result["invalid_targets"], ["bin/bogus"])

    def test_empty_selection_passes(self):
        from validate_selection import validate

        result = validate([], {"bin/a"})
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["n_selected"], 0)

    def test_bin_prefix_must_match_exactly(self):
        from validate_selection import validate

        # depmap uses 'bin/'-prefixed names; a bare name must NOT match.
        result = validate(["test_gemm"], {"bin/test_gemm"})
        self.assertEqual(result["verdict"], "fail")
        self.assertEqual(result["invalid_targets"], ["test_gemm"])

    def test_ctest_secondary_check_can_fail_verdict(self):
        from validate_selection import validate

        # All targets valid, but basename not a registered ctest test.
        result = validate(
            ["bin/test_gemm"],
            {"bin/test_gemm"},
            ctest_tests={"test_other"},
        )
        self.assertEqual(result["verdict"], "fail")
        self.assertEqual(result["invalid_tests"], ["bin/test_gemm"])

    def test_ctest_secondary_check_passes_when_registered(self):
        from validate_selection import validate

        result = validate(
            ["bin/test_gemm"],
            {"bin/test_gemm"},
            ctest_tests={"test_gemm"},
        )
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["invalid_tests"], [])


class TestJunit(unittest.TestCase):
    def test_junit_reports_failure_per_invalid_target(self):
        from validate_selection import render_junit, validate

        result = validate(["bin/a", "bin/bogus"], {"bin/a"})
        xml = render_junit(result)
        self.assertIn('failures="1"', xml)
        self.assertIn("bin/bogus", xml)
        self.assertIn("<failure", xml)

    def test_junit_pass_has_no_failures(self):
        from validate_selection import render_junit, validate

        result = validate(["bin/a"], {"bin/a"})
        xml = render_junit(result)
        self.assertIn('failures="0"', xml)
        self.assertNotIn("<failure", xml)

    def test_no_mode_keeps_untagged_classname(self):
        from validate_selection import render_junit, validate

        xml = render_junit(validate(["bin/a"], {"bin/a"}))
        self.assertIn('classname="smart-build.selection"', xml)
        self.assertIn('name="smart-build-selection"', xml)

    def test_mode_tags_suite_and_classname(self):
        from validate_selection import render_junit, validate

        result = validate(["bin/a"], {"bin/a"}, mode="full")
        self.assertEqual(result["mode"], "full")
        self.assertTrue(result["advisory"])
        xml = render_junit(result)
        self.assertIn('classname="smart-build.selection.full"', xml)
        self.assertIn('name="smart-build-selection-full"', xml)

    def test_selective_mode_is_not_advisory(self):
        from validate_selection import validate

        self.assertFalse(validate(["bin/a"], {"bin/a"}, mode="selective")["advisory"])

    def test_none_mode_is_advisory(self):
        from validate_selection import validate

        result = validate([], {"bin/a"}, mode="none")
        self.assertEqual(result["mode"], "none")
        self.assertTrue(result["advisory"])

    def test_junit_xml_is_well_formed(self):
        import xml.etree.ElementTree as ET
        from validate_selection import render_junit, validate

        result = validate(["bin/a", "bin/bogus"], {"bin/a"})
        ET.fromstring(render_junit(result))

    def test_label_tags_suite_and_classname(self):
        from validate_selection import render_junit, validate

        result = validate(["bin/a"], {"bin/a"}, mode="full", label="gfx942")
        self.assertEqual(result["label"], "gfx942")
        xml = render_junit(result)
        self.assertIn('classname="smart-build.selection.full.gfx942"', xml)
        self.assertIn('name="smart-build-selection-full-gfx942"', xml)
        # the leaf case name carries the arch too, so the row isn't a duplicate
        self.assertIn('name="all-selected-targets-exist (gfx942)"', xml)

    def test_label_without_mode_still_tags(self):
        from validate_selection import render_junit, validate

        xml = render_junit(validate(["bin/a"], {"bin/a"}, label="gfx950"))
        self.assertIn('classname="smart-build.selection.gfx950"', xml)

    def test_junit_pass_has_properties(self):
        import xml.etree.ElementTree as ET
        from validate_selection import render_junit, validate

        result = validate(["bin/a", "bin/b"], {"bin/a", "bin/b"})
        root = ET.fromstring(render_junit(result))
        props = {p.get("name"): p.get("value") for p in root.findall("./properties/property")}
        self.assertEqual(props["n_selected"], "2")
        self.assertEqual(props["n_known_targets"], "2")
        self.assertEqual(props["n_invalid_targets"], "0")

    def test_junit_fail_has_properties(self):
        import xml.etree.ElementTree as ET
        from validate_selection import render_junit, validate

        result = validate(["bin/a", "bin/bogus"], {"bin/a"})
        root = ET.fromstring(render_junit(result))
        props = {p.get("name"): p.get("value") for p in root.findall("./properties/property")}
        self.assertEqual(props["n_selected"], "2")
        self.assertEqual(props["n_invalid_targets"], "1")

    def test_junit_properties_include_advisory_when_mode_set(self):
        import xml.etree.ElementTree as ET
        from validate_selection import render_junit, validate

        result = validate(["bin/a"], {"bin/a"}, mode="full")
        root = ET.fromstring(render_junit(result))
        props = {p.get("name"): p.get("value") for p in root.findall("./properties/property")}
        self.assertEqual(props["advisory"], "true")

    def test_junit_properties_no_advisory_without_mode(self):
        import xml.etree.ElementTree as ET
        from validate_selection import render_junit, validate

        root = ET.fromstring(render_junit(validate(["bin/a"], {"bin/a"})))
        props = {p.get("name") for p in root.findall("./properties/property")}
        self.assertNotIn("advisory", props)

    def test_junit_properties_include_n_invalid_tests_when_ctest_checked(self):
        import xml.etree.ElementTree as ET
        from validate_selection import render_junit, validate

        result = validate(["bin/a"], {"bin/a"}, ctest_tests={"a"})
        root = ET.fromstring(render_junit(result))
        props = {p.get("name") for p in root.findall("./properties/property")}
        self.assertIn("n_invalid_tests", props)


class TestCli(unittest.TestCase):
    """End-to-end exit-code checks through main.py validate."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.targets = os.path.join(self.tmp, "targets.txt")
        with open(self.targets, "w") as f:
            f.write("bin/test_gemm: phony\nbin/test_conv: rule\n")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self, executables):
        sel = os.path.join(self.tmp, "sel.json")
        with open(sel, "w") as f:
            json.dump({"executables": executables}, f)
        out = os.path.join(self.tmp, "result.json")
        proc = subprocess.run(
            [
                sys.executable,
                str(MAIN_PY),
                "validate",
                sel,
                "--ninja-targets",
                self.targets,
                "--output",
                out,
            ],
            capture_output=True,
            text=True,
        )
        with open(out) as f:
            result = json.load(f)
        return proc.returncode, result

    def test_cli_pass_exit_zero(self):
        rc, result = self._run(["bin/test_gemm"])
        self.assertEqual(rc, 0)
        self.assertEqual(result["verdict"], "pass")

    def test_cli_fail_exit_one(self):
        rc, result = self._run(["bin/test_gemm", "bin/nope"])
        self.assertEqual(rc, 1)
        self.assertEqual(result["verdict"], "fail")
        self.assertEqual(result["invalid_targets"], ["bin/nope"])

    def test_cli_missing_file_exit_two(self):
        out = os.path.join(self.tmp, "r.json")
        proc = subprocess.run(
            [
                sys.executable,
                str(MAIN_PY),
                "validate",
                os.path.join(self.tmp, "does_not_exist.json"),
                "--ninja-targets",
                self.targets,
                "--output",
                out,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
