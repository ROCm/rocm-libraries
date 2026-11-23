# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_CTEST_DIR = _TESTS_DIR.parent

sys.path.insert(0, str(_CTEST_DIR))
import parse_ctest_categories as pcc  # noqa: E402

# `.*` is the shape that needs exclude_tests: every test is a candidate and one
# entry must be left out. The other categories show that skipping is per category.
CATEGORIES = {
    "quick": {"test_patterns": [".*"]},
    "ffm-quick": {
        "test_patterns": [".*"],
        "exclude_tests": ["plain_binary"],
        "labels": ["ffm-full"],
    },
    "unit": {"test_patterns": ["plain_binary", "other_binary"]},
}


class TestExcludeTestsInstallLabels(unittest.TestCase):
    def test_skipped_test_keeps_every_other_category_label(self):
        labels = pcc._compute_install_labels(
            CATEGORIES, {}, ["plain_binary", "other_binary"]
        )

        self.assertEqual(
            labels["plain_binary"],
            ["quick", "standard", "comprehensive", "full", "unit"],
        )
        self.assertEqual(
            labels["other_binary"],
            [
                "quick",
                "standard",
                "comprehensive",
                "full",
                "ffm-quick",
                "ffm-full",
                "unit",
            ],
        )

    def test_regex_entry_skips_every_match(self):
        categories = {
            "ffm-quick": {
                "test_patterns": [".*"],
                "exclude_tests": ["skipped_.*"],
            }
        }

        labels = pcc._compute_install_labels(
            categories, {}, ["skipped_one", "skipped_two", "kept"]
        )

        self.assertEqual(labels["skipped_one"], [])
        self.assertEqual(labels["skipped_two"], [])
        self.assertEqual(labels["kept"], ["ffm-quick"])


class TestExcludeTestsBuildTreeLabels(unittest.TestCase):
    def test_skipped_test_is_unlabelled_in_the_build_tree(self):
        cmake = shutil.which("cmake")
        ctest = shutil.which("ctest")
        if not cmake or not ctest:
            self.skipTest("CMake and CTest are required for category integration")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            build = source / "build"
            (source / "categories.yaml").write_text(
                "test_categories:\n"
                "  ffm-quick:\n"
                "    test_patterns:\n"
                "      - '.*'\n"
                "    exclude_tests:\n"
                "      - 'plain_binary'\n",
                encoding="utf-8",
            )
            module = (_CTEST_DIR / "TestCategories.cmake").as_posix()
            # No EXPLICIT_TESTS: this is the get_property(DIRECTORY ... TESTS) loop,
            # whose regex match is evaluated by CMake rather than Python.
            (source / "CMakeLists.txt").write_text(
                f"""cmake_minimum_required(VERSION 3.20)
project(ExcludeTests NONE)
enable_testing()
include("{module}")
foreach(name plain_binary other_binary)
    add_test(NAME ${{name}} COMMAND "${{CMAKE_COMMAND}}" -E true)
endforeach()
apply_ctest_category_labels("${{CMAKE_CURRENT_SOURCE_DIR}}/categories.yaml")
""",
                encoding="utf-8",
            )
            configured = subprocess.run(
                [
                    cmake,
                    "-S",
                    str(source),
                    "-B",
                    str(build),
                    f"-DPython3_EXECUTABLE={sys.executable}",
                    f"-DROCM_LIBRARIES_ROOT={_CTEST_DIR.parent.parent}",
                ],
                capture_output=True,
                cwd=source,
                text=True,
                check=False,
            )
            self.assertEqual(
                configured.returncode, 0, configured.stdout + configured.stderr
            )
            # A multi-config generator (Visual Studio) lists no tests without -C.
            selected = subprocess.run(
                [
                    ctest,
                    "--test-dir",
                    str(build),
                    "-C",
                    "Debug",
                    "--show-only=json-v1",
                    "-L",
                    "^ffm-quick$",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(selected.returncode, 0, selected.stderr)
            self.assertEqual(
                {test["name"] for test in json.loads(selected.stdout)["tests"]},
                {"other_binary"},
            )


if __name__ == "__main__":
    unittest.main()
