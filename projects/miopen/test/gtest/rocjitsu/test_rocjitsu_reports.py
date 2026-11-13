# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Execution completeness and diagnostic preservation, independent of rocjitsu."""

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from rocjitsu_reports import merge_reports


def write_report(path, rows):
    """Build GTest-format executions from explicit (identity, outcome) fixtures."""
    root = ET.Element("testsuites", tests=str(len(rows)), errors="0", time="1")
    suites = {}
    failures = 0
    for identity, outcome in rows:
        suite_name, name = identity.split(".", 1)
        if suite_name not in suites:
            suites[suite_name] = ET.SubElement(
                root,
                "testsuite",
                name=suite_name,
                tests="0",
                failures="0",
                skipped="0",
                disabled="0",
                errors="0",
                time="1",
            )
        suite = suites[suite_name]
        suite.set("tests", str(int(suite.get("tests")) + 1))
        case = ET.SubElement(
            suite,
            "testcase",
            name=name,
            classname=suite_name,
            status="notrun" if outcome == "disabled" else "run",
            result=(
                "suppressed"
                if outcome == "disabled"
                else (
                    "skipped"
                    if outcome in {"skipped", "failed-and-skipped"}
                    else "completed"
                )
            ),
            time="0.25",
            type_param="float",
            value_param="shape",
        )
        if outcome in {"failed", "failed-and-skipped"}:
            ET.SubElement(case, "failure", message="numerical mismatch").text = (
                "expected 7, got 9"
            )
            suite.set("failures", str(int(suite.get("failures")) + 1))
            failures += 1
        if outcome in {"skipped", "failed-and-skipped"}:
            ET.SubElement(case, "skipped", message="unsupported geometry").text = (
                "skip detail"
            )
            if outcome == "skipped":
                suite.set("skipped", str(int(suite.get("skipped")) + 1))
        if outcome == "disabled":
            suite.set("disabled", str(int(suite.get("disabled")) + 1))
    root.set("failures", str(failures))
    ET.ElementTree(root).write(path)
    return root


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.output = self.directory / "merged.xml"

    def report(self, name, rows):
        path = self.directory / name
        write_report(path, rows)
        return path

    def test_disjoint_shards_preserve_failure_skip_disabled_and_parameters(self):
        names = [
            "CPU_Config_NONE.pass",
            "CPU_Config_NONE.fail",
            "GPU_Op_FP32.skip",
            "Smoke/GPU_Op_FP16.DISABLED_case/0",
        ]
        first = self.report("cpu.xml", [(names[0], "passed"), (names[1], "failed")])
        second = self.report("gpu.xml", [(names[2], "skipped"), (names[3], "disabled")])
        counts = merge_reports([first, second], names, self.output, 3.5)
        self.assertEqual(
            counts, {"passed": 1, "failed": 1, "skipped": 1, "disabled": 1, "total": 4}
        )
        root = ET.parse(self.output).getroot()
        self.assertEqual(root.get("time"), "3.5")
        cases = {
            f"{case.get('classname')}.{case.get('name')}": case
            for case in root.iter("testcase")
        }
        self.assertEqual(set(cases), set(names))
        self.assertEqual(cases[names[1]].find("failure").text, "expected 7, got 9")
        self.assertEqual(cases[names[2]].find("skipped").text, "skip detail")
        self.assertEqual(cases[names[3]].get("status"), "notrun")
        self.assertEqual(cases[names[0]].get("type_param"), "float")
        self.assertEqual(cases[names[0]].get("value_param"), "shape")

    def test_suppressed_shard_copy_never_overwrites_execution(self):
        name = "CPU_Config_NONE.case"
        disabled = "CPU_Config_NONE.DISABLED_case"
        first = self.report("first.xml", [(name, "disabled"), (disabled, "disabled")])
        second = self.report("second.xml", [(name, "failed"), (disabled, "disabled")])
        for paths in ([first, second], [second, first]):
            with self.subTest(order=paths):
                counts = merge_reports(paths, [name, disabled], self.output, 1)
                self.assertEqual(counts["failed"], 1)
                self.assertEqual(counts["disabled"], 1)
                self.assertEqual(
                    ET.parse(self.output).find(".//failure").text, "expected 7, got 9"
                )

    def test_skip_after_failure_cannot_hide_failure(self):
        path = self.report("both.xml", [("GPU_Op_FP32.case", "failed-and-skipped")])
        counts = merge_reports([path], ["GPU_Op_FP32.case"], self.output, 1)
        self.assertEqual(
            (counts["failed"], counts["skipped"], counts["passed"]), (1, 0, 0)
        )
        case = ET.parse(self.output).find(".//testcase")
        self.assertEqual(case.find("failure").get("message"), "numerical mismatch")
        self.assertEqual(case.find("skipped").get("message"), "unsupported geometry")

    def test_missing_execution_report_is_not_success(self):
        path = self.report("one.xml", [("CPU_A_NONE.case", "passed")])
        with self.assertRaisesRegex(ValueError, "Cannot read"):
            merge_reports(
                [path, self.directory / "missing.xml"],
                ["CPU_A_NONE.case"],
                self.output,
                1,
            )
        self.assertFalse(self.output.exists())

    def test_absent_identity_is_not_a_disabled_test(self):
        path = self.report("partial.xml", [("CPU_A_NONE.case", "passed")])
        with self.assertRaisesRegex(ValueError, "Missing expected GTest identities"):
            merge_reports(
                [path], ["CPU_A_NONE.case", "CPU_A_NONE.DISABLED_other"], self.output, 1
            )
        self.assertFalse(self.output.exists())

    def test_runnable_suppressed_case_is_not_a_skip(self):
        path = self.report("notrun.xml", [("GPU_A_FP32.case", "disabled")])
        with self.assertRaisesRegex(ValueError, "did not run"):
            merge_reports([path], ["GPU_A_FP32.case"], self.output, 1)
        self.assertFalse(self.output.exists())

    def test_duplicate_actual_execution_across_or_within_reports_is_rejected(self):
        name = "GPU_A_FP32.case"
        first = self.report("first.xml", [(name, "passed")])
        second = self.report("second.xml", [(name, "failed")])
        repeated = self.report("repeated.xml", [(name, "passed"), (name, "passed")])
        for paths in ([first, second], [repeated]):
            with self.subTest(paths=paths), self.assertRaisesRegex(
                ValueError, "Duplicate actual"
            ):
                merge_reports(paths, [name], self.output, 1)
        self.assertFalse(self.output.exists())

    def test_phase_cannot_execute_identity_from_other_phase(self):
        path = self.report("cpu.xml", [("GPU_A_FP32.case", "passed")])
        with self.assertRaisesRegex(ValueError, "Unexpected GTest identity"):
            merge_reports([path], ["CPU_A_NONE.case"], self.output, 1)
        self.assertFalse(self.output.exists())

    def test_catalogue_xml_is_not_an_execution(self):
        path = self.report("list.xml", [("CPU_A_NONE.case", "passed")])
        tree = ET.parse(path)
        case = tree.find(".//testcase")
        del case.attrib["status"]
        del case.attrib["result"]
        tree.write(path)
        with self.assertRaisesRegex(ValueError, "listing XML is not a report"):
            merge_reports([path], ["CPU_A_NONE.case"], self.output, 1)
        self.assertFalse(self.output.exists())

    def test_corrupt_counts_or_durations_are_rejected(self):
        for attribute, value, message in (
            ("failures", "1", "Inconsistent failures"),
            ("errors", "1", "nonzero GTest errors"),
            ("tests", "2", "Inconsistent tests"),
        ):
            with self.subTest(attribute=attribute):
                path = self.report("bad.xml", [("CPU_A_NONE.case", "passed")])
                tree = ET.parse(path)
                tree.getroot().set(attribute, value)
                tree.write(path)
                with self.assertRaisesRegex(ValueError, message):
                    merge_reports([path], ["CPU_A_NONE.case"], self.output, 1)
        for value in ("nan", "inf", "-1", "not-a-time"):
            with self.subTest(time=value):
                path = self.report("bad.xml", [("CPU_A_NONE.case", "passed")])
                tree = ET.parse(path)
                tree.find(".//testcase").set("time", value)
                tree.write(path)
                with self.assertRaisesRegex(ValueError, "time"):
                    merge_reports([path], ["CPU_A_NONE.case"], self.output, 1)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
