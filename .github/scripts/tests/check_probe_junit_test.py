# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import check_probe_junit as cpj

MANIFEST = ["hkp-probe-gfx950_attention_dense", "hkp-probe-tools"]


def case_xml(name: str, child: str = "") -> str:
    body = f"<{child}/>" if child else ""
    status = "notrun" if child == "skipped" else "run"
    return f'<testcase name="{name}" classname="{name}" time="1" status="{status}">{body}</testcase>'


def junit_xml(cases: list[str]) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<testsuite name="(empty)" tests="{len(cases)}">{"".join(cases)}</testsuite>'
    )


class CheckProbeJunitTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def run_check(self, xml: str, manifest: list[str]) -> list[str]:
        junit_path = self.dir / "probe-junit.xml"
        manifest_path = self.dir / "manifest.txt"
        junit_path.write_text(xml)
        manifest_path.write_text("".join(f"{n}\n" for n in manifest))
        return [check_id for check_id, _ in cpj.check(junit_path, manifest_path)]

    def test_all_declared_tests_passed(self):
        xml = junit_xml([case_xml(n) for n in MANIFEST])
        self.assertEqual(self.run_check(xml, MANIFEST), [])

    def test_skipped_test_fails(self):
        xml = junit_xml([case_xml(MANIFEST[0]), case_xml(MANIFEST[1], "skipped")])
        self.assertEqual(self.run_check(xml, MANIFEST), ["skipped"])

    def test_fewer_testcases_than_manifest_fails(self):
        xml = junit_xml([case_xml(MANIFEST[0])])
        self.assertEqual(
            self.run_check(xml, MANIFEST), ["count-mismatch", "name-missing"]
        )

    def test_empty_manifest_fails(self):
        xml = junit_xml([case_xml(n) for n in MANIFEST])
        self.assertIn("manifest-too-small", self.run_check(xml, []))

    def test_manifest_without_probe_fails(self):
        xml = junit_xml([case_xml("hkp-probe-tools")])
        self.assertEqual(
            self.run_check(xml, ["hkp-probe-tools"]), ["manifest-too-small"]
        )

    def test_failure_fails(self):
        xml = junit_xml([case_xml(MANIFEST[0], "failure"), case_xml(MANIFEST[1])])
        self.assertEqual(self.run_check(xml, MANIFEST), ["failure"])

    def test_error_fails(self):
        xml = junit_xml([case_xml(MANIFEST[0]), case_xml(MANIFEST[1], "error")])
        self.assertEqual(self.run_check(xml, MANIFEST), ["error"])

    def test_declared_name_missing_with_equal_count_fails(self):
        xml = junit_xml([case_xml(MANIFEST[0]), case_xml("hkp-probe-other")])
        self.assertEqual(self.run_check(xml, MANIFEST), ["name-missing"])

    def test_unparseable_junit_fails(self):
        self.assertEqual(self.run_check("<testsuite", MANIFEST), ["junit-unreadable"])

    def test_missing_manifest_fails(self):
        junit_path = self.dir / "probe-junit.xml"
        junit_path.write_text(junit_xml([case_xml(n) for n in MANIFEST]))
        failures = cpj.check(junit_path, self.dir / "absent.txt")
        self.assertEqual([c for c, _ in failures], ["manifest-unreadable"])

    def test_main_exit_codes(self):
        junit_path = self.dir / "probe-junit.xml"
        manifest_path = self.dir / "manifest.txt"
        manifest_path.write_text("".join(f"{n}\n" for n in MANIFEST))
        argv = [
            "--junit",
            os.fspath(junit_path),
            "--manifest",
            os.fspath(manifest_path),
        ]
        junit_path.write_text(junit_xml([case_xml(n) for n in MANIFEST]))
        self.assertEqual(cpj.main(argv), 0)
        junit_path.write_text(junit_xml([case_xml(MANIFEST[0])]))
        self.assertEqual(cpj.main(argv), 1)


if __name__ == "__main__":
    unittest.main()
