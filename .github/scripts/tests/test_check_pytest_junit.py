# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import contextlib
import io
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import check_pytest_junit as cpj


def case_xml(classname: str, name: str, skip: str | None = None) -> str:
    """One testcase. ``skip`` is None (ran), "skip" (a skip) or "xfail"."""
    child = ""
    if skip == "skip":
        child = '<skipped type="pytest.skip" message="no input"/>'
    elif skip == "xfail":
        child = '<skipped type="pytest.xfail" message="known bad"/>'
    return f'<testcase classname="{classname}" name="{name}">{child}</testcase>'


def junit_xml(cases: list[str]) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?><testsuites>'
        f'<testsuite name="pytest">{"".join(cases)}</testsuite></testsuites>'
    )


class CheckPytestJunitTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def junit(self, cases: list[str], stem: str = "suite") -> Path:
        path = self.dir / f"{stem}.xml"
        path.write_text(junit_xml(cases))
        return path

    def allow(self, text: str, stem: str = "allow") -> Path:
        path = self.dir / f"{stem}.txt"
        path.write_text(text)
        return path

    def ids(self, junits: list[Path], allows: list[Path]) -> list[str]:
        return [kind for kind, _ in cpj.check(junits, allows)]

    def run_main(self, junits: list[Path], allows: list[Path]) -> tuple[int, str]:
        argv = []
        for junit in junits:
            argv += ["--junit", str(junit)]
        for allow in allows:
            argv += ["--allow-skips", str(allow)]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = cpj.main(argv)
        return rc, out.getvalue()

    def test_all_passing_with_an_empty_allow_list_is_clean(self):
        junit = self.junit([case_xml("tests.a", "test_one")])
        self.assertEqual(self.ids([junit], [self.allow("# nothing\n")]), [])

    def test_an_unlisted_skip_fails(self):
        junit = self.junit([case_xml("tests.a", "test_one", "skip")])
        self.assertEqual(self.ids([junit], [self.allow("")]), ["unexpected-skip"])

    def test_an_unlisted_xfail_fails_like_a_skip(self):
        junit = self.junit([case_xml("tests.a", "test_one", "xfail")])
        self.assertEqual(self.ids([junit], [self.allow("")]), ["unexpected-skip"])

    def test_an_allow_listed_skip_passes(self):
        junit = self.junit([case_xml("tests.a.TestReal", "test_x[p1]", "skip")])
        allow = self.allow("suite::tests.a.TestReal::* # needs input by design\n")
        self.assertEqual(self.ids([junit], [allow]), [])

    def test_a_bracketed_id_is_matched_by_a_star_or_an_escaped_bracket(self):
        """`[` opens a glob character class, so a literal `[p1]` matches nothing."""
        junit = self.junit([case_xml("tests.a", "test_x[p1]", "skip")])
        escaped = self.allow("suite::tests.a::test_x[[]p1] # escaped\n")
        literal = self.allow("suite::tests.a::test_x[p1] # unescaped\n", "literal")
        self.assertEqual(self.ids([junit], [escaped]), [])
        self.assertEqual(
            self.ids([junit], [literal]), ["unexpected-skip", "stale-allow"]
        )

    def test_a_pattern_that_matches_no_skip_is_stale(self):
        junit = self.junit([case_xml("tests.a", "test_one")])
        allow = self.allow("suite::tests.gone::* # was skipped once\n")
        self.assertEqual(self.ids([junit], [allow]), ["stale-allow"])

    def test_a_pattern_is_scoped_to_its_junit_file(self):
        junit = self.junit([case_xml("tests.a", "test_one", "skip")], stem="other")
        allow = self.allow("suite::tests.a::* # wrong suite\n")
        self.assertEqual(self.ids([junit], [allow]), ["unexpected-skip", "stale-allow"])

    def test_a_missing_junit_is_unreadable(self):
        allow = self.allow("")
        self.assertEqual(
            self.ids([self.dir / "absent.xml"], [allow]), ["junit-unreadable"]
        )

    def test_a_junit_that_is_not_xml_is_unreadable(self):
        junit = self.dir / "suite.xml"
        junit.write_text("not xml")
        self.assertEqual(self.ids([junit], [self.allow("")]), ["junit-unreadable"])

    def test_a_junit_without_testcases_is_empty(self):
        self.assertEqual(self.ids([self.junit([])], [self.allow("")]), ["junit-empty"])

    def test_an_allow_list_line_without_a_reason_is_malformed(self):
        junit = self.junit([case_xml("tests.a", "test_one", "skip")])
        allow = self.allow("suite::tests.a::*\n")
        self.assertEqual(
            self.ids([junit], [allow]), ["allow-list-malformed", "unexpected-skip"]
        )

    def test_a_missing_allow_list_is_unreadable(self):
        junit = self.junit([case_xml("tests.a", "test_one")])
        self.assertEqual(
            self.ids([junit], [self.dir / "absent.txt"]), ["allow-list-unreadable"]
        )

    def test_every_junit_file_is_checked(self):
        clean = self.junit([case_xml("tests.t", "test_one")], stem="a")
        skipping = self.junit([case_xml("tests.t", "test_two", "skip")], stem="b")
        failures = cpj.check([clean, skipping], [self.allow("")])
        self.assertEqual([kind for kind, _ in failures], ["unexpected-skip"])
        self.assertIn("b::tests.t::test_two", failures[0][1])
        self.assertIn("no input", failures[0][1])

    def test_several_allow_lists_are_pooled(self):
        junit = self.junit(
            [
                case_xml("tests.a", "test_one", "skip"),
                case_xml("tests.b", "test_two", "skip"),
            ]
        )
        shared = self.allow("suite::tests.a::* # shared skip\n", "shared")
        lane = self.allow("suite::tests.b::* # lane-only skip\n", "lane")
        self.assertEqual(self.ids([junit], [shared, lane]), [])

    def test_a_lane_entry_matching_nothing_is_stale_even_when_the_shared_list_matches(
        self,
    ):
        junit = self.junit([case_xml("tests.a", "test_one", "skip")])
        shared = self.allow("suite::tests.a::* # shared skip\n", "shared")
        lane = self.allow("suite::tests.gone::* # no longer skipped\n", "lane")
        self.assertEqual(self.ids([junit], [shared, lane]), ["stale-allow"])

    def test_main_reports_ok_with_exit_zero(self):
        junit = self.junit([case_xml("tests.a", "test_one")])
        rc, out = self.run_main([junit], [self.allow("")])
        self.assertEqual(rc, 0)
        self.assertTrue(out.startswith("check_pytest_junit: OK "), out)

    def test_main_reports_the_counts_it_checked(self):
        """The OK line is what a CI log shows, so its numbers have to be right: they
        sum over every junit file."""
        first = self.junit(
            [case_xml("tests.a", "test_one"), case_xml("tests.a", "test_two", "skip")],
            stem="a",
        )
        second = self.junit([case_xml("tests.b", "test_three", "skip")], stem="b")
        allow = self.allow("a::tests.a::* # first\nb::tests.b::* # second\n")
        rc, out = self.run_main([first, second], [allow])
        self.assertEqual(rc, 0)
        match = re.search(
            r"OK (\d+) testcase\(s\) in (\d+) junit file\(s\), (\d+) allow-listed skip\(s\)",
            out,
        )
        self.assertIsNotNone(match, out)
        self.assertEqual(tuple(map(int, match.groups())), (3, 2, 2))

    def test_main_prints_one_fail_line_naming_the_failure_and_why(self):
        junit = self.junit([case_xml("tests.a", "test_one", "skip")])
        rc, out = self.run_main([junit], [self.allow("")])
        self.assertEqual(rc, 1)
        lines = out.splitlines()
        self.assertEqual(len(lines), 1, out)
        self.assertTrue(
            lines[0].startswith("check_pytest_junit: FAIL unexpected-skip:"), out
        )
        self.assertIn("no input", lines[0])


if __name__ == "__main__":
    unittest.main()
