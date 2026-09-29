# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for compare_forwarding_runs.py.

The comparator is the only thing that can turn a forwarding divergence into a
red build, so the cases worth covering are the ones where it could report
agreement that is not there: a name that appears twice in a run, an XML that a
crashed replay left half-written, a pair of reports left over from an earlier
build, and two runs that skipped everything. Comparing two well-formed, agreeing
files only ever exercises the passing path.

The known-divergence list is covered from the same angle: that it tolerates only
the divergence it names, that a line which has stopped applying fails rather than
lingering, and that tolerating one is said out loud instead of reported as a
clean pass.

Written against the standard library's unittest rather than pytest: this runs as
a ctest entry in a wrapper-enabled build, and nothing provisions pytest for a
machine that builds MIOpen.

Everything here works on string fixtures, so no build, toolchain or GPU is
needed:

    python3 -m unittest test_compare_forwarding_runs
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Colocated with the script under test, so the module also runs under a bare
# `python3 -m unittest` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_forwarding_runs as cmp  # noqa: E402


def suite(*cases):
    """Build a JUnit document from (name, status) pairs, statuses as gtest emits them.

    "skipped" is a DISABLED_ test, which gtest never starts. "gtest_skipped" is a
    test that started and called GTEST_SKIP(), which gtest records as run.

    A case may carry a third element, a dict of recorded properties, written the
    way gtest writes them: a properties element inside the testcase.
    """
    body = []
    for case in cases:
        name, status = case[0], case[1]
        inner = ""
        if len(case) > 2:
            inner = "<properties>{}</properties>".format(
                "".join(
                    '<property name="{}" value="{}"/>'.format(key, value)
                    for key, value in sorted(case[2].items())
                )
            )
        attrs = ""
        if status == "failed":
            inner += '<failure message="x"/>'
        elif status == "skipped":
            attrs = ' status="notrun"'
        elif status == "gtest_skipped":
            attrs = ' status="run" result="skipped"'
            inner += '<skipped message="x"/>'
        body.append(
            '<testcase name="{}" classname="Shim"{}>{}</testcase>'.format(
                name, attrs, inner
            )
        )
    return '<?xml version="1.0"?><testsuites>{}</testsuites>'.format("".join(body))


SERVED = ("A", "passed", {"parity_served_case": "true"})
DECLINED = ("A", "passed", {"parity_served_case": "false"})
KNOWN_LINE = (
    "Shim.A | passed[parity_served_case=true] | passed[parity_served_case=false] "
    "| the forwarded path cannot express this problem"
)


def aged(path, seconds):
    """Backdate a file, so freshness can be tested without waiting for the clock."""
    stamp = os.path.getmtime(path) - seconds
    os.utime(path, (stamp, stamp))
    return path


class ComparatorTest(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.tmp_path = Path(holder.name)

    def write(self, name, text):
        path = self.tmp_path / name
        path.write_text(text)
        return str(path)

    def compare(self, disabled_xml, enabled_xml, newer_than=None, known=None):
        """Run the comparator on two fixtures, returning (exit code, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cmp.main(
                self.write("disabled.xml", disabled_xml),
                self.write("enabled.xml", enabled_xml),
                newer_than,
                self.write("known.txt", known) if known is not None else None,
            )
        return rc, out.getvalue(), err.getvalue()

    def test_identical_runs_agree(self):
        xml = suite(("A", "passed"), ("B", "skipped"))
        rc, out, _ = self.compare(xml, xml)
        self.assertEqual(rc, 0)
        self.assertIn("2 tests identical", out)

    def test_status_divergence_is_reported(self):
        rc, _, err = self.compare(suite(("A", "passed")), suite(("A", "failed")))
        self.assertEqual(rc, 1)
        self.assertIn("disabled=passed enabled=failed", err)

    def test_test_only_in_disabled_run_is_reported(self):
        rc, _, err = self.compare(
            suite(("A", "passed"), ("B", "passed")), suite(("A", "passed"))
        )
        self.assertEqual(rc, 1)
        self.assertIn("only in disabled run: Shim.B", err)

    def test_test_only_in_enabled_run_is_reported(self):
        rc, _, err = self.compare(
            suite(("A", "passed")), suite(("A", "passed"), ("B", "passed"))
        )
        self.assertEqual(rc, 1)
        self.assertIn("only in enabled run: Shim.B", err)

    def test_duplicate_name_divergence_is_not_collapsed(self):
        # Same name twice in each run, disagreeing on one of the two. Keyed by name
        # alone, the second entry would overwrite the first and the two runs would
        # look identical.
        rc, _, err = self.compare(
            suite(("A", "passed"), ("A", "passed")),
            suite(("A", "passed"), ("A", "failed")),
        )
        self.assertEqual(rc, 1)
        self.assertIn("2 entries", err)
        self.assertIn("failed", err)

    def test_duplicate_name_agreement_still_passes(self):
        # Emission order is not part of the claim, so the same outcomes in the other
        # order are agreement, not a divergence.
        rc, _, _ = self.compare(
            suite(("A", "passed"), ("A", "failed")),
            suite(("A", "failed"), ("A", "passed")),
        )
        self.assertEqual(rc, 0)

    def test_truncated_xml_gives_a_diagnostic_not_a_traceback(self):
        good = suite(("A", "passed"))
        rc, _, err = self.compare(good, good[: len(good) // 2])
        self.assertEqual(rc, 1)
        self.assertIn("not well-formed XML", err)
        self.assertIn("crashed partway", err)

    def test_two_empty_runs_do_not_pass(self):
        empty = suite()
        rc, _, err = self.compare(empty, empty)
        self.assertEqual(rc, 1)
        self.assertIn("neither run executed a test", err)

    def test_two_all_skipped_runs_do_not_pass(self):
        # Non-empty and identical, so every other check here is satisfied. Nothing ran,
        # so the agreement says nothing about forwarding.
        skipped = suite(("A", "skipped"), ("B", "skipped"))
        rc, _, err = self.compare(skipped, skipped)
        self.assertEqual(rc, 1)
        self.assertIn("neither run executed a test", err)

    def test_gtest_skip_counts_as_skipped(self):
        xml = suite(("A", "passed"), ("B", "gtest_skipped"))
        rc, out, _ = self.compare(xml, xml)
        self.assertEqual(rc, 0)
        self.assertIn("1 executed, 1 skipped", out)

    def test_two_all_gtest_skip_runs_do_not_pass(self):
        # What a GPU-less machine produces: every test starts, finds no device and
        # calls GTEST_SKIP(). gtest marks these run, so only result= says otherwise.
        skipped = suite(("A", "gtest_skipped"), ("B", "gtest_skipped"))
        rc, _, err = self.compare(skipped, skipped)
        self.assertEqual(rc, 1)
        self.assertIn("neither run executed a test", err)

    def test_gtest_skip_in_one_run_only_is_reported(self):
        rc, _, err = self.compare(
            suite(("A", "passed"), ("B", "passed")),
            suite(("A", "passed"), ("B", "gtest_skipped")),
        )
        self.assertEqual(rc, 1)
        self.assertIn("disabled=passed enabled=skipped", err)

    def test_one_executed_test_is_enough(self):
        xml = suite(("A", "passed"), ("B", "skipped"))
        rc, out, _ = self.compare(xml, xml)
        self.assertEqual(rc, 0)
        self.assertIn("1 executed, 1 skipped", out)

    def test_a_missing_report_is_not_agreement(self):
        disabled = self.write("disabled.xml", suite(("A", "passed")))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = cmp.main(disabled, str(self.tmp_path / "never_written.xml"))
        self.assertEqual(rc, 1)
        self.assertIn("did not run", err.getvalue())

    def test_reports_older_than_the_binary_are_rejected(self):
        # The harness's worst failure mode: both files are well-formed and agree, but
        # they are the previous build's output and neither replay ran this time.
        xml = suite(("A", "passed"))
        binary = self.tmp_path / "miopen_gtest"
        binary.touch()
        rc, _, _ = self.compare(xml, xml, newer_than=str(binary))
        self.assertEqual(rc, 0, "a report written after the binary is current")

        aged(self.tmp_path / "disabled.xml", 60)
        aged(self.tmp_path / "enabled.xml", 60)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = cmp.main(
                str(self.tmp_path / "disabled.xml"),
                str(self.tmp_path / "enabled.xml"),
                str(binary),
            )
        self.assertEqual(rc, 1)
        self.assertIn("left over from an earlier build", err.getvalue())

    def test_serving_a_case_and_declining_it_is_a_divergence(self):
        # Both runs pass. The difference is that one computed a result and the other was
        # told the problem could not be expressed, which is what the recorded property
        # carries and what the bare verdicts throw away.
        rc, _, err = self.compare(suite(SERVED), suite(DECLINED))
        self.assertEqual(rc, 1)
        self.assertIn("parity_served_case=true", err)
        self.assertIn("parity_served_case=false", err)

    def test_a_property_written_as_an_attribute_is_also_read(self):
        # Older gtest releases write recorded properties as attributes on the testcase
        # rather than as children of it. Reading only one shape would mean a gtest
        # update silently dropping the distinction the property carries.
        as_attribute = (
            '<?xml version="1.0"?><testsuites><testcase name="A" classname="Shim" '
            'parity_served_case="{}"/></testsuites>'
        )
        rc, _, err = self.compare(
            as_attribute.format("true"), as_attribute.format("false")
        )
        self.assertEqual(rc, 1)
        self.assertIn("parity_served_case=false", err)

    def test_a_listed_divergence_is_tolerated_and_announced(self):
        rc, out, _ = self.compare(suite(SERVED), suite(DECLINED), known=KNOWN_LINE)
        self.assertEqual(rc, 0)
        self.assertIn("1 known divergence tolerated", out)
        self.assertIn("Shim.A", out)
        self.assertIn("cannot express this problem", out)
        # The ordinary success line would read as two modes that agreed.
        self.assertNotIn("tests identical under both modes", out)

    def test_a_listed_divergence_that_stopped_happening_fails(self):
        rc, _, err = self.compare(suite(SERVED), suite(SERVED), known=KNOWN_LINE)
        self.assertEqual(rc, 1)
        self.assertIn("no longer diverges", err)
        self.assertIn("remove its line", err)

    def test_a_listed_divergence_declined_in_both_modes_is_not_stale(self):
        # On a device with no engine for the case, forwarding off declines it too. The
        # runs agree, but the gap is still open on devices where MIOpen serves the case.
        rc, _, err = self.compare(suite(DECLINED), suite(DECLINED), known=KNOWN_LINE)
        self.assertEqual(rc, 0, err)

    def test_a_listed_divergence_skipped_in_both_modes_is_not_stale(self):
        xml = suite(("A", "gtest_skipped"), ("B", "passed"))
        rc, _, err = self.compare(xml, xml, known=KNOWN_LINE)
        self.assertEqual(rc, 0, err)

    def test_a_listed_test_absent_from_the_run_is_not_stale(self):
        # The discrete build registers the harness against several binaries, and only
        # one of them holds any given test. Absence is not a line that has gone stale.
        xml = suite(("B", "passed"))
        rc, _, _ = self.compare(xml, xml, known=KNOWN_LINE)
        self.assertEqual(rc, 0)

    def test_the_list_does_not_suppress_a_divergence_it_does_not_name_exactly(self):
        # Same test, different divergence. A line keyed on the name alone would
        # swallow it.
        rc, _, err = self.compare(
            suite(SERVED), suite(("A", "failed")), known=KNOWN_LINE
        )
        self.assertEqual(rc, 1)
        self.assertIn("disabled=passed[parity_served_case=true] enabled=failed", err)

    def test_a_malformed_list_line_is_not_skipped_over(self):
        rc, _, err = self.compare(
            suite(SERVED), suite(DECLINED), known="Shim.A | passed | failed"
        )
        self.assertEqual(rc, 1)
        self.assertIn("line 1", err)
        self.assertIn("why it is accepted", err)

    def test_comments_and_blank_lines_in_the_list_are_ignored(self):
        listing = "# a comment\n\n{}\n".format(KNOWN_LINE)
        rc, _, _ = self.compare(suite(SERVED), suite(DECLINED), known=listing)
        self.assertEqual(rc, 0)

    def test_a_missing_list_is_not_treated_as_an_empty_one(self):
        # Otherwise an installed tree that shipped the scripts without the list would
        # quietly enforce a stricter check than the one it was configured with.
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = cmp.main(
                self.write("disabled.xml", suite(SERVED)),
                self.write("enabled.xml", suite(SERVED)),
                None,
                str(self.tmp_path / "no_such_list.txt"),
            )
        self.assertEqual(rc, 1)
        self.assertIn("could not be read", err.getvalue())

    def test_a_missing_newer_than_target_is_rejected(self):
        # Nothing to date the reports against means their freshness is unknown, which
        # is not the same as fresh.
        xml = suite(("A", "passed"))
        rc, _, err = self.compare(
            xml, xml, newer_than=str(self.tmp_path / "no_such_binary")
        )
        self.assertEqual(rc, 1)
        self.assertIn("cannot be shown to be current", err)


if __name__ == "__main__":
    unittest.main()
