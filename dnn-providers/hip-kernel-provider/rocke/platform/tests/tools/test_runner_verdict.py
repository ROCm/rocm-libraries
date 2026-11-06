# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The verdict and exit-code contract every rocKE runner reports through."""

import io

import pytest

from rocke.tests.runner_verdict import (
    EXIT_GREEN,
    EXIT_INCOMPLETE,
    EXIT_RED,
    Outcome,
    RunRecord,
    Verdict,
    junit_counts,
    user_scratch_dir,
)


def _record(*outcomes: Outcome, partial: bool = False, allow: bool = False):
    record = RunRecord(allow_incomplete=allow)
    for index, outcome in enumerate(outcomes):
        record.add(f"stage{index}", outcome, "why")
    if partial:
        record.partial("--only x")
    return record


@pytest.mark.parametrize(
    ("outcomes", "partial", "verdict", "code"),
    [
        ((Outcome.PASSED,), False, Verdict.GREEN, EXIT_GREEN),
        ((Outcome.PASSED,), True, Verdict.PARTIAL, EXIT_GREEN),
        ((Outcome.PASSED, Outcome.SKIPPED), False, Verdict.INCOMPLETE, EXIT_INCOMPLETE),
        ((Outcome.PASSED, Outcome.EMPTY), False, Verdict.INCOMPLETE, EXIT_INCOMPLETE),
        ((Outcome.FAILED, Outcome.NOT_RUN), False, Verdict.RED, EXIT_RED),
        # Precedence: RED > INCOMPLETE > PARTIAL > GREEN.
        ((Outcome.SKIPPED, Outcome.FAILED), True, Verdict.RED, EXIT_RED),
        ((Outcome.SKIPPED,), True, Verdict.INCOMPLETE, EXIT_INCOMPLETE),
        # A run that recorded nothing proved nothing.
        ((), False, Verdict.INCOMPLETE, EXIT_INCOMPLETE),
    ],
)
def test_verdict_and_exit_code_follow_one_table(outcomes, partial, verdict, code):
    record = _record(*outcomes, partial=partial)
    assert record.verdict() is verdict
    assert record.exit_code() == code


def test_allow_incomplete_changes_the_code_but_never_the_verdict():
    record = _record(Outcome.PASSED, Outcome.SKIPPED, allow=True)
    out = io.StringIO()
    assert record.report("runner", out) == EXIT_GREEN
    text = out.getvalue()
    assert "RESULT: INCOMPLETE (allowed by --allow-incomplete)" in text
    assert "GREEN" not in text


def test_allow_incomplete_never_hides_a_failure():
    assert _record(Outcome.FAILED, allow=True).exit_code() == EXIT_RED


@pytest.mark.parametrize(
    ("code", "kwargs", "outcome"),
    [
        (0, {}, Outcome.PASSED),
        (0, {"executed": 0}, Outcome.EMPTY),
        (0, {"executed": 3}, Outcome.PASSED),
        (77, {"skip_allowed": True}, Outcome.SKIPPED),
        (77, {}, Outcome.FAILED),
        (3, {"delegated_runner": True}, Outcome.SKIPPED),
        # For pytest 3 is an internal error, not a delegated INCOMPLETE.
        (3, {}, Outcome.FAILED),
        (5, {}, Outcome.FAILED),
        (2, {}, Outcome.FAILED),
        (124, {}, Outcome.FAILED),
    ],
)
def test_child_exit_status_maps_to_one_outcome(code, kwargs, outcome):
    record = RunRecord()
    assert record.returncode("child", code, **kwargs) is outcome


def test_pytest_no_tests_collected_is_a_failure_that_says_so():
    record = RunRecord()
    record.returncode("pytest", 5)
    assert record.stages[0].detail == "selected no tests"
    assert record.verdict() is Verdict.RED


def test_report_lists_every_stage_and_its_reason():
    record = RunRecord()
    record.passed("guard")
    record.skipped("both-engine pytest", "rocke_engine is not importable")
    record.not_run("ctest", "an earlier stage failed")
    record.partial("--no-gate")
    out = io.StringIO()
    record.report("platform run_all", out)
    text = out.getvalue()
    assert "skipped" in text and "rocke_engine is not importable" in text
    assert "not run" in text and "an earlier stage failed" in text
    assert "narrowed by: --no-gate" in text
    assert text.rstrip().endswith("RESULT: INCOMPLETE")


def test_junit_counts_separates_executed_from_skipped(tmp_path):
    report = tmp_path / "junit.xml"
    report.write_text(
        "<testsuites><testsuite>"
        "<testcase name='a'/>"
        "<testcase name='b'><skipped message='no gpu'/></testcase>"
        "<testcase name='c'><failure/></testcase>"
        "</testsuite></testsuites>"
    )
    assert junit_counts(report) == (2, 1)


def test_junit_counts_reports_unknown_rather_than_zero(tmp_path):
    assert junit_counts(tmp_path / "absent.xml") is None
    broken = tmp_path / "broken.xml"
    broken.write_text("<testsuites>")
    assert junit_counts(broken) is None


def test_user_scratch_dir_is_per_user(monkeypatch):
    monkeypatch.setattr("getpass.getuser", lambda: "alice smith")
    assert user_scratch_dir("rocke_verify").name == "rocke_verify-alice_smith"
    monkeypatch.setattr("getpass.getuser", lambda: "bob")
    assert user_scratch_dir("rocke_verify").name == "rocke_verify-bob"
