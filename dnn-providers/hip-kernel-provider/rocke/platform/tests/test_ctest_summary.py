# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the ctest --output-junit -> step-summary renderer.

These cover the honesty-critical paths a green CI run can never exercise: a
failing/errored/skipped test rendering correctly, the no-op when no summary
file is configured, and that a broken/missing report is loud-but-non-fatal.
"""

from __future__ import annotations

from pathlib import Path

import ctest_summary as cs

_JUNIT = """<?xml version="1.0"?>
<testsuite name="hip_kernel_provider" tests="4" failures="1" skipped="1">
  <testcase name="hip_kernel_provider_integration_tests" time="517.11" status="run"/>
  <testcase name="rocke_pytest" time="4.85"><failure message="boom">trace</failure></testcase>
  <testcase name="rocke_crashed" time="0.0"><error message="segv"/></testcase>
  <testcase name="rocke_gfx950_only" time="0.0"><skipped/></testcase>
</testsuite>
"""


def _write(tmp_path: Path, xml: str) -> Path:
    p = tmp_path / "ctest.xml"
    p.write_text(xml, encoding="utf-8")
    return p


def test_parse_classifies_pass_fail_error_skip(tmp_path):
    results = {r.name: r for r in cs.parse_junit(_write(tmp_path, _JUNIT))}
    assert results["hip_kernel_provider_integration_tests"].status == "Passed"
    assert results["rocke_pytest"].status == "Failed"
    assert results["rocke_crashed"].status == "Failed"  # <error> is never green
    assert results["rocke_gfx950_only"].status == "Skipped"
    assert results["hip_kernel_provider_integration_tests"].seconds == 517.11


def test_summary_renders_failed_and_skipped_rows(tmp_path):
    summary = tmp_path / "summary.md"
    cs.write_step_summary(_write(tmp_path, _JUNIT), "gfx94X-dcgpu", str(summary))
    out = summary.read_text(encoding="utf-8")
    assert "gfx94X-dcgpu" in out
    assert "❌ | `rocke_pytest`" in out
    assert "❌ | `rocke_crashed`" in out  # crash surfaces, never omitted/green
    assert "⚪ | `rocke_gfx950_only`" in out
    assert "1/4 ctest entries passed" in out  # 1 pass, 2 fail (failure+error), 1 skip


def test_no_op_when_summary_path_unset(tmp_path):
    # local runs have no $GITHUB_STEP_SUMMARY; must not raise or write anything
    cs.write_step_summary(_write(tmp_path, _JUNIT), "gfx94X", None)
    cs.write_step_summary(_write(tmp_path, _JUNIT), "gfx94X", "")


def test_missing_report_is_loud_but_not_fatal(tmp_path):
    summary = tmp_path / "summary.md"
    # must not raise even though the JUnit file does not exist
    cs.write_step_summary(tmp_path / "does-not-exist.xml", "gfx94X", str(summary))
    assert "summary generation failed" in summary.read_text(encoding="utf-8")
