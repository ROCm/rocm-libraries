# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for check_tier_partition.py (pytest)."""

import textwrap

import pytest

import check_tier_partition as ctp

_YAML_TEXT = """\
    test_categories:
      quick:
        test_patterns: ["Quick/*", "TestHost.Named"]
        exclude: ["*DISABLED*"]
      standard:
        test_patterns: ["*"]
        exclude: ["*DISABLED*", "Comprehensive*", "Full*", "Quick/*", "TestHost.Named"]
      comprehensive:
        test_patterns: ["Comprehensive*"]
      full:
        test_patterns: ["Full*"]
      ffm-quick:
        test_patterns: ["Quick/*"]
"""
YAML = textwrap.dedent(_YAML_TEXT)

TESTS = [
    "Quick/TestGpuA.Compare/0",
    "Standard/TestGpuA.Compare/0",
    "TestHost.Named",
    "TestHost.Other",
    "Comprehensive/TestGpuA.Compare/0",
    "Full/TestGpuA.Compare/0",
    "TestGpuA.DISABLED_Broken",
]


@pytest.fixture
def run(tmp_path, monkeypatch):
    def _run(tests, yaml_text=YAML, require_match=("quick",), no_device_skips=()):
        path = tmp_path / "tiers.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        monkeypatch.setattr(ctp, "list_tests", lambda exe: tests)
        ctp.check(str(path), "exe", list(require_match), list(no_device_skips))

    return _run


def test_every_test_in_exactly_one_tier(run, capsys):
    run(TESTS)
    assert "each in exactly one tier" in capsys.readouterr().out


def test_test_selected_by_no_tier_fails(run):
    yaml_text = YAML.replace(
        'exclude: ["*DISABLED*", "Comprehensive*"',
        'exclude: ["*DISABLED*", "TestHost.Other", "Comprehensive*"',
    )
    with pytest.raises(SystemExit, match=r"TestHost.Other: selected by no tier"):
        run(TESTS, yaml_text)


def test_test_selected_by_two_tiers_fails(run):
    yaml_text = YAML.replace(
        'test_patterns: ["Quick/*", "TestHost.Named"]',
        'test_patterns: ["Quick/*", "TestHost.Named", "Standard/*"]',
    )
    with pytest.raises(
        SystemExit,
        match=r"Standard/TestGpuA.Compare/0: selected by \['quick', 'standard'\]",
    ):
        run(TESTS, yaml_text)


def test_quick_pattern_that_selects_nothing_fails(run):
    tests = [t for t in TESTS if t != "TestHost.Named"]
    with pytest.raises(
        SystemExit, match=r"quick test_patterns entry 'TestHost.Named' selects no test"
    ):
        run(tests)


def test_no_listed_tests_fails(run):
    with pytest.raises(SystemExit, match="listed no tests"):
        run([])


def test_question_mark_and_bracket_follow_gtest_rules():
    assert ctp.matches("Suite?.Test", "Suite1.Test")
    assert not ctp.matches("Suite?.Test", "Suite12.Test")
    assert ctp.matches("Suite.Test[0]", "Suite.Test[0]")
    assert not ctp.matches("Suite.Test[0]", "Suite.Test0")


def test_gtest_filter_is_positive_minus_negative():
    assert (
        ctp.gtest_filter({"test_patterns": ["*"], "exclude": ["A*", "B*"]}) == "*-A*:B*"
    )
    assert ctp.gtest_filter({"test_patterns": ["Quick/*", "X.Y"]}) == "Quick/*:X.Y"


def test_device_test_in_a_tier_that_must_run_without_a_device_fails(run, monkeypatch):
    seen = []

    def skipped(exe, filter_):
        seen.append(filter_)
        return ["TestHost.Named"]

    monkeypatch.setattr(ctp, "tests_skipped_without_device", skipped)
    with pytest.raises(
        SystemExit,
        match=r"TestHost.Named: in quick, but skips when no device is visible",
    ):
        run(TESTS, require_match=("quick",), no_device_skips=("quick",))
    assert seen == ["Quick/*:TestHost.Named-*DISABLED*"]


def test_tier_without_device_skips_passes(run, monkeypatch):
    monkeypatch.setattr(ctp, "tests_skipped_without_device", lambda exe, filter_: [])
    run(TESTS, require_match=("quick",), no_device_skips=("quick",))


def test_no_device_skip_parser_reads_the_run_line_of_the_skipping_test(monkeypatch):
    output = (
        "[ RUN      ] TestA.Host\n"
        "[       OK ] TestA.Host (0 ms)\n"
        "[ RUN      ] TestA.Device\n"
        "x.cpp(1): Skipped\n"
        "No devices available. Skipping test.\n"
        "[  SKIPPED ] TestA.Device (0 ms)\n"
        "[==========] 2 tests ran.\n"
    )

    class Completed:
        stdout = output
        stderr = ""

    monkeypatch.setattr(ctp.subprocess, "run", lambda *a, **k: Completed())
    assert ctp.tests_skipped_without_device("exe", "*") == ["TestA.Device"]
