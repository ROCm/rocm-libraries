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


def test_only_the_star_is_a_wildcard():
    assert ctp.matches("Suite*.Test", "Suite12.Test")
    assert not ctp.matches("Suite?.Test", "Suite1.Test")
    assert ctp.matches("Suite?.Test", "Suite?.Test")
    assert ctp.matches("Suite.Test[0]", "Suite.Test[0]")
    assert not ctp.matches("Suite.Test[0]", "Suite.Test0")


@pytest.mark.parametrize("pattern", ["Suite?.Test", "Suite.Test[0]", "A:B", "A B"])
def test_pattern_the_shared_parser_rejects_fails(run, pattern):
    yaml_text = YAML.replace(
        'test_patterns: ["Full*"]', f'test_patterns: ["Full*", "{pattern}"]'
    )
    with pytest.raises(SystemExit, match="Invalid gtest pattern"):
        run(TESTS, yaml_text)


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
    monkeypatch.setattr(ctp, "device_is_hidden", lambda exe, filter_: True)
    with pytest.raises(
        SystemExit,
        match=r"TestHost.Named: in quick, but skips when no device is visible",
    ):
        run(TESTS, require_match=("quick",), no_device_skips=("quick",))
    assert seen == ["Quick/*:TestHost.Named-*DISABLED*"]


def test_tier_without_device_skips_passes(run, monkeypatch):
    controls = []
    monkeypatch.setattr(ctp, "tests_skipped_without_device", lambda exe, filter_: [])
    monkeypatch.setattr(
        ctp, "device_is_hidden", lambda exe, filter_: controls.append(filter_) or True
    )
    run(TESTS, require_match=("quick",), no_device_skips=("quick",))
    # The control runs the first tier that was not named.
    assert controls == ["*-*DISABLED*:Comprehensive*:Full*:Quick/*:TestHost.Named"]


def test_device_that_is_not_hidden_fails_the_check(run, monkeypatch):
    monkeypatch.setattr(ctp, "tests_skipped_without_device", lambda exe, filter_: [])
    monkeypatch.setattr(ctp, "device_is_hidden", lambda exe, filter_: False)
    with pytest.raises(SystemExit, match="the device is not hidden here"):
        run(TESTS, require_match=("quick",), no_device_skips=("quick",))


_CLOSING_BANNER = "[==========] 2 tests from 1 test suite ran. (0 ms total)\n"


class _Completed:
    def __init__(self, stdout, returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def test_no_device_skip_parser_reads_the_run_line_of_the_skipping_test(monkeypatch):
    output = (
        "[ RUN      ] TestA.Host\n"
        "[       OK ] TestA.Host (0 ms)\n"
        "[ RUN      ] TestA.Device\n"
        "x.cpp(1): Skipped\n"
        "No devices available. Skipping test.\n"
        "[  SKIPPED ] TestA.Device (0 ms)\n" + _CLOSING_BANNER
    )
    monkeypatch.setattr(ctp.subprocess, "run", lambda *a, **k: _Completed(output))
    assert ctp.tests_skipped_without_device("exe", "*") == ["TestA.Device"]


def test_no_device_run_hides_the_device_from_hip_and_rocr(monkeypatch):
    seen = {}

    def run(*args, **kwargs):
        seen.update(kwargs["env"])
        return _Completed(_CLOSING_BANNER)

    monkeypatch.setattr(ctp.subprocess, "run", run)
    ctp.tests_skipped_without_device("exe", "*")
    assert seen["HIP_VISIBLE_DEVICES"] == "-1"
    assert seen["ROCR_VISIBLE_DEVICES"] == "-1"


@pytest.mark.parametrize(
    "completed",
    [
        # An aborted run: the opening banner was printed, the closing one never was.
        _Completed("[==========] Running 2 tests from 1 test suite.\n", returncode=139),
        # A test that fails instead of skipping.
        _Completed(
            "[ RUN      ] TestA.Device\n[  FAILED  ] TestA.Device\n" + _CLOSING_BANNER,
            returncode=1,
        ),
        # An exit status of 0 with no closing banner: the run did not finish its tests.
        _Completed("[==========] Running 2 tests from 1 test suite.\n", returncode=0),
    ],
)
def test_no_device_run_that_does_not_pass_fails(monkeypatch, completed):
    monkeypatch.setattr(ctp.subprocess, "run", lambda *a, **k: completed)
    with pytest.raises(SystemExit, match="with no visible device exited with"):
        ctp.tests_skipped_without_device("exe", "*")


def test_control_run_counts_skips_whatever_the_exit_status(monkeypatch):
    skipping = (
        "[ RUN      ] TestA.Device\nNo devices available. Skipping test.\n"
        "[  SKIPPED ] TestA.Device (0 ms)\n" + _CLOSING_BANNER
    )
    monkeypatch.setattr(
        ctp.subprocess, "run", lambda *a, **k: _Completed(skipping, returncode=1)
    )
    assert ctp.device_is_hidden("exe", "*")
    monkeypatch.setattr(
        ctp.subprocess, "run", lambda *a, **k: _Completed(_CLOSING_BANNER)
    )
    assert not ctp.device_is_hidden("exe", "*")
