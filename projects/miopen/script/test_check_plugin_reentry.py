# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for check_plugin_reentry.py.

The script decides between pass, fail and skip from what two runs of the re-entry test
print, so the tests stand in for that test with a program whose output each case chooses.
No build or GPU is needed:

    python -m pytest projects/miopen/script/test_check_plugin_reentry.py
"""

import json
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_plugin_reentry as reentry  # noqa: E402

# Behaves as the re-entry test does in each situation. The run with the stand-in is the one
# that has HIPDNN_PLUGIN_DIR set; every run is appended to FAKE_LOG with the environment it saw.
FAKE_GTEST = f"""#!{sys.executable}
import json, os, sys, time

KEYS = {list(reentry.CALLER_SETTINGS)!r}
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps({{
        "stand_in": "HIPDNN_PLUGIN_DIR" in os.environ,
        "plugin_dir": os.environ.get("HIPDNN_PLUGIN_DIR"),
        "expect": os.environ.get("{reentry.EXPECT_REJECTION}"),
        "forwarding": os.environ.get("MIOPEN_HIPDNN_FORWARDING"),
        "leaked": sorted(k for k in KEYS if k in os.environ
                         and k not in ("HIPDNN_PLUGIN_DIR", "{reentry.EXPECT_REJECTION}")),
        "args": sys.argv[1:],
    }}) + "\\n")

mode = os.environ["FAKE_STAND_IN" if "HIPDNN_PLUGIN_DIR" in os.environ else "FAKE_CONTROL"]
if mode == "message_pass":
    print("[MIOpen] [hipDNN-forwarded] miopenStatusInternalError: {reentry.MESSAGE} "
          "miopenConvolutionForward while hipDNN was running a forwarded call.")
    print("{reentry.PASSED_LINE}.")
    sys.exit(0)
if mode == "message_fail":
    print("[MIOpen] [hipDNN-forwarded] miopenStatusInternalError: {reentry.MESSAGE} "
          "miopenConvolutionForward while hipDNN was running a forwarded call.")
    print("[  FAILED  ] the application's call returned success")
    sys.exit(1)
if mode == "pass_no_message":
    print("{reentry.PASSED_LINE}.")
    sys.exit(0)
if mode == "pass":
    print("{reentry.PASSED_LINE}.")
    sys.exit(0)
if mode == "no_test":
    print("[==========] {reentry.NO_TEST_LINE} from 0 test suites.")
    sys.exit(0)
if mode == "hang":
    time.sleep(60)
print("some unrelated failure")
sys.exit(1)
"""


@pytest.fixture
def setup(tmp_path, monkeypatch):
    gtest = tmp_path / "test_binary"
    gtest.write_text(FAKE_GTEST)
    gtest.chmod(gtest.stat().st_mode | stat.S_IXUSR)
    stand_in = tmp_path / "public_stand_in"
    stand_in.mkdir()
    (stand_in / "libstand_in.so").write_bytes(b"")
    log = tmp_path / "runs.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))
    return gtest, stand_in, log


def runs(log):
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def check(
    gtest, stand_in, monkeypatch, stand_in_mode, control_mode, extra=(), timeout="5"
):
    monkeypatch.setenv("FAKE_STAND_IN", stand_in_mode)
    monkeypatch.setenv("FAKE_CONTROL", control_mode)
    return reentry.main(
        [
            "--gtest",
            str(gtest),
            "--stand-in-dir",
            str(stand_in),
            "--timeout",
            timeout,
            *extra,
        ]
    )


def test_a_passing_test_with_the_message_is_a_pass_and_needs_no_control_run(
    setup, monkeypatch, capsys
):
    gtest, stand_in, log = setup
    assert check(gtest, stand_in, monkeypatch, "message_pass", "fail") == reentry.PASS
    assert len(runs(log)) == 1
    assert "PASS" in capsys.readouterr().out


def test_the_message_alone_is_not_a_pass(setup, monkeypatch, capsys):
    """The message printed but the test failed, for example the application's call still
    returned success: the old script reported PASS here."""
    gtest, stand_in, log = setup
    assert check(gtest, stand_in, monkeypatch, "message_fail", "pass") == reentry.FAIL
    out = capsys.readouterr().out
    assert "failed" in out
    assert "the application's call returned success" in out


def test_a_pass_without_the_message_is_a_failure(setup, monkeypatch, capsys):
    gtest, stand_in, log = setup
    assert (
        check(gtest, stand_in, monkeypatch, "pass_no_message", "pass") == reentry.FAIL
    )
    assert "passed without the wrapper's message" in capsys.readouterr().out


def test_a_silent_failure_on_a_machine_that_runs_the_test_is_a_failure(
    setup, monkeypatch, capsys
):
    """The guard is gone: the call went wrong some other way, silently."""
    gtest, stand_in, log = setup
    assert check(gtest, stand_in, monkeypatch, "silent", "pass") == reentry.FAIL
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "some unrelated failure" in out


def test_a_call_that_never_returns_is_a_failure_on_a_machine_that_runs_the_test(
    setup, monkeypatch, capsys
):
    """What the unguarded recursion, or a handle that cannot be destroyed, looks like."""
    gtest, stand_in, log = setup
    assert (
        check(gtest, stand_in, monkeypatch, "hang", "pass", timeout="1") == reentry.FAIL
    )
    assert "did not finish" in capsys.readouterr().out


@pytest.mark.parametrize("control", ["fail", "hang", "no_test"])
def test_a_machine_that_cannot_run_the_test_skips(setup, monkeypatch, capsys, control):
    """No GPU, an unsupported GPU or no hipDNN: the control run fails the same way."""
    gtest, stand_in, log = setup
    assert (
        check(gtest, stand_in, monkeypatch, "silent", control, timeout="1")
        == reentry.SKIP
    )
    assert "SKIP" in capsys.readouterr().out


def test_a_binary_without_the_test_is_a_failure_not_a_skip(setup, monkeypatch, capsys):
    """A renamed or removed test would otherwise fail its control run too and skip forever."""
    gtest, stand_in, log = setup
    assert check(gtest, stand_in, monkeypatch, "no_test", "no_test") == reentry.FAIL
    assert "selects no test" in capsys.readouterr().out
    assert len(runs(log)) == 1


def test_a_missing_test_binary_skips(setup, monkeypatch, capsys):
    gtest, stand_in, log = setup
    gtest.unlink()
    assert check(gtest, stand_in, monkeypatch, "message_pass", "pass") == reentry.SKIP
    assert "is not built" in capsys.readouterr().out
    assert runs(log) == []


def test_a_missing_stand_in_skips(setup, monkeypatch, capsys):
    gtest, stand_in, log = setup
    (stand_in / "libstand_in.so").unlink()
    assert check(gtest, stand_in, monkeypatch, "message_pass", "pass") == reentry.SKIP
    assert "stand-in" in capsys.readouterr().out
    assert runs(log) == []


def test_a_missing_stand_in_directory_skips(setup, monkeypatch):
    gtest, stand_in, log = setup
    assert (
        check(gtest, stand_in / "absent", monkeypatch, "message_pass", "pass")
        == reentry.SKIP
    )


def test_only_the_stand_in_run_is_pointed_at_the_stand_in_and_expects_a_rejection(
    setup, monkeypatch
):
    gtest, stand_in, log = setup
    check(gtest, stand_in, monkeypatch, "silent", "pass")
    stand_in_run, control_run = runs(log)
    assert stand_in_run["plugin_dir"] == str(stand_in)
    assert stand_in_run["expect"] == "1"
    assert not control_run["stand_in"]
    assert control_run["expect"] is None


def test_both_runs_forward_and_use_the_requested_test(setup, monkeypatch):
    gtest, stand_in, log = setup
    check(gtest, stand_in, monkeypatch, "silent", "pass", ["--filter", "Suite.Test"])
    for run in runs(log):
        assert run["forwarding"] == "1"
        assert run["args"] == ["--gtest_filter=Suite.Test"]


@pytest.mark.parametrize(
    "setting, value",
    [
        ("GTEST_TOTAL_SHARDS", "2"),
        ("GTEST_SHARD_INDEX", "1"),
        ("GTEST_FILTER", "Nothing.Matches"),
        ("MIOPEN_DISABLE_HIPDNN_FOR", "*"),
        ("MIOPEN_LOG_LEVEL", "1"),
    ],
)
def test_the_callers_settings_do_not_reach_either_run(
    setup, monkeypatch, setting, value
):
    """A shard or filter could select no test, a deny list would keep the call on MIOpen and
    a log level of 1 or 2 hides the message."""
    gtest, stand_in, log = setup
    monkeypatch.setenv(setting, value)
    monkeypatch.setenv("HIPDNN_PLUGIN_DIR", "/somewhere/else")
    check(gtest, stand_in, monkeypatch, "silent", "pass")
    stand_in_run, control_run = runs(log)
    assert stand_in_run["leaked"] == [] and control_run["leaked"] == []
    assert stand_in_run["plugin_dir"] == str(stand_in)
    assert not control_run["stand_in"]


def test_a_relative_stand_in_directory_reaches_hipdnn_as_an_absolute_one(
    setup, monkeypatch
):
    """hipDNN ignores a relative HIPDNN_PLUGIN_DIR and loads its default plugins instead."""
    gtest, stand_in, log = setup
    monkeypatch.chdir(stand_in.parent)
    check(gtest, Path(stand_in.name), monkeypatch, "silent", "pass")
    stand_in_run, _ = runs(log)
    assert Path(stand_in_run["plugin_dir"]).is_absolute()
    assert stand_in_run["plugin_dir"] == str(stand_in)
