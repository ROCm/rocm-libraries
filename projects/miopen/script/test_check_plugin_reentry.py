# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for check_plugin_reentry.py.

The script decides between pass, fail and skip from what two runs of the shim test print, so
the tests stand in for that test with a program whose output each case chooses. No build or
GPU is needed:

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

# Behaves as the shim test does in each situation. The run with the stand-in is the one that
# has HIPDNN_PLUGIN_DIR set; every run is appended to FAKE_LOG with the environment it saw.
FAKE_GTEST = f"""#!{sys.executable}
import json, os, sys, time

with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps({{
        "stand_in": "HIPDNN_PLUGIN_DIR" in os.environ,
        "plugin_dir": os.environ.get("HIPDNN_PLUGIN_DIR"),
        "forwarding": os.environ.get("MIOPEN_HIPDNN_FORWARDING"),
        "deny": os.environ.get("MIOPEN_DISABLE_HIPDNN_FOR"),
        "args": sys.argv[1:],
    }}) + "\\n")

mode = os.environ["FAKE_STAND_IN" if "HIPDNN_PLUGIN_DIR" in os.environ else "FAKE_CONTROL"]
if mode == "message":
    print("[MIOpen] [hipDNN-forwarded] miopenStatusInternalError: {reentry.MESSAGE} "
          "miopenConvolutionForward while hipDNN was running a forwarded call.")
    sys.exit(1)
if mode == "pass":
    print("{reentry.PASSED_LINE}.")
    sys.exit(0)
if mode == "nomatch":
    print("Running 0 tests")
    sys.exit(0)
if mode == "hang":
    time.sleep(60)
print("some unrelated failure")
sys.exit(1)
"""


@pytest.fixture
def setup(tmp_path, monkeypatch):
    gtest = tmp_path / "miopen_gtest"
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


def check(gtest, stand_in, monkeypatch, stand_in_mode, control_mode, extra=()):
    monkeypatch.setenv("FAKE_STAND_IN", stand_in_mode)
    monkeypatch.setenv("FAKE_CONTROL", control_mode)
    return reentry.main(
        [
            "--gtest",
            str(gtest),
            "--stand-in-dir",
            str(stand_in),
            "--timeout",
            "5",
            *extra,
        ]
    )


def test_the_message_is_a_pass_and_needs_no_control_run(setup, monkeypatch, capsys):
    gtest, stand_in, log = setup
    assert check(gtest, stand_in, monkeypatch, "message", "fail") == reentry.PASS
    assert len(runs(log)) == 1
    assert "PASS" in capsys.readouterr().out


def test_no_message_on_a_machine_that_runs_the_test_is_a_failure(
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
    """What the unguarded recursion looks like when it does not crash first."""
    gtest, stand_in, log = setup
    monkeypatch.setenv("FAKE_STAND_IN", "hang")
    monkeypatch.setenv("FAKE_CONTROL", "pass")
    result = reentry.main(
        ["--gtest", str(gtest), "--stand-in-dir", str(stand_in), "--timeout", "1"]
    )
    assert result == reentry.FAIL
    assert "did not finish" in capsys.readouterr().out


@pytest.mark.parametrize("control", ["fail", "nomatch", "hang"])
def test_a_machine_that_cannot_run_the_test_skips(setup, monkeypatch, capsys, control):
    """No GPU, an unsupported GPU or no hipDNN: the control run fails the same way."""
    gtest, stand_in, log = setup
    monkeypatch.setenv("FAKE_STAND_IN", "silent")
    monkeypatch.setenv("FAKE_CONTROL", control)
    result = reentry.main(
        ["--gtest", str(gtest), "--stand-in-dir", str(stand_in), "--timeout", "1"]
    )
    assert result == reentry.SKIP
    assert "SKIP" in capsys.readouterr().out


def test_a_missing_test_binary_skips(setup, monkeypatch, capsys):
    gtest, stand_in, log = setup
    gtest.unlink()
    assert check(gtest, stand_in, monkeypatch, "message", "pass") == reentry.SKIP
    assert "is not built" in capsys.readouterr().out
    assert runs(log) == []


def test_a_missing_stand_in_skips(setup, monkeypatch, capsys):
    gtest, stand_in, log = setup
    (stand_in / "libstand_in.so").unlink()
    assert check(gtest, stand_in, monkeypatch, "message", "pass") == reentry.SKIP
    assert "stand-in" in capsys.readouterr().out
    assert runs(log) == []


def test_a_missing_stand_in_directory_skips(setup, monkeypatch):
    gtest, stand_in, log = setup
    assert (
        check(gtest, stand_in / "absent", monkeypatch, "message", "pass")
        == reentry.SKIP
    )


def test_only_the_stand_in_run_is_pointed_at_the_stand_in(setup, monkeypatch):
    gtest, stand_in, log = setup
    check(gtest, stand_in, monkeypatch, "silent", "pass")
    stand_in_run, control_run = runs(log)
    assert stand_in_run["plugin_dir"] == str(stand_in)
    assert not control_run["stand_in"]


def test_both_runs_forward_and_use_the_requested_test(setup, monkeypatch):
    gtest, stand_in, log = setup
    check(gtest, stand_in, monkeypatch, "silent", "pass", ["--filter", "Suite.Test"])
    for run in runs(log):
        assert run["forwarding"] == "1"
        assert run["args"] == ["--gtest_filter=Suite.Test"]


def test_the_callers_deny_list_and_plugin_directory_do_not_reach_either_run(
    setup, monkeypatch
):
    """A deny list would keep the call on MIOpen; a plugin directory would replace the
    stand-in or change what the control run loads."""
    gtest, stand_in, log = setup
    monkeypatch.setenv("MIOPEN_DISABLE_HIPDNN_FOR", "*")
    monkeypatch.setenv("HIPDNN_PLUGIN_DIR", "/somewhere/else")
    check(gtest, stand_in, monkeypatch, "silent", "pass")
    stand_in_run, control_run = runs(log)
    assert stand_in_run["deny"] is None and control_run["deny"] is None
    assert stand_in_run["plugin_dir"] == str(stand_in)
    assert not control_run["stand_in"]
