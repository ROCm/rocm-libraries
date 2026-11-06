# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The examples driver reports through the shared runner verdict contract."""

import json
import os
import tempfile

import pytest
from rocke.examples import run_all
from rocke.tests.runner_verdict import EXIT_GREEN, EXIT_INCOMPLETE, EXIT_RED

_REGISTRY = [
    run_all.Example(name="add_demo", module="demo.add", family="elementwise"),
    run_all.Example(name="gemm_demo", module="demo.gemm", family="gemm"),
]


@pytest.fixture
def driver(monkeypatch, tmp_path):
    """Run ``main`` against a fake registry; no example process ever starts.

    ``statuses`` maps an example name to the ``RunResult.status`` its fake run
    reports. Returns ``(exit code, ran example names)``.
    """
    ran = []

    def run(argv, statuses):
        def fake_run_one(ex, *, bless, live_arch):
            ran.append(ex.name)
            status = statuses[ex.name]
            reason = "no GPU visible" if status == "SKIPPED" else ""
            return run_all.RunResult(
                name=ex.name, family=ex.family, status=status, reason=reason
            )

        monkeypatch.setattr(run_all, "run_one", fake_run_one)
        return run_all.main([*argv, "--json", str(tmp_path / "dash.json")]), ran

    monkeypatch.setattr(run_all, "REGISTRY", _REGISTRY)
    monkeypatch.setattr(run_all, "_live_arch", lambda: None)
    return run


def test_all_examples_skipped_is_incomplete(driver, capsys):
    code, ran = driver(["--check"], {"add_demo": "SKIPPED", "gemm_demo": "SKIPPED"})
    out = capsys.readouterr().out
    assert code == EXIT_INCOMPLETE
    assert ran == ["add_demo", "gemm_demo"]
    assert out.rstrip().endswith("RESULT: INCOMPLETE")
    assert "skipped           elementwise/add_demo (no GPU visible)" in out
    # The legacy summary block survives.
    assert "PASS=0 FAIL=0 DRIFT=0 SKIPPED=2 ERROR=0" in out


def test_allow_incomplete_exits_zero_but_still_says_incomplete(driver, capsys):
    code, _ = driver(
        ["--check", "--allow-incomplete"],
        {"add_demo": "SKIPPED", "gemm_demo": "SKIPPED"},
    )
    out = capsys.readouterr().out
    assert code == EXIT_GREEN
    assert "RESULT: INCOMPLETE (allowed by --allow-incomplete)" in out
    assert "RESULT: GREEN" not in out


@pytest.mark.parametrize("bad", ["FAIL", "DRIFT", "ERROR"])
def test_a_failing_example_is_red_even_with_allow_incomplete(driver, capsys, bad):
    code, _ = driver(
        ["--check", "--allow-incomplete"],
        {"add_demo": "SKIPPED", "gemm_demo": bad},
    )
    out = capsys.readouterr().out
    assert code == EXIT_RED
    assert f"failed            gemm/gemm_demo ({bad})" in out
    assert out.rstrip().endswith("RESULT: RED")


def test_only_matching_nothing_is_red(driver, capsys):
    code, ran = driver(["--check", "--only", "nomatch"], {})
    out = capsys.readouterr().out
    assert code == EXIT_RED
    assert ran == []
    assert "selected no examples" in out
    assert out.rstrip().endswith("RESULT: RED")


def test_passing_narrowed_run_is_partial_not_green(driver, capsys):
    code, ran = driver(["--check", "--only", "gemm"], {"gemm_demo": "PASS"})
    out = capsys.readouterr().out
    assert code == EXIT_GREEN
    assert ran == ["gemm_demo"]
    assert "narrowed by: --only gemm" in out
    assert out.rstrip().endswith("RESULT: PARTIAL")
    assert "RESULT: GREEN" not in out


def test_full_passing_run_is_green(driver, capsys, tmp_path):
    code, _ = driver(["--check"], {"add_demo": "PASS", "gemm_demo": "PASS"})
    out = capsys.readouterr().out
    assert code == EXIT_GREEN
    assert out.rstrip().endswith("RESULT: GREEN")
    rows = json.loads((tmp_path / "dash.json").read_text())
    assert [row["status"] for row in rows] == ["PASS", "PASS"]


def test_default_dashboard_path_needs_no_uid(monkeypatch, tmp_path, capsys):
    """The default ``--json`` is a per-user temp file, built without ``os.getuid``."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delattr(os, "getuid", raising=False)
    monkeypatch.setenv("LOGNAME", "tester")
    monkeypatch.setattr(run_all, "REGISTRY", _REGISTRY[:1])
    monkeypatch.setattr(run_all, "_live_arch", lambda: None)
    monkeypatch.setattr(
        run_all,
        "run_one",
        lambda ex, *, bless, live_arch: run_all.RunResult(
            name=ex.name, family=ex.family, status="PASS"
        ),
    )
    assert run_all.main(["--check"]) == EXIT_GREEN
    out = capsys.readouterr().out
    dashboard = tmp_path / "ck_examples_run-tester.json"
    assert json.loads(dashboard.read_text())[0]["status"] == "PASS"
    assert f"dashboard: {dashboard}" in out
