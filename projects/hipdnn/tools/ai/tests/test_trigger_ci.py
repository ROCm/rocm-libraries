# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the hipdnn-trigger-ci skill's trigger_ci.py.

Every subprocess call goes to FakeGh, so no test reaches GitHub or dispatches
a real workflow.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS_AI_DIR = Path(__file__).resolve().parent.parent
TRIGGER_CI = TOOLS_AI_DIR / "skills" / "hipdnn-trigger-ci" / "scripts" / "trigger_ci.py"

REPO = "ROCm/rocm-libraries"
HEAD_SHA = "a" * 40
OTHER_SHA = "b" * 40
BASELINE_ID = 36880722738


@pytest.fixture(scope="module")
def trigger_mod() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hipdnn_trigger_ci", TRIGGER_CI)
    assert spec and spec.loader, f"could not load {TRIGGER_CI}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeGh:
    """Stands in for subprocess.run; answers the gh/git calls trigger_ci makes."""

    def __init__(self):
        self.calls = []
        self.dispatched = []
        self.active_runs = []
        self.baseline = {
            "path": ".github/workflows/therock-multi-arch-ci.yml",
            "head_sha": HEAD_SHA,
            "status": "completed",
            "repo": REPO,
        }
        self.pr_head = {
            "headRefName": "users/someone/feature",
            "isCrossRepository": False,
        }
        self.last_dispatch_id = 100

    def __call__(self, cmd, capture_output=False, text=False, check=False, **_):
        self.calls.append(cmd)
        stdout = self._answer(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    def _answer(self, cmd):
        if cmd[:3] == ["gh", "auth", "status"]:
            return ""
        if cmd[:3] == ["gh", "pr", "view"]:
            return json.dumps(self.pr_head)
        if cmd[:2] == ["gh", "api"] and "/commits/" in cmd[2]:
            return HEAD_SHA
        if cmd[:2] == ["gh", "api"] and "/actions/runs/" in cmd[2]:
            return json.dumps(self.baseline)
        if cmd[:3] == ["gh", "run", "list"] and "--commit" in cmd:
            return json.dumps(self.active_runs)
        if cmd[:3] == ["gh", "run", "list"] and "--event" in cmd:
            return json.dumps([{"databaseId": self.last_dispatch_id}])
        if cmd[:3] == ["gh", "workflow", "run"]:
            self.dispatched.append(cmd)
            self.last_dispatch_id += 1
            return ""
        if cmd[0] == "git":
            # No checkout root: skips the TheRock pin-drift check.
            return ""
        raise AssertionError(f"unexpected command: {cmd}")


@pytest.fixture
def gh(trigger_mod, monkeypatch) -> FakeGh:
    fake = FakeGh()
    monkeypatch.setattr(trigger_mod.subprocess, "run", fake)
    monkeypatch.setattr(trigger_mod.time, "sleep", lambda _seconds: None)
    return fake


def _run(trigger_mod, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["trigger_ci.py", *argv])
    trigger_mod.main()


def _exit_code(trigger_mod, monkeypatch, *argv):
    with pytest.raises(SystemExit) as exc:
        _run(trigger_mod, monkeypatch, *argv)
    return exc.value.code


def _inputs(dispatch_cmd):
    return dict(
        dispatch_cmd[i + 1].split("=", 1)
        for i, arg in enumerate(dispatch_cmd)
        if arg == "-f"
    )


MULTI_ARCH = ["--branch", "users/someone/feature", "dispatch", "-w", "multi-arch"]


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", [[], ["--dry-run", "--yes"]])
def test_dispatch_needs_exactly_one_of_dry_run_and_yes(
    trigger_mod, gh, monkeypatch, mode
):
    assert _exit_code(trigger_mod, monkeypatch, *MULTI_ARCH, *mode) == 2
    assert gh.dispatched == []


def test_therock_ci_requires_projects(trigger_mod, gh, monkeypatch, capsys):
    code = _exit_code(
        trigger_mod,
        monkeypatch,
        "--branch",
        "users/someone/feature",
        "dispatch",
        "-w",
        "therock-ci",
        "--gfx",
        "gfx94X",
        "--yes",
    )
    assert code == 1
    assert "needs --projects" in capsys.readouterr().err
    assert gh.dispatched == []


@pytest.mark.parametrize(
    "workflow_args",
    [["therock-ci", "--projects", "all"], ["hipdnn-superbuild"]],
    ids=["therock-ci", "hipdnn-superbuild"],
)
def test_reuse_build_is_multi_arch_only(
    trigger_mod, gh, monkeypatch, capsys, workflow_args
):
    code = _exit_code(
        trigger_mod,
        monkeypatch,
        "--branch",
        "users/someone/feature",
        "dispatch",
        "-w",
        *workflow_args,
        "--reuse-build",
        str(BASELINE_ID),
        "--dry-run",
    )
    assert code == 1
    assert "--reuse-build is only valid for -w multi-arch" in capsys.readouterr().err
    assert gh.dispatched == []


def test_fork_pr_is_refused(trigger_mod, gh, monkeypatch, capsys):
    gh.pr_head = {"headRefName": "main", "isCrossRepository": True}
    code = _exit_code(
        trigger_mod, monkeypatch, "--pr", "1", "dispatch", "-w", "multi-arch", "--yes"
    )
    assert code == 1
    assert "is from a fork" in capsys.readouterr().err
    assert gh.dispatched == []


# --------------------------------------------------------------------------- #
# Shared branches
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("branch", ["develop", "release/therock-7.11"])
def test_shared_branch_is_refused(trigger_mod, gh, monkeypatch, capsys, branch):
    code = _exit_code(
        trigger_mod,
        monkeypatch,
        "--branch",
        branch,
        "dispatch",
        "-w",
        "multi-arch",
        "--yes",
    )
    assert code == 1
    assert "--allow-shared-branch" in capsys.readouterr().err
    assert gh.dispatched == []


def test_allow_shared_branch_overrides_refusal(trigger_mod, gh, monkeypatch, capsys):
    _run(
        trigger_mod,
        monkeypatch,
        "--branch",
        "develop",
        "dispatch",
        "-w",
        "multi-arch",
        "--allow-shared-branch",
        "--dry-run",
    )
    assert "gh workflow run therock-multi-arch-ci.yml" in capsys.readouterr().out
    assert gh.dispatched == []


# --------------------------------------------------------------------------- #
# Runs cancelled through the concurrency group
# --------------------------------------------------------------------------- #


def _run_entry(run_id, status, event, branch="users/someone/feature"):
    return {
        "databaseId": run_id,
        "status": status,
        "event": event,
        "headBranch": branch,
    }


def test_runs_cancelled_by_dispatch_filters_completed_and_pr_runs(trigger_mod, gh):
    gh.active_runs = [
        _run_entry(1, "in_progress", "workflow_dispatch"),
        _run_entry(2, "queued", "push", "develop"),
        _run_entry(3, "in_progress", "schedule"),
        _run_entry(4, "completed", "workflow_dispatch"),
        _run_entry(5, "in_progress", "pull_request"),
        _run_entry(6, "in_progress", "pull_request_target"),
    ]
    doomed = trigger_mod.runs_cancelled_by_dispatch(
        "therock-multi-arch-ci.yml", HEAD_SHA
    )
    assert [run["databaseId"] for run in doomed] == [1, 2, 3]
    list_cmd = gh.calls[-1]
    assert list_cmd[list_cmd.index("--commit") + 1] == HEAD_SHA
    assert list_cmd[list_cmd.index("--workflow") + 1] == "therock-multi-arch-ci.yml"


def test_dry_run_lists_runs_it_would_cancel(trigger_mod, gh, monkeypatch, capsys):
    gh.active_runs = [_run_entry(37055366239, "in_progress", "workflow_dispatch")]
    _run(trigger_mod, monkeypatch, *MULTI_ARCH, "--dry-run")
    err = capsys.readouterr().err
    assert "cancels 1 active run(s)" in err
    assert "37055366239" in err
    assert "also needs --cancel-active" in err
    assert gh.dispatched == []


def test_yes_refuses_to_cancel_active_runs(trigger_mod, gh, monkeypatch, capsys):
    gh.active_runs = [_run_entry(37055366239, "in_progress", "workflow_dispatch")]
    assert _exit_code(trigger_mod, monkeypatch, *MULTI_ARCH, "--yes") == 1
    assert "pass --cancel-active" in capsys.readouterr().err
    assert gh.dispatched == []


def test_yes_with_cancel_active_dispatches(trigger_mod, gh, monkeypatch, capsys):
    gh.active_runs = [_run_entry(37055366239, "in_progress", "workflow_dispatch")]
    _run(trigger_mod, monkeypatch, *MULTI_ARCH, "--cancel-active", "--yes")
    assert len(gh.dispatched) == 1
    assert "run 101" in capsys.readouterr().out


def test_yes_dispatches_when_nothing_is_active(trigger_mod, gh, monkeypatch, capsys):
    _run(
        trigger_mod,
        monkeypatch,
        *MULTI_ARCH,
        "--gfx",
        "gfx94X",
        "--test-labels",
        "test:hipdnn",
        "--yes",
    )
    assert len(gh.dispatched) == 1
    cmd = gh.dispatched[0]
    assert cmd[cmd.index("--ref") + 1] == "users/someone/feature"
    assert _inputs(cmd) == {
        "linux_amdgpu_families": "gfx94X",
        "linux_test_labels": "test:hipdnn",
    }
    assert "cancels" not in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# --reuse-build
# --------------------------------------------------------------------------- #


def test_reuse_build_sets_baseline_and_prebuilt_stages(
    trigger_mod, gh, monkeypatch, capsys
):
    _run(
        trigger_mod,
        monkeypatch,
        *MULTI_ARCH,
        "--gfx",
        "gfx94X",
        "--reuse-build",
        str(BASELINE_ID),
        "--yes",
    )
    assert len(gh.dispatched) == 1
    assert _inputs(gh.dispatched[0]) == {
        "linux_amdgpu_families": "gfx94X",
        "baseline_run_id": str(BASELINE_ID),
        "prebuilt_stages": "all",
    }


def test_reuse_build_accepts_path_with_ref_suffix(trigger_mod, gh):
    gh.baseline["path"] = ".github/workflows/therock-multi-arch-ci.yml@refs/heads/x"
    trigger_mod.check_reuse_baseline(
        BASELINE_ID, "therock-multi-arch-ci.yml", "x", HEAD_SHA
    )


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("path", ".github/workflows/therock-ci.yml", "is a run of"),
        ("repo", "someone/rocm-libraries", "built code from someone/rocm-libraries"),
        ("head_sha", OTHER_SHA, f"built {OTHER_SHA[:11]}, but 'x' is at"),
        ("status", "in_progress", "is still in_progress"),
    ],
    ids=["wrong-workflow", "fork", "other-commit", "unfinished"],
)
def test_reuse_build_rejects_unusable_baseline(
    trigger_mod, gh, capsys, field, value, message
):
    gh.baseline[field] = value
    with pytest.raises(SystemExit) as exc:
        trigger_mod.check_reuse_baseline(
            BASELINE_ID, "therock-multi-arch-ci.yml", "x", HEAD_SHA
        )
    assert exc.value.code == 1
    assert (
        f"error: --reuse-build run {BASELINE_ID} {message}" in capsys.readouterr().err
    )


def test_reuse_build_rejection_blocks_dispatch(trigger_mod, gh, monkeypatch):
    gh.baseline["status"] = "in_progress"
    code = _exit_code(
        trigger_mod,
        monkeypatch,
        *MULTI_ARCH,
        "--reuse-build",
        str(BASELINE_ID),
        "--cancel-active",
        "--yes",
    )
    assert code == 1
    assert gh.dispatched == []
