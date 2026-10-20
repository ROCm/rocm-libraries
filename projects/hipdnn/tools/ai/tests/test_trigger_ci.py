# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the hipdnn-trigger-ci skill's trigger_ci.py.

Every subprocess call goes to FakeGh, so no test reaches GitHub or dispatches
a real workflow. FakeGh serves the workflow files from this checkout, so the
tests also catch a workflow change that breaks the script or SKILL.md.
"""

import importlib.util
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS_AI_DIR = Path(__file__).resolve().parent.parent
SKILL_DIR = TOOLS_AI_DIR / "skills" / "hipdnn-trigger-ci"
TRIGGER_CI = SKILL_DIR / "scripts" / "trigger_ci.py"
WORKFLOWS_DIR = TOOLS_AI_DIR.parents[3] / ".github" / "workflows"

REPO = "ROCm/rocm-libraries"
REPO_ARG = f"github.com/{REPO}"
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
        # Workflow file name -> text served instead of the checkout's copy.
        self.workflows = {}
        self.auth_error = ""
        self.local_head = HEAD_SHA

    def __call__(self, cmd, capture_output=False, text=False, check=False, **_):
        self.calls.append(cmd)
        if cmd[0] == "gh":
            _assert_on_github_com(cmd)
        if cmd[:3] == ["gh", "auth", "status"] and self.auth_error:
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr=self.auth_error
            )
        if cmd[:2] == ["gh", "api"] and "/contents/.github/workflows/" in cmd[-1]:
            return self._workflow_file(cmd)
        stdout = self._answer(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    def _workflow_file(self, cmd):
        name = cmd[-1].split("/contents/.github/workflows/")[1].split("?")[0]
        text = self.workflows.get(name)
        if text is None and (WORKFLOWS_DIR / name).is_file():
            text = (WORKFLOWS_DIR / name).read_text()
        if text is None:
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="gh: Not Found (HTTP 404)"
            )
        return subprocess.CompletedProcess(cmd, 0, stdout=text, stderr="")

    def _answer(self, cmd):
        if cmd[:3] == ["gh", "auth", "status"]:
            return ""
        if cmd[:3] == ["gh", "pr", "view"]:
            return json.dumps(self.pr_head)
        if cmd[:2] == ["gh", "api"] and "/commits/" in cmd[4]:
            return HEAD_SHA
        if cmd[:2] == ["gh", "api"] and "/actions/runs/" in cmd[4]:
            return json.dumps(self.baseline)
        if cmd[:3] == ["gh", "run", "list"] and "--commit" in cmd:
            return json.dumps(self.active_runs)
        if cmd[:3] == ["gh", "run", "list"] and "--event" in cmd:
            return json.dumps([{"databaseId": self.last_dispatch_id}])
        if cmd[:3] == ["gh", "run", "list"] and "--status" in cmd:
            return json.dumps([{"databaseId": 7, "workflowName": "CI"}])
        if cmd[:3] in (["gh", "run", "list"], ["gh", "pr", "checks"]):
            return ""
        if cmd[:3] == ["gh", "run", "watch"]:
            return ""
        if cmd[:3] == ["gh", "workflow", "run"]:
            self.dispatched.append(cmd)
            self.last_dispatch_id += 1
            return ""
        if cmd[:4] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return "users/someone/feature"
        if cmd == ["git", "rev-parse", "HEAD"]:
            return self.local_head
        if cmd[0] == "git":
            return ""
        raise AssertionError(f"unexpected command: {cmd}")


def _assert_on_github_com(cmd):
    """Every gh call names github.com, so GH_HOST cannot send it elsewhere."""
    if cmd[1] == "api":
        assert cmd[2:4] == ["--hostname", "github.com"], cmd
    elif cmd[1] == "auth":
        assert cmd[-2:] == ["--hostname", "github.com"], cmd
    else:
        assert cmd[cmd.index("--repo") + 1] == REPO_ARG, cmd


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


BRANCH = ["--branch", "users/someone/feature"]
MULTI_ARCH = [*BRANCH, "dispatch", "-w", "multi-arch"]

# A workflow named like therock-ci.yml, so the tests below do not depend on
# which inputs the real one has a default for.
FAKE_WORKFLOW = """\
name: Fake
on:
  workflow_dispatch:
    inputs:
      projects:
        type: string
      families:
        type: string
        default: "gfx94X"  # comment
"""


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", [[], ["--dry-run", "--yes"]])
def test_dispatch_needs_exactly_one_of_dry_run_and_yes(
    trigger_mod, gh, monkeypatch, mode
):
    assert _exit_code(trigger_mod, monkeypatch, *MULTI_ARCH, *mode) == 2
    assert gh.dispatched == []


def test_gh_auth_failure_shows_gh_error(trigger_mod, gh, monkeypatch, capsys):
    gh.auth_error = "github.com\n  X Failed to log in to github.com account someone"
    assert _exit_code(trigger_mod, monkeypatch, *MULTI_ARCH, "--dry-run") == 1
    err = capsys.readouterr().err
    assert "X Failed to log in to github.com account someone" in err
    assert "gh auth login --hostname github.com" in err
    assert len(gh.calls) == 1


def test_input_without_default_is_required(trigger_mod, gh, monkeypatch, capsys):
    gh.workflows["therock-ci.yml"] = FAKE_WORKFLOW
    code = _exit_code(
        trigger_mod,
        monkeypatch,
        *BRANCH,
        "dispatch",
        "-w",
        "therock-ci",
        "-f",
        "families=gfx950",
        "-f",
        "projects=",
        "--yes",
    )
    assert code == 1
    assert "input projects has no default" in capsys.readouterr().err
    assert gh.dispatched == []


def test_unknown_input_is_rejected_with_the_current_list(
    trigger_mod, gh, monkeypatch, capsys
):
    gh.workflows["therock-ci.yml"] = FAKE_WORKFLOW
    code = _exit_code(
        trigger_mod,
        monkeypatch,
        *BRANCH,
        "dispatch",
        "-w",
        "therock-ci",
        "-f",
        "projects=all",
        "-f",
        "linux_test_labels=test:hipdnn",
        "--dry-run",
    )
    assert code == 1
    err = capsys.readouterr().err
    assert (
        "therock-ci.yml at 'users/someone/feature' has no input linux_test_labels"
        in err
    )
    assert "its inputs are: projects, families (default 'gfx94X')" in err
    assert gh.dispatched == []


@pytest.mark.parametrize(
    "fields, message",
    [
        (["-f", "projects"], "-f 'projects' is not name=value"),
        (["-f", "=all"], "-f '=all' is not name=value"),
        (["-f", "projects=a", "-f", "projects=b"], "-f projects given twice"),
    ],
    ids=["no-equals", "no-name", "twice"],
)
def test_malformed_field_is_rejected(
    trigger_mod, gh, monkeypatch, capsys, fields, message
):
    code = _exit_code(
        trigger_mod,
        monkeypatch,
        *BRANCH,
        "dispatch",
        "-w",
        "therock-ci",
        *fields,
        "--dry-run",
    )
    assert code == 1
    assert message in capsys.readouterr().err
    assert gh.dispatched == []


def test_missing_workflow_at_ref_stops_dispatch(trigger_mod, gh, monkeypatch, capsys):
    monkeypatch.setitem(trigger_mod.WORKFLOWS, "multi-arch", "no-such-workflow.yml")
    assert _exit_code(trigger_mod, monkeypatch, *MULTI_ARCH, "--dry-run") == 1
    assert "Not Found (HTTP 404)" in capsys.readouterr().err
    assert gh.dispatched == []


def test_workflow_without_dispatch_trigger_is_rejected(
    trigger_mod, gh, monkeypatch, capsys
):
    gh.workflows["therock-multi-arch-ci.yml"] = (
        "on:\n  push:\n    branches: [develop]\n"
    )
    assert _exit_code(trigger_mod, monkeypatch, *MULTI_ARCH, "--dry-run") == 1
    assert "has no workflow_dispatch trigger" in capsys.readouterr().err


def test_reuse_build_needs_the_baseline_inputs(trigger_mod, gh, monkeypatch, capsys):
    gh.workflows["therock-ci.yml"] = FAKE_WORKFLOW
    code = _exit_code(
        trigger_mod,
        monkeypatch,
        *BRANCH,
        "dispatch",
        "-w",
        "therock-ci",
        "-f",
        "projects=all",
        "--reuse-build",
        str(BASELINE_ID),
        "--dry-run",
    )
    assert code == 1
    assert (
        "--reuse-build needs baseline_run_id, prebuilt_stages, which therock-ci.yml"
        in capsys.readouterr().err
    )
    assert gh.dispatched == []


def test_reuse_build_refuses_its_inputs_via_field(trigger_mod, gh, monkeypatch, capsys):
    code = _exit_code(
        trigger_mod,
        monkeypatch,
        *MULTI_ARCH,
        "-f",
        "prebuilt_stages=all",
        "--reuse-build",
        str(BASELINE_ID),
        "--dry-run",
    )
    assert code == 1
    assert "--reuse-build sets prebuilt_stages" in capsys.readouterr().err


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
        "-f",
        "linux_amdgpu_families=gfx94X",
        "-f",
        "linux_test_labels=test:hipdnn",
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
# Local HEAD against the remote head of a branch taken from the checkout
# --------------------------------------------------------------------------- #

INFERRED_MULTI_ARCH = ["dispatch", "-w", "multi-arch"]


def test_dry_run_warns_when_local_head_differs(trigger_mod, gh, monkeypatch, capsys):
    gh.local_head = OTHER_SHA
    _run(trigger_mod, monkeypatch, *INFERRED_MULTI_ARCH, "--dry-run")
    err = capsys.readouterr().err
    assert f"local HEAD is {OTHER_SHA[:11]}" in err
    assert f"the run tests {HEAD_SHA[:11]}" in err
    assert "also needs --allow-head-mismatch" in err
    assert gh.dispatched == []


def test_yes_refuses_when_local_head_differs(trigger_mod, gh, monkeypatch, capsys):
    gh.local_head = OTHER_SHA
    assert _exit_code(trigger_mod, monkeypatch, *INFERRED_MULTI_ARCH, "--yes") == 1
    assert "pass --allow-head-mismatch" in capsys.readouterr().err
    assert gh.dispatched == []


def test_allow_head_mismatch_dispatches(trigger_mod, gh, monkeypatch):
    gh.local_head = OTHER_SHA
    _run(
        trigger_mod,
        monkeypatch,
        *INFERRED_MULTI_ARCH,
        "--allow-head-mismatch",
        "--yes",
    )
    assert len(gh.dispatched) == 1


def test_matching_local_head_dispatches(trigger_mod, gh, monkeypatch, capsys):
    _run(trigger_mod, monkeypatch, *INFERRED_MULTI_ARCH, "--yes")
    assert len(gh.dispatched) == 1
    assert "local HEAD" not in capsys.readouterr().err


@pytest.mark.parametrize("target", [BRANCH, ["--pr", "1"]], ids=["branch", "pr"])
def test_explicit_ref_skips_local_head_check(trigger_mod, gh, monkeypatch, target):
    gh.local_head = OTHER_SHA
    _run(trigger_mod, monkeypatch, *target, *INFERRED_MULTI_ARCH, "--yes")
    assert len(gh.dispatched) == 1
    assert ["git", "rev-parse", "HEAD"] not in gh.calls


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
        "-f",
        "linux_amdgpu_families=gfx94X",
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


# --------------------------------------------------------------------------- #
# Workflow parsing, and the script against this checkout's workflows and SKILL.md
# --------------------------------------------------------------------------- #


def test_dispatch_inputs_reads_defaults_and_flow_style(trigger_mod):
    assert trigger_mod.dispatch_inputs(FAKE_WORKFLOW) == {
        "projects": None,
        "families": "gfx94X",
    }
    assert trigger_mod.dispatch_inputs("on: [push, workflow_dispatch]\n") == {}
    assert trigger_mod.dispatch_inputs("on: [push]\n") is None


@pytest.mark.parametrize("alias", ["therock-ci", "multi-arch", "hipdnn-superbuild"])
def test_checkout_workflows_are_dispatchable(trigger_mod, alias):
    path = WORKFLOWS_DIR / trigger_mod.WORKFLOWS[alias]
    assert path.is_file(), f"{alias} points to missing {path}"
    declared = trigger_mod.dispatch_inputs(path.read_text())
    assert declared is not None, f"{path.name} has no workflow_dispatch trigger"
    if alias == "multi-arch":
        assert set(trigger_mod.REUSE_INPUTS) <= set(declared)


def _yaml_dispatch_inputs(data):
    on = data.get("on", data.get(True))  # YAML 1.1 loads a bare on: as True
    if isinstance(on, (str, list)):
        return {} if "workflow_dispatch" in on else None
    if not isinstance(on, dict) or "workflow_dispatch" not in on:
        return None
    inputs = (on["workflow_dispatch"] or {}).get("inputs") or {}
    result = {}
    for name, spec in inputs.items():
        default = (spec or {}).get("default")
        if isinstance(default, bool):
            default = str(default).lower()
        result[name] = None if default is None else str(default)
    return result


@pytest.mark.parametrize(
    "path", sorted(WORKFLOWS_DIR.glob("*.y*ml")), ids=lambda path: path.name
)
def test_dispatch_inputs_matches_pyyaml(trigger_mod, path):
    yaml = pytest.importorskip("yaml")
    expected = _yaml_dispatch_inputs(yaml.safe_load(path.read_text()) or {})
    assert trigger_mod.dispatch_inputs(path.read_text()) == expected


SKILL_PLACEHOLDERS = {
    "<pr-number>": "1",
    "<run-id>": str(BASELINE_ID),
    "<branch>": "users/someone/feature",
}


def _skill_commands():
    """The runnable trigger_ci.py lines in SKILL.md (templates with [...] skipped)."""
    text = (SKILL_DIR / "SKILL.md").read_text()
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        for line in block.splitlines():
            if "scripts/trigger_ci.py" not in line or "[" in line:
                continue
            for placeholder, value in SKILL_PLACEHOLDERS.items():
                line = line.replace(placeholder, value)
            argv = shlex.split(line.split("scripts/trigger_ci.py", 1)[1])
            commands.append(argv)
    return commands


def test_skill_md_has_dispatch_examples_for_every_workflow(trigger_mod):
    used = {argv[argv.index("-w") + 1] for argv in _skill_commands() if "-w" in argv}
    assert used == set(trigger_mod.WORKFLOWS)


@pytest.mark.parametrize("argv", _skill_commands(), ids=shlex.join)
def test_skill_md_command_runs(trigger_mod, gh, monkeypatch, capsys, argv):
    """Each SKILL.md example parses, and its -f inputs exist in the workflow."""
    try:
        _run(trigger_mod, monkeypatch, *argv)
    except SystemExit as exc:
        assert exc.code in (0, None), capsys.readouterr().err
    assert gh.dispatched == []
