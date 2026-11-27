# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A hook runs for a change that deletes files it takes and edits none.

pre-commit gives a hook only the changed files that still exist, so a change
that deletes a hook's files and edits none skips it. ``rocke/tools/
run_hook_on_deletion.py`` runs from an always_run hook and runs each named hook
for such a change, from the staged changes or, with PRE_COMMIT_FROM_REF and
PRE_COMMIT_TO_REF set, from the diff between them. Each case builds a real git
repository whose hooks record that they ran. The last cases hold the
monorepo's own deletion hook to what the script needs of it, and to naming
every rocKE hook that takes no file names.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from rocke.assets import platform_root
from rocke.tests.lanes import SOURCE_TREE, skip_until_reason
from rocke.tests.source_modules import load_module_from_path

pytest.importorskip("yaml", reason="reading a pre-commit config needs PyYAML")

ROCKE = platform_root().resolve().parent
GATE = ROCKE / "tools" / "run_hook_on_deletion.py"
# rocke -> hip-kernel-provider -> dnn-providers -> monorepo.
MONOREPO = ROCKE.parents[2]
# The monorepo's always_run hook that runs this script.
DELETION_HOOK = "rocke-on-deletion"
pytestmark = [
    pytest.mark.skipif(
        not GATE.is_file(),
        reason=skip_until_reason(
            SOURCE_TREE,
            reason="rocke/tools/run_hook_on_deletion.py is absent from this tree",
        ),
    ),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
]

# The hook "registry" records that it ran and fails with HOOK_STATUS, so a case
# sees both whether it ran and that its status comes back; "notes" records that
# it ran and passes.
HOOK_STATUS = 3
CONFIG = f"""\
repos:
  - repo: local
    hooks:
      - id: registry
        name: registry
        entry: python -c "import pathlib, sys; pathlib.Path('ran').touch(); sys.exit({HOOK_STATUS})"
        language: python
        files: ^lib/
        exclude: \\.md$
        pass_filenames: false
      - id: notes
        name: notes
        entry: python -c "import pathlib; pathlib.Path('ran-notes').touch()"
        language: python
        files: \\.md$
        pass_filenames: false
      - id: per-file
        name: per-file
        entry: python -c "pass"
        language: python
        files: ^lib/
"""
# The scratch repositories are throwaway: their commits need an identity, and
# no user or system git config (commit signing, a hooks path, init templates)
# may apply to them, so a developer's or runner's settings cannot fail a case.
_SCRATCH_GIT = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        env={**os.environ, **_SCRATCH_GIT},
    )


@pytest.fixture
def repo(tmp_path):
    """A repository whose one commit holds the config, two files the hook takes,
    one it excludes and one outside its files."""
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / ".pre-commit-config.yaml").write_text(CONFIG)
    (tmp_path / "lib").mkdir()
    for name in ("lib/kept.py", "lib/dropped.py", "lib/notes.md", "other.py"):
        (tmp_path / name).write_text("# file\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def _run_gate(
    repo: Path, *hook_ids: str, refs: tuple[str, str] = ()
) -> subprocess.CompletedProcess:
    """Run the script from ``repo`` for ``hook_ids`` (default: registry)."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in ("PRE_COMMIT_FROM_REF", "PRE_COMMIT_TO_REF")
    }
    environment.update(_SCRATCH_GIT)
    if refs:
        environment["PRE_COMMIT_FROM_REF"], environment["PRE_COMMIT_TO_REF"] = refs
    # pre-commit puts the hook environment's bin directory first on PATH, so
    # an entry's "python" is that environment's.
    environment["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), environment.get("PATH", "")]
    )
    return subprocess.run(
        [sys.executable, str(GATE), *(hook_ids or ("registry",))],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _ran(repo: Path) -> bool:
    return (repo / "ran").exists()


def _ran_notes(repo: Path) -> bool:
    return (repo / "ran-notes").exists()


def _commit_on_branch(repo: Path, *git_steps: tuple[str, ...]) -> None:
    _git(repo, "checkout", "-q", "-b", "change")
    for step in git_steps:
        _git(repo, *step)
    _git(repo, "commit", "-q", "-m", "change")


def test_a_users_git_config_does_not_reach_the_scratch_repository(
    tmp_path, monkeypatch
):
    """A global config that signs every commit and runs a failing hook would
    fail each commit of a scratch repository; it does not apply there."""
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    hook = hooks / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    hostile = tmp_path / "gitconfig"
    hostile.write_text(
        f"[commit]\n\tgpgsign = true\n[core]\n\thooksPath = {hooks.as_posix()}\n"
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(hostile))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _git(scratch, "init", "-q", "-b", "main")
    (scratch / "file.txt").write_text("text\n")
    _git(scratch, "add", "-A")
    _git(scratch, "commit", "-q", "-m", "base")
    log = subprocess.run(
        ["git", "log", "--format=%s"],
        cwd=scratch,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **_SCRATCH_GIT},
    )
    assert log.stdout.split() == ["base"]


def test_a_change_that_deletes_nothing_reads_no_config_and_runs_no_hook(repo):
    (repo / "lib/kept.py").write_text("# edited\n")
    _git(repo, "add", "lib/kept.py")
    # An id the config lacks: the script returns before it reads the config.
    assert _run_gate(repo, "no-such-hook").returncode == 0
    assert not _ran(repo)


def test_a_staged_deletion_of_a_file_the_hook_takes_runs_it(repo):
    _git(repo, "rm", "-q", "lib/dropped.py")
    result = _run_gate(repo)
    assert result.returncode == HOOK_STATUS
    assert _ran(repo)
    assert (
        "registry: the change deletes 1 file(s) it takes and edits none, so it runs "
        "here; first: lib/dropped.py"
    ) in result.stdout


def test_a_rename_out_of_the_hooks_files_counts_as_a_deletion(repo):
    _git(repo, "mv", "lib/dropped.py", "other_dropped.py")
    assert _run_gate(repo).returncode == HOOK_STATUS
    assert _ran(repo)


@pytest.mark.parametrize(
    "deleted",
    [
        # The hook's exclude.
        "lib/notes.md",
        # Outside the hook's files.
        "other.py",
    ],
)
def test_deleting_a_file_the_hook_does_not_take_runs_no_hook(repo, deleted):
    _git(repo, "rm", "-q", deleted)
    assert _run_gate(repo).returncode == 0
    assert not _ran(repo)


def test_a_deletion_beside_an_edit_the_hook_takes_leaves_it_to_pre_commit(repo):
    _git(repo, "rm", "-q", "lib/dropped.py")
    (repo / "lib/kept.py").write_text("# edited\n")
    _git(repo, "add", "lib/kept.py")
    assert _run_gate(repo).returncode == 0
    assert not _ran(repo)


def test_a_deletion_between_two_refs_runs_the_hook(repo):
    _commit_on_branch(repo, ("rm", "-q", "lib/dropped.py"))
    assert _run_gate(repo, refs=("main", "change")).returncode == HOOK_STATUS
    assert _ran(repo)


def test_a_rename_between_two_refs_counts_as_a_deletion(repo):
    _commit_on_branch(repo, ("mv", "lib/dropped.py", "other_dropped.py"))
    assert _run_gate(repo, refs=("main", "change")).returncode == HOOK_STATUS
    assert _ran(repo)


def test_between_two_refs_a_deletion_beside_an_edit_is_left_to_pre_commit(repo):
    (repo / "lib/kept.py").write_text("# edited\n")
    _commit_on_branch(repo, ("rm", "-q", "lib/dropped.py"), ("add", "lib/kept.py"))
    assert _run_gate(repo, refs=("main", "change")).returncode == 0
    assert not _ran(repo)


def test_between_two_refs_a_staged_deletion_is_not_the_change(repo):
    _git(repo, "checkout", "-q", "-b", "change")
    _git(repo, "rm", "-q", "lib/dropped.py")
    assert _run_gate(repo, refs=("main", "change")).returncode == 0
    assert not _ran(repo)


def test_two_refs_with_no_merge_base_compare_whole_trees(repo):
    # A root commit with no shared history: "main...change" has no merge base,
    # so the change is the whole difference between the two trees.
    _git(repo, "checkout", "-q", "--orphan", "change")
    # The orphan branch's index holds the files as new, so removing one needs -f.
    _git(repo, "rm", "-q", "-f", "lib/dropped.py")
    _git(repo, "commit", "-q", "-m", "unrelated")
    assert _run_gate(repo, refs=("main", "change")).returncode == HOOK_STATUS
    assert _ran(repo)


def test_a_hook_that_takes_file_names_is_refused(repo):
    _git(repo, "rm", "-q", "lib/dropped.py")
    result = _run_gate(repo, "per-file")
    assert result.returncode == 2
    assert result.stderr.strip() == (
        "running a hook for deleted files failed - hook='per-file' takes file "
        "names, expected pass_filenames: false (a change that only deletes files "
        "has none to give it)"
    )


def test_an_unknown_hook_is_refused_when_a_change_deletes_files(repo):
    _git(repo, "rm", "-q", "lib/dropped.py")
    result = _run_gate(repo, "no-such-hook")
    assert result.returncode == 2
    assert result.stderr.strip() == (
        "loading a pre-commit hook failed - id='no-such-hook', expected a local "
        "hook of .pre-commit-config.yaml with that id"
    )


def test_each_named_hook_runs_for_the_deletions_it_takes(repo):
    _git(repo, "rm", "-q", "lib/dropped.py", "lib/notes.md")
    result = _run_gate(repo, "registry", "notes")
    # registry fails, and notes still runs after it.
    assert result.returncode == HOOK_STATUS
    assert _ran(repo) and _ran_notes(repo)


def test_a_named_hook_that_takes_no_deleted_path_does_not_run(repo):
    _git(repo, "rm", "-q", "lib/notes.md")
    assert _run_gate(repo, "registry", "notes").returncode == 0
    assert _ran_notes(repo)
    assert not _ran(repo)


def test_every_named_hook_is_checked_before_any_runs(repo):
    _git(repo, "rm", "-q", "lib/dropped.py")
    result = _run_gate(repo, "registry", "per-file")
    assert result.returncode == 2
    assert not _ran(repo)


@pytest.fixture(scope="module")
def monorepo_hooks():
    config = MONOREPO / ".pre-commit-config.yaml"
    if not config.is_file():
        pytest.skip(f"no monorepo pre-commit config at {config}")
    hooks = load_module_from_path(
        "precommit_hooks_for_deletion", ROCKE / "tools" / "precommit_hooks.py"
    )
    return {hook.hook_id: hook for hook in hooks.load_local_hooks(config)}


def test_the_monorepos_deletion_hook_runs_hooks_this_script_can_run(monorepo_hooks):
    deletion = monorepo_hooks[DELETION_HOOK]
    checked_out, gate, *hook_ids = deletion.args
    assert (MONOREPO / checked_out).resolve() == ROCKE
    assert (MONOREPO / gate).resolve() == GATE
    assert hook_ids
    for hook_id in hook_ids:
        target = monorepo_hooks[hook_id]
        assert not target.pass_filenames, hook_id
        # The script runs each target's command in the deletion hook's
        # environment.
        assert (target.language, target.additional_dependencies) == (
            deletion.language,
            deletion.additional_dependencies,
        ), hook_id


def test_the_monorepos_deletion_hook_names_every_rocke_hook_without_file_names(
    monorepo_hooks,
):
    # A pure deletion can leave any whole-tree check stale (a registry record,
    # an allowlist line, a source file CMake still lists), and pre-commit runs
    # none of them for it. A hook that takes file names is given none for a
    # deletion, and one that always runs needs no help.
    expected = {
        hook_id
        for hook_id, hook in monorepo_hooks.items()
        if hook_id.startswith("rocke-")
        and not hook.pass_filenames
        and hook_id != DELETION_HOOK
    }
    _, _, *named = monorepo_hooks[DELETION_HOOK].args
    assert sorted(named) == sorted(set(named)), "a hook is named twice"
    assert set(named) == expected, (
        f"{DELETION_HOOK} in .pre-commit-config.yaml must name exactly the rocKE "
        f"hooks that take no file names; add {sorted(expected - set(named))}, "
        f"drop {sorted(set(named) - expected)}"
    )
