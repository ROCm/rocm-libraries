# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No new torch use anywhere in the rocke tree.

rocKE does not use torch. The uses that remain are listed, one line per file
with the kinds of use it makes, in ``no_torch_allowlist.txt`` beside this file,
and are being removed. The gate reads every Python, requirements, pyproject and
CMake file under ``rocke/`` with :mod:`rocke.tests.torch_uses` and fails when a
file not on the list uses torch, when a listed file gains a kind of use, and
when a listed file or kind no longer uses torch (delete or shorten its line:
the list only shrinks). TESTING.md ("No torch") states the rule.

It reads the source checkout through git, so it runs in the build tree only
(``rocke_no_torch_check``), never in the installed test tree.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from rocke.assets import platform_root
from rocke.tests.torch_uses import (
    TorchUseKind,
    allowlist_breaches,
    file_kinds,
    format_allowlist_entry,
    read_allowlist,
    tree_files,
    tree_torch_uses,
)

ROCKE = platform_root().parent
ALLOWLIST = Path(__file__).with_name("no_torch_allowlist.txt")
GUIDANCE = (
    "rocKE does not use torch; use numpy references / rocke.numeric / DeviceArray "
    '(TESTING.md, "No torch"). The allowlist only shrinks: remove the lines of '
    "uses you removed, never add one."
)


def test_the_tree_adds_no_torch_use():
    breaches = allowlist_breaches(tree_torch_uses(ROCKE), read_allowlist(ALLOWLIST))
    assert not breaches, GUIDANCE + "\n" + "\n".join(breaches)


def test_the_allowlist_is_sorted_one_canonical_line_per_file():
    listed = [
        line
        for line in ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    canonical = [
        format_allowlist_entry(path, kinds)
        for path, kinds in sorted(read_allowlist(ALLOWLIST).items())
    ]
    assert listed == canonical


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        [shutil.which("git") or "git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """A git work tree with one committed torch import, one listed in the
    allowlist below."""
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("scratch/\n", encoding="utf-8")
    (tmp_path / "listed.py").write_text("import torch\n", encoding="utf-8")
    (tmp_path / "clean.py").write_text("import numpy\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    return tmp_path


LISTED = {"listed.py": frozenset({TorchUseKind.IMPORT})}


def test_the_tree_is_what_git_tracks_or_would_track(repository):
    (repository / "untracked.py").write_text("x = 1\n", encoding="utf-8")
    (repository / "CMakeLists.txt").write_text("project(x)\n", encoding="utf-8")
    (repository / "notes.md").write_text("import torch\n", encoding="utf-8")
    (repository / "scratch").mkdir()
    (repository / "scratch" / "probe.py").write_text("import torch\n", encoding="utf-8")
    assert tree_files(repository) == [
        "CMakeLists.txt",
        "clean.py",
        "listed.py",
        "untracked.py",
    ]


def test_a_directory_outside_a_git_work_tree_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    if (
        subprocess.run(
            [shutil.which("git") or "git", "rev-parse", "--is-inside-work-tree"],
            cwd=outside,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    ):
        pytest.fail(f"expected {outside} outside any git work tree")
    with pytest.raises(
        RuntimeError, match=r"tracked_files failed - root=.*git work tree"
    ):
        tree_files(outside)


def test_a_planted_new_file_fails_the_gate(repository):
    (repository / "new.py").write_text("def f():\n    import torch\n", encoding="utf-8")
    assert allowlist_breaches(tree_torch_uses(repository), LISTED) == [
        "new.py: new torch use: line 2 import: import torch"
    ]


def test_a_planted_new_kind_in_a_listed_file_fails_the_gate(repository):
    (repository / "listed.py").write_text(
        "import torch\ntorch.cuda.synchronize()\n", encoding="utf-8"
    )
    assert allowlist_breaches(tree_torch_uses(repository), LISTED) == [
        "listed.py: new kind of torch use: line 2 torch_api: torch.cuda"
    ]


def test_a_listed_file_whose_torch_use_was_removed_fails_the_gate(repository):
    (repository / "listed.py").write_text("import numpy\n", encoding="utf-8")
    assert allowlist_breaches(tree_torch_uses(repository), LISTED) == [
        "listed.py: stale entry, no torch use left: delete its line"
    ]


def test_the_planted_files_pass_once_listed(repository):
    (repository / "requirements.txt").write_text("torch\n", encoding="utf-8")
    found = tree_torch_uses(repository)
    allowed = {path: file_kinds(uses) for path, uses in found.items()}
    assert allowed == {
        "listed.py": frozenset({TorchUseKind.IMPORT}),
        "requirements.txt": frozenset({TorchUseKind.CONFIG}),
    }
    assert allowlist_breaches(found, allowed) == []
