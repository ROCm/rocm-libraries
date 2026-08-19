# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the blanket-snapshot-regeneration CI guard (AIHPBLAS-3876).

This tool is the unbypassable backstop against a blanket `.ambr` regeneration
(see the characterization ``README.md``'s "Legitimate bulk regeneration"
section), so its own decision logic -- the threshold, the override lookup, and
the diff-filter choices (added/modified only, never deleted/renamed) -- has to
be pinned directly. Tests exercise real, throwaway git repos (via ``git init``
in ``tmp_path``) rather than mocking git, since the whole point of the tool is
correct git plumbing (merge-base resolution, diff-filter semantics, `git show`
path handling).
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parent
_MODULE_PATH = _TOOLS_DIR / "check_snapshot_diff.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_snapshot_diff", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


csd = _load_module()

pytestmark = pytest.mark.unit

CHAR_DIR = "Tensile/Tests/unit/characterization"


# --------------------------------------------------------------------------- #
# git repo fixture                                                            #
# --------------------------------------------------------------------------- #
class _Repo:
    def __init__(self, root: Path):
        self.root = root

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    def write(self, rel_path: str, content: str) -> None:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def remove(self, rel_path: str) -> None:
        (self.root / rel_path).unlink()

    def commit(self, message: str) -> str:
        self._git("add", "-A")
        self._git("commit", "-q", "-m", message)
        return self._git("rev-parse", "HEAD").strip()


@pytest.fixture
def repo(tmp_path: Path) -> _Repo:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    r = _Repo(tmp_path)
    r._git("config", "user.email", "test@example.com")
    r._git("config", "user.name", "Test")
    r.write(f"{CHAR_DIR}/DataType/__snapshots__/test_datatype_char.ambr", "# base\n")
    r.write(f"{CHAR_DIR}/adr/0001-placeholder.md", "# ADR 0001: placeholder\n\nStatus: Accepted\n")
    r.commit("base commit")
    return r


def _args(base: str, head: str, repo_root: Path, **overrides) -> argparse.Namespace:
    defaults = dict(
        base=base,
        head=head,
        repo_root=str(repo_root),
        characterization_dir=CHAR_DIR,
        threshold=csd.DEFAULT_THRESHOLD,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# --------------------------------------------------------------------------- #
# OVERRIDE_MARKER_RE                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "Bulk-Snapshot-Update: yes\n",
        "Bulk-Snapshot-Update:yes\n",
        "bulk-snapshot-update: YES\n",
        "  Bulk-Snapshot-Update: yes  \n",
        "Status: Accepted\nBulk-Snapshot-Update: yes\nDefect: none\n",
    ],
)
def test_override_marker_matches_valid_forms(text):
    assert csd.OVERRIDE_MARKER_RE.search(text)


@pytest.mark.parametrize(
    "text",
    [
        "Bulk-Snapshot-Update: no\n",
        "Bulk-Snapshot-Update:\n",
        "This ADR does not grant a Bulk-Snapshot-Update: yes style override.\n",
        "Status: Accepted\n",
        "",
    ],
)
def test_override_marker_rejects_invalid_forms(text):
    assert not csd.OVERRIDE_MARKER_RE.search(text)


# --------------------------------------------------------------------------- #
# merge_base                                                                  #
# --------------------------------------------------------------------------- #
def test_merge_base_resolves_common_ancestor(repo: _Repo):
    base_sha = repo._git("rev-parse", "HEAD").strip()
    repo.write(f"{CHAR_DIR}/DataType/__snapshots__/test_datatype_char.ambr", "# changed\n")
    head_sha = repo.commit("change one golden")
    mb = csd.merge_base(base_sha, head_sha, repo.root)
    assert mb == base_sha


def test_merge_base_raises_on_bad_ref(repo: _Repo):
    with pytest.raises(csd.SnapshotDiffError):
        csd.merge_base("not-a-real-ref", "HEAD", repo.root)


# --------------------------------------------------------------------------- #
# changed_ambr_files                                                          #
# --------------------------------------------------------------------------- #
def test_changed_ambr_files_filters_to_ambr_only(repo: _Repo):
    base_sha = repo._git("rev-parse", "HEAD").strip()
    repo.write(f"{CHAR_DIR}/DataType/__snapshots__/test_datatype_char.ambr", "# changed\n")
    repo.write(f"{CHAR_DIR}/DataType/test_datatype_char.py", "# not a golden\n")
    repo.write("README.md", "outside the characterization dir entirely\n")
    head_sha = repo.commit("touch golden, source, and unrelated file")

    changed = csd.changed_ambr_files(base_sha, head_sha, repo.root, CHAR_DIR)
    assert changed == [f"{CHAR_DIR}/DataType/__snapshots__/test_datatype_char.ambr"]


def test_changed_ambr_files_counts_new_goldens_too(repo: _Repo):
    base_sha = repo._git("rev-parse", "HEAD").strip()
    repo.write(f"{CHAR_DIR}/Naming/__snapshots__/test_naming_char.ambr", "# new\n")
    head_sha = repo.commit("add a new golden")

    changed = csd.changed_ambr_files(base_sha, head_sha, repo.root, CHAR_DIR)
    assert changed == [f"{CHAR_DIR}/Naming/__snapshots__/test_naming_char.ambr"]


def test_changed_ambr_files_empty_when_nothing_changed(repo: _Repo):
    base_sha = repo._git("rev-parse", "HEAD").strip()
    repo.write("README.md", "unrelated\n")
    head_sha = repo.commit("unrelated change only")

    assert csd.changed_ambr_files(base_sha, head_sha, repo.root, CHAR_DIR) == []


def test_changed_ambr_files_excludes_pure_renames(repo: _Repo):
    # git detects a byte-identical move as a rename (status R) by default; a pure
    # rename changes no pinned content (e.g. a test function renamed, carrying its
    # golden along untouched), so it must not count toward the threshold.
    base_sha = repo._git("rev-parse", "HEAD").strip()
    original = repo.root / f"{CHAR_DIR}/DataType/__snapshots__/test_datatype_char.ambr"
    renamed = repo.root / f"{CHAR_DIR}/DataType/__snapshots__/test_datatype_v2_char.ambr"
    renamed.write_bytes(original.read_bytes())
    original.unlink()
    head_sha = repo.commit("rename a golden, content untouched")

    # Confirm git actually detected this as a rename (not add+delete) before
    # asserting on the guard's behavior, so the test fails loudly if git's default
    # rename-detection heuristics ever change out from under it.
    statuses = repo._git("diff", "--name-status", "--diff-filter=R", base_sha, head_sha).strip()
    assert statuses, "expected git to detect this as a rename"

    assert csd.changed_ambr_files(base_sha, head_sha, repo.root, CHAR_DIR) == []


# --------------------------------------------------------------------------- #
# changed_adr_files                                                          #
# --------------------------------------------------------------------------- #
def test_changed_adr_files_includes_added_and_modified(repo: _Repo):
    base_sha = repo._git("rev-parse", "HEAD").strip()
    repo.write(f"{CHAR_DIR}/adr/0001-placeholder.md", "# ADR 0001: placeholder (edited)\n")
    repo.write(f"{CHAR_DIR}/adr/0002-new-decision.md", "# ADR 0002: new decision\n")
    head_sha = repo.commit("edit one ADR, add another")

    changed = csd.changed_adr_files(base_sha, head_sha, repo.root, CHAR_DIR)
    assert changed == [
        f"{CHAR_DIR}/adr/0001-placeholder.md",
        f"{CHAR_DIR}/adr/0002-new-decision.md",
    ]


def test_changed_adr_files_excludes_deletions(repo: _Repo):
    base_sha = repo._git("rev-parse", "HEAD").strip()
    repo.remove(f"{CHAR_DIR}/adr/0001-placeholder.md")
    repo.write(f"{CHAR_DIR}/adr/0002-new-decision.md", "# ADR 0002: new decision\n")
    head_sha = repo.commit("delete one ADR, add another")

    # A deleted ADR cannot be read at `head` at all, so it must never be
    # treated as a candidate override -- only the added one should surface.
    changed = csd.changed_adr_files(base_sha, head_sha, repo.root, CHAR_DIR)
    assert changed == [f"{CHAR_DIR}/adr/0002-new-decision.md"]


def test_changed_adr_files_ignores_non_adr_paths(repo: _Repo):
    base_sha = repo._git("rev-parse", "HEAD").strip()
    repo.write(f"{CHAR_DIR}/README.md", "not under adr/\n")
    head_sha = repo.commit("touch a non-adr markdown file")

    assert csd.changed_adr_files(base_sha, head_sha, repo.root, CHAR_DIR) == []


# --------------------------------------------------------------------------- #
# find_override                                                              #
# --------------------------------------------------------------------------- #
def test_find_override_detects_marker(repo: _Repo):
    repo.write(
        f"{CHAR_DIR}/adr/0002-bulk.md",
        "# ADR 0002: bulk regen\n\nStatus: Accepted\nBulk-Snapshot-Update: yes\n",
    )
    head_sha = repo.commit("add override ADR")

    found = csd.find_override([f"{CHAR_DIR}/adr/0002-bulk.md"], head_sha, repo.root)
    assert found == f"{CHAR_DIR}/adr/0002-bulk.md"


def test_find_override_returns_none_without_marker(repo: _Repo):
    repo.write(f"{CHAR_DIR}/adr/0002-no-marker.md", "# ADR 0002: unrelated\n\nStatus: Accepted\n")
    head_sha = repo.commit("add unrelated ADR")

    assert csd.find_override([f"{CHAR_DIR}/adr/0002-no-marker.md"], head_sha, repo.root) is None


def test_find_override_on_empty_candidate_list(repo: _Repo):
    head_sha = repo._git("rev-parse", "HEAD").strip()
    assert csd.find_override([], head_sha, repo.root) is None


# --------------------------------------------------------------------------- #
# cmd_check / main (end-to-end)                                              #
# --------------------------------------------------------------------------- #
def _change_n_goldens(repo: _Repo, n: int, message: str) -> str:
    for i in range(n):
        repo.write(f"{CHAR_DIR}/Module{i}/__snapshots__/test_char.ambr", f"# v{i}\n")
    return repo.commit(message)


def test_cmd_check_passes_within_threshold(repo: _Repo, capsys):
    base_sha = repo._git("rev-parse", "HEAD").strip()
    head_sha = _change_n_goldens(repo, csd.DEFAULT_THRESHOLD, "scoped change")

    rc = csd.cmd_check(_args(base_sha, head_sha, repo.root))
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_cmd_check_fails_over_threshold_without_override(repo: _Repo, capsys):
    base_sha = repo._git("rev-parse", "HEAD").strip()
    head_sha = _change_n_goldens(repo, csd.DEFAULT_THRESHOLD + 1, "blanket regen")

    rc = csd.cmd_check(_args(base_sha, head_sha, repo.root))
    assert rc == 1
    err = capsys.readouterr().err
    assert "FAIL" in err
    assert "Module0" in err  # names an offending file
    assert "Bulk-Snapshot-Update: yes" in err  # remediation is printed


def test_cmd_check_passes_over_threshold_with_override(repo: _Repo, capsys):
    base_sha = repo._git("rev-parse", "HEAD").strip()
    for i in range(csd.DEFAULT_THRESHOLD + 1):
        repo.write(f"{CHAR_DIR}/Module{i}/__snapshots__/test_char.ambr", f"# v{i}\n")
    repo.write(
        f"{CHAR_DIR}/adr/0002-bulk.md",
        "# ADR 0002: intentional bulk regen\n\n"
        "Status: Accepted\nBulk-Snapshot-Update: yes\n\n"
        "## Context\nSnapshot format changed.\n\n## Decision\nRegenerate all goldens.\n"
        "\n## Consequences\nNone.\n",
    )
    head_sha = repo.commit("intentional blanket regen with ADR override")

    rc = csd.cmd_check(_args(base_sha, head_sha, repo.root))
    out = capsys.readouterr().out
    assert rc == 0
    assert "0002-bulk.md" in out


def test_cmd_check_fails_when_adr_present_but_marker_missing(repo: _Repo, capsys):
    base_sha = repo._git("rev-parse", "HEAD").strip()
    for i in range(csd.DEFAULT_THRESHOLD + 1):
        repo.write(f"{CHAR_DIR}/Module{i}/__snapshots__/test_char.ambr", f"# v{i}\n")
    repo.write(
        f"{CHAR_DIR}/adr/0002-unrelated.md",
        "# ADR 0002: unrelated decision\n\nStatus: Accepted\n",
    )
    head_sha = repo.commit("blanket regen with an unrelated ADR (no marker)")

    rc = csd.cmd_check(_args(base_sha, head_sha, repo.root))
    assert rc == 1
    assert "FAIL" in capsys.readouterr().err


def test_cmd_check_cli_threshold_override(repo: _Repo):
    base_sha = repo._git("rev-parse", "HEAD").strip()
    head_sha = _change_n_goldens(repo, csd.DEFAULT_THRESHOLD + 5, "wider change")

    assert csd.cmd_check(_args(base_sha, head_sha, repo.root)) == 1
    assert csd.cmd_check(_args(base_sha, head_sha, repo.root, threshold=100)) == 0


def test_main_returns_two_on_bad_base_ref(repo: _Repo, capsys):
    rc = csd.main(
        [
            "--base",
            "not-a-real-ref",
            "--head",
            "HEAD",
            "--repo-root",
            str(repo.root),
        ]
    )
    assert rc == 2
    assert "error" in capsys.readouterr().err


def test_main_end_to_end_pass(repo: _Repo):
    base_sha = repo._git("rev-parse", "HEAD").strip()
    head_sha = _change_n_goldens(repo, 1, "single scoped change")

    rc = csd.main(
        [
            "--base",
            base_sha,
            "--head",
            head_sha,
            "--repo-root",
            str(repo.root),
            "--characterization-dir",
            CHAR_DIR,
        ]
    )
    assert rc == 0


def test_main_end_to_end_fail(repo: _Repo):
    base_sha = repo._git("rev-parse", "HEAD").strip()
    head_sha = _change_n_goldens(repo, csd.DEFAULT_THRESHOLD + 1, "blanket regen")

    rc = csd.main(
        [
            "--base",
            base_sha,
            "--head",
            head_sha,
            "--repo-root",
            str(repo.root),
            "--characterization-dir",
            CHAR_DIR,
        ]
    )
    assert rc == 1
