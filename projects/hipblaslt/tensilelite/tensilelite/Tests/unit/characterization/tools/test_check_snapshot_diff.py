# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the blanket-snapshot-regeneration CI guard (AIHPBLAS-3876).

This tool is the unbypassable backstop against a blanket `.ambr` regeneration
(see the characterization ``README.md``'s "Legitimate bulk regeneration"
section), so its own decision logic -- the threshold, the override lookup, and
which changes count (content changes only, never deletions or byte-identical
renames) -- has to be pinned directly. Tests exercise real, throwaway git repos (via ``git init``
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
# has_override_marker                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "Bulk-Snapshot-Update: yes\n",
        "Bulk-Snapshot-Update:yes\n",
        "bulk-snapshot-update: YES\n",
        "Bulk-Snapshot-Update:  yes  \n",
        "Status: Accepted\nBulk-Snapshot-Update: yes\nDefect: none\n",
        "# ADR 0002: bulk\n\nStatus: Accepted\nBulk-Snapshot-Update: yes\n\n## Context\nx\n",
    ],
)
def test_override_marker_matches_valid_forms(text):
    assert csd.has_override_marker(text)


@pytest.mark.parametrize(
    "text",
    [
        "Bulk-Snapshot-Update: no\n",
        "Bulk-Snapshot-Update:\n",
        "This ADR does not grant a Bulk-Snapshot-Update: yes style override.\n",
        "Status: Accepted\n",
        "",
        # Indented code block, not a metadata field.
        "Status: Accepted\n\n    Bulk-Snapshot-Update: yes\n",
        # Inside a fenced example in the header block.
        "Status: Accepted\n\n```markdown\nBulk-Snapshot-Update: yes\n```\n",
        "Status: Accepted\n\n~~~\nBulk-Snapshot-Update: yes\n~~~\n",
        # In a Nygard section body rather than the header metadata block.
        "# ADR 0002: x\n\nStatus: Accepted\n\n## Context\nBulk-Snapshot-Update: yes\n",
        "# ADR 0002: x\n\nStatus: Accepted\n\n## Context\n```\nBulk-Snapshot-Update: yes\n```\n",
    ],
)
def test_override_marker_rejects_invalid_forms(text):
    assert not csd.has_override_marker(text)


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


def _long_golden(tag: str) -> str:
    # Enough shared body that git's default rename detection pairs an edited
    # copy with its original (similarity well above the 50% default).
    body = "".join(f"# shared snapshot line {n}\n" for n in range(40))
    return body + f"# {tag}\n"


def test_changed_ambr_files_counts_renames_with_edited_content(repo: _Repo):
    n = csd.DEFAULT_THRESHOLD + 1
    for i in range(n):
        repo.write(f"{CHAR_DIR}/Module{i}/__snapshots__/test_char.ambr", _long_golden("v1"))
    base_sha = repo.commit("seed goldens")
    for i in range(n):
        repo.remove(f"{CHAR_DIR}/Module{i}/__snapshots__/test_char.ambr")
        repo.write(f"{CHAR_DIR}/Module{i}/__snapshots__/test_char_v2.ambr", _long_golden("v2"))
    head_sha = repo.commit("rename and edit every golden")

    # Default rename detection must actually see these as (inexact) renames,
    # or this test would not exercise the rename path at all.
    statuses = repo._git("diff", "--name-status", base_sha, head_sha).split()
    rename_statuses = [s for s in statuses if s.startswith("R")]
    assert len(rename_statuses) == n
    assert all(s != "R100" for s in rename_statuses)

    changed = csd.changed_ambr_files(base_sha, head_sha, repo.root, CHAR_DIR)
    assert changed == [f"{CHAR_DIR}/Module{i}/__snapshots__/test_char_v2.ambr" for i in range(n)]
    assert csd.cmd_check(_args(base_sha, head_sha, repo.root)) == 1


def test_changed_ambr_files_counts_rename_with_reordered_content(repo: _Repo):
    # git's similarity score is line-multiset based, so a rename whose lines were
    # only reordered can still report R100 even though the blob changed. Only a
    # byte-identical blob may be exempted.
    lines = [f"# snapshot line {n}\n" for n in range(40)]
    repo.write(f"{CHAR_DIR}/Module0/__snapshots__/test_char.ambr", "".join(lines))
    base_sha = repo.commit("seed golden")
    repo.remove(f"{CHAR_DIR}/Module0/__snapshots__/test_char.ambr")
    repo.write(f"{CHAR_DIR}/Module0/__snapshots__/test_char_v2.ambr", "".join(reversed(lines)))
    head_sha = repo.commit("rename and reorder")

    assert csd.changed_ambr_files(base_sha, head_sha, repo.root, CHAR_DIR) == [
        f"{CHAR_DIR}/Module0/__snapshots__/test_char_v2.ambr"
    ]


def test_changed_ambr_files_counts_non_ascii_filenames(repo: _Repo):
    # git quotes non-ASCII paths in its default (non -z) output, e.g.
    # "caf\303\251_0.ambr", which must not hide them from the count.
    n = csd.DEFAULT_THRESHOLD + 1
    for i in range(n):
        repo.write(f"{CHAR_DIR}/Module/__snapshots__/café_{i}.ambr", "# v1\n")
    base_sha = repo.commit("seed non-ascii goldens")
    for i in range(n):
        repo.write(f"{CHAR_DIR}/Module/__snapshots__/café_{i}.ambr", "# v2\n")
    head_sha = repo.commit("modify non-ascii goldens")

    changed = csd.changed_ambr_files(base_sha, head_sha, repo.root, CHAR_DIR)
    assert changed == [f"{CHAR_DIR}/Module/__snapshots__/café_{i}.ambr" for i in range(n)]
    assert csd.cmd_check(_args(base_sha, head_sha, repo.root)) == 1


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


def test_changed_adr_files_ignores_unnumbered_adr_dir_files(repo: _Repo):
    # adr/README.md documents the override (with examples); only numbered
    # decision records can grant it.
    base_sha = repo._git("rev-parse", "HEAD").strip()
    repo.write(f"{CHAR_DIR}/adr/README.md", "Bulk-Snapshot-Update: yes\n")
    repo.write(f"{CHAR_DIR}/adr/notes.md", "Bulk-Snapshot-Update: yes\n")
    head_sha = repo.commit("touch non-decision-record files under adr/")

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


def test_cmd_check_override_found_when_repo_root_is_a_subdirectory(repo: _Repo):
    # git reports top-level-relative paths even when run from a subdirectory with
    # subdirectory-relative pathspecs (e.g. from projects/hipblaslt/tensilelite
    # with the default --characterization-dir).
    sub = "projects/tensilelite"
    base_sha = repo._git("rev-parse", "HEAD").strip()
    for i in range(csd.DEFAULT_THRESHOLD + 1):
        repo.write(f"{sub}/{CHAR_DIR}/Module{i}/__snapshots__/test_char.ambr", f"# v{i}\n")
    repo.write(
        f"{sub}/{CHAR_DIR}/adr/0002-bulk.md",
        "# ADR 0002: bulk\n\nStatus: Accepted\nBulk-Snapshot-Update: yes\n",
    )
    head_sha = repo.commit("bulk regen with override, nested project layout")

    assert csd.cmd_check(_args(base_sha, head_sha, repo.root / sub)) == 0


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


def test_cmd_check_fails_when_readme_example_shows_marker(repo: _Repo, capsys):
    base_sha = repo._git("rev-parse", "HEAD").strip()
    for i in range(csd.DEFAULT_THRESHOLD + 1):
        repo.write(f"{CHAR_DIR}/Module{i}/__snapshots__/test_char.ambr", f"# v{i}\n")
    repo.write(
        f"{CHAR_DIR}/adr/README.md",
        "# ADRs\n\n## Template\n\n```markdown\nStatus: Accepted\n"
        "Bulk-Snapshot-Update: yes\n```\n",
    )
    head_sha = repo.commit("blanket regen plus a documentation example of the marker")

    assert csd.cmd_check(_args(base_sha, head_sha, repo.root)) == 1
    assert "FAIL" in capsys.readouterr().err


def test_cmd_check_fails_when_decision_record_only_quotes_marker(repo: _Repo, capsys):
    base_sha = repo._git("rev-parse", "HEAD").strip()
    for i in range(csd.DEFAULT_THRESHOLD + 1):
        repo.write(f"{CHAR_DIR}/Module{i}/__snapshots__/test_char.ambr", f"# v{i}\n")
    repo.write(
        f"{CHAR_DIR}/adr/0002-explains-override.md",
        "# ADR 0002: how the override works\n\nStatus: Accepted\n\n"
        "## Context\nA bulk PR adds this line:\n\n```\nBulk-Snapshot-Update: yes\n```\n",
    )
    head_sha = repo.commit("blanket regen plus an ADR that only quotes the marker")

    assert csd.cmd_check(_args(base_sha, head_sha, repo.root)) == 1
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


def test_main_returns_two_on_missing_repo_root(tmp_path: Path, capsys):
    # In a container job, ${{ github.workspace }} is the host path; the error
    # must name the missing directory rather than blame a missing git binary.
    missing = tmp_path / "does-not-exist"
    rc = csd.main(["--base", "HEAD~1", "--head", "HEAD", "--repo-root", str(missing)])
    assert rc == 2
    err = capsys.readouterr().err
    assert str(missing) in err
    assert "git executable not found" not in err


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
