# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every repo path the hipdnn-ingestor-engine skill cites must resolve.

`Results/ingestor-skill-review.md` found `native-pack.md` sending an agent to
read `packs/AttentionDenseNative.cpp`, which did not exist on the branch the
skill shipped on -- one grep would have caught it, and nothing ever ran that
grep. This makes that grep permanent.

Lives here (not under the skill directory, which has no test runner of its
own) because this tool already has a venv, a pytest story, and the shortest
path from the repo root the skill's `$REPO`-relative paths are written
against. `tests/skill_paths.py` holds the conservative extractor; this file
is the assertions plus the fixtures proving the extractor isn't the kind of
check that only ever sees good data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.skill_paths import (
    extract_candidates,
    git_tracked_paths,
    line_ref_is_valid,
    resolves,
)

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SKILL_DIR = _REPO_ROOT / "projects/hipdnn/tools/ai/skills/hipdnn-ingestor-engine"
# The seven files the skill ships; prompt.md is deleted (its content
# redistributed) and is deliberately not in this list -- a resurrected prompt.md
# would need adding back here explicitly, not picked up by a glob, so an
# accidental revert is a visible test-file diff.
#
# The list is asserted complete against the directory below, because the failure
# this whole file exists to prevent is a cited path nobody checked, and a file
# missing from this tuple is exactly that: unchecked, silently. workloads.md was
# added to the skill and spent its first commits uncovered for that reason.
_SKILL_FILES = (
    "SKILL.md",
    "RUNBOOK.md",
    "graph-contract.md",
    "rocke-mining.md",
    "native-pack.md",
    "extend.md",
    "workloads.md",
)


def test_every_shipped_skill_file_is_covered():
    """A new skill file must be added here, not silently escape path checking."""
    on_disk = {p.name for p in _SKILL_DIR.glob("*.md")}
    assert on_disk == set(_SKILL_FILES), (
        f"skill directory and _SKILL_FILES disagree: "
        f"only on disk {sorted(on_disk - set(_SKILL_FILES))}, "
        f"only listed {sorted(set(_SKILL_FILES) - on_disk)}"
    )


@pytest.fixture(scope="module")
def repo_tracked_paths():
    return git_tracked_paths(_REPO_ROOT)


def _all_candidates():
    """(path, placeholders) pairs collected across every real skill file,
    keeping file/line provenance for readable failure messages."""
    all_paths = []
    all_placeholders = []
    for name in _SKILL_FILES:
        text = (_SKILL_DIR / name).read_text(encoding="utf-8")
        paths, placeholders = extract_candidates(text, name)
        all_paths.extend(paths)
        all_placeholders.extend(placeholders)
    return all_paths, all_placeholders


class TestEveryRepoPathTheSkillCitesExists:
    """The positive check: run the extractor over the CURRENT skill files and
    require every unambiguous path to resolve somewhere in the tree."""

    def test_no_dangling_repo_paths(self, repo_tracked_paths):
        files, dirs = repo_tracked_paths
        candidates, _ = _all_candidates()
        assert candidates, (
            "extractor found zero repo-path candidates across the skill -- "
            "that almost certainly means the extraction rules regressed, "
            "not that the skill stopped citing any paths"
        )
        dangling = [c for c in candidates if not resolves(c.path, files, dirs)]
        assert not dangling, "dangling path(s) the skill cites:\n" + "\n".join(
            f"  {c.file}:{c.line}: `{c.raw}`" for c in dangling
        )

    def test_no_stale_line_numbers(self, repo_tracked_paths):
        """A `path:N`/`path:N-M` citation claims line N exists TODAY --
        resolving the bare path is not enough (a suffix match survives a
        file rename, but not edits that shrank the file below N)."""
        files, _dirs = repo_tracked_paths
        candidates, _ = _all_candidates()
        stale = [
            c for c in candidates if not line_ref_is_valid(c.path, files, _REPO_ROOT)
        ]
        assert not stale, "stale line reference(s) the skill cites:\n" + "\n".join(
            f"  {c.file}:{c.line}: `{c.raw}`" for c in stale
        )

    def test_placeholder_skip_count_is_sane(self):
        """Placeholders (`<op>`, `$REPO`, `{{ }}`) are deliberately SKIPPED,
        not silently ignored -- assert the count is in the range a human
        reading the skill would expect, so a change that suddenly stops
        producing placeholders (extractor regression: everything now
        misclassified as a placeholder, or the reverse) is visible instead of
        just quietly changing the denominator."""
        _, placeholders = _all_candidates()
        # Verified by hand against the current six files: RUNBOOK.md alone
        # carries $GEN/$SLUG/$BUILD/<op>-style generics; graph-contract.md
        # and native-pack.md each carry the general-form `<op>`/`<incumbent>`
        # placeholders the path-vs-example rule requires. Zero would mean the
        # extractor broke; three figures' worth would mean it started
        # flagging ordinary prose.
        assert 5 <= len(placeholders) <= 40, (
            f"placeholder count {len(placeholders)} is outside the sane "
            "range -- inspect tests/skill_paths.py's placeholder detection: "
            + ", ".join(f"{p.file}:{p.line} `{p.raw}`" for p in placeholders)
        )


class TestExtractorCatchesARealDanglingPath:
    """Negative test: an assertion that only ever sees valid data is
    decoration. Prove the extractor actually flags a known-bad path, using
    the literal defect the review found (`packs/AttentionDenseNative.cpp`,
    which real `find`/`git ls-files` confirms is absent from this branch)."""

    def test_synthetic_dangling_path_is_flagged(self, repo_tracked_paths):
        files, dirs = repo_tracked_paths
        fixture_text = (
            "Reference implementation to read before writing anything: "
            "`packs/AttentionDenseNative.cpp`.\n"
        )
        candidates, _ = extract_candidates(fixture_text, "fixture.md")
        assert len(candidates) == 1
        assert not resolves(candidates[0].path, files, dirs), (
            "packs/AttentionDenseNative.cpp resolved against the tree -- "
            "either it now exists on this branch (re-verify with `git "
            "ls-files | grep AttentionDenseNative`) or the resolver's suffix "
            "match got too permissive"
        )

    def test_synthetic_valid_path_is_not_flagged(self, repo_tracked_paths):
        """Companion sanity check: a real path in the same fixture shape must
        NOT be flagged, so the negative case above is attributable to the
        path being genuinely absent, not to every path failing."""
        files, dirs = repo_tracked_paths
        fixture_text = (
            "Reference implementations to read: `packs/ConvNative.cpp` and "
            "`packs/PointwiseNative.cpp`.\n"
        )
        candidates, _ = extract_candidates(fixture_text, "fixture.md")
        assert len(candidates) == 2
        assert all(resolves(c.path, files, dirs) for c in candidates)

    def test_synthetic_stale_line_number_is_flagged(self, repo_tracked_paths):
        """A `path:N` citation whose N exceeds the real file's line count is
        a stale reference -- bare-path resolution alone would miss it."""
        files, _dirs = repo_tracked_paths
        fixture_text = "See `projects/hipdnn/CMakeLists.txt:999999` for the option.\n"
        candidates, _ = extract_candidates(fixture_text, "fixture.md")
        assert len(candidates) == 1
        assert not line_ref_is_valid(candidates[0].path, files, _REPO_ROOT), (
            "projects/hipdnn/CMakeLists.txt:999999 was accepted as a valid "
            "line reference -- either the file grew past 999999 lines or "
            "line_ref_is_valid stopped checking line counts"
        )

    def test_synthetic_valid_line_number_is_not_flagged(self, repo_tracked_paths):
        """Companion sanity check: a real, in-range line citation on the
        same real file must NOT be flagged."""
        files, _dirs = repo_tracked_paths
        fixture_text = "See `projects/hipdnn/CMakeLists.txt:1` for the top.\n"
        candidates, _ = extract_candidates(fixture_text, "fixture.md")
        assert len(candidates) == 1
        assert line_ref_is_valid(candidates[0].path, files, _REPO_ROOT)


class TestExtractionIsConservative:
    """Bare words, `$VAR`s, `<placeholder>`s, and shell fragments must never
    be flagged as dangling repo paths -- a test that cries wolf gets deleted
    by the next person, per the plan's own design principle."""

    @pytest.mark.parametrize(
        "line",
        [
            "Ask the user to point at the kernel source.",
            "Set `$REPO` to your checkout root before anything else.",
            "The general form is `src/engines/<incumbent-engine>/plans/`.",
            "Fill in `{{ engine_name }}` before rendering.",
            "Run `git log --all --diff-filter=A -- '**/packs/*Native.cpp'`.",
            "The stub body is `// TODO - FILL THIS OUT`.",
            "See `quick/SdpaFwd` for a shipped bundle.",
            "Glob `projects/hipdnn/**/*.fbs` for every schema.",
        ],
    )
    def test_ambiguous_forms_are_not_flagged_as_dangling_candidates(self, line):
        candidates, _ = extract_candidates(line, "fixture.md")
        assert candidates == [], (
            f"conservative extractor should not treat {line!r} as an "
            f"unambiguous repo path, but flagged: {candidates}"
        )
