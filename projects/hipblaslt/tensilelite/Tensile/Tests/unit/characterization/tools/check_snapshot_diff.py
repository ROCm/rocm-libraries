#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CI guard against blanket `.ambr` golden regeneration (AIHPBLAS-3876).

The characterization suite's `.ambr` goldens are a safety net only if they change
**deliberately, one reviewed diff at a time** (see the characterization
``README.md``'s "Snapshot / golden discipline" section). A bare, suite-wide
``pytest --snapshot-update`` silently rewrites every golden; a PR that does this
(accidentally, or to force a red suite green) can pin a real regression, because
the tests then pass against the very output that is wrong. Local tooling
(the pre-commit hook, the README's guidance) discourages this, but neither is
unbypassable: the hook is opt-in and only re-runs tests, which pass fine against
freshly-regenerated goldens.

This script is the backstop that CANNOT be bypassed with ``git commit
--no-verify``, because it runs in CI against the PR's actual diff. It compares
the `.ambr` files changed between a PR's merge-base and its head; if more than a
small threshold changed, it fails unless the same diff also adds or updates an
Architecture Decision Record (ADR, under ``characterization/adr/``) carrying an
explicit ``Bulk-Snapshot-Update: yes`` line -- a conscious, reviewed opt-in for a
genuine mass update (e.g. a change to the snapshot format itself), documented the
same way as any other characterization decision (see ``adr/README.md``).

Usage (run from the TensileLite root, ``projects/hipblaslt/tensilelite``, same
convention as ``coverage_ratchet.py``)::

    python Tensile/Tests/unit/characterization/tools/check_snapshot_diff.py \\
        --base <base-sha-or-ref> --head <head-sha-or-ref>

Exit codes: ``0`` OK (within threshold, or a valid override was found), ``1`` a
blanket regeneration was detected with no override, ``2`` a setup/usage problem
(bad refs, git failure) rather than a detected violation.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# A PR touching more than this many `.ambr` files without an override reads as a
# blanket regeneration rather than a scoped, reviewed behavior change. Small and
# deliberate: the characterization suite has ~100 golden files across ~30 module
# directories, so a real, scoped fix touches a small handful of nodes, not dozens.
DEFAULT_THRESHOLD = 3

DEFAULT_CHARACTERIZATION_DIR = "Tensile/Tests/unit/characterization"

# Matches the ADR template field documented in adr/README.md. Deliberately strict
# (exact "yes", own line) so a stray mention of "bulk" in an ADR's prose can never
# accidentally grant an override.
OVERRIDE_MARKER_RE = re.compile(
    r"^[ \t]*Bulk-Snapshot-Update:[ \t]*yes[ \t]*$", re.IGNORECASE | re.MULTILINE
)

# Printed verbatim so a failing CI log tells the developer exactly how to move
# forward (never "just re-run with --snapshot-update and force it through").
REMEDIATION = (
    "This looks like a blanket snapshot regeneration rather than a scoped,\n"
    "reviewed change. Either:\n"
    "  (a) split this into smaller PRs, each scoped to the behavior it actually\n"
    "      changes (see 'Surgical, never blanket' in the characterization\n"
    "      README's snapshot discipline section), or\n"
    "  (b) if this bulk update is genuinely reviewed and intentional (e.g. a\n"
    "      change to the snapshot format itself), add or update an ADR under\n"
    "      characterization/adr/ (see adr/README.md for the template) with the\n"
    "      line:\n"
    "          Bulk-Snapshot-Update: yes\n"
    "      explaining why the bulk regeneration is correct, then push again."
)

MAX_FILES_SHOWN = 20


class SnapshotDiffError(Exception):
    """A setup/usage problem (bad refs, git failure), not a detected violation."""


def _run_git(args: list[str], repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise SnapshotDiffError(f"git executable not found: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise SnapshotDiffError(f"'git {' '.join(args)}' failed: {stderr or exc}") from exc
    return result.stdout


def merge_base(base: str, head: str, repo_root: Path) -> str:
    """Merge-base of ``base`` and ``head``.

    Using the merge-base (rather than a plain two-ref diff) means the guard only
    ever judges what the PR branch itself changed, ignoring unrelated commits that
    landed on the base branch after the PR forked from it -- the same semantics as
    GitHub's PR "Files changed" tab (``base...head``, not ``base..head``).
    """
    return _run_git(["merge-base", base, head], repo_root).strip()


def _changed_files(
    ref_range: tuple[str, str], diff_filter: str, pathspec: str, repo_root: Path
) -> list[str]:
    base_ref, head_ref = ref_range
    out = _run_git(
        [
            "diff",
            "--name-only",
            f"--diff-filter={diff_filter}",
            base_ref,
            head_ref,
            "--",
            pathspec,
        ],
        repo_root,
    )
    return [line for line in out.splitlines() if line]


def changed_ambr_files(
    merge_base_ref: str, head: str, repo_root: Path, characterization_dir: str
) -> list[str]:
    """`.ambr` goldens added/copied/modified between merge-base and head.

    Deliberately excludes renames: a pure rename (e.g. a test function renamed,
    carrying its golden along byte-for-byte) changes no pinned content, so it
    must not count toward the blanket-regeneration threshold.
    """
    names = _changed_files((merge_base_ref, head), "ACM", characterization_dir, repo_root)
    return sorted(p for p in names if p.endswith(".ambr"))


def changed_adr_files(
    merge_base_ref: str, head: str, repo_root: Path, characterization_dir: str
) -> list[str]:
    """ADR markdown files added/modified between merge-base and head.

    Deliberately excludes deletions/renames: the override must be a real, present
    ADR file at ``head``, not a file that merely existed somewhere in the range.
    """
    adr_dir = f"{characterization_dir.rstrip('/')}/adr"
    names = _changed_files((merge_base_ref, head), "ACM", adr_dir, repo_root)
    return sorted(p for p in names if p.endswith(".md"))


def file_at_ref(ref: str, path: str, repo_root: Path) -> str:
    """Contents of ``path`` as of ``ref``.

    ``path`` must be the repo-top-level-relative form ``git diff --name-only``
    returns (which is what this module always passes in here) -- that form is
    valid for ``git show <ref>:<path>`` regardless of the process's cwd, unlike a
    cwd-relative path, which ``git show`` requires prefixing with ``./``.
    """
    return _run_git(["show", f"{ref}:{path}"], repo_root)


def find_override(adr_paths: list[str], head: str, repo_root: Path) -> str | None:
    """First ADR path (of ``adr_paths``, as of ``head``) granting the override, if any."""
    for path in adr_paths:
        if OVERRIDE_MARKER_RE.search(file_at_ref(head, path, repo_root)):
            return path
    return None


def _format_offenders(paths: list[str]) -> list[str]:
    shown = paths[:MAX_FILES_SHOWN]
    lines = [f"  {p}" for p in shown]
    remaining = len(paths) - len(shown)
    if remaining > 0:
        lines.append(f"  ... and {remaining} more")
    return lines


def cmd_check(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root)
    mb = merge_base(args.base, args.head, repo_root)

    changed_ambr = changed_ambr_files(mb, args.head, repo_root, args.characterization_dir)
    if len(changed_ambr) <= args.threshold:
        print(
            f"check_snapshot_diff: OK - {len(changed_ambr)} .ambr file(s) changed "
            f"(threshold {args.threshold})."
        )
        return 0

    adr_candidates = changed_adr_files(mb, args.head, repo_root, args.characterization_dir)
    override_path = find_override(adr_candidates, args.head, repo_root)
    if override_path is not None:
        print(
            f"check_snapshot_diff: OK - {len(changed_ambr)} .ambr file(s) changed "
            f"(over threshold {args.threshold}), overridden by {override_path} "
            "(Bulk-Snapshot-Update: yes)."
        )
        return 0

    print(
        f"check_snapshot_diff: FAIL - {len(changed_ambr)} .ambr golden file(s) "
        f"changed, more than the threshold ({args.threshold}):\n",
        file=sys.stderr,
    )
    for line in _format_offenders(changed_ambr):
        print(line, file=sys.stderr)
    print("\n" + REMEDIATION, file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--base",
        required=True,
        help="base ref/sha to diff against (e.g. the PR's base branch sha)",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="head ref/sha to diff (default: %(default)s)",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="directory to run git commands in (default: cwd, i.e. %(default)r)",
    )
    parser.add_argument(
        "--characterization-dir",
        default=DEFAULT_CHARACTERIZATION_DIR,
        help="characterization suite root, relative to --repo-root " "(default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help="max changed .ambr files allowed without an override " "(default: %(default)s)",
    )
    parser.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SnapshotDiffError as exc:
        print(f"check_snapshot_diff: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
