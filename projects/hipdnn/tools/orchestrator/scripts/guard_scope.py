#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Prove an agent changed only what it was allowed to change.

The loop this serves gives an agent write access to a checkout and then measures
success with the integration suite and a benchmark. Both of those measurements can
be made to succeed by editing the wrong file: weaken a tolerance in the harness,
edit a bundle's shapes, add a `test_skips` entry to the engine TOML, or "fix" the
kernel by changing the reference engine it is compared against. None of that shows
up as a failing test -- it shows up as a green run nobody should believe.

So the files that define the *measurement* are hashed before the loop starts and
re-hashed after every agent step. The files that define the *work* are simply not
watched. A change to anything in the first set is a hard failure; that is the whole
contract.

    # once, before the loop
    guard_scope.py --repo R --mode baseline --out baseline.json --watch "<glob>" ...

    # after each agent step, same --watch and --expect-change sets
    guard_scope.py --repo R --mode check --baseline baseline.json --out scope.json \
        --watch "<glob>" ... --expect-change "<glob>" ...

`--watch` is the negative half: those files must be byte-identical to the baseline.
`--expect-change` is the positive half: at least one file matching those globs must
differ. Without it, an agent that wrote nothing at all still passes every gate after
this one -- the build succeeds, the suite reports what it reported last round, and
the benchmark runs the previous engine.

Some watched trees have a third, narrower shape: not "must not change" and not
"unwatched", but "may only grow". The integration-test bundle tree is exactly this --
the work this guard serves requires an agent to ADD a new bundle case (a new
template+sweep directory, or a new case appended to an existing topology's
sweep.json), and a flat --watch on that tree forbids the very thing the agent is
asked to do. Not watching it instead reopens the cheat this guard exists to close:
edit an existing case's shapes, or widen a tolerance in its metadata, until the pack
that cannot serve them passes.

`--allow-added GLOB` (repeatable) excuses a matching file from `added`: a brand-new
file under the glob is not a violation, though it is still reported (in
`allowed_added`). `--allow-grow GLOB` (repeatable) excuses a matching file from
`modified` when its baseline content, parsed as JSON, is a structural *prefix* of
its current content -- lists may only gain trailing elements, dicts may only gain
keys, at every level (see `_json_grew`). A file that matches `--allow-grow` but was
edited rather than appended to (an inserted or reordered element, an edited value at
an existing key) stays in `modified` and stays a violation, because that edit is the
cheat this guard exists to catch. Proving growth needs the baseline's *content*, not
just its hash, so baseline mode additionally records the text of every file matching
an `--allow-grow` glob under `grow_files`; a baseline written before this existed has
no `grow_files`, and a check against it cannot prove growth for anything, so it fails
closed -- every `--allow-grow` candidate stays a violation until the baseline is
retaken.

Patterns are repo-relative globs (`**` recurses). Directories and files that do not
exist are not an error: a pattern matching nothing contributes nothing, which is
reported as `watched_count`/`work_count` so a typo'd pattern is visible as zero
rather than as a silent pass.

Exit codes: 0 the report was written (check mode does NOT exit non-zero on a
violation -- the flow asserts on `outside_count` and `work_changed_count`, so the
report is written either way and the evidence survives), 1 usage or I/O failure.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path


def _hash_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _manifest(repo: Path, patterns: list[str]) -> dict[str, str]:
    """path (repo-relative, forward slashes) -> sha256, over every file any pattern
    matches. Sorted output so two runs of the same tree produce the same file.

    `include_hidden` is not optional here. Without it `glob` silently drops every
    dot-prefixed name, so `.clang-tidy` next to a watched source is unprotected and a
    dot-prefixed file added under the engine does not count as work. A guard with a
    blind spot reads exactly like a guard without one."""
    found: dict[str, str] = {}
    for pattern in patterns:
        for match in glob.glob(
            pattern, root_dir=repo, recursive=True, include_hidden=True
        ):
            candidate = repo / match
            if not candidate.is_file():
                continue
            found[Path(match).as_posix()] = _hash_file(candidate)
    return dict(sorted(found.items()))


def _json_grew(old: object, new: object) -> bool:
    """True if `new` extends `old` by appending only: new list elements land after
    the old ones and new dict keys land alongside the old ones, with everything the
    old value already had left byte-for-byte in place. This is the shape
    `import_graph.py` produces when it appends a case to an existing topology's
    sweep.json, and it is exactly what an edited-in-place case (a changed shape, a
    widened tolerance) is NOT -- an altered or reordered existing entry always fails
    this check, however small the edit."""
    if isinstance(old, list) and isinstance(new, list):
        return len(new) >= len(old) and all(new[i] == old[i] for i in range(len(old)))
    if isinstance(old, dict) and isinstance(new, dict):
        return all(key in new and _json_grew(old[key], new[key]) for key in old)
    return old == new


def _matched_paths(repo: Path, patterns: list[str]) -> set[str]:
    """repo-relative posix paths, as the tree stands right now, that any pattern
    matches. Used to test whether a path already classified as added or modified
    also falls under a narrower --allow-added/--allow-grow carve-out; the carve-out
    is layered on top of --watch, not a replacement for it."""
    found: set[str] = set()
    for pattern in patterns:
        for match in glob.glob(
            pattern, root_dir=repo, recursive=True, include_hidden=True
        ):
            candidate = repo / match
            if candidate.is_file():
                found.add(Path(match).as_posix())
    return found


def _grow_manifest(repo: Path, patterns: list[str]) -> dict[str, str]:
    """path -> file text, over every file an --allow-grow glob matches. Recorded as
    text rather than a hash because proving "this only grew" needs the old content
    to diff structurally against; a hash only proves "this changed", which is the
    fact already available from `_manifest` and not the one growth needs."""
    found: dict[str, str] = {}
    for pattern in patterns:
        for match in glob.glob(
            pattern, root_dir=repo, recursive=True, include_hidden=True
        ):
            candidate = repo / match
            if not candidate.is_file():
                continue
            try:
                found[Path(match).as_posix()] = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
    return dict(sorted(found.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", required=True, help="checkout root the patterns are relative to"
    )
    parser.add_argument("--mode", required=True, choices=("baseline", "check"))
    parser.add_argument("--out", required=True, help="where to write the JSON report")
    parser.add_argument("--baseline", help="baseline manifest (required in check mode)")
    parser.add_argument(
        "--watch",
        action="append",
        default=[],
        metavar="GLOB",
        help="repo-relative glob of files that must not change; repeatable",
    )
    parser.add_argument(
        "--expect-change",
        action="append",
        default=[],
        metavar="GLOB",
        help="repo-relative glob of files that MUST differ from the baseline; repeatable",
    )
    parser.add_argument(
        "--allow-added",
        action="append",
        default=[],
        metavar="GLOB",
        help="repo-relative glob of files allowed to be newly added under --watch; repeatable",
    )
    parser.add_argument(
        "--allow-grow",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "repo-relative glob of files allowed to change under --watch if the "
            "change is JSON-prefix growth only (see _json_grew); repeatable"
        ),
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        print(f"error: --repo is not a directory: {repo}", file=sys.stderr)
        return 1
    if not args.watch:
        print("error: at least one --watch pattern is required", file=sys.stderr)
        return 1

    manifest = _manifest(repo, args.watch)
    work = _manifest(repo, args.expect_change) if args.expect_change else {}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "baseline":
        grow_manifest = _grow_manifest(repo, args.allow_grow) if args.allow_grow else {}
        out_path.write_text(
            json.dumps(
                {
                    "repo": repo.as_posix(),
                    "watched_count": len(manifest),
                    "work_count": len(work),
                    "files": manifest,
                    "work_files": work,
                    "grow_files": grow_manifest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary = f"baseline: {len(manifest)} watched file(s), {len(work)} work file(s)"
        if args.allow_grow:
            summary += f", {len(grow_manifest)} grow file(s) recorded"
        summary += f" -> {out_path}"
        print(summary)
        return 0

    if not args.baseline:
        print("error: --baseline is required in check mode", file=sys.stderr)
        return 1
    try:
        snapshot = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        baseline = snapshot["files"]
        work_baseline = snapshot.get("work_files", {})
        grow_baseline = snapshot.get("grow_files", {})
    except (OSError, ValueError, KeyError) as error:
        print(
            f"error: could not read baseline {args.baseline}: {error}", file=sys.stderr
        )
        return 1

    modified_all = sorted(
        k for k, v in manifest.items() if k in baseline and baseline[k] != v
    )
    removed = sorted(k for k in baseline if k not in manifest)
    added_all = sorted(k for k in manifest if k not in baseline)

    # Both carve-outs are narrower globs layered on top of --watch, not a
    # replacement for it: a path only leaves `added`/`modified` if it BOTH matches
    # one of these AND proves the specific shape (freshly added, or JSON-prefix
    # growth) the carve-out promises. Anything else is still a violation.
    allow_added_paths = (
        _matched_paths(repo, args.allow_added) if args.allow_added else set()
    )
    allow_grow_paths = (
        _matched_paths(repo, args.allow_grow) if args.allow_grow else set()
    )

    allowed_added = sorted(k for k in added_all if k in allow_added_paths)
    added = sorted(k for k in added_all if k not in allow_added_paths)

    grown: list[str] = []
    grow_rejected: list[str] = []
    for path in modified_all:
        if path not in allow_grow_paths:
            continue
        old_text = grow_baseline.get(path)
        if old_text is None:
            # No recorded content -- an older baseline predates --allow-grow, or
            # this glob wasn't passed at baseline time. Either way there is nothing
            # to prove growth against, so this stays a violation: fail closed.
            continue
        try:
            new_text = (repo / path).read_text(encoding="utf-8")
            grew = _json_grew(json.loads(old_text), json.loads(new_text))
        except (OSError, UnicodeDecodeError, ValueError):
            # Not parseable as JSON on one side, or unreadable now: "could not
            # check whether this grew" is not the same answer as "this is fine".
            grew = False
        if grew:
            grown.append(path)
        else:
            grow_rejected.append(path)

    grown_set = set(grown)
    modified = sorted(k for k in modified_all if k not in grown_set)
    outside = modified + removed + added

    # The other half of the contract. Every gate after this one can be satisfied by an
    # agent that changed nothing at all -- the build still succeeds, the suite reports
    # whatever it reported last round, and the benchmark still runs. "The engine is
    # byte-for-byte what it was before you started" has to be its own failure.
    work_changed = sorted(
        set(k for k, v in work.items() if work_baseline.get(k) != v)
        | set(k for k in work_baseline if k not in work)
    )

    feedback_parts = []
    if outside:
        lines = [
            "The following files are part of how this run is *measured* and must not"
            " be edited. Revert them and make the change in the engine instead:",
            "",
        ]
        lines += [f"- modified: {path}" for path in modified]
        lines += [f"- deleted:  {path}" for path in removed]
        lines += [f"- added:    {path}" for path in added]
        feedback_parts.append("\n".join(lines))
    if grow_rejected:
        lines = [
            "The following files may only grow -- gain a new case appended to what "
            "was already there -- but their existing content changed instead. That "
            'is a different instruction than "do not touch this file": put back '
            "what was there and add the new case alongside it, not in place of it:",
            "",
        ]
        lines += [f"- {path}" for path in grow_rejected]
        feedback_parts.append("\n".join(lines))
    if args.expect_change and not work_changed:
        feedback_parts.append(
            "No file under the engine changed. Whatever was reported, nothing was "
            "written to disk: the build, the test suite and the benchmark all ran "
            "against the previous round's engine."
        )

    out_path.write_text(
        json.dumps(
            {
                "watched_count": len(manifest),
                "outside_count": len(outside),
                "modified": modified,
                "removed": removed,
                "added": added,
                "allowed_added": allowed_added,
                "allowed_added_count": len(allowed_added),
                "grown": grown,
                "grown_count": len(grown),
                "work_count": len(work),
                "work_changed_count": len(work_changed),
                "work_changed": work_changed,
                "feedback": "\n\n".join(feedback_parts),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"watched {len(manifest)} file(s); {len(outside)} unexpected change(s), "
        f"{len(allowed_added)} allowed-added, {len(grown)} grown. "
        f"engine: {len(work)} file(s), {len(work_changed)} changed."
    )
    for path in outside:
        print(f"  unexpected: {path}")
    for path in allowed_added:
        print(f"  allowed:    {path}")
    for path in grown:
        print(f"  grown:      {path}")
    for path in work_changed:
        print(f"  engine:     {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
