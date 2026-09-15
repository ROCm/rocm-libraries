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
        out_path.write_text(
            json.dumps(
                {
                    "repo": repo.as_posix(),
                    "watched_count": len(manifest),
                    "work_count": len(work),
                    "files": manifest,
                    "work_files": work,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"baseline: {len(manifest)} watched file(s), {len(work)} work file(s) -> {out_path}"
        )
        return 0

    if not args.baseline:
        print("error: --baseline is required in check mode", file=sys.stderr)
        return 1
    try:
        snapshot = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        baseline = snapshot["files"]
        work_baseline = snapshot.get("work_files", {})
    except (OSError, ValueError, KeyError) as error:
        print(
            f"error: could not read baseline {args.baseline}: {error}", file=sys.stderr
        )
        return 1

    modified = sorted(
        k for k, v in manifest.items() if k in baseline and baseline[k] != v
    )
    removed = sorted(k for k in baseline if k not in manifest)
    added = sorted(k for k in manifest if k not in baseline)
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
        f"watched {len(manifest)} file(s); {len(outside)} unexpected change(s). "
        f"engine: {len(work)} file(s), {len(work_changed)} changed."
    )
    for path in outside:
        print(f"  unexpected: {path}")
    for path in work_changed:
        print(f"  engine:     {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
