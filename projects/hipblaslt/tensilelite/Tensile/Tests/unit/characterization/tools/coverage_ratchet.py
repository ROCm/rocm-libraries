#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-file coverage floors (and the ratchet that raises them) for TensileLite.

This tool maintains a *per-file floor*: a minimum coverage percentage for each
file, recorded in a committed baseline (``coverage-baseline.json``). It is the
per-file complement to the whole-project floor (a single minimum for the
combined total, set by ``[tool.coverage.report] fail_under`` in
``pyproject.toml``, AIHPBLAS-3877). Per-file floors catch a single file
backsliding even when the overall total still clears the whole-project floor
(AIHPBLAS-3878).

The floors move one way only: up, and on purpose. That upward move is the
*ratchet*. Like a ratchet wrench that only turns one way, each click locks in a
higher floor. Raising the floors is a deliberate maintenance step done in a
reviewed PR (this tool's ``update`` mode); everyday coverage gains never lower
anything on their own.

Two modes:

* ``check``  - enforce the floors: compare a fresh ``coverage.json`` against the
  committed per-file floors and fail (exit 1) if any file dropped below its floor
  by more than ``--tolerance`` percentage points (see ``DEFAULT_TOLERANCE`` for
  why that buffer is deliberately wide for now). This runs in CI on every
  coverage run. Prints exactly which files dropped and the one command to fix it.
* ``update`` - the ratchet click: raise each file's floor to the level in the
  current ``coverage.json``. A floor never moves down on its own; lowering one
  requires naming the file with ``--allow-lower``, so a reviewed intentional
  reduction stays possible but cannot happen as a side effect of raising
  something else. This is the single reviewed command for a deliberate floor
  move; the resulting diff is reviewed like any other change.

Input is the coverage.py JSON report (``--cov-report=json``), whose per-file
line-coverage percentage is ``files[<path>]["summary"]["percent_covered"]``. In
the ``coverage-unit`` lane that report is the *combined* dataset (the
characterization and pure-unit suites unioned by ``coverage combine``), so the
ratchet is agnostic to which suite reaches a line. That is why it does not track
the characterization-to-unit migration; the split-summary card does that.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Iterable
from pathlib import Path

# Decimal places the baseline is written at. Comparisons round to this first, so
# a difference the file cannot represent is never treated as a change.
_BASELINE_PRECISION = 2

# A file must drop by more than this many percentage points to count as a
# regression. This absorbs measurement noise rather than real coverage loss, and
# there is more of that noise than a tight budget can hold:
#
# * A merge that *deletes covered* code lowers the ratio without anything
#   becoming less tested. Removing 7 covered statements from a 770-unit file
#   moved it 88.16 -> 88.05 with identical missed statements and branches.
# * xdist worker scheduling and toolchain differences flip individual branch
#   arcs between otherwise identical runs.
# * A tolerance in percentage points does not scale with file size. One arc in a
#   770-unit file is worth ~0.13 pp, so a 0.1 pp budget cannot absorb even a
#   single-arc wobble in any file smaller than ~1000 measurable units.
#
# Set deliberately wide (1 pp) while the gate first lands, so it fails on real
# regressions instead of on noise. Tighten it once the numbers have proven stable
# across a few develop merges; the whole-project floor is the backstop meanwhile.
DEFAULT_TOLERANCE = 1.0


def remediation(paths: list[str]) -> str:
    """The exact command to move these floors on purpose, file by file.

    Emitted with one ``--allow-lower`` per regressed file so a failing CI log
    can be copy-pasted without also becoming a blanket "regenerate everything":
    the command it prints can only lower the files named in it.

    Each path is shell-quoted and passed as ``--allow-lower=<value>`` rather
    than ``--allow-lower <value>``: a path with a space would otherwise split
    into two shell arguments, a path starting with ``-`` would otherwise be
    read as another option, and quoting alone stops shell metacharacters (for
    example ``$(...)``) in a path from being interpreted when this command is
    copy-pasted, all of which coverage.py and git happily allow in a filename.
    """
    named = "".join(
        f" \\\n        --allow-lower={shlex.quote(path)}" for path in paths
    )
    return (
        "If these drops are intentional, review them and update the baseline with:\n"
        "    tox -e coverage-unit            # regenerate coverage.json, then\n"
        "    python Tensile/Tests/unit/characterization/tools/coverage_ratchet.py \\\n"
        f"        update --current coverage.json{named}\n"
        "and commit the reviewed coverage-baseline.json diff."
    )


class RatchetError(Exception):
    """A setup/usage problem (missing or malformed input), not a regression."""


def _load_json(path: Path, what: str) -> dict:
    if not path.is_file():
        raise RatchetError(f"{what} not found: {path}")
    try:
        # utf-8-sig tolerates a UTF-8 BOM (some tooling / shells emit one) while
        # reading plain UTF-8 unchanged; coverage.py itself writes no BOM.
        with path.open(encoding="utf-8-sig") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise RatchetError(f"could not read {what} ({path}): {exc}") from exc


def per_file_coverage(coverage_json: dict) -> dict[str, float]:
    """Extract ``{file_path: percent_covered}`` from a coverage.py JSON report."""
    files = coverage_json.get("files")
    if not isinstance(files, dict):
        raise RatchetError(
            "coverage json has no 'files' object; is it a coverage.py report?"
        )
    result: dict[str, float] = {}
    for path, data in files.items():
        try:
            result[path] = float(data["summary"]["percent_covered"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RatchetError(f"malformed coverage entry for {path}: {exc}") from exc
    return result


def find_regressions(
    baseline: dict[str, float],
    current: dict[str, float],
    tolerance: float,
) -> list[tuple[str, float, float]]:
    """Return ``(path, baseline_pct, current_pct)`` for every regressed file.

    A file present in the baseline but absent from the current report is treated
    as removed source (not a regression) and skipped. New files not yet in the
    baseline are ignored here; they get pinned on the next ``update``.
    """
    regressions: list[tuple[str, float, float]] = []
    for path, base_pct in baseline.items():
        if path not in current:
            continue
        cur_pct = current[path]
        if cur_pct < base_pct - tolerance:
            regressions.append((path, base_pct, cur_pct))
    regressions.sort(key=lambda row: row[2] - row[1])  # biggest drop first
    return regressions


def ratchet_floors(
    existing: dict[str, float],
    current: dict[str, float],
    allow_lower: Iterable[str] = (),
    tolerance: float = 0.0,
) -> tuple[dict[str, float], list[tuple[str, float, float]]]:
    """Return ``(new_floors, refused)`` for an ``update``.

    A floor rises to the current level, and a file with no floor yet is pinned at
    its current level. A floor that would *fall* is held at its existing value
    and reported in ``refused``, unless its path was named in ``allow_lower``.
    That is what makes ``update`` a ratchet rather than a snapshot of whichever
    coverage run happened to be on disk: raising one file's floor can no longer
    quietly lower another's.

    A drop no bigger than ``tolerance`` is held at the existing floor without
    being added to ``refused``. This keeps the command ``check`` prints in sync
    with what ``update`` actually demands: ``check`` only names files whose drop
    exceeds the same tolerance, so an unnamed in-tolerance wobble must not force
    a second, separate authorization here.

    Percentages are rounded to the precision the baseline is written at before
    being compared, so a difference too small to appear in the file is not
    treated as a fall.
    """
    allow_lower = set(allow_lower)
    floors: dict[str, float] = {}
    refused: list[tuple[str, float, float]] = []
    for path, cur_pct in current.items():
        cur_pct = round(cur_pct, _BASELINE_PRECISION)
        base_pct = existing.get(path)
        base_pct = (
            round(base_pct, _BASELINE_PRECISION) if base_pct is not None else None
        )
        if base_pct is None or cur_pct >= base_pct or path in allow_lower:
            floors[path] = cur_pct
        elif cur_pct >= base_pct - tolerance:
            floors[path] = base_pct
        else:
            floors[path] = base_pct
            refused.append((path, base_pct, cur_pct))
    refused.sort(key=lambda row: row[2] - row[1])  # biggest drop first
    return floors, refused


def write_baseline(
    current: dict[str, float], baseline_path: Path, tolerance: float
) -> None:
    payload = {
        # Written here rather than hand-edited in the JSON, so it survives every
        # regeneration instead of being silently overwritten by the next update.
        "_comment": (
            "Per-file coverage ratchet baseline for the TensileLite "
            "characterization suite (AIHPBLAS-3878). Coverage may rise but not "
            "fall. Regenerate with coverage_ratchet.py update; review the diff. "
            "'tolerance' is the noise buffer in percentage points, set wide "
            "while the gate first lands; see DEFAULT_TOLERANCE in "
            "tools/coverage_ratchet.py for why, and tighten it once the numbers "
            "prove stable."
        ),
        "tolerance": tolerance,
        "files": {
            path: round(pct, _BASELINE_PRECISION)
            for path, pct in sorted(current.items())
        },
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with baseline_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")


def _baseline_files(baseline_json: dict) -> dict[str, float]:
    files = baseline_json.get("files")
    if not isinstance(files, dict):
        raise RatchetError(
            "baseline has no 'files' object; regenerate it with 'update'."
        )
    return {path: float(pct) for path, pct in files.items()}


def cmd_check(args: argparse.Namespace) -> int:
    current_path = Path(args.current)
    if not current_path.is_file():
        # Most likely the coverage run failed before emitting a report. Don't
        # mask that upstream failure with a confusing second error.
        print(
            f"coverage_ratchet: no coverage report at {current_path}; "
            "skipping ratchet (did the coverage run fail?).",
            file=sys.stderr,
        )
        return 0

    baseline_json = _load_json(Path(args.baseline), "baseline")
    tolerance = (
        args.tolerance
        if args.tolerance is not None
        else baseline_json.get("tolerance", DEFAULT_TOLERANCE)
    )
    baseline = _baseline_files(baseline_json)
    current = per_file_coverage(_load_json(current_path, "coverage report"))

    regressions = find_regressions(baseline, current, tolerance)
    if not regressions:
        print(
            f"coverage_ratchet: OK - no per-file regression "
            f"(checked {len(baseline)} files, tolerance {tolerance:g} pp)."
        )
        return 0

    print(
        f"coverage_ratchet: FAIL - {len(regressions)} file(s) regressed:\n",
        file=sys.stderr,
    )
    print(f"  {'baseline':>9}  {'current':>9}  {'delta':>8}  file", file=sys.stderr)
    for path, base_pct, cur_pct in regressions:
        print(
            f"  {base_pct:8.2f}%  {cur_pct:8.2f}%  {cur_pct - base_pct:+7.2f}  {path}",
            file=sys.stderr,
        )
    print("\n" + remediation([path for path, _, _ in regressions]), file=sys.stderr)
    return 1


def cmd_update(args: argparse.Namespace) -> int:
    current = per_file_coverage(_load_json(Path(args.current), "coverage report"))
    tolerance = args.tolerance if args.tolerance is not None else DEFAULT_TOLERANCE
    baseline_path = Path(args.baseline)
    allow_lower = set(args.allow_lower or ())

    # Absent on the very first update, which pins every file at its current
    # level. After that the existing floors are what the new ones ratchet from.
    existing = (
        _baseline_files(_load_json(baseline_path, "baseline"))
        if baseline_path.is_file()
        else {}
    )
    floors, refused = ratchet_floors(existing, current, allow_lower, tolerance)

    if refused:
        print(
            f"coverage_ratchet: refusing to lower {len(refused)} floor(s). "
            "Name each file you mean to lower:\n",
            file=sys.stderr,
        )
        print(f"  {'floor':>9}  {'current':>9}  {'delta':>8}  file", file=sys.stderr)
        for path, base_pct, cur_pct in refused:
            print(
                f"  {base_pct:8.2f}%  {cur_pct:8.2f}%  {cur_pct - base_pct:+7.2f}  {path}",
                file=sys.stderr,
            )
        print("\n" + remediation([path for path, _, _ in refused]), file=sys.stderr)
        return 1

    # A path that is not in the report cannot be the one you meant to lower, so
    # it is a typo or a stale entry. Silence here would look like consent.
    for path in sorted(allow_lower - set(current)):
        print(
            f"coverage_ratchet: warning: --allow-lower {path} is not in the "
            "coverage report; nothing to lower.",
            file=sys.stderr,
        )

    raised = sum(1 for p, pct in floors.items() if p in existing and pct > existing[p])
    lowered = sum(1 for p, pct in floors.items() if p in existing and pct < existing[p])
    added = len(set(floors) - set(existing))

    write_baseline(floors, baseline_path, tolerance)
    print(
        f"coverage_ratchet: wrote baseline for {len(floors)} files to "
        f"{baseline_path} ({raised} raised, {added} newly pinned, "
        f"{lowered} lowered on request)."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    default_baseline = str(
        Path(__file__).resolve().parent.parent / "coverage-baseline.json"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--baseline",
        default=default_baseline,
        help="path to the committed coverage-baseline.json (default: %(default)s)",
    )
    common.add_argument(
        "--current",
        default="coverage.json",
        help="path to the fresh coverage.py JSON report (default: %(default)s)",
    )
    common.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="allowed drop in percentage points before it counts as a regression "
        f"(default: baseline's value or {DEFAULT_TOLERANCE})",
    )

    p_check = sub.add_parser(
        "check", parents=[common], help="fail on any per-file regression"
    )
    p_check.set_defaults(func=cmd_check)

    p_update = sub.add_parser(
        "update", parents=[common], help="raise the baseline to the current coverage"
    )
    p_update.add_argument(
        "--allow-lower",
        action="append",
        metavar="PATH",
        help="lower this file's floor to its current coverage (repeatable). "
        "Without it, update never moves a floor down; every deliberate "
        "reduction has to be named here and reviewed in the diff.",
    )
    p_update.set_defaults(func=cmd_update)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RatchetError as exc:
        print(f"coverage_ratchet: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
