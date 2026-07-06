#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-file coverage ratchet for the TensileLite characterization suite.

The coverage *floor* (a single whole-project minimum) lives in
``[tool.coverage.report] fail_under`` in ``pyproject.toml`` and is enforced by
pytest-cov. This ratchet is the complementary no-silent-regression mechanism
(AIHPBLAS-3878): it pins the *current* per-file coverage in a committed
baseline so coverage may rise but never quietly fall, even while the whole
project stays above the floor.

Two modes:

* ``check``  - compare a fresh ``coverage.json`` against the committed baseline
  and fail (exit 1) if any file regressed by more than ``--tolerance``
  percentage points. Prints exactly which files dropped and the one command to
  fix it.
* ``update`` - rewrite the baseline from the current ``coverage.json``. This is
  the single reviewed command for an intentional baseline move; the resulting
  diff is reviewed like any other change.

Input is the coverage.py JSON report (``--cov-report=json``), whose per-file
line-coverage percentage is ``files[<path>]["summary"]["percent_covered"]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Small guard against floating-point / xdist-ordering noise: a file must drop by
# more than this many percentage points to count as a regression.
DEFAULT_TOLERANCE = 0.1

# Printed verbatim so a failing CI log tells the developer exactly how to move
# the baseline on purpose (never a blanket "just regenerate everything").
REMEDIATION = (
    "If these drops are intentional, review them and update the baseline with:\n"
    "    tox -e coverage-unit            # regenerate coverage.json, then\n"
    "    python Tensile/Tests/unit/characterization/tools/coverage_ratchet.py \\\n"
    "        update --current coverage.json\n"
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


def write_baseline(
    current: dict[str, float], baseline_path: Path, tolerance: float
) -> None:
    payload = {
        "_comment": (
            "Per-file coverage ratchet baseline for the TensileLite "
            "characterization suite (AIHPBLAS-3878). Coverage may rise but not "
            "fall. Regenerate with coverage_ratchet.py update; review the diff."
        ),
        "tolerance": tolerance,
        "files": {path: round(pct, 2) for path, pct in sorted(current.items())},
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
    print("\n" + REMEDIATION, file=sys.stderr)
    return 1


def cmd_update(args: argparse.Namespace) -> int:
    current = per_file_coverage(_load_json(Path(args.current), "coverage report"))
    tolerance = args.tolerance if args.tolerance is not None else DEFAULT_TOLERANCE
    baseline_path = Path(args.baseline)
    write_baseline(current, baseline_path, tolerance)
    print(
        f"coverage_ratchet: wrote baseline for {len(current)} files to {baseline_path}"
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

    p_update = sub.add_parser("update", parents=[common], help="rewrite the baseline")
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
