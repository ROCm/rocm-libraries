#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Which tests did no CI lane run, with no recorded reason? A per-test audit.

``platform/tests/tools/test_pending_lanes.py`` checks, file by file, that every
test file is in a CI-run CTest entry or recorded in ``rocke.tests.pending_lanes``.
It cannot see a single test that skips or is deselected in every CI lane while
the rest of its file runs, nor a file recorded under one lane whose other
tests skip for another reason. This audit sees both, from what CI lanes did:
the run records ``rocke.tests.run_records`` writes when a run sets
``ROCKE_RUN_RECORDS_DIR`` and ``PYTEST_PLUGINS=rocke.tests.run_records``. It
needs those records, so it is not a unit test: a scheduled job, or the
coordinator after a CI emulation, runs it over one records directory per CI
lane, and fails on any test in limbo.

A test is in limbo when no lane ran it and the registry does not account for
it. The registry accounts for it when each lane that collected it but did not
run it gives a reason that ``pending_lanes.waiting_entry`` finds a record for,
or a reason that names the GPU of another lane given (a test that needs a
gfx942 GPU skips on the gfx1100 lane for that lane's gfx942 run to decide),
and at least one lane gives a recorded reason. A lane with no GPU defers every
test a GPU lane collected. A test no lane collected is accounted for only by a
record that covers every test of its file (a quarantine, or a lane that no CI
entry runs yet); pass ``--universe`` (the records of a ``--collect-only`` run
over this source tree) to see those tests.

Name each lane by the GPU arch its runner had (``gfx942``, ``gfx1100``), or by
any other name for a lane without a GPU::

    python tools/limbo_audit.py --lane gfx942=<records> --lane gfx1100=<records> \\
        --universe <records of a --collect-only run> --report limbo.md

Records name test files by their path in the tree that ran them (an installed
tree or a source checkout); the audit maps each to the source test file whose
path is the longest unique suffix of it.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

# A source checkout keeps the rocke package beside this tool's tree.
_SOURCE_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "platform" / "python"
if _SOURCE_PACKAGE_ROOT.is_dir() and str(_SOURCE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_PACKAGE_ROOT))

from rocke.core.arch.target import known_arches
from rocke.tests.pending_lanes import TEST_FILE_ROOTS, waiting_entry
from rocke.tests.run_records import OTHER_SHARD, RunRecord, read_run_records

__all__ = [
    "LaneRun",
    "LimboTest",
    "SourcePaths",
    "audit",
    "lane_run",
    "main",
    "python_files",
    "universe_of",
]

ROCKE_DIR = Path(__file__).resolve().parents[1]
_GFX = re.compile(r"\bgfx[0-9a-f]+\b")
# pytest's skip reason for the placeholder case of a test whose parameter set is
# empty in that run (a parameter set that depends on the GPU, for one).
_EMPTY_PARAMETER_SET = "got empty parameter set"
# How far a test got in a lane, lowest first; a lane keeps the furthest. An
# unfinished test was to run, but its process ended first (run_records).
_PROGRESS = ("collected", "deselected", "unfinished", "skipped", "ran")


class SourcePaths:
    """Maps a test file's path in the tree that ran it to its source path,
    relative to the rocke directory: the source file whose path is the longest
    unique suffix of it, or None."""

    def __init__(self, source_files: Iterable[str]) -> None:
        self._by_suffix: dict[str, list[str]] = collections.defaultdict(list)
        for source in source_files:
            parts = source.split("/")
            for start in range(len(parts)):
                self._by_suffix["/".join(parts[start:])].append(source)
        self._mapped: dict[str, str | None] = {}

    def source_of(self, path: str) -> str | None:
        if path not in self._mapped:
            parts = Path(path).as_posix().split("/")
            self._mapped[path] = next(
                (
                    hits[0]
                    for start in range(len(parts))
                    if len(hits := self._by_suffix.get("/".join(parts[start:]), ()))
                    == 1
                ),
                None,
            )
        return self._mapped[path]


def python_files(rocke_dir: Path) -> list[str]:
    """Every Python file under the test file roots of ``rocke_dir``, relative to
    it: the files a run may collect tests from (an entry may name a test module
    that is not ``test_*.py``)."""
    return [
        file.relative_to(rocke_dir).as_posix()
        for root in TEST_FILE_ROOTS
        for file in (rocke_dir / root).rglob("*.py")
        if "__pycache__" not in file.parts
    ]


TestId = tuple[str, str]  # (source test file, pytest node name)


@dataclass
class LaneRun:
    """What one CI lane did with each test: its furthest event and, when it
    did not run the test, why."""

    name: str
    gpu_arch: str | None  # None: the lane had no GPU
    progress: dict[TestId, str] = field(default_factory=dict)
    reasons: dict[TestId, str] = field(default_factory=dict)
    # Test files skipped whole at collection -> the reason.
    skipped_files: dict[str, str] = field(default_factory=dict)
    # Test files that failed to collect, which fails the lane visibly.
    failed_files: set[str] = field(default_factory=set)
    unmapped: set[str] = field(default_factory=set)

    def note(self, test: TestId, event: str, reason: str) -> None:
        current = self.progress.get(test)
        if current is None or _PROGRESS.index(event) >= _PROGRESS.index(current):
            self.progress[test] = event
            if reason:
                self.reasons[test] = reason

    def ran(self, test: TestId) -> bool:
        return self.progress.get(test) == "ran" or test[0] in self.failed_files

    def collected(self, test: TestId) -> bool:
        return test in self.progress or test[0] in self.skipped_files

    def deselected(self, test: TestId) -> bool:
        """Whether the lane's entries left ``test`` out (a filter removed it)
        rather than running it to a skip."""
        return self.progress.get(test) == "deselected"

    def why_not_run(self, test: TestId) -> str:
        """Why the lane, which collected ``test``, did not run it."""
        if test[0] in self.skipped_files:
            return self.skipped_files[test[0]]
        return self.reasons.get(test, "collected, but no run of it was recorded")


def lane_run(
    name: str, records: Iterable[RunRecord], source_paths: SourcePaths
) -> LaneRun:
    """One lane's records, by source test file and node name. A lane named by a
    known GPU arch had that GPU; any other name, none. A test a shard left to
    another shard of its entry has its verdict in that shard's records; when
    those are missing (a lane run in part), the lane did not observe it."""
    lane = LaneRun(name, name if name in known_arches() else None)
    other_shard: set[TestId] = set()
    for record in records:
        source = source_paths.source_of(record.file)
        if source is None:
            lane.unmapped.add(record.file)
        elif record.event == "module_skipped":
            lane.skipped_files[source] = record.reason
        elif record.event == "collect_error":
            lane.failed_files.add(source)
        elif record.event == "deselected" and record.reason == OTHER_SHARD:
            other_shard.add((source, record.node))
        else:
            lane.note((source, record.node), record.event, record.reason)
    for test in other_shard:
        if lane.progress.get(test) == "collected":
            del lane.progress[test]
    return lane


def universe_of(records: Iterable[RunRecord], source_paths: SourcePaths) -> set[TestId]:
    """The tests a ``--collect-only`` run over the source tree collected."""
    return {
        (source, record.node)
        for record in records
        if record.event == "collected"
        and (source := source_paths.source_of(record.file)) is not None
    }


@dataclass(frozen=True)
class LimboTest:
    """A test no lane ran, with no record of why: what each lane that
    collected it said."""

    path: str
    node: str
    reasons: Mapping[str, str]  # lane name -> why it did not run the test


def _defers(lane: LaneRun, test: TestId, reason: str, lanes: list[LaneRun]) -> bool:
    """Whether ``lane`` leaves ``test`` to another lane: its reason names
    another lane's GPU, or ``lane`` has no GPU and a GPU lane collected it."""
    others = {other.gpu_arch for other in lanes if other is not lane} - {None}
    if lane.gpu_arch is None:
        return any(
            other.gpu_arch is not None and other.collected(test) for other in lanes
        )
    return bool(set(_GFX.findall(reason)) & (others - {lane.gpu_arch}))


def _accounted_for(test: TestId, lanes: list[LaneRun]) -> bool:
    path, node = test
    observed = [lane for lane in lanes if lane.collected(test)]
    if not observed:
        return waiting_entry(path, node, None) is not None
    recorded = 0
    for lane in observed:
        reason = lane.why_not_run(test)
        if (
            waiting_entry(path, node, reason, deselected=lane.deselected(test))
            is not None
        ):
            recorded += 1
        elif not _defers(lane, test, reason, lanes):
            return False
    return recorded > 0


def _placeholder_of_a_test_that_ran(
    test: TestId, lanes: list[LaneRun], ran_functions: set[TestId]
) -> bool:
    """Whether ``test`` is the placeholder pytest makes where a test's parameter
    set is empty, of a test another case of which ran: no case is missing."""
    path, node = test
    return (path, node.partition("[")[0]) in ran_functions and all(
        lane.why_not_run(test).startswith(_EMPTY_PARAMETER_SET)
        for lane in lanes
        if lane.collected(test)
    )


def audit(lanes: list[LaneRun], universe: Iterable[TestId] = ()) -> list[LimboTest]:
    """The tests in limbo among those any lane collected and ``universe``."""
    tests = set(universe).union(*(lane.progress for lane in lanes))
    # A test file skipped whole in a lane stands for its tests seen elsewhere,
    # or for itself (node "") when no lane collected any of them.
    known_files = {path for path, _ in tests}
    tests |= {
        (path, "")
        for lane in lanes
        for path in lane.skipped_files
        if path not in known_files
    }
    ran = {test for test in tests if any(lane.ran(test) for lane in lanes)}
    ran_functions = {(path, node.partition("[")[0]) for path, node in ran}
    limbo = []
    for test in sorted(tests - ran):
        if _accounted_for(test, lanes) or _placeholder_of_a_test_that_ran(
            test, lanes, ran_functions
        ):
            continue
        limbo.append(
            LimboTest(
                *test,
                {
                    lane.name: lane.why_not_run(test)
                    for lane in lanes
                    if lane.collected(test)
                },
            )
        )
    return limbo


def report(limbo: list[LimboTest], lanes: list[LaneRun]) -> str:
    """The audit's report: a count, then each test in limbo with its reasons."""
    lines = [
        "# Tests in limbo",
        "",
        (
            f"Lanes: {', '.join(lane.name for lane in lanes)}. Tests no lane ran "
            f"and rocke.tests.pending_lanes does not account for: {len(limbo)}."
        ),
    ]
    unmapped = sorted(set().union(*(lane.unmapped for lane in lanes)))
    if unmapped:
        lines += ["", f"Records naming no source test file: {len(unmapped)}"]
        lines += [f"- `{path}`" for path in unmapped]
    by_file = collections.Counter(test.path for test in limbo)
    if by_file:
        lines += ["", "| test file | tests |", "|---|---|"]
        lines += [f"| `{path}` | {count} |" for path, count in by_file.most_common()]
        lines += [
            "",
            "| test file | test | why each lane did not run it |",
            "|---|---|---|",
        ]
        for test in limbo:
            why = "; ".join(
                f"{name}: {reason}" for name, reason in test.reasons.items()
            )
            why = why.replace("|", "/").replace("\n", " ")
            lines.append(
                f"| `{test.path}` | `{test.node}` | {why or 'no lane collected it'} |"
            )
    return "\n".join(lines) + "\n"


def _named_directory(text: str) -> tuple[str, Path]:
    name, separator, directory = text.partition("=")
    if not separator or not name or not directory:
        raise argparse.ArgumentTypeError(
            f"--lane failed - value={text!r}, expected NAME=RECORDS_DIR"
        )
    return name, Path(directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--lane",
        action="append",
        required=True,
        type=_named_directory,
        help="NAME=RECORDS_DIR of one CI lane's run; NAME is its GPU arch, if any",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        help="records of a --collect-only run over this source tree",
    )
    parser.add_argument("--report", type=Path, help="write the report here too")
    parser.add_argument(
        "--tree",
        type=Path,
        default=ROCKE_DIR,
        help="the rocke source directory the records ran (default: this checkout)",
    )
    args = parser.parse_args(argv)
    source_paths = SourcePaths(python_files(args.tree))
    lanes = [
        lane_run(name, read_run_records(directory), source_paths)
        for name, directory in args.lane
    ]
    universe = (
        universe_of(read_run_records(args.universe), source_paths)
        if args.universe
        else set()
    )
    limbo = audit(lanes, universe)
    text = report(limbo, lanes)
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    print(
        text
        if len(text) < 20000
        else text[:20000] + "\n[report truncated; see --report]"
    )
    return 1 if limbo else 0


if __name__ == "__main__":
    raise SystemExit(main())
