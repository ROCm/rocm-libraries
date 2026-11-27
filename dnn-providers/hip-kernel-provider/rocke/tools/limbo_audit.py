#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Which tests did no CI lane run, with no recorded reason, and which failed in
one? A per-test audit of CI runs.

``platform/tests/tools/test_pending_lanes.py`` checks, file by file, that every
test file is assigned to a CI entry or recorded in ``rocke.tests.pending_lanes``.
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
tree or a source checkout), on any operating system; the audit maps each to the
source test file whose path is the longest unique suffix of it. A lane none of
whose records maps to a source test file, or that ran no test, is no evidence,
and fails the audit as a test in limbo does. The report counts the tests each
lane ran.

A test that failed in a lane fails the audit too: one whose body failed or
whose setup or teardown raised, whose process ended before it did, or whose
file failed to collect, in any run the lane's records hold, even if another
run of it passed. A lane's records directory kept across a CI step's retries
holds every attempt, so a test that passed only on a retry is red here.
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
from rocke.tests.pending_lanes import (
    TEST_FILE_ROOTS,
    python_files_under_test_roots,
    waiting_entry,
)
from rocke.tests.run_records import (
    FAILED_OUTCOMES,
    OTHER_SHARD,
    RunRecord,
    read_run_records,
)

__all__ = [
    "FailedTest",
    "LaneRun",
    "LimboTest",
    "SourcePaths",
    "audit",
    "failed_tests",
    "lane_run",
    "main",
    "universe_of",
]

ROCKE_DIR = Path(__file__).resolve().parents[1]
_GFX = re.compile(r"\bgfx[0-9a-f]+\b")
# The parameter id of the placeholder case pytest makes for a test whose
# parameter set is empty in a run (a parameter set that depends on the GPU, for
# one); one part of the id of a test with stacked parametrizes.
_EMPTY_PARAMETER_SET_ID = "NOTSET"
# How a failed attempt of a test ended, besides FAILED_OUTCOMES: its process
# ended first, or its file failed to collect (node "").
_UNFINISHED = "unfinished"
_COLLECT_ERROR = "collect error"
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
            # Records name files with "/" (run_records); a record a Windows
            # run wrote with "\\" maps the same on any reader's OS.
            parts = path.replace("\\", "/").split("/")
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


TestId = tuple[str, str]  # (source test file, pytest node name)


@dataclass
class LaneRun:
    """What one CI lane did with each test: its furthest event and, when it
    did not run the test, every reason its entries gave for that event (one
    lane runs several entries, so one test can have several)."""

    name: str
    gpu_arch: str | None  # None: the lane had no GPU
    progress: dict[TestId, str] = field(default_factory=dict)
    reasons: dict[TestId, list[str]] = field(default_factory=dict)
    # Test files skipped whole at collection -> the reason.
    skipped_files: dict[str, str] = field(default_factory=dict)
    # Test files that failed to collect, which fails the lane visibly.
    failed_files: set[str] = field(default_factory=set)
    unmapped: set[str] = field(default_factory=set)
    # Each test that failed in a run of the lane -> how each failed run ended.
    failures: dict[TestId, list[str]] = field(default_factory=dict)

    def note(self, test: TestId, event: str, reason: str) -> None:
        """Keep the furthest event of ``test`` and every reason given for it."""
        current = self.progress.get(test)
        if current is not None and _PROGRESS.index(event) < _PROGRESS.index(current):
            return
        if event != current:
            self.progress[test] = event
            self.reasons[test] = []
        if reason and reason not in self.reasons[test]:
            self.reasons[test].append(reason)

    def note_failure(self, test: TestId, outcome: str) -> None:
        """Keep how one run of ``test`` failed."""
        self.failures.setdefault(test, []).append(outcome)

    @property
    def observed_nothing(self) -> bool:
        """Whether none of the lane's records named a source test file: its
        records are missing or from another tree, so it is no evidence."""
        return not (self.progress or self.skipped_files or self.failed_files)

    @property
    def tests_run(self) -> int:
        """How many tests the lane ran (to any outcome)."""
        return sum(event == "ran" for event in self.progress.values())

    def ran(self, test: TestId) -> bool:
        return self.progress.get(test) == "ran" or test[0] in self.failed_files

    def collected(self, test: TestId) -> bool:
        return test in self.progress or test[0] in self.skipped_files

    def deselected(self, test: TestId) -> bool:
        """Whether the lane's entries left ``test`` out (a filter removed it)
        rather than running it to a skip."""
        return self.progress.get(test) == "deselected"

    def reasons_not_run(self, test: TestId) -> tuple[str, ...]:
        """Every reason the lane, which collected ``test``, gave for not
        running it."""
        if test[0] in self.skipped_files:
            return (self.skipped_files[test[0]],)
        return tuple(self.reasons.get(test) or ()) or (
            "collected, but no run of it was recorded",
        )

    def why_not_run(self, test: TestId) -> str:
        """Why the lane, which collected ``test``, did not run it: its reasons,
        joined."""
        return " / ".join(self.reasons_not_run(test))


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
            lane.note_failure((source, ""), _COLLECT_ERROR)
        elif record.event == "deselected" and record.reason == OTHER_SHARD:
            other_shard.add((source, record.node))
        else:
            test = (source, record.node)
            if record.event == "unfinished":
                lane.note_failure(test, _UNFINISHED)
            elif record.event == "ran" and record.reason in FAILED_OUTCOMES:
                lane.note_failure(test, record.reason)
            lane.note(test, record.event, record.reason)
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


@dataclass(frozen=True)
class FailedTest:
    """A test that failed in a lane: how each failed run of it ended, by lane.
    ``node`` is "" for a test file that failed to collect."""

    path: str
    node: str
    outcomes: Mapping[str, tuple[str, ...]]  # lane name -> failed runs' outcomes


def failed_tests(lanes: list[LaneRun]) -> list[FailedTest]:
    """Every test that failed in a run of any of ``lanes``, whether or not
    another run of it passed."""
    tests = sorted(set().union(*(lane.failures for lane in lanes)))
    return [
        FailedTest(
            *test,
            {
                lane.name: tuple(lane.failures[test])
                for lane in lanes
                if test in lane.failures
            },
        )
        for test in tests
    ]


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
        reasons = lane.reasons_not_run(test)
        if any(
            waiting_entry(path, node, reason, deselected=lane.deselected(test))
            is not None
            for reason in reasons
        ):
            recorded += 1
        elif not any(_defers(lane, test, reason, lanes) for reason in reasons):
            return False
    return recorded > 0


def _is_empty_parameter_set_placeholder(node: str) -> bool:
    """Whether the test ``node`` is the placeholder pytest makes where a test's
    parameter set is empty."""
    _, _, parameter_id = node.partition("[")
    return _EMPTY_PARAMETER_SET_ID in parameter_id.removesuffix("]").split("-")


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
    # A test whose parameter set is empty in one run and not in another has
    # its real cases audited one by one, so its placeholder stands for no case.
    functions_with_cases = {
        (path, node.partition("[")[0])
        for path, node in tests
        if not _is_empty_parameter_set_placeholder(node)
    }
    limbo = []
    for test in sorted(tests - ran):
        path, node = test
        if _accounted_for(test, lanes) or (
            _is_empty_parameter_set_placeholder(node)
            and (path, node.partition("[")[0]) in functions_with_cases
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


def _no_evidence(lane: LaneRun) -> bool:
    """Whether ``lane``'s records show nothing it ran."""
    return lane.observed_nothing or not lane.tests_run


def report(
    limbo: list[LimboTest], failed: list[FailedTest], lanes: list[LaneRun]
) -> str:
    """The audit's report: the counts, each lane's tests run, each test that
    failed, then each test in limbo with its reasons."""
    lines = [
        "# Audit of CI runs",
        "",
        (
            f"Lanes: {', '.join(lane.name for lane in lanes)}. Tests no lane ran "
            f"and rocke.tests.pending_lanes does not account for: {len(limbo)}."
        ),
        "",
        f"Tests that failed in a lane, a passed retry included: {len(failed)}.",
        "",
        "| lane | GPU | tests run | tests failed |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {lane.name} | {lane.gpu_arch or 'none'} | {lane.tests_run} | "
        f"{len(lane.failures)} |"
        for lane in lanes
    ]
    empty = [lane.name for lane in lanes if lane.observed_nothing]
    if empty:
        evidence = "Lanes with no record naming a source test file, so no evidence"
        lines += ["", f"{evidence}: {', '.join(empty)}"]
    idle = [
        lane.name for lane in lanes if not (lane.observed_nothing or lane.tests_run)
    ]
    if idle:
        lines += ["", f"Lanes that ran no test, so no evidence: {', '.join(idle)}"]
    if failed:
        lines += [
            "",
            "| test file | test | how each failed run ended |",
            "|---|---|---|",
        ]
        for test in failed:
            how = "; ".join(
                f"{name}: {', '.join(outcomes)}"
                for name, outcomes in test.outcomes.items()
            )
            lines.append(f"| `{test.path}` | `{test.node}` | {how} |")
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
    try:
        tree_files = python_files_under_test_roots(args.tree)
    except RuntimeError as error:
        parser.error(str(error))
    if not tree_files:
        parser.error(
            f"limbo_audit failed - tree={str(args.tree)!r}, expected a rocke "
            f"source checkout with Python files under {', '.join(TEST_FILE_ROOTS)}"
        )
    source_paths = SourcePaths(tree_files)
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
    failed = failed_tests(lanes)
    text = report(limbo, failed, lanes)
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    print(
        text
        if len(text) < 20000
        else text[:20000] + "\n[report truncated; see --report]"
    )
    no_evidence = any(_no_evidence(lane) for lane in lanes)
    return 1 if limbo or failed or no_evidence else 0


if __name__ == "__main__":
    raise SystemExit(main())
