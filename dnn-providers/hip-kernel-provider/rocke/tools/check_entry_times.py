#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Would rocKE's CTest entries fit their CI shards? A per-entry time check.

CI runs each test tier as a fixed number of shards inside a 30-minute step, and
rocKE shares that step with the rest of the provider, so each tier gives rocKE a
time budget per shard (``CI_TIERS``). Every shard starts every entry of its tier:
an entry whose command takes its shard from the gtest variables
(``rocke.tests.ci_shard.ACTIVATION_FLAG``: the ``rocke.tests.ctest_shard``
pytest plugin's entries and the golden runners) runs only its share, any other
entry runs whole on every shard.

The check combines two inputs. The time each ``rocke_*`` entry costs when run
whole, per operating system, comes from ``entry_times.json`` beside this file,
recorded from CI logs. Which tiers an entry is in, and whether it splits across
shards, comes from the live CTest registration, so relabelling an entry is
judged without re-measuring it.

A tier fails, on any operating system, when an entry alone is over the tier's
budget, or when one shard's sum is over the tier's limit: its budget, or the
ceiling ``entry_times.json`` records for that tier and operating system. A
ceiling holds a tier that is already over budget where it is today, so it
cannot grow further while the fix lands. Ceilings only go down: lower one with
``lower-ceiling`` as the tier gets cheaper (it refuses to raise one, and drops
it once the tier fits its budget); ``record`` keeps them as they are. Each run
prints every tier's margin to its limit and its gap to the budget.

A split entry is projected as dividing evenly across shards, so the figure is
the average shard: an uneven split or per-shard start-up cost puts CI's slowest
shard above it.

The installed test tree carries this file and ``entry_times.json`` in
``tests/tools``. Check the registration CTest lists in a directory; run from the
installed tree's CTest directory, this is the command a CTest entry for the
check would run::

    python check_entry_times.py check --ctest-dir .

``check`` judges one operating system: the one it runs on, or ``--column``.
The registration it lists is that system's, so another system's column cannot
be judged from it. ``check`` also fails when an entry a CI tier runs has no
time in that column, so a new entry cannot count as 0 s, and when a registered
entry has a time in the committed ``entry_times.json`` but none in the
``--entry-times`` file it projects from, so a time cannot drop out of the
projection unnoticed.

Refresh one operating system's times from ``ctest -V`` logs of CI test jobs (any
tier, any shard; the median over the logs is kept, and entries the logs do not
show keep their previous time)::

    python check_entry_times.py record --column linux --source "<what these are>" LOG...

Lower a ceiling to the current projection, rounded up to 10 s::

    python check_entry_times.py lower-ceiling --column linux --tier standard --ctest-dir .
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import math
import re
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

# A source checkout keeps the rocke package beside this tool's tree; the
# installed tests find it through PYTHONPATH.
_SOURCE_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "platform" / "python"
if _SOURCE_PACKAGE_ROOT.is_dir() and str(_SOURCE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_PACKAGE_ROOT))

from rocke.tests.ci_shard import ACTIVATION_FLAG
from rocke.tests.ctest_listing import list_ctest_entries

__all__ = [
    "CI_TIERS",
    "ENTRY_TIMES_PATH",
    "STEP_LIMIT_SECONDS",
    "RegisteredEntry",
    "TierLayout",
    "TierVerdict",
    "entry_seconds_from_ctest_logs",
    "evaluate_tiers",
    "host_column",
    "load_entry_times",
    "lowered_ceiling",
    "registered_entries",
    "times_dropped_from",
    "untimed_entries",
]

ENTRY_TIMES_PATH = Path(__file__).with_name("entry_times.json")

# Only rocKE's entries count against rocKE's share of a shard.
ROCKE_ENTRY_PREFIX = "rocke_"
# CI selects a tier with `-L '^<tier>$' -LE ex_gpu`.
EXCLUDED_LABEL = "ex_gpu"
STEP_LIMIT_SECONDS = 1800.0
# A lowered ceiling is the projection rounded up to this many seconds.
CEILING_STEP_SECONDS = 10


@dataclasses.dataclass(frozen=True)
class TierLayout:
    """How CI runs one tier: its shard count and rocKE's time budget per shard."""

    label: str
    shards: int
    budget_seconds: float


# Shard counts and the 30-minute step are TheRock's test configuration for the
# provider (fetch_test_configurations.py and test_component.yml).
# Budgets: quick runs as one shard and gives rocKE 300 s of it; a standard shard
# gives rocKE 420 s, which keeps rocKE plus the rest of the provider under half
# of the 30-minute step. The budgets are ratchets: fix a breach at its source (a
# cheaper test, a smaller shard key, a different tier), never by raising them.
CI_TIERS = (
    TierLayout("quick", shards=1, budget_seconds=300.0),
    TierLayout("standard", shards=4, budget_seconds=420.0),
)


def host_column() -> str:
    """The ``entry_times.json`` column of the operating system this runs on."""
    return "windows" if sys.platform == "win32" else "linux"


@dataclasses.dataclass(frozen=True)
class RegisteredEntry:
    """One CTest entry as CI selects and shards it."""

    name: str
    labels: frozenset[str]
    splits_across_shards: bool

    def runs_in(self, tier: TierLayout) -> bool:
        return tier.label in self.labels and EXCLUDED_LABEL not in self.labels

    def seconds_per_shard(self, whole_seconds: float, tier: TierLayout) -> float:
        """Time on one shard, assuming a split entry divides evenly."""
        if self.splits_across_shards:
            return whole_seconds / tier.shards
        return whole_seconds


@dataclasses.dataclass(frozen=True)
class TierVerdict:
    """One tier on one operating system: what each shard is projected to cost."""

    column: str
    tier: TierLayout
    seconds_per_shard: dict[str, float]
    ceiling_seconds: float | None = None

    @property
    def shard_seconds(self) -> float:
        return sum(self.seconds_per_shard.values())

    @property
    def limit_seconds(self) -> float:
        """What one shard may cost: the ceiling while there is one, else the budget."""
        if self.ceiling_seconds is None:
            return self.tier.budget_seconds
        return self.ceiling_seconds

    def summary(self) -> str:
        """One line: the shard's cost, its margin to the limit and its budget gap."""
        line = (
            f"{self.column} {self.tier.label} ({self.tier.shards} shard(s)): "
            f"{self.shard_seconds:.0f} s per shard"
        )
        if self.ceiling_seconds is not None:
            line += (
                f", ceiling {self.ceiling_seconds:.0f} s (margin "
                f"{self.limit_seconds - self.shard_seconds:.0f} s)"
            )
        gap = self.shard_seconds - self.tier.budget_seconds
        state = f"{gap:.0f} s over" if gap > 0 else f"{-gap:.0f} s under"
        return f"{line}, {state} the {self.tier.budget_seconds:.0f} s budget"

    def breaches(self) -> list[str]:
        """Each way this tier would cross its limits, worst entry first."""
        where = f"{self.column} {self.tier.label}"
        found = [
            f"{where}: {name} alone takes {seconds:.0f} s of a shard, over the "
            f"{self.tier.budget_seconds:.0f} s budget"
            for name, seconds in self._by_cost()
            if seconds > self.tier.budget_seconds
        ]
        if self.shard_seconds > self.limit_seconds:
            limit = (
                f"{self.tier.budget_seconds:.0f} s budget"
                if self.ceiling_seconds is None
                else f"{self.ceiling_seconds:.0f} s ceiling"
            )
            over_step = (
                f" and over the {STEP_LIMIT_SECONDS:.0f} s step limit"
                if self.shard_seconds > STEP_LIMIT_SECONDS
                else ""
            )
            top = ", ".join(f"{name} {s:.0f} s" for name, s in self._by_cost()[:5])
            found.append(
                f"{where}: one shard takes {self.shard_seconds:.0f} s, over the "
                f"{limit}{over_step}; largest: {top}"
            )
        return found

    def _by_cost(self) -> list[tuple[str, float]]:
        return sorted(self.seconds_per_shard.items(), key=lambda item: -item[1])


def registered_entries(tests: Iterable[Mapping]) -> list[RegisteredEntry]:
    """The ``rocke_*`` entries among ``ctest --show-only=json-v1`` tests."""
    entries = []
    for test in tests:
        name = test["name"]
        if not name.startswith(ROCKE_ENTRY_PREFIX):
            continue
        if "command" not in test:
            raise ValueError(
                f"registered_entries failed - {name} has no command in the listing, "
                "expected one; CTest omits it when it cannot find the program, so "
                "list a built tree with the test interpreter on PATH"
            )
        properties = {p["name"]: p["value"] for p in test.get("properties", [])}
        entries.append(
            RegisteredEntry(
                name=name,
                labels=frozenset(properties.get("LABELS", ())),
                splits_across_shards=ACTIVATION_FLAG in test["command"],
            )
        )
    return entries


def load_entry_times(path: Path = ENTRY_TIMES_PATH) -> dict[str, dict]:
    """``{column: {"source": str, "seconds": {entry: whole seconds},
    "ceilings": {tier label: seconds}}}``; ``ceilings`` is optional."""
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_tiers(
    entries: Iterable[RegisteredEntry],
    entry_times: Mapping[str, Mapping],
    tiers: Sequence[TierLayout] = CI_TIERS,
) -> list[TierVerdict]:
    """Project every tier's shard on every operating system column of
    ``entry_times``. An entry without a time in a column is left out of that
    column's projection: :func:`untimed_entries` names it."""
    entries = list(entries)
    verdicts = []
    for column, recorded in entry_times.items():
        whole_seconds = recorded["seconds"]
        ceilings = recorded.get("ceilings", {})
        for tier in tiers:
            in_tier = [entry for entry in entries if entry.runs_in(tier)]
            verdicts.append(
                TierVerdict(
                    column=column,
                    tier=tier,
                    seconds_per_shard={
                        entry.name: entry.seconds_per_shard(
                            whole_seconds[entry.name], tier
                        )
                        for entry in in_tier
                        if entry.name in whole_seconds
                    },
                    ceiling_seconds=ceilings.get(tier.label),
                )
            )
    return verdicts


def untimed_entries(
    entries: Iterable[RegisteredEntry],
    entry_times: Mapping[str, Mapping],
    tiers: Sequence[TierLayout] = CI_TIERS,
) -> list[str]:
    """One failure per column and entry that a CI tier runs and the column has
    no time for: such an entry would otherwise count as 0 s."""
    in_ci = sorted(
        {entry.name for entry in entries if any(entry.runs_in(t) for t in tiers)}
    )
    return [
        f"{column}: {name} has no time in the data, so no shard counts it; run "
        f"it in a CI job and record its time from the job's `ctest -V` log: "
        f"check_entry_times.py record --column {column} --source "
        f'"<which jobs and dates>" LOG...'
        for column, recorded in entry_times.items()
        for name in in_ci
        if name not in recorded["seconds"]
    ]


def times_dropped_from(
    entries: Iterable[RegisteredEntry],
    committed: Mapping[str, Mapping],
    projected: Mapping[str, Mapping],
) -> list[str]:
    """Registered entries ``committed`` has a time for and ``projected`` lacks.

    An entry without a time is not counted, so losing one would make a tier
    look cheaper than it is.
    """
    registered = {entry.name for entry in entries}
    dropped = []
    for column, recorded in committed.items():
        kept = projected.get(column, {}).get("seconds", {})
        dropped += [
            f"{column}: {name} has a committed time but none in the projected data"
            for name in sorted(registered & set(recorded["seconds"]))
            if name not in kept
        ]
    return dropped


def lowered_ceiling(verdict: TierVerdict) -> float | None:
    """The ceiling ``verdict``'s tier may move down to: its projection rounded up
    to ``CEILING_STEP_SECONDS``, or ``None`` once the tier fits its budget.

    Refuses anything that is not a decrease, because a ceiling only goes down.
    """
    where = f"{verdict.column} {verdict.tier.label}"
    if verdict.ceiling_seconds is None:
        raise ValueError(
            f"lowered_ceiling failed - {where} has ceiling=None, expected a "
            f"ceiling to lower; a tier without one is held to its "
            f"{verdict.tier.budget_seconds:.0f} s budget"
        )
    if verdict.shard_seconds <= verdict.tier.budget_seconds:
        return None
    rounded = CEILING_STEP_SECONDS * math.ceil(
        verdict.shard_seconds / CEILING_STEP_SECONDS
    )
    if rounded >= verdict.ceiling_seconds:
        raise ValueError(
            f"lowered_ceiling failed - {where} projects {verdict.shard_seconds:.0f} "
            f"s, ceiling={verdict.ceiling_seconds:.0f} s, expected the projection "
            f"rounded up ({rounded} s) below the ceiling; ceilings only go down"
        )
    return float(rounded)


# `ctest -V` lines, as ctest prints them or after the timestamp a CI log
# prefixes to each line.
_RESULT = re.compile(r"\d+/\d+ Test +#(\d+): (\S+) \.+.*?\s([\d.]+) sec")
_COMMAND = re.compile(r"(?:^|\s)(\d+): Test command: (.*)$")
_SHARD = re.compile(r"(?:^|\s)(\d+): rocke-ctest-shard: \d+/(\d+)\b")


def _whole_seconds_in_log(text: str) -> dict[str, float]:
    """Each ``rocke_*`` entry's whole-entry seconds as one shard's log shows it."""
    split, shard_count, seconds = set(), {}, {}
    for line in text.splitlines():
        if match := _COMMAND.search(line):
            if ACTIVATION_FLAG in match.group(2):
                split.add(match.group(1))
        elif match := _SHARD.search(line):
            shard_count[match.group(1)] = int(match.group(2))
        elif match := _RESULT.search(line):
            number, name, elapsed = match.groups()
            if name.startswith(ROCKE_ENTRY_PREFIX):
                # A split entry ran 1/n of its items here; n is 1 when the
                # plugin reported no sharding.
                scale = shard_count.get(number, 1) if number in split else 1
                seconds[name] = float(elapsed) * scale
    return seconds


def entry_seconds_from_ctest_logs(logs: Iterable[Path]) -> dict[str, float]:
    """Median whole-entry seconds per ``rocke_*`` entry over ``ctest -V`` logs."""
    samples: dict[str, list[float]] = {}
    for log in logs:
        opener = gzip.open if log.suffix == ".gz" else open
        with opener(log, "rt", encoding="utf-8", errors="replace") as handle:
            for name, seconds in _whole_seconds_in_log(handle.read()).items():
                samples.setdefault(name, []).append(seconds)
    return {
        name: round(statistics.median(values), 2)
        for name, values in sorted(samples.items())
    }


def _registration(arguments: argparse.Namespace) -> list[RegisteredEntry]:
    if arguments.listing:
        listing = json.loads(arguments.listing.read_text(encoding="utf-8"))
        return registered_entries(listing["tests"])
    return registered_entries(list_ctest_entries(arguments.ctest_dir))


def _write_entry_times(entry_times: Mapping, path: Path) -> None:
    path.write_text(json.dumps(entry_times, indent=2) + "\n", encoding="utf-8")


def _column_of(entry_times: Mapping[str, Mapping], column: str, command: str) -> dict:
    """``{column: its data}``, or a refusal naming the columns there are."""
    if column not in entry_times:
        raise KeyError(
            f"{command} failed - column={column!r}, expected one of "
            f"{sorted(entry_times)}"
        )
    return {column: entry_times[column]}


def _check(arguments: argparse.Namespace) -> int:
    entries = _registration(arguments)
    entry_times = load_entry_times(arguments.entry_times)
    column = _column_of(entry_times, arguments.column, "check")
    breaches = untimed_entries(entries, column)
    breaches += times_dropped_from(
        entries, load_entry_times(arguments.committed_times), entry_times
    )
    for verdict in evaluate_tiers(entries, column):
        print(verdict.summary())
        breaches += verdict.breaches()
    for breach in breaches:
        print(f"FAIL {breach}")
    return 1 if breaches else 0


def _record(arguments: argparse.Namespace) -> int:
    entry_times = load_entry_times(arguments.entry_times)
    seconds = entry_seconds_from_ctest_logs(arguments.logs)
    if not seconds:
        raise ValueError(
            "record failed - logs contain no rocke_* entry times, expected "
            "`ctest -V` output of a test job"
        )
    column = entry_times.setdefault(arguments.column, {})
    column["source"] = arguments.source
    column["seconds"] = dict(sorted({**column.get("seconds", {}), **seconds}.items()))
    _write_entry_times(entry_times, arguments.entry_times)
    print(f"recorded {len(seconds)} entries into {arguments.column}")
    return 0


def _lower_ceiling(arguments: argparse.Namespace) -> int:
    entry_times = load_entry_times(arguments.entry_times)
    column = _column_of(entry_times, arguments.column, "lower-ceiling")
    [tier] = [t for t in CI_TIERS if t.label == arguments.tier]
    entries = _registration(arguments)
    untimed = untimed_entries(entries, column, tiers=[tier])
    if untimed:
        raise ValueError(
            f"lower-ceiling failed - {tier.label} untimed entries={len(untimed)}, "
            "expected 0 before a ceiling moves; " + "; ".join(untimed)
        )
    [verdict] = evaluate_tiers(entries, column, tiers=[tier])
    lowered = lowered_ceiling(verdict)
    ceilings = entry_times[arguments.column]["ceilings"]
    if lowered is None:
        del ceilings[tier.label]
        print(f"{arguments.column} {tier.label} fits its budget; ceiling removed")
    else:
        ceilings[tier.label] = lowered
        print(f"{arguments.column} {tier.label} ceiling lowered to {lowered:.0f} s")
    if not ceilings:
        del entry_times[arguments.column]["ceilings"]
    _write_entry_times(entry_times, arguments.entry_times)
    return 0


def _add_registration_source(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ctest-dir", type=Path, help="list the registration here")
    source.add_argument("--listing", type=Path, help="a saved json-v1 listing")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--entry-times", type=Path, default=ENTRY_TIMES_PATH)
    parser.add_argument(
        "--committed-times",
        type=Path,
        default=ENTRY_TIMES_PATH,
        help="data whose registered entries `check` may not lose a time for",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="fail if a tier would exceed its limit")
    check.add_argument(
        "--column",
        default=host_column(),
        help="the operating system whose registration is listed (default: this one)",
    )
    _add_registration_source(check)
    check.set_defaults(run=_check)

    record = commands.add_parser(
        "record", help="update one column's times from CI logs"
    )
    record.add_argument("--column", required=True, help="operating system, e.g. linux")
    record.add_argument("--source", required=True, help="which jobs and dates")
    record.add_argument("logs", type=Path, nargs="+", help="ctest -V logs (.gz ok)")
    record.set_defaults(run=_record)

    lower = commands.add_parser(
        "lower-ceiling", help="move one ceiling down to the current projection"
    )
    lower.add_argument("--column", required=True, help="operating system, e.g. linux")
    lower.add_argument(
        "--tier", required=True, choices=[tier.label for tier in CI_TIERS]
    )
    _add_registration_source(lower)
    lower.set_defaults(run=_lower_ceiling)

    arguments = parser.parse_args(argv)
    return arguments.run(arguments)


if __name__ == "__main__":
    sys.exit(main())
