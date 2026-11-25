#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Would rocKE's CTest entries fit their CI shards? A per-shard time check.

CI runs each test tier as a fixed number of shards inside a 30-minute step, and
rocKE shares that step with the rest of the provider, so each tier gives rocKE a
time budget per shard (``CI_TIERS``). Every shard starts every entry of its tier:
an entry whose command takes its shard from the gtest variables
(``rocke.tests.ci_shard.ACTIVATION_FLAG``: the ``rocke.tests.ctest_shard``
pytest plugin's entries and the golden runners) runs only its share, any other
entry runs whole on every shard.

The check combines three inputs:

- The time each ``rocke_*`` entry costs when run whole, per operating system,
  from ``entry_times.json`` beside this file, recorded from CI logs; and the
  time of each slow test of a split pytest entry (``test_seconds``), recorded
  from logs of runs with pytest's ``--durations`` report.
- The live CTest registration: which tiers an entry is in, and whether it
  splits across shards. Relabelling an entry is judged without re-measuring it.
- Each split entry of a sharded tier, listed with its own command, working
  directory and environment but no shard: which shard key each case has (a
  pytest entry's tests, through this file's ``list-keys``; any other split
  runner's cases, through its ``rocke.tests.ci_shard.LIST_KEYS_FLAG``), each
  key's ``rocke_shard_cost`` mark, and which tests run alone. The keys are
  placed by CI's own rule, ``rocke.tests.ci_shard.place_keys``.

Each shard is projected on its own. A split entry's whole time ``W`` (the sum
of its shards) is divided over the shards in proportion to how long each
shard's share takes:

1. A shard key weighs its cases' recorded seconds. A key without them weighs
   its declared cost, scaled by the median ratio of recorded seconds to
   declared cost over the entry's keys that have both (as seconds when none
   has both). The keys with neither share what is left of ``W`` times the
   entry's ``--rocke-workers`` count (the time its processes spend together)
   in proportion to their number of cases.
2. On each shard, the keys of the cases that do not run alone are spread over
   the workers by ``rocke.tests.ctest_workers.worker_keys``. The shard takes as
   long as its busiest worker, plus its cases that run alone, one after another.

An entry whose listing exits with its ``SKIP_RETURN_CODE`` on this host (a GPU
suite of another target) has no cases to place: it divides ``W`` evenly, and
``check`` names it. The projection is an estimate: case times vary between
runs, a declared cost is a rough figure, cases without a time count alike, and
the shared remainder assumes every worker is busy for the whole run.

A tier fails when an entry alone is over the tier's budget on a shard, or when
a shard's sum is over the tier's limit: its budget, or the ceiling
``entry_times.json`` records for that tier and operating system. A ceiling
holds a tier that is already over budget where it is today, so it cannot grow
further while the fix lands. Ceilings only go down: lower one with
``lower-ceiling`` as the tier gets cheaper (it refuses to raise one, and drops
it once the tier's slowest shard fits its budget); ``record`` keeps them as
they are. Each run prints every shard's projection, the margin of the slowest
shard to the limit and its gap to the budget.

``check`` judges one operating system: the one it runs on, or ``--column``.
The registration it lists is that system's, so another system's column cannot
be judged from it. ``check`` also fails when an entry a CI tier runs has no
time in that column, so a new entry cannot count as 0 s, and when a registered
entry has a time in the committed ``entry_times.json`` but none in the
``--entry-times`` file it projects from, so a time cannot drop out of the
projection unnoticed.

The installed test tree carries this file and ``entry_times.json`` in
``tests/tools``, and the check needs that tree: the registration CTest lists
there, with the provider's tier labels, and the installed tests it collects.
Run from the installed tree's CTest directory, this is the command its CTest
entry runs::

    python check_entry_times.py check --ctest-dir .

Refresh one operating system's times from ``ctest -V`` logs of CI test jobs
(any tier, any shard; entries the logs do not show keep their previous time).
A split entry's whole time is the sum over its shards of each shard's median;
when the logs lack a shard, it is the shard count times the median shard. Test
seconds come from the logs that carry pytest's ``--durations`` report, for the
tests of at least ``TEST_SECONDS_FLOOR``::

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
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

# A source checkout keeps the rocke package beside this tool's tree; the
# installed tests find it through PYTHONPATH.
_SOURCE_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "platform" / "python"
if _SOURCE_PACKAGE_ROOT.is_dir() and str(_SOURCE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_PACKAGE_ROOT))

from rocke.tests.ci_shard import (
    ACTIVATION_FLAG,
    GTEST_INDEX_ENV,
    GTEST_TOTAL_ENV,
    LIST_KEYS_FLAG,
    SHARD_ENV,
    WORKER_ENV,
    ShardKeyListing,
    current_shard,
    header_line,
    place_keys,
)
from rocke.tests.ctest_listing import list_ctest_entries
from rocke.tests.ctest_shard import shard_costs, shard_key
from rocke.tests.ctest_workers import RUN_ALONE_MARKER, WORKERS_OPTION, worker_keys
from rocke.tests.host_cpus import available_cpus
from rocke.tests.run_records import fixture_session_environment

__all__ = [
    "CI_TIERS",
    "ENTRY_TIMES_PATH",
    "STEP_LIMIT_SECONDS",
    "TEST_SECONDS_FLOOR",
    "RegisteredEntry",
    "TierLayout",
    "TierVerdict",
    "collect_entry_keys",
    "entry_seconds_from_ctest_logs",
    "evaluate_tiers",
    "host_column",
    "load_entry_times",
    "lowered_ceiling",
    "registered_entries",
    "slow_test_seconds_from_ctest_logs",
    "split_entry_shard_seconds",
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
# `record` keeps the test seconds of the tests at least this slow; the time of
# the faster ones stays in what the keys without a weight share.
TEST_SECONDS_FLOOR = 1.0
# Collecting one entry takes seconds; a collection still running after this
# long is stuck.
_COLLECTION_TIMEOUT_SECONDS = 600.0
# The variables that would make a collection take a CI shard's or a worker's
# share instead of the whole selection.
_SHARE_VARIABLES = (SHARD_ENV, WORKER_ENV, GTEST_INDEX_ENV, GTEST_TOTAL_ENV)
_PYTEST_MODULE = ("-m", "pytest")
# `check` is the one case of its own CTest entry, so a sharded CI tier runs it
# on the shard place_keys gives this key, not on every shard. A lone key lands
# on its hash shard whatever its cost; the unit cost makes it a declared key,
# placed by the cost rule like a `rocke_shard_cost` mark.
_CHECK_KEY = "rocke_entry_time_check"
_CHECK_COSTS = {_CHECK_KEY: 1.0}


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
    """One CTest entry as CI selects, shards and runs it."""

    name: str
    labels: frozenset[str]
    command: tuple[str, ...]
    environment: tuple[str, ...] = ()
    working_directory: str | None = None
    skip_return_code: int | None = None

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError(
                f"RegisteredEntry failed - {self.name} {WORKERS_OPTION}="
                f"{self.workers}, expected at least 1"
            )

    def runs_in(self, tier: TierLayout) -> bool:
        return tier.label in self.labels and EXCLUDED_LABEL not in self.labels

    @property
    def splits_across_shards(self) -> bool:
        return ACTIVATION_FLAG in self.command

    @property
    def pytest_arguments(self) -> tuple[str, ...] | None:
        """The arguments after ``-m pytest``, or ``None`` for an entry whose
        command is not a pytest run."""
        index = self._pytest_module_index()
        return None if index is None else self.command[index + 2 :]

    @property
    def interpreter(self) -> tuple[str, ...]:
        """The command before ``-m pytest``: the interpreter and its options;
        the whole command for an entry that is not a pytest run."""
        index = self._pytest_module_index()
        return self.command if index is None else self.command[:index]

    @property
    def workers(self) -> int:
        """The worker processes the command asks ``rocke.tests.ctest_workers``
        for; 1 without the option."""
        for index, argument in enumerate(self.command):
            if argument == WORKERS_OPTION and index + 1 < len(self.command):
                value = self.command[index + 1]
            elif argument.startswith(f"{WORKERS_OPTION}="):
                value = argument.partition("=")[2]
            else:
                continue
            if not (value.isascii() and value.isdigit()):
                raise ValueError(
                    f"RegisteredEntry failed - {self.name} {WORKERS_OPTION}="
                    f"{value!r}, expected a whole number"
                )
            return int(value)
        return 1

    def _pytest_module_index(self) -> int | None:
        for index in range(1, len(self.command) - 1):
            if self.command[index : index + 2] == _PYTEST_MODULE:
                return index
        return None


@dataclasses.dataclass(frozen=True)
class TierVerdict:
    """One tier on one operating system: what each shard is projected to cost.

    ``shards`` holds, per shard in CI's order, the seconds of each entry there.
    """

    column: str
    tier: TierLayout
    shards: tuple[Mapping[str, float], ...]
    ceiling_seconds: float | None = None

    @property
    def shard_seconds(self) -> tuple[float, ...]:
        return tuple(sum(shard.values()) for shard in self.shards)

    @property
    def slowest_seconds(self) -> float:
        return max(self.shard_seconds, default=0.0)

    @property
    def limit_seconds(self) -> float:
        """What one shard may cost: the ceiling while there is one, else the budget."""
        if self.ceiling_seconds is None:
            return self.tier.budget_seconds
        return self.ceiling_seconds

    def summary(self) -> str:
        """One line: each shard's cost, the slowest one's margin to the limit and
        its gap to the budget."""
        per_shard = " / ".join(f"{seconds:.0f}" for seconds in self.shard_seconds)
        line = (
            f"{self.column} {self.tier.label} ({self.tier.shards} shard(s)): "
            f"{per_shard} s per shard, slowest {self.slowest_seconds:.0f} s"
        )
        if self.ceiling_seconds is not None:
            line += (
                f", ceiling {self.ceiling_seconds:.0f} s (margin "
                f"{self.limit_seconds - self.slowest_seconds:.0f} s)"
            )
        gap = self.slowest_seconds - self.tier.budget_seconds
        state = f"{gap:.0f} s over" if gap > 0 else f"{-gap:.0f} s under"
        return f"{line}, {state} the {self.tier.budget_seconds:.0f} s budget"

    def breaches(self) -> list[str]:
        """Each way a shard of this tier would cross its limits, worst entry
        first. Shards are numbered from 1, as CI's job names number them."""
        found = []
        limit = (
            f"{self.tier.budget_seconds:.0f} s budget"
            if self.ceiling_seconds is None
            else f"{self.ceiling_seconds:.0f} s ceiling"
        )
        for number, shard in enumerate(self.shards, start=1):
            where = (
                f"{self.column} {self.tier.label} shard {number} of {self.tier.shards}"
            )
            by_cost = sorted(shard.items(), key=lambda item: -item[1])
            found += [
                f"{where}: {name} alone takes {seconds:.0f} s, over the "
                f"{self.tier.budget_seconds:.0f} s budget"
                for name, seconds in by_cost
                if seconds > self.tier.budget_seconds
            ]
            total = sum(shard.values())
            if total > self.limit_seconds:
                over_step = (
                    f" and over the {STEP_LIMIT_SECONDS:.0f} s step limit"
                    if total > STEP_LIMIT_SECONDS
                    else ""
                )
                top = ", ".join(f"{name} {s:.0f} s" for name, s in by_cost[:5])
                found.append(
                    f"{where} takes {total:.0f} s, over the {limit}{over_step}; "
                    f"largest: {top}"
                )
        return found


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
        if "ENVIRONMENT_MODIFICATION" in properties:
            raise ValueError(
                f"registered_entries failed - {name} has ENVIRONMENT_MODIFICATION, "
                "expected ENVIRONMENT only; the check collects an entry in the "
                "environment ENVIRONMENT gives it"
            )
        entries.append(
            RegisteredEntry(
                name=name,
                labels=frozenset(properties.get("LABELS", ())),
                command=tuple(test["command"]),
                environment=tuple(properties.get("ENVIRONMENT", ())),
                working_directory=properties.get("WORKING_DIRECTORY"),
                skip_return_code=properties.get("SKIP_RETURN_CODE"),
            )
        )
    return entries


def load_entry_times(path: Path = ENTRY_TIMES_PATH) -> dict[str, dict]:
    """``{column: {"source": str, "seconds": {entry: whole seconds},
    "test_seconds": {entry: {test node id: seconds}},
    "ceilings": {tier label: seconds}}}``; ``test_seconds`` and ``ceilings``
    are optional."""
    return json.loads(path.read_text(encoding="utf-8"))


def _cost_scale(measured: Mapping[str, float], declared: Mapping[str, float]) -> float:
    """Seconds per unit of declared cost: the median ratio over the keys with
    both a recorded time and a declared cost; 1 when no key has both."""
    ratios = [measured[key] / declared[key] for key in measured if key in declared]
    return statistics.median(ratios) if ratios else 1.0


def split_entry_shard_seconds(
    whole_seconds: float,
    shards: int,
    workers: int = 1,
    keys: ShardKeyListing | None = None,
    test_seconds: Mapping[str, float] | None = None,
) -> tuple[float, ...]:
    """A split entry's seconds on each of ``shards``, by the rule in the module
    docstring: ``keys`` places its tests (``None``: an even split), a key
    weighs its tests' ``test_seconds`` else its scaled declared cost, and a
    shard takes as long as its busiest of ``workers`` plus its tests that run
    alone."""
    for name, value in (("shards", shards), ("workers", workers)):
        if value < 1:
            raise ValueError(
                f"split_entry_shard_seconds failed - {name}={value}, expected at "
                "least 1"
            )
    if keys is None or not keys.key_of_case:
        return (whole_seconds / shards,) * shards
    recorded = {} if test_seconds is None else test_seconds
    ordered_keys = list(dict.fromkeys(keys.key_of_case.values()))
    tests_of: dict[str, list[str]] = {key: [] for key in ordered_keys}
    for test, key in keys.key_of_case.items():
        tests_of[key].append(test)
    # Each key's seconds on a worker and alone, after the workers.
    on_worker: dict[str, float] = {}
    alone: dict[str, float] = {}
    measured = {}
    for key, tests in tests_of.items():
        timed = [test for test in tests if test in recorded]
        if timed:
            on_worker[key] = sum(
                recorded[test] for test in timed if test not in keys.run_alone_cases
            )
            alone[key] = sum(
                recorded[test] for test in timed if test in keys.run_alone_cases
            )
            measured[key] = on_worker[key] + alone[key]
    scale = _cost_scale(measured, keys.costs)
    for key, cost in keys.costs.items():
        if key not in measured:
            on_worker[key] = keys.worker_costs.get(key, 0.0) * scale
            alone[key] = cost * scale - on_worker[key]
    unweighed = [key for key in ordered_keys if key not in on_worker]
    if unweighed:
        spent = sum(on_worker.values()) + sum(alone.values())
        unweighed_cases = sum(len(tests_of[key]) for key in unweighed)
        per_case = max(whole_seconds * workers - spent, 0.0) / unweighed_cases
        for key in unweighed:
            share = per_case * len(tests_of[key])
            runs_alone = all(test in keys.run_alone_cases for test in tests_of[key])
            on_worker[key], alone[key] = (0.0, share) if runs_alone else (share, 0.0)
    placement = place_keys(ordered_keys, shards, keys.costs)
    busy = []
    for shard in range(shards):
        on_shard = [key for key in ordered_keys if placement[key] == shard]
        worked = [
            key
            for key in on_shard
            if any(test not in keys.run_alone_cases for test in tests_of[key])
        ]
        worker_of = worker_keys(worked, dict(keys.worker_costs), workers)
        load = [0.0] * workers
        for key in worked:
            load[worker_of[key]] += on_worker[key]
        busy.append(max(load) + sum(alone[key] for key in on_shard))
    total = sum(busy)
    if total == 0:
        return (whole_seconds / shards,) * shards
    return tuple(whole_seconds * seconds / total for seconds in busy)


def evaluate_tiers(
    entries: Iterable[RegisteredEntry],
    entry_times: Mapping[str, Mapping],
    entry_keys: Mapping[str, ShardKeyListing | None] | None = None,
    tiers: Sequence[TierLayout] = CI_TIERS,
) -> list[TierVerdict]:
    """Project every shard of every tier on every operating system column of
    ``entry_times``.

    ``entry_keys`` holds the collected selection of split pytest entries; a
    split entry with none, or with ``None``, divides evenly. An entry without a
    time in a column is left out of that column's projection:
    :func:`untimed_entries` names it."""
    entries = list(entries)
    keys_by_entry = {} if entry_keys is None else entry_keys
    verdicts = []
    for column, recorded in entry_times.items():
        whole_seconds = recorded["seconds"]
        test_seconds = recorded.get("test_seconds", {})
        ceilings = recorded.get("ceilings", {})
        for tier in tiers:
            shards: list[dict[str, float]] = [{} for _ in range(tier.shards)]
            for entry in entries:
                if not entry.runs_in(tier) or entry.name not in whole_seconds:
                    continue
                whole = whole_seconds[entry.name]
                if entry.splits_across_shards:
                    per_shard = split_entry_shard_seconds(
                        whole,
                        tier.shards,
                        entry.workers,
                        keys_by_entry.get(entry.name),
                        test_seconds.get(entry.name),
                    )
                else:
                    per_shard = (whole,) * tier.shards
                for shard, seconds in zip(shards, per_shard):
                    shard[entry.name] = seconds
            verdicts.append(
                TierVerdict(
                    column=column,
                    tier=tier,
                    shards=tuple(shards),
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
    """The ceiling ``verdict``'s tier may move down to: its slowest shard
    rounded up to ``CEILING_STEP_SECONDS``, or ``None`` once that shard fits
    its budget.

    Refuses anything that is not a decrease, because a ceiling only goes down.
    """
    where = f"{verdict.column} {verdict.tier.label}"
    if verdict.ceiling_seconds is None:
        raise ValueError(
            f"lowered_ceiling failed - {where} has ceiling=None, expected a "
            f"ceiling to lower; a tier without one is held to its "
            f"{verdict.tier.budget_seconds:.0f} s budget"
        )
    slowest = verdict.slowest_seconds
    if slowest <= verdict.tier.budget_seconds:
        return None
    rounded = CEILING_STEP_SECONDS * math.ceil(slowest / CEILING_STEP_SECONDS)
    if rounded >= verdict.ceiling_seconds:
        raise ValueError(
            f"lowered_ceiling failed - {where} projects {slowest:.0f} s on its "
            f"slowest shard, ceiling={verdict.ceiling_seconds:.0f} s, expected the "
            f"projection rounded up ({rounded} s) below the ceiling; ceilings only "
            "go down"
        )
    return float(rounded)


def _collection_environment(entry: RegisteredEntry) -> dict[str, str]:
    """The environment ``entry`` runs in (this process's, with the entry's
    ``ENVIRONMENT`` over it), less what would take a share of the selection
    and the run-records directory, as a collection is not a run."""
    environment = dict(fixture_session_environment())
    for assignment in entry.environment:
        name, _, value = assignment.partition("=")
        environment[name] = value
    for name in _SHARE_VARIABLES:
        environment.pop(name, None)
    return environment


# Runs this file in the child, with `-c` keeping the working directory first on
# sys.path as `python -m pytest` does, so the entry's imports resolve as in its
# own run. (In a source checkout, this file then puts the source package first.)
_RUN_THIS_FILE = (
    "import runpy, sys; sys.argv = sys.argv[1:]; "
    "runpy.run_path(sys.argv[0], run_name='__main__')"
)


def _listing_command(entry: RegisteredEntry, listing: Path) -> list[str]:
    """The command that makes ``entry`` write its :class:`ShardKeyListing` to
    ``listing``: a pytest entry through this file's ``list-keys``, any other
    split entry (a golden runner, the emitter-pair check) with
    ``LIST_KEYS_FLAG``."""
    pytest_arguments = entry.pytest_arguments
    if pytest_arguments is None:
        return [*entry.command, LIST_KEYS_FLAG, str(listing)]
    return [
        *entry.interpreter,
        "-c",
        _RUN_THIS_FILE,
        str(Path(__file__).resolve()),
        "list-keys",
        "--output",
        str(listing),
        "--",
        *pytest_arguments,
    ]


def collect_entry_keys(
    entry: RegisteredEntry, timeout_seconds: float = _COLLECTION_TIMEOUT_SECONDS
) -> ShardKeyListing | None:
    """List the cases of ``entry``, a split entry, as CI's shards would before
    taking their share: its own command, working directory and environment,
    with no shard. ``None`` when the listing exits with the entry's
    ``SKIP_RETURN_CODE``: the entry skips on this host (a GPU suite of another
    target), so there is no selection to place."""
    if not entry.splits_across_shards:
        raise ValueError(
            f"collect_entry_keys failed - {entry.name} command="
            f"{' '.join(entry.command)!r}, expected a split entry ({ACTIVATION_FLAG})"
        )
    with tempfile.TemporaryDirectory() as scratch:
        listing = Path(scratch) / "keys.json"
        try:
            result = subprocess.run(
                _listing_command(entry, listing),
                check=False,
                cwd=entry.working_directory,
                env=_collection_environment(entry),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as expired:
            raise RuntimeError(
                f"collect_entry_keys failed - {entry.name} timeout_seconds="
                f"{timeout_seconds:g}, expected its listing to finish within it"
            ) from expired
        if (
            entry.skip_return_code is not None
            and result.returncode == entry.skip_return_code
        ):
            return None
        if result.returncode != 0 or not listing.is_file():
            output = (result.stdout + result.stderr).strip().splitlines()[-20:]
            written = "written" if listing.is_file() else "not written"
            raise RuntimeError(
                f"collect_entry_keys failed - {entry.name} returncode="
                f"{result.returncode} with the key listing {written}, expected 0 "
                "with the listing written; its last lines:\n" + "\n".join(output)
            )
        return ShardKeyListing.read(listing)


def _entries_to_collect(
    entries: Iterable[RegisteredEntry], tiers: Sequence[TierLayout] = CI_TIERS
) -> list[RegisteredEntry]:
    """The split entries of the tiers that run as more than one shard."""
    return [
        entry
        for entry in entries
        if entry.splits_across_shards
        and any(tier.shards > 1 and entry.runs_in(tier) for tier in tiers)
    ]


def _collect_all(
    entries: Sequence[RegisteredEntry],
) -> dict[str, ShardKeyListing | None]:
    """:func:`collect_entry_keys` of every entry :func:`_entries_to_collect`
    names, in parallel."""
    to_collect = _entries_to_collect(entries)
    if not to_collect:
        return {}
    with ThreadPoolExecutor(max_workers=available_cpus()) as pool:
        collected = list(pool.map(collect_entry_keys, to_collect))
    return {entry.name: keys for entry, keys in zip(to_collect, collected)}


# `ctest -V` lines, as ctest prints them or after the timestamp a CI log
# prefixes to each line.
_RESULT = re.compile(r"\d+/\d+ Test +#(\d+): (\S+) \.+.*?\s([\d.]+) sec")
_COMMAND = re.compile(r"(?:^|\s)(\d+): Test command: (.*)$")
_SHARD = re.compile(r"(?:^|\s)(\d+): rocke-ctest-shard: (\d+)/(\d+)\b")
# One line of pytest's `--durations` report, as an entry prints it (a worker's
# lines carry its `[w<index>]` prefix).
_DURATION = re.compile(
    r"(?:^|\s)(\d+): (?:\[\w+\] )?([\d.]+)s (?:setup|call|teardown) +(\S.*?)\s*$"
)


@dataclasses.dataclass(frozen=True)
class _ShardRun:
    """One entry's run in one CI shard's log: which shard of how many, and
    how long it took. An entry that does not split ran whole: shard 0 of 1."""

    index: int
    count: int
    seconds: float


def _shard_runs_in_log(text: str) -> dict[str, _ShardRun]:
    """Each ``rocke_*`` entry's run as one shard's log shows it."""
    split, shard_of_number, runs = set(), {}, {}
    for line in text.splitlines():
        if match := _COMMAND.search(line):
            if ACTIVATION_FLAG in match.group(2):
                split.add(match.group(1))
        elif match := _SHARD.search(line):
            number, index, count = match.groups()
            shard_of_number[number] = (int(index), int(count))
        elif match := _RESULT.search(line):
            number, name, elapsed = match.groups()
            if name.startswith(ROCKE_ENTRY_PREFIX):
                # The plugin reports no shard line when the run is unsharded.
                index, count = (
                    shard_of_number.get(number, (0, 1)) if number in split else (0, 1)
                )
                runs[name] = _ShardRun(index, count, float(elapsed))
    return runs


def _test_seconds_in_log(text: str) -> dict[str, dict[str, float]]:
    """Each ``rocke_*`` entry's tests with their setup, call and teardown
    seconds summed, from the ``--durations`` reports in one shard's log."""
    by_number: dict[str, dict[str, float]] = {}
    names = {}
    for line in text.splitlines():
        if match := _DURATION.search(line):
            number, seconds, test = match.groups()
            tests = by_number.setdefault(number, {})
            tests[test] = tests.get(test, 0.0) + float(seconds)
        elif match := _RESULT.search(line):
            names[match.group(1)] = match.group(2)
    return {
        names[number]: tests
        for number, tests in by_number.items()
        if names.get(number, "").startswith(ROCKE_ENTRY_PREFIX)
    }


def _read_log(log: Path) -> str:
    opener = gzip.open if log.suffix == ".gz" else open
    with opener(log, "rt", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _whole_seconds(shard_seconds: Mapping[int, Mapping[int, list[float]]]) -> float:
    """An entry's whole time from its shard runs, grouped by shard count and
    shard: per shard count, the sum of each shard's median when every shard
    is there, else the count times the median shard; the median over counts."""
    wholes = []
    for count, by_index in shard_seconds.items():
        if set(by_index) == set(range(count)):
            wholes.append(sum(statistics.median(by_index[i]) for i in range(count)))
        else:
            every = [seconds for runs in by_index.values() for seconds in runs]
            wholes.append(count * statistics.median(every))
    return statistics.median(wholes)


def entry_seconds_from_ctest_logs(logs: Iterable[Path]) -> dict[str, float]:
    """Whole-entry seconds per ``rocke_*`` entry over ``ctest -V`` logs of CI
    shards (:func:`_whole_seconds`)."""
    samples: dict[str, dict[int, dict[int, list[float]]]] = {}
    for log in logs:
        for name, run in _shard_runs_in_log(_read_log(log)).items():
            by_index = samples.setdefault(name, {}).setdefault(run.count, {})
            by_index.setdefault(run.index, []).append(run.seconds)
    return {
        name: round(_whole_seconds(by_count), 2)
        for name, by_count in sorted(samples.items())
    }


def slow_test_seconds_from_ctest_logs(
    logs: Iterable[Path],
) -> dict[str, dict[str, float]]:
    """Per ``rocke_*`` entry, the median seconds of each test that takes at
    least ``TEST_SECONDS_FLOOR``, over the logs whose ``--durations`` report
    shows it. A test runs in one shard of a run, so each log of one run adds
    its own tests."""
    samples: dict[str, dict[str, list[float]]] = {}
    for log in logs:
        for name, tests in _test_seconds_in_log(_read_log(log)).items():
            for test, seconds in tests.items():
                samples.setdefault(name, {}).setdefault(test, []).append(seconds)
    recorded = {}
    for name, tests in sorted(samples.items()):
        medians = {test: statistics.median(values) for test, values in tests.items()}
        slow = {
            test: round(seconds, 2)
            for test, seconds in sorted(medians.items())
            if seconds >= TEST_SECONDS_FLOOR
        }
        if slow:
            recorded[name] = slow
    return recorded


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


def _list_check_case(path: Path) -> int:
    """Write the check's one case to ``path`` (``LIST_KEYS_FLAG``)."""
    ShardKeyListing({_CHECK_KEY: _CHECK_KEY}, costs=_CHECK_COSTS).write(path)
    return 0


def _check_runs_here(arguments: argparse.Namespace) -> bool:
    """Whether this run of ``check`` checks: always without the shard flag;
    with it, only on the shard that keeps the check's key. With the flag it
    prints the shard lines the other split runs print."""
    if not arguments.shard_from_gtest_env:
        return True
    print(header_line(arguments.shard))
    if arguments.shard is None:
        return True
    print(arguments.shard.split_line([_CHECK_KEY], _CHECK_COSTS))
    return _CHECK_KEY in arguments.shard.keeps([_CHECK_KEY], _CHECK_COSTS)


def _check(arguments: argparse.Namespace) -> int:
    if arguments.list_shard_keys is not None:
        return _list_check_case(arguments.list_shard_keys)
    if not _check_runs_here(arguments):
        return 0
    entries = _registration(arguments)
    entry_times = load_entry_times(arguments.entry_times)
    column = _column_of(entry_times, arguments.column, "check")
    entry_keys = _collect_all(entries)
    not_placed = sorted(name for name, keys in entry_keys.items() if keys is None)
    if not_placed:
        print(
            "not projected per shard, as their listing skips on this host "
            "(divided evenly): " + ", ".join(not_placed)
        )
    breaches = untimed_entries(entries, column)
    breaches += times_dropped_from(
        entries, load_entry_times(arguments.committed_times), entry_times
    )
    for verdict in evaluate_tiers(entries, column, entry_keys):
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
    test_seconds = slow_test_seconds_from_ctest_logs(arguments.logs)
    if test_seconds:
        column["test_seconds"] = dict(
            sorted({**column.get("test_seconds", {}), **test_seconds}.items())
        )
    _write_entry_times(entry_times, arguments.entry_times)
    print(
        f"recorded {len(seconds)} entries and the slow tests of "
        f"{len(test_seconds)} into {arguments.column}"
    )
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
    [verdict] = evaluate_tiers(entries, column, _collect_all(entries), tiers=[tier])
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


class _KeyListing:
    """A pytest plugin that writes the selection's :class:`ShardKeyListing` to
    ``output`` once every filter has run."""

    def __init__(self, output: Path) -> None:
        self.output = output

    def pytest_collection_finish(self, session) -> None:
        items = session.items
        alone = {
            item.nodeid for item in items if item.get_closest_marker(RUN_ALONE_MARKER)
        }
        ShardKeyListing(
            key_of_case={item.nodeid: shard_key(item) for item in items},
            costs=shard_costs(items),
            worker_costs=shard_costs(
                [item for item in items if item.nodeid not in alone]
            ),
            run_alone_cases=frozenset(alone),
        ).write(self.output)


def _list_keys(arguments: argparse.Namespace) -> int:
    """Collect the pytest run ``arguments.pytest_arguments`` names into
    ``arguments.output`` (:class:`_KeyListing`). Runs in the entry's own
    working directory and environment (:func:`collect_entry_keys`)."""
    pytest_arguments = list(arguments.pytest_arguments)
    if pytest_arguments[:1] == ["--"]:
        pytest_arguments = pytest_arguments[1:]
    status = pytest.main(
        [*pytest_arguments, "--collect-only", "-q", "-p", "no:cacheprovider"],
        plugins=[_KeyListing(arguments.output)],
    )
    # An empty selection is a listing with no keys, not a failure.
    collected = status in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED)
    return 0 if collected else int(status)


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

    check = commands.add_parser("check", help="fail if a shard would exceed its limit")
    check.add_argument(
        "--column",
        default=host_column(),
        help="the operating system whose registration is listed (default: this one)",
    )
    _add_registration_source(check)
    check.add_argument(
        ACTIVATION_FLAG,
        dest="shard_from_gtest_env",
        action="store_true",
        help="check only on the CI shard that keeps this check's one case",
    )
    check.add_argument(
        LIST_KEYS_FLAG,
        dest="list_shard_keys",
        type=Path,
        default=None,
        help="write this check's one case to this path instead of checking",
    )
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

    list_keys = commands.add_parser(
        "list-keys",
        help="(used by check) write a pytest run's shard keys and declared costs",
    )
    list_keys.add_argument("--output", type=Path, required=True)
    list_keys.add_argument("pytest_arguments", nargs=argparse.REMAINDER)
    list_keys.set_defaults(run=_list_keys)

    arguments = parser.parse_args(argv)
    if arguments.run is _check:
        if arguments.list_shard_keys is not None and not arguments.shard_from_gtest_env:
            parser.error(f"{LIST_KEYS_FLAG} needs {ACTIVATION_FLAG}")
        try:
            arguments.shard = (
                current_shard(from_gtest=True)
                if arguments.shard_from_gtest_env
                else None
            )
        except ValueError as error:
            parser.error(f"rocke-ctest-shard: {error}")
    return arguments.run(arguments)


if __name__ == "__main__":
    sys.exit(main())
