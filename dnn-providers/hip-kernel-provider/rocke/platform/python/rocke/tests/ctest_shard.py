# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pytest plugin: run one CI shard's share of the selected tests.

Load it with ``python -m pytest -p rocke.tests.ctest_shard``. The shard comes
from these sources, in this order:

1. A non-blank ``ROCKE_TEST_SHARD`` (``i/n`` or ``i,j,.../n``, 0-based).
2. The 0-based gtest pair ``GTEST_SHARD_INDEX`` and ``GTEST_TOTAL_SHARDS``,
   only when the command line has ``--rocke-shard-from-gtest-env``.
3. No shard: every test runs.

When ``ROCKE_TEST_SHARD`` is non-blank, the plugin does not read the gtest
pair. The plugin does not read TheRock's 1-based ``SHARD_INDEX`` and
``TOTAL_SHARDS``. A malformed value of the source that gives the shard is a
usage error (exit 4) before collection. A gtest pair with only one variable
set is also a usage error, when it is that source.

The plugin hashes a shard key per test and keeps the tests whose hash falls in
this shard. The shard sources and this rule live in ``rocke.tests.ci_shard``,
which runners that are not pytest share. The key is the module path plus the
parameter id, so all tests of one parameter set run in the same shard.

A heavy test declares its cost with ``@pytest.mark.rocke_shard_cost(seconds)``
(on a function, a class, a module's ``pytestmark`` or one ``pytest.param``; the
closest mark wins). A shard key's cost is the sum over its tests, and the keys
with a cost are spread over the shards by ``ci_shard.place_keys`` instead of by
their hash, heaviest first, so an entry's few heavy groups do not pile onto one
shard by chance. The seconds are a rough single-core measurement; only their
relative size matters. Tests without the mark keep the hash rule. The
plugin filters after ``-k``, ``-m``, ``--deselect`` and the conftest filters.
Thus every shard divides the same set of tests, and each test runs in exactly
one shard.

The marks are placement hints, not the projection of shard time (that is the
per-entry times file). So that a mark does not drift silently from what its
tests cost, the plugin times the marked tests of every run. A shard key counts
when every marked test of it passed; its ratio is measured seconds over
declared cost, and the run's scale is the median ratio of the keys that count,
so the check needs no reference machine. A key whose ratio is more than
``COST_MARK_TOLERANCE`` times the scale, or less than its inverse, gets a
``rocke-ctest-shard: cost mark ...`` line in the terminal summary. The line
warns and never changes the exit status. Under pytest-xdist the tests run in
the workers, which print no summary, so there is no check.

A header line names the shard and its source, for example
``rocke-ctest-shard: 1/4 (GTEST_SHARD_INDEX=1, GTEST_TOTAL_SHARDS=4)``, or
``rocke-ctest-shard: off``. The plugin adds this line to the report header
and prints it again in the terminal summary, so it also shows with ``-q``
or ``--no-header``. Each shard then prints the whole split, for example
``rocke-ctest-shard 1/4: split 3/2/4/3 of 12; kept 2``. The split part of
this line is the same on every shard of one run. When the selection has tests
but none of them falls in this shard, the run passes (exit 0). An empty
selection still exits with 5, and a collection error still exits with 2.

A conftest or a runner can also call ``select_items`` without the plugin to
keep the ``ROCKE_TEST_SHARD`` items. A conftest calls ``owns_selection`` first:
when it is true, the plugin does the selection.
"""

from __future__ import annotations

import collections
import dataclasses
import os
import statistics
from collections.abc import Mapping

import pytest

from .ci_shard import (
    ACTIVATION_FLAG,
    CHILD_ENV_EXCLUDED,
    GTEST_INDEX_ENV,
    GTEST_TOTAL_ENV,
    SHARD_ENV,
    Shard,
    check_cost,
    header_line,
    parse_shard,
    resolve_shard,
    shard_of,
    unsharded_environment,
)
from .run_records import OTHER_SHARD, note_deselection

__all__ = [
    "ACTIVATION_FLAG",
    "CHILD_ENV_EXCLUDED",
    "COST_MARK_TOLERANCE",
    "GTEST_INDEX_ENV",
    "GTEST_TOTAL_ENV",
    "PLUGIN_NAME",
    "SHARD_COST_MARKER",
    "SHARD_ENV",
    "drifted_cost_marks",
    "owns_selection",
    "parse_shard",
    "register_shard_cost_marker",
    "resolve_shard",
    "select_items",
    "shard_costs",
    "shard_key",
    "shard_of",
    "unsharded_environment",
]

PLUGIN_NAME = "rocke.tests.ctest_shard"
SHARD_COST_MARKER = "rocke_shard_cost"
COST_MARK_TOLERANCE = 2.0


def shard_key(item) -> str:
    """Module path plus parameter id (the test function name drops out, so
    the tests of one parameter set share a shard); the node id without
    parameters."""
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return item.nodeid
    return f"{item.nodeid.split('::', 1)[0]}[{callspec.id}]"


def register_shard_cost_marker(config) -> None:
    """Declare the ``rocke_shard_cost`` mark, so a run that marks tests
    without loading this plugin does not warn about an unknown mark."""
    config.addinivalue_line(
        "markers",
        f"{SHARD_COST_MARKER}(seconds): the rough cost of a heavy test, which "
        "places its shard key by cost instead of by hash (rocke.tests.ctest_shard)",
    )


def _declared_cost(item) -> float | None:
    marker = item.get_closest_marker(SHARD_COST_MARKER)
    if marker is None:
        return None
    if marker.kwargs or len(marker.args) != 1:
        raise pytest.UsageError(
            f"{item.nodeid}: {SHARD_COST_MARKER} takes one positional cost in "
            f"seconds, got args={marker.args!r} kwargs={marker.kwargs!r}"
        )
    return marker.args[0]


def shard_costs(items) -> dict[str, float]:
    """The declared cost of each shard key of ``items`` that has one: the sum
    of its tests' ``rocke_shard_cost`` marks. Raises ``pytest.UsageError``
    for a malformed mark or a cost that is not a positive finite number."""
    costs: dict[str, float] = {}
    for item in items:
        cost = _declared_cost(item)
        if cost is None:
            continue
        key = shard_key(item)
        try:
            check_cost(key, cost)
        except ValueError as error:
            raise pytest.UsageError(f"{item.nodeid}: {error}") from error
        costs[key] = costs.get(key, 0) + cost
    return costs


def drifted_cost_marks(
    declared: Mapping[str, float],
    measured: Mapping[str, float],
    tolerance: float = COST_MARK_TOLERANCE,
) -> dict[str, float]:
    """The shard keys whose cost mark is off, each with its ratio to the run's
    scale. A key's ratio is ``measured[key] / declared[key]``; the scale is the
    median ratio over the keys in both mappings; a key is off when its ratio is
    more than ``tolerance`` times the scale or less than the scale divided by
    ``tolerance``. Raises ``ValueError`` unless ``tolerance`` is at least 1."""
    if not (tolerance >= 1):  # also refuses NaN
        raise ValueError(f"cost mark tolerance {tolerance!r} is not at least 1")
    ratios = {key: measured[key] / declared[key] for key in declared if key in measured}
    if not ratios:
        return {}
    scale = statistics.median(ratios.values())
    if scale <= 0:
        return {}
    relative = {key: ratio / scale for key, ratio in ratios.items()}
    return {
        key: value
        for key, value in sorted(relative.items())
        if not 1 / tolerance <= value <= tolerance
    }


def select_items(
    config,
    items: list,
    spec: str | None = None,
    *,
    costs: Mapping[str, float] | None = None,
) -> None:
    """``pytest_collection_modifyitems`` body: keep the items of shard
    ``spec`` (default: ``ROCKE_TEST_SHARD``; blank keeps every item).
    ``costs`` are the shard costs of ``items`` when the caller already has
    them (default: computed here with ``shard_costs``)."""
    text = (os.environ.get(SHARD_ENV, "") if spec is None else spec).strip()
    if not text:
        return
    shard = Shard(text)
    keys = [shard_key(item) for item in items]
    kept = shard.keeps(keys, shard_costs(items) if costs is None else costs)
    keep, drop = [], []
    for item, key in zip(items, keys):
        (keep if key in kept else drop).append(item)
    if drop:
        for item in drop:
            note_deselection(item, OTHER_SHARD)
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


@dataclasses.dataclass
class _ShardState:
    shard: Shard | None
    before: int | None = None
    split_line: str | None = None
    kept: int | None = None
    costs: dict[str, float] | None = None

    def empty_share(self, exitstatus) -> bool:
        """True when the selection had tests, none fell in this shard, and
        pytest found no other problem."""
        return (
            self.shard is not None
            and exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED
            and bool(self.before)
            and self.kept == 0
        )


_STATE = pytest.StashKey[_ShardState]()


# eq=False keeps the instance hashable: it is registered as a plugin.
@dataclasses.dataclass(eq=False)
class _CostMarkTiming:
    """The measured time of the run's marked tests, per shard key."""

    key_of_test: dict[str, str] = dataclasses.field(default_factory=dict)
    declared: dict[str, float] = dataclasses.field(default_factory=dict)
    measured: collections.Counter[str] = dataclasses.field(
        default_factory=collections.Counter
    )
    passed: collections.Counter[str] = dataclasses.field(
        default_factory=collections.Counter
    )
    spoiled: set[str] = dataclasses.field(default_factory=set)

    def pytest_collection_finish(self, session) -> None:
        """Time the marked tests of the run's final selection, reusing the
        shard costs the selection computed when there was a shard."""
        state = session.config.stash.get(_STATE, None)
        costs = None if state is None else state.costs
        self.declared = shard_costs(session.items) if costs is None else costs
        self.key_of_test = {
            item.nodeid: shard_key(item)
            for item in session.items
            if _declared_cost(item) is not None
        }

    def pytest_runtest_logreport(self, report) -> None:
        key = self.key_of_test.get(report.nodeid)
        if key is None:
            return
        self.measured[key] += report.duration
        if report.outcome != "passed":
            self.spoiled.add(key)
        elif report.when == "call":
            self.passed[key] += 1

    def complete(self) -> dict[str, float]:
        """Measured seconds of each key whose marked tests all passed."""
        tests_per_key = collections.Counter(self.key_of_test.values())
        return {
            key: self.measured[key]
            for key, count in tests_per_key.items()
            if key not in self.spoiled and self.passed[key] == count
        }

    def warning_lines(self) -> list[str]:
        drifted = drifted_cost_marks(self.declared, self.complete())
        return [
            f"rocke-ctest-shard: cost mark of {key} is off: declared "
            f"{self.declared[key]:g} s, measured {self.measured[key]:.3g} s, "
            f"{relative:.2g}x the run's median ratio; update its "
            f"{SHARD_COST_MARKER} mark"
            for key, relative in drifted.items()
        ]


_TIMING = pytest.StashKey[_CostMarkTiming]()


def owns_selection(config) -> bool:
    """True when this module is a registered plugin of the run, so it does
    the shard selection. An import of the module alone gives False."""
    return config.pluginmanager.has_plugin(PLUGIN_NAME)


def pytest_addoption(parser):
    parser.addoption(
        ACTIVATION_FLAG,
        action="store_true",
        default=False,
        help=(
            f"take the shard from {GTEST_INDEX_ENV} and {GTEST_TOTAL_ENV} "
            f"when {SHARD_ENV} is blank"
        ),
    )


def pytest_configure(config):
    register_shard_cost_marker(config)
    try:
        resolved = resolve_shard(from_gtest=config.getoption(ACTIVATION_FLAG))
    except ValueError as error:
        raise pytest.UsageError(f"rocke-ctest-shard: {error}") from error
    shard = None if resolved is None else Shard(*resolved)
    config.stash[_STATE] = _ShardState(shard=shard)
    timing = _CostMarkTiming()
    config.stash[_TIMING] = timing
    config.pluginmanager.register(timing, f"{PLUGIN_NAME}.cost_mark_timing")


def _shard_of_run(config) -> Shard | None:
    state = config.stash.get(_STATE, None)
    return None if state is None else state.shard


def pytest_report_header(config):
    return header_line(_shard_of_run(config))


# trylast: the split must see the final selection of every other hook.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    state = config.stash.get(_STATE, None)
    if state is None or state.shard is None:
        return
    state.before = len(items)
    state.costs = shard_costs(items)
    state.split_line = state.shard.split_line(
        [shard_key(item) for item in items], state.costs
    )
    select_items(config, items, state.shard.spec, costs=state.costs)
    state.kept = len(items)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    state = config.stash.get(_STATE, None)
    terminalreporter.write_line(header_line(_shard_of_run(config)))
    timing = config.stash.get(_TIMING, None)
    if timing is not None:
        for line in timing.warning_lines():
            terminalreporter.write_line(line)
    if state is None or state.split_line is None:
        return
    terminalreporter.write_line(state.split_line)
    if state.empty_share(exitstatus):
        terminalreporter.write_line(
            f"rocke-ctest-shard {state.shard.spec}: no test of this entry falls in "
            "this shard; the entry passes"
        )


def pytest_sessionfinish(session, exitstatus):
    state = session.config.stash.get(_STATE, None)
    if state is not None and state.empty_share(exitstatus):
        session.exitstatus = pytest.ExitCode.OK
