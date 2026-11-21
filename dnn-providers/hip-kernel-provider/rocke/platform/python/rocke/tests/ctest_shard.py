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
parameter id, so all tests of one parameter set run in the same shard. The
plugin filters after ``-k``, ``-m``, ``--deselect`` and the conftest filters.
Thus every shard divides the same set of tests, and each test runs in exactly
one shard.

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

import dataclasses
import os

import pytest

from .ci_shard import (
    ACTIVATION_FLAG,
    CHILD_ENV_EXCLUDED,
    GTEST_INDEX_ENV,
    GTEST_TOTAL_ENV,
    SHARD_ENV,
    Shard,
    header_line,
    parse_shard,
    resolve_shard,
    shard_of,
    unsharded_environment,
)

__all__ = [
    "ACTIVATION_FLAG",
    "CHILD_ENV_EXCLUDED",
    "GTEST_INDEX_ENV",
    "GTEST_TOTAL_ENV",
    "PLUGIN_NAME",
    "SHARD_ENV",
    "owns_selection",
    "parse_shard",
    "resolve_shard",
    "select_items",
    "shard_key",
    "shard_of",
    "unsharded_environment",
]

PLUGIN_NAME = "rocke.tests.ctest_shard"


def shard_key(item) -> str:
    """Module path plus parameter id (the test function name drops out, so
    the tests of one parameter set share a shard); the node id without
    parameters."""
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return item.nodeid
    return f"{item.nodeid.split('::', 1)[0]}[{callspec.id}]"


def select_items(config, items: list, spec: str | None = None) -> None:
    """``pytest_collection_modifyitems`` body: keep the items of shard
    ``spec`` (default: ``ROCKE_TEST_SHARD``; blank keeps every item)."""
    text = (os.environ.get(SHARD_ENV, "") if spec is None else spec).strip()
    if not text:
        return
    shard = Shard(text)
    keep, drop = [], []
    for item in items:
        (keep if shard.owns(shard_key(item)) else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


@dataclasses.dataclass
class _ShardState:
    shard: Shard | None
    before: int | None = None
    split_line: str | None = None
    kept: int | None = None

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
    try:
        resolved = resolve_shard(from_gtest=config.getoption(ACTIVATION_FLAG))
    except ValueError as error:
        raise pytest.UsageError(f"rocke-ctest-shard: {error}") from error
    shard = None if resolved is None else Shard(*resolved)
    config.stash[_STATE] = _ShardState(shard=shard)


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
    state.split_line = state.shard.split_line(shard_key(item) for item in items)
    select_items(config, items, state.shard.spec)
    state.kept = len(items)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    state = config.stash.get(_STATE, None)
    terminalreporter.write_line(header_line(_shard_of_run(config)))
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
