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
this shard. The key is the module path plus the parameter id, so all tests of
one parameter set run in the same shard. The plugin filters after ``-k``,
``-m``, ``--deselect`` and the conftest filters. Thus every shard divides the
same set of tests, and each test runs in exactly one shard.

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
import hashlib
import os
from collections.abc import Mapping

import pytest

PLUGIN_NAME = "rocke.tests.ctest_shard"
SHARD_ENV = "ROCKE_TEST_SHARD"
GTEST_INDEX_ENV = "GTEST_SHARD_INDEX"
GTEST_TOTAL_ENV = "GTEST_TOTAL_SHARDS"
ACTIVATION_FLAG = "--rocke-shard-from-gtest-env"
# A child pytest that must run unsharded, with only the plugins that it names,
# must not inherit these variables. SHARD_INDEX and TOTAL_SHARDS are
# TheRock's 1-based pair.
CHILD_ENV_EXCLUDED = (
    SHARD_ENV,
    GTEST_INDEX_ENV,
    GTEST_TOTAL_ENV,
    "SHARD_INDEX",
    "TOTAL_SHARDS",
    "PYTEST_ADDOPTS",
    "PYTEST_PLUGINS",
)


def parse_shard(text: str) -> tuple[frozenset[int], int]:
    """``"i/n"`` or ``"i,j,.../n"`` -> ``({i, j, ...}, n)``."""
    picks, sep, total = text.strip().partition("/")
    if not sep:
        raise ValueError(f"shard {text!r} is not i/n")
    n = int(total)
    idx = frozenset(int(p) for p in picks.split(",") if p.strip())
    if n < 1 or not idx or any(not 0 <= i < n for i in idx):
        raise ValueError(f"shard {text!r} out of range")
    return idx, n


def shard_key(item) -> str:
    """Module path plus parameter id (the test function name drops out, so
    the tests of one parameter set share a shard); the node id without
    parameters."""
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return item.nodeid
    return f"{item.nodeid.split('::', 1)[0]}[{callspec.id}]"


def shard_of(key: str, n: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % n


def select_items(config, items: list, spec: str | None = None) -> None:
    """``pytest_collection_modifyitems`` body: keep the items of shard
    ``spec`` (default: ``ROCKE_TEST_SHARD``; blank keeps every item)."""
    text = (os.environ.get(SHARD_ENV, "") if spec is None else spec).strip()
    if not text:
        return
    picks, n = parse_shard(text)
    keep, drop = [], []
    for it in items:
        (keep if shard_of(shard_key(it), n) in picks else drop).append(it)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def _count(name: str, raw: str) -> int:
    if not (raw.isascii() and raw.isdigit()):
        raise ValueError(f"{name}={raw!r} is not a non-negative integer")
    return int(raw)


def _gtest_shard(env: Mapping[str, str]) -> str | None:
    """``"<index>/<total>"`` from the gtest pair; ``None`` when both are
    blank."""
    index_text = env.get(GTEST_INDEX_ENV, "").strip()
    total_text = env.get(GTEST_TOTAL_ENV, "").strip()
    if not index_text and not total_text:
        return None
    if not index_text:
        raise ValueError(f"{GTEST_TOTAL_ENV}={total_text} needs {GTEST_INDEX_ENV}")
    if not total_text:
        raise ValueError(f"{GTEST_INDEX_ENV}={index_text} needs {GTEST_TOTAL_ENV}")
    total = _count(GTEST_TOTAL_ENV, total_text)
    if total < 1:
        raise ValueError(f"{GTEST_TOTAL_ENV}={total_text!r} is not at least 1")
    index = _count(GTEST_INDEX_ENV, index_text)
    if index >= total:
        raise ValueError(
            f"{GTEST_INDEX_ENV}={index_text!r} is not in 0..{total - 1} "
            f"({GTEST_TOTAL_ENV}={total_text})"
        )
    return f"{index}/{total}"


def _gtest_source(env: Mapping[str, str]) -> str:
    index = env.get(GTEST_INDEX_ENV, "").strip()
    total = env.get(GTEST_TOTAL_ENV, "").strip()
    return f"{GTEST_INDEX_ENV}={index}, {GTEST_TOTAL_ENV}={total}"


def unsharded_environment(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """A copy of ``env`` (default ``os.environ``) without the variables in
    ``CHILD_ENV_EXCLUDED``, for a child pytest that runs every test."""
    env = os.environ if env is None else env
    return {k: v for k, v in env.items() if k not in CHILD_ENV_EXCLUDED}


def resolve_shard(
    env: Mapping[str, str] | None = None, from_gtest: bool = False
) -> tuple[str, str] | None:
    """``(spec, source)`` of this run, or ``None`` when there is no shard.

    ``spec`` is ``i/n`` or ``i,j,.../n``. ``source`` names the variables
    that give the shard. A non-blank ``ROCKE_TEST_SHARD`` gives the shard,
    and then the gtest pair is not read. When ``from_gtest`` is true, the
    gtest pair gives the shard if ``ROCKE_TEST_SHARD`` is blank. The 1-based
    ``SHARD_INDEX`` and ``TOTAL_SHARDS`` are not read. Raises ``ValueError``
    for a malformed or out-of-range value of the source that gives the
    shard, and for a half-set gtest pair when that pair is the source."""
    env = os.environ if env is None else env
    text = env.get(SHARD_ENV, "").strip()
    if text:
        parse_shard(text)
        return text, f"{SHARD_ENV}={text}"
    if not from_gtest:
        return None
    spec = _gtest_shard(env)
    return None if spec is None else (spec, _gtest_source(env))


@dataclasses.dataclass
class _ShardState:
    spec: str | None
    source: str = ""
    before: int | None = None
    split: list[int] | None = None
    kept: int | None = None

    def empty_share(self, exitstatus) -> bool:
        """True when the selection had tests, none fell in this shard, and
        pytest found no other problem."""
        return (
            self.spec is not None
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
    if resolved is None:
        config.stash[_STATE] = _ShardState(spec=None)
    else:
        config.stash[_STATE] = _ShardState(spec=resolved[0], source=resolved[1])


def _header_line(state: _ShardState | None) -> str:
    if state is None or state.spec is None:
        return "rocke-ctest-shard: off"
    return f"rocke-ctest-shard: {state.spec} ({state.source})"


def pytest_report_header(config):
    return _header_line(config.stash.get(_STATE, None))


# trylast: the split must see the final selection of every other hook.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    state = config.stash.get(_STATE, None)
    if state is None or state.spec is None:
        return
    _picks, n = parse_shard(state.spec)
    split = [0] * n
    for item in items:
        split[shard_of(shard_key(item), n)] += 1
    state.before = len(items)
    state.split = split
    select_items(config, items, state.spec)
    state.kept = len(items)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    state = config.stash.get(_STATE, None)
    terminalreporter.write_line(_header_line(state))
    if state is None or state.split is None:
        return
    counts = "/".join(str(c) for c in state.split)
    terminalreporter.write_line(
        f"rocke-ctest-shard {state.spec}: split {counts} of {state.before}; "
        f"kept {state.kept}"
    )
    if state.empty_share(exitstatus):
        terminalreporter.write_line(
            f"rocke-ctest-shard {state.spec}: no test of this entry falls in "
            "this shard; the entry passes"
        )


def pytest_sessionfinish(session, exitstatus):
    state = session.config.stash.get(_STATE, None)
    if state is not None and state.empty_share(exitstatus):
        session.exitstatus = pytest.ExitCode.OK
