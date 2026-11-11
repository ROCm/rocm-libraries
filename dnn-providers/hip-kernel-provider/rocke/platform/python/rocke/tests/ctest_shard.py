# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pytest plugin: run one CI shard's share of the selected tests.

Load it with ``python -m pytest -p rocke.tests.ctest_shard``. The shard comes
from ``ROCKE_TEST_SHARD`` (``i/n`` or ``i,j,.../n``, 0-based). When that
variable is blank, the 0-based gtest pair ``GTEST_SHARD_INDEX`` and
``GTEST_TOTAL_SHARDS`` gives the shard. With neither, every test runs.

The plugin hashes a shard key per test and keeps the tests whose hash falls in
this shard. The key is the module path plus the parameter id, so all tests of
one parameter set run in the same shard. The plugin filters after ``-k``,
``-m``, ``--deselect`` and the conftest filters. Thus every shard divides the
same set of tests, and each test runs in exactly one shard.

Each shard prints the whole split, for example
``rocke-ctest-shard 1/4: split 3/2/4/3 of 12; kept 2``. The split line of one
run is the same on every shard. When the selection has tests but none of them
falls in this shard, pytest exits with 77. CTest reports that exit as skipped
under ``SKIP_RETURN_CODE 77``. An empty selection still exits with 5, and a
collection error still exits with 2.

Tiers: a test without a tier marker is in the quick tier. The markers
``tier_standard``, ``tier_comprehensive`` and ``tier_full`` put a test in that
tier. ``--rocke-tier <tier>`` keeps only the tests of that tier.
``--rocke-suite-tiers <tier,...>`` names every tier that has a CTest entry for
this suite, and must include ``quick`` and the ``--rocke-tier`` value. A test
whose tier has no entry is an error, so a marked test always runs in some
entry. Without the two options, every test runs, and a marked test is an error.
Two different tier markers on one test, or an unknown ``tier_*`` marker, is an
error. The plugin selects the tier before the shard, and prints the result,
for example ``rocke-ctest-tier standard: kept 3 of 12``. When the selection
has tests but no test is in this tier, pytest exits with 77, as for a shard.

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
SKIP_EXIT_CODE = 77
TIERS = ("quick", "standard", "comprehensive", "full")
TIER_MARKERS = {f"tier_{tier}": tier for tier in TIERS[1:]}
TIER_MARKER_HELP = {
    "tier_standard": "run the test in the standard tier and above, not in quick",
    "tier_comprehensive": "run the test in the comprehensive and full tiers only",
    "tier_full": "run the test in the full tier only",
}
TIER_OPT_IN = "rocke/platform/CMakeLists.txt"
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
    """``"<index>/<total>"`` from the gtest pair; ``None`` when the total is
    blank or 1."""
    total_text = env.get(GTEST_TOTAL_ENV, "").strip()
    if not total_text:
        return None
    total = _count(GTEST_TOTAL_ENV, total_text)
    if total < 1:
        raise ValueError(f"{GTEST_TOTAL_ENV}={total_text!r} is not at least 1")
    index_text = env.get(GTEST_INDEX_ENV, "").strip()
    if not index_text:
        raise ValueError(f"{GTEST_TOTAL_ENV}={total_text} needs {GTEST_INDEX_ENV}")
    index = _count(GTEST_INDEX_ENV, index_text)
    if index >= total:
        raise ValueError(
            f"{GTEST_INDEX_ENV}={index_text!r} is not in 0..{total - 1} "
            f"({GTEST_TOTAL_ENV}={total_text})"
        )
    if total == 1:
        return None
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


def resolve_shard(env: Mapping[str, str] | None = None) -> tuple[str, str] | None:
    """``(spec, source)`` of this run, or ``None`` when there is no shard.

    ``spec`` is ``i/n`` or ``i,j,.../n``. ``source`` names the variables
    that give the shard. A non-blank ``ROCKE_TEST_SHARD`` always wins. The
    1-based ``SHARD_INDEX`` and ``TOTAL_SHARDS`` are not read. Raises
    ``ValueError`` for a malformed or out-of-range value."""
    env = os.environ if env is None else env
    text = env.get(SHARD_ENV, "").strip()
    if not text:
        spec = _gtest_shard(env)
        return None if spec is None else (spec, _gtest_source(env))
    parsed = parse_shard(text)
    source = f"{SHARD_ENV}={text}"
    if env.get(GTEST_TOTAL_ENV, "").strip():
        try:
            gtest = _gtest_shard(env)
            same = gtest is not None and parse_shard(gtest) == parsed
        except ValueError:
            same = False
        if not same:
            source += f" overrides {_gtest_source(env)}"
    return text, source


def parse_suite_tiers(text: str) -> tuple[str, ...]:
    """``"quick,standard"`` -> ``("quick", "standard")``, in tier order."""
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = sorted(set(names) - set(TIERS))
    if unknown:
        raise ValueError(f"unknown tier {', '.join(unknown)} in {text!r}")
    if "quick" not in names:
        raise ValueError(f"suite tiers {text!r} do not include quick")
    return tuple(tier for tier in TIERS if tier in names)


def item_tier(item) -> str:
    """The tier of ``item`` from its ``tier_*`` markers; ``quick`` without one.
    Raises ``ValueError`` for an unknown ``tier_*`` marker or for two tiers."""
    names = {mark.name for mark in item.iter_markers() if mark.name.startswith("tier_")}
    unknown = sorted(names - TIER_MARKERS.keys())
    if unknown:
        raise ValueError(
            f"{item.nodeid} has the unknown marker {', '.join(unknown)}; "
            f"the tier markers are {', '.join(TIER_MARKERS)}"
        )
    if len(names) > 1:
        raise ValueError(
            f"{item.nodeid} has more than one tier marker: {', '.join(sorted(names))}"
        )
    return TIER_MARKERS[names.pop()] if names else "quick"


@dataclasses.dataclass
class _ShardState:
    spec: str | None
    source: str = ""
    tier: str | None = None
    suite_tiers: tuple[str, ...] = ("quick",)
    before: int | None = None
    tiered: int | None = None
    split: list[int] | None = None
    kept: int | None = None

    def reports_skip(self, exitstatus) -> bool:
        """True when this run exits with ``SKIP_EXIT_CODE``: the selection had
        tests, none fell in this tier and shard, and pytest found no other
        problem."""
        return (
            (self.spec is not None or self.tier is not None)
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
    group = parser.getgroup("rocke-ctest-shard")
    group.addoption(
        "--rocke-tier",
        choices=TIERS,
        default=None,
        help="run only the tests of this tier (unmarked tests are quick)",
    )
    group.addoption(
        "--rocke-suite-tiers",
        default=None,
        metavar="TIER,...",
        help="the tiers that have a CTest entry for this suite",
    )


def _tier_options(config) -> tuple[str | None, tuple[str, ...]]:
    tier = config.getoption("rocke_tier")
    suite_text = config.getoption("rocke_suite_tiers")
    if tier is None and suite_text is None:
        return None, ("quick",)
    if tier is None or suite_text is None:
        raise ValueError("--rocke-tier and --rocke-suite-tiers need each other")
    suite_tiers = parse_suite_tiers(suite_text)
    if tier not in suite_tiers:
        raise ValueError(
            f"--rocke-tier {tier} is not in --rocke-suite-tiers {suite_text}"
        )
    return tier, suite_tiers


def pytest_configure(config):
    for marker, text in TIER_MARKER_HELP.items():
        config.addinivalue_line("markers", f"{marker}: {text}")
    try:
        resolved = resolve_shard()
        tier, suite_tiers = _tier_options(config)
    except ValueError as error:
        raise pytest.UsageError(f"rocke-ctest-shard: {error}") from error
    spec, source = (None, "") if resolved is None else resolved
    config.stash[_STATE] = _ShardState(
        spec=spec, source=source, tier=tier, suite_tiers=suite_tiers
    )


def pytest_report_header(config):
    state = config.stash.get(_STATE, None)
    if state is None or state.spec is None:
        lines = ["rocke-ctest-shard: off"]
    else:
        lines = [f"rocke-ctest-shard: {state.spec} ({state.source})"]
    if state is not None and state.tier is not None:
        lines.append(
            f"rocke-ctest-tier: {state.tier} "
            f"(suite tiers {','.join(state.suite_tiers)})"
        )
    return lines


def _no_entry_message(item, tier: str, state: _ShardState, found) -> str:
    """``found`` holds the tier of every collected test, so the suggested
    suite tiers cover every marked test of this run."""
    opt_in = f"_ROCKE_PYTEST_TIERS_<CTest entry> in {TIER_OPT_IN}"
    if state.tier is None:
        suite_tiers = [name for name in TIERS if name == "quick" or name in found]
        return (
            f"{item.nodeid} is marked tier_{tier}, and this run has no "
            f"--rocke-tier and --rocke-suite-tiers. Give both options, for "
            f"example --rocke-tier {tier} --rocke-suite-tiers "
            f"{','.join(suite_tiers)}. The CTest entries get them when "
            f"{opt_in} lists {' '.join(suite_tiers[1:])}"
        )
    return (
        f"{item.nodeid} is marked tier_{tier}, but this suite has no "
        f"{tier} entry (suite tiers: {','.join(state.suite_tiers)}). "
        f"Add {tier} to {opt_in}, or remove the marker"
    )


def _select_tier(config, state: _ShardState, items: list) -> None:
    """Check the tier of every item, then keep the items of ``state.tier``."""
    errors, tiered_items = [], []
    for item in items:
        try:
            tiered_items.append((item, item_tier(item)))
        except ValueError as error:
            errors.append(str(error))
    found = {tier for _item, tier in tiered_items}
    keep, drop = [], []
    for item, tier in tiered_items:
        if tier not in state.suite_tiers:
            errors.append(_no_entry_message(item, tier, state, found))
            continue
        (keep if state.tier in (None, tier) else drop).append(item)
    if errors:
        shown = errors[:10]
        if len(errors) > len(shown):
            shown.append(f"... and {len(errors) - len(shown)} more")
        raise pytest.UsageError("rocke-ctest-shard: " + "\n".join(shown))
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


# trylast: the split must see the final selection of every other hook.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    state = config.stash.get(_STATE, None)
    if state is None:
        return
    state.before = len(items)
    _select_tier(config, state, items)
    state.tiered = len(items)
    if state.spec is None:
        state.kept = len(items)
        return
    _picks, n = parse_shard(state.spec)
    split = [0] * n
    for item in items:
        index = shard_of(shard_key(item), n)
        if not 0 <= index < n:
            raise pytest.UsageError(
                f"rocke-ctest-shard: {item.nodeid} maps to shard {index}, "
                f"not in 0..{n - 1}"
            )
        split[index] += 1
    state.split = split
    select_items(config, items, state.spec)
    state.kept = len(items)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    state = config.stash.get(_STATE, None)
    if state is None or state.tiered is None:
        return
    if state.tier is not None:
        terminalreporter.write_line(
            f"rocke-ctest-tier {state.tier}: kept {state.tiered} of {state.before}"
        )
    if state.split is not None:
        counts = "/".join(str(c) for c in state.split)
        terminalreporter.write_line(
            f"rocke-ctest-shard {state.spec}: split {counts} of {sum(state.split)}; "
            f"kept {state.kept}"
        )
    if not state.reports_skip(exitstatus):
        return
    if not state.tiered:
        terminalreporter.write_line(
            f"no test of this entry is in the {state.tier} tier; reported as skipped"
        )
    else:
        terminalreporter.write_line(
            f"no test of this entry falls in shard {state.spec}; reported as skipped"
        )


def pytest_sessionfinish(session, exitstatus):
    state = session.config.stash.get(_STATE, None)
    if state is not None and state.reports_skip(exitstatus):
        session.exitstatus = SKIP_EXIT_CODE
