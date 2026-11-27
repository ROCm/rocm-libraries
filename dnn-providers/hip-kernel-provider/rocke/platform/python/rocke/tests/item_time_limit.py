# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pytest plugin: an entry's per-test time limit, raised only by a measured time.

Load it with ``-p rocke.tests.item_time_limit`` beside pytest-timeout's
``--timeout`` (the entry's per-test limit, in seconds). Every installed GPU
pytest entry does (``platform/CMakeLists.txt``). The limit holds for every test
of the entry, with one way to raise it for one test: the test declares its
slowest measured time on CI hardware with
``@pytest.mark.rocke_measured_seconds(seconds)`` (on a function, a class, a
module's ``pytestmark`` or one ``pytest.param``; the closest mark wins). Its
limit becomes ``LIMIT_HEADROOM`` times that time when that is above the
entry's limit, so the test keeps its worst case under half its limit; the mark
never lowers a limit. A test that is slower than half the entry's limit and
carries no mark is a test to make cheaper or to mark, with the measurement.
Keep the raised limit under half the entry's CTest ``TIMEOUT``: past that, a
stall again ends the entry before the test is named.

So that the mark stays the only way up, a ``pytest.mark.timeout`` must give a
positive limit no higher than the entry's (it can only tighten); a timeout mark
above the entry's limit, one that turns the limit off (zero, a negative value
or None), one that is not a number, or one beside the mark is a usage error at
collection (exit 4) that names the test.

The mark changes nothing in a run without this plugin; a conftest declares it
there with ``register_measured_seconds_marker``.
"""

from __future__ import annotations

import math

import pytest

__all__ = [
    "LIMIT_HEADROOM",
    "MEASURED_SECONDS_MARKER",
    "PLUGIN_NAME",
    "per_test_limit",
    "register_measured_seconds_marker",
]

PLUGIN_NAME = __name__
MEASURED_SECONDS_MARKER = "rocke_measured_seconds"
# A test's worst case stays under half its limit.
LIMIT_HEADROOM = 2.0
_TIMEOUT_MARKER = "timeout"


def register_measured_seconds_marker(config: pytest.Config) -> None:
    """Declare the ``rocke_measured_seconds`` mark, so a run that marks tests
    without loading this plugin does not warn about an unknown mark."""
    config.addinivalue_line(
        "markers",
        f"{MEASURED_SECONDS_MARKER}(seconds): the slowest measured CI time of a test "
        f"slower than half its entry's per-test limit; the entry raises that test's "
        f"limit to {LIMIT_HEADROOM:g}x it ({PLUGIN_NAME})",
    )


def _is_positive_seconds(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _measured_seconds(item: pytest.Item) -> float | None:
    marker = item.get_closest_marker(MEASURED_SECONDS_MARKER)
    if marker is None:
        return None
    arguments = [*marker.args, *marker.kwargs.values()]
    if set(marker.kwargs) - {"seconds"} or len(arguments) != 1:
        arguments = [None]
    if not _is_positive_seconds(arguments[0]):
        raise pytest.UsageError(
            f"{item.nodeid}: {MEASURED_SECONDS_MARKER} - args={marker.args!r} "
            f"kwargs={marker.kwargs!r}, expected one positive finite number of seconds"
        )
    return float(arguments[0])


def _timeout_mark_seconds(item: pytest.Item, entry_limit: float) -> float | None:
    """The limit of the item's own ``timeout`` mark, or None without one.
    Raises ``pytest.UsageError`` unless it is a positive number no higher than
    ``entry_limit``."""
    marker = item.get_closest_marker(_TIMEOUT_MARKER)
    if marker is None:
        return None
    seconds = marker.kwargs.get("timeout", marker.args[0] if marker.args else None)
    if not _is_positive_seconds(seconds) or seconds > entry_limit:
        raise pytest.UsageError(
            f"{item.nodeid}: {_TIMEOUT_MARKER} mark - seconds={seconds!r}, expected "
            f"a positive number of seconds at most the entry's per-test limit "
            f"{entry_limit:g}; a slower test declares its slowest measured CI time "
            f"with {MEASURED_SECONDS_MARKER}(seconds) instead"
        )
    return float(seconds)


def per_test_limit(item: pytest.Item, entry_limit: float) -> float:
    """The per-test limit of ``item`` in an entry whose limit is
    ``entry_limit`` seconds: ``LIMIT_HEADROOM`` times its
    ``rocke_measured_seconds`` mark when that is higher, else the entry's
    limit, or the lower limit of its own ``pytest.mark.timeout``. Raises
    ``pytest.UsageError`` for a malformed mark, a ``timeout`` mark that does not
    tighten the entry's limit, or a ``timeout`` mark beside a measured time."""
    measured = _measured_seconds(item)
    own_limit = _timeout_mark_seconds(item, entry_limit)
    if measured is not None and own_limit is not None:
        raise pytest.UsageError(
            f"{item.nodeid}: {MEASURED_SECONDS_MARKER} and {_TIMEOUT_MARKER} marks - "
            f"seconds={measured:g}, timeout={own_limit:g}, expected one; keep "
            f"{MEASURED_SECONDS_MARKER}"
        )
    if own_limit is not None:
        return own_limit
    if measured is not None:
        return max(entry_limit, LIMIT_HEADROOM * measured)
    return entry_limit


def pytest_configure(config: pytest.Config) -> None:
    register_measured_seconds_marker(config)
    timeout = config.getoption("timeout", None)
    if not _is_positive_seconds(timeout):
        raise pytest.UsageError(
            f"{PLUGIN_NAME}: --timeout - value={timeout!r}, expected pytest-timeout's "
            f"--timeout with the entry's per-test limit in seconds"
        )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Give each item that declares a measured time above half the entry's
    limit its own limit; pytest-timeout reads the item's closest ``timeout``
    mark when the item runs."""
    entry_limit = float(config.getoption("timeout"))
    for item in items:
        limit = per_test_limit(item, entry_limit)
        if limit > entry_limit:
            item.add_marker(pytest.mark.timeout(limit), append=False)
