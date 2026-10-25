# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""pytest plugin that turns a skipped test into a failure, so a build that should run a suite
cannot pass by silently not running part of it.

ctest reports a pytest run as passed when every case skips, and a test that needs a compiler,
a built binary or an installed package skips without a word when it is missing. The hipDNN
build runs the DescriptorGenerator and IngestorGenerator suites with this plugin
(``-p hipdnn_strict_skips``, see tools/CMakeLists.txt). Run by hand it is not loaded, so those
tests skip as usual.

A skip that is correct by design (an author-supplied input, a platform that cannot run the
test) opts out with ``@pytest.mark.allow_skip(reason="...")`` on the test or its class; the
reason is required. A skip raised while a module is imported (``pytest.importorskip(...)``,
``pytest.skip(allow_module_level=True)``) has no test to carry the marker, so it fails the
collection.
"""

import pytest

MARKER = "allow_skip"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"{MARKER}(reason): this test may skip under hipdnn_strict_skips, for the stated "
        "reason (an author-supplied input, or a platform that cannot run it).",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    # xfail also reports as skipped, with `wasxfail` set; a strict run has no use for it.
    if not report.skipped:
        return

    if isinstance(report.longrepr, tuple):
        reason = report.longrepr[2].removeprefix("Skipped: ")
    else:
        reason = getattr(report, "wasxfail", None) or str(report.longrepr)

    marker = item.get_closest_marker(MARKER)
    if marker is not None:
        if marker.kwargs.get("reason"):
            return
        report.outcome = "failed"
        report.longrepr = (
            f"@pytest.mark.{MARKER} needs a reason: "
            f'@pytest.mark.{MARKER}(reason="...") (the test skipped: {reason})'
        )
        return

    report.outcome = "failed"
    report.longrepr = (
        f"skipped in a strict run: {reason}\n"
        "Provide the missing prerequisite. If the skip is by design, mark the test "
        f'@pytest.mark.{MARKER}(reason="...").'
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    report = outcome.get_result()
    if not report.skipped:
        return

    reason = (
        report.longrepr[2].removeprefix("Skipped: ")
        if isinstance(report.longrepr, tuple)
        else str(report.longrepr)
    )
    report.outcome = "failed"
    report.longrepr = (
        f"skipped while collecting {collector.nodeid} in a strict run: {reason}\n"
        "Provide the missing prerequisite. A module-level skip cannot be marked "
        f"@pytest.mark.{MARKER}; skip inside the tests that need it instead."
    )
