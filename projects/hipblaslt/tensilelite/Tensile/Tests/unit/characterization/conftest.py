################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Characterization-suite shared configuration.

Adds the ``_codegen`` helper directory to ``sys.path`` so any characterization
suite can ``from codegen_harness import ...`` (the CPU-only assembly-emit
harness used by the codegen coverage suites), and exposes the cached
toolchain / cap-map / data-dir as session fixtures.

This file only *adds* an import path and read-only fixtures; it changes no
existing behavior.
"""

import os
import sys

import pytest

_CODEGEN_DIR = os.path.join(os.path.dirname(__file__), "_codegen")
if _CODEGEN_DIR not in sys.path:
    sys.path.insert(0, _CODEGEN_DIR)


def _foreign_path_instantiable() -> bool:
    """Return True if a Windows-flavored ``pathlib.Path`` can be built here.

    On POSIX, CPython refuses to instantiate ``WindowsPath`` (``NotImplementedError``
    through 3.12, the ``NotImplementedError`` subclass ``UnsupportedOperation`` in
    3.13+).  Characterization tests that flip ``os.name`` to ``"nt"`` and let
    production code construct a concrete ``pathlib.Path`` therefore raise inside the
    function under test *and* inside pytest's own failure reporting, which builds
    ``Path(os.getcwd())`` while ``os.name`` is still ``"nt"`` -- taking down the
    whole xdist worker with an INTERNALERROR.  Detect the capability directly
    rather than sniffing the interpreter version.
    """
    from pathlib import Path

    saved = os.name
    try:
        os.name = "nt"
        Path("probe")
        return True
    except NotImplementedError:
        return False
    finally:
        os.name = saved


def _snapshot_plugin_available() -> bool:
    """Return True if syrupy's ``snapshot`` fixture is available here.

    The characterization suite asserts against syrupy snapshots via the
    ``snapshot`` fixture.  The tox ``unit``/``coverage-unit`` environments install
    syrupy, but the installed-artifact test tree (``share/hipblaslt/tensilelite``,
    run by TheRock) executes against a leaner interpreter that does not ship
    syrupy and does not receive this package's ``pyproject``/``tox`` dependency
    declarations.  There every ``snapshot``-using test errors at setup with
    ``fixture 'snapshot' not found``.  Detect the capability directly so the suite
    is cleanly skipped where the plugin is absent rather than erroring the run.
    """
    import importlib.util

    return importlib.util.find_spec("syrupy") is not None


def pytest_collection_modifyitems(config, items):
    """Skip capability-gated characterization tests where the env can't run them.

    Two independent gates:

    * ``nt_path_simulation`` tests flip ``os.name`` to ``"nt"`` and exercise
      production code that constructs a ``pathlib.Path``.  On any POSIX
      interpreter this cannot succeed, so they run only where a Windows-flavored
      ``Path`` is instantiable.
    * ``snapshot``-using tests require the syrupy plugin, which the
      installed-artifact test environment does not provide; they run only where
      syrupy is importable.
    """
    skip_nt = None
    if not _foreign_path_instantiable():
        skip_nt = pytest.mark.skip(
            reason="WindowsPath not instantiable on this interpreter; nt-path simulation skipped"
        )

    skip_snapshot = None
    if not _snapshot_plugin_available():
        skip_snapshot = pytest.mark.skip(
            reason="syrupy not installed; snapshot characterization tests skipped"
        )

    if skip_nt is None and skip_snapshot is None:
        return

    for item in items:
        if skip_nt is not None and "nt_path_simulation" in item.keywords:
            item.add_marker(skip_nt)
        if skip_snapshot is not None and "snapshot" in getattr(item, "fixturenames", ()):
            item.add_marker(skip_snapshot)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Tag a failed snapshot test so the terminal summary can route the author to
    the golden-update policy.

    Keys off the real ``snapshot`` fixture usage (not failure-text heuristics) and
    records the tag on ``report.user_properties`` so it survives xdist worker ->
    controller serialization (a plain module global would be lost under ``-n``).
    """
    outcome = yield
    report = outcome.get_result()
    if (
        report.when == "call"
        and report.failed
        and "snapshot" in getattr(item, "fixturenames", ())
    ):
        report.user_properties.append(("char_snapshot_failure", True))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """When a characterization snapshot test fails, print the snapshot-update
    policy so a failing author never reaches for a blanket ``--snapshot-update``.

    Driven by the ``char_snapshot_failure`` tag set in ``pytest_runtest_makereport``,
    which gates on ``report.when == "call"`` -- so this fires on a call-phase failure
    of a ``snapshot``-fixture test (usually a golden mismatch, occasionally an
    unrelated error in the test body; the guidance is harmless in the latter case).
    Setup-/teardown-phase errors are not tagged and do not trigger this banner.
    """
    failed = terminalreporter.stats.get("failed", [])
    nodeids = [
        rep.nodeid
        for rep in failed
        if any(k == "char_snapshot_failure" for k, _ in getattr(rep, "user_properties", []))
    ]
    if not nodeids:
        return

    bar = "=" * 78
    write = terminalreporter.write_line
    write("")
    write(bar)
    write("Characterization snapshot test failed (likely a golden mismatch)")
    write("-" * 78)
    write("These goldens pin TensileLite's current behavior as a refactor safety net.")
    write("")
    write("If this behavior change was NOT intended, you found a regression -- fix")
    write("your code; do NOT touch the .ambr files.")
    write("")
    write("If it WAS intended, update ONLY the affected node(s) and review the diff:")
    for nodeid in nodeids[:10]:
        write("    pytest '%s' --snapshot-update" % nodeid)
    if len(nodeids) > 10:
        write("    ... and %d more" % (len(nodeids) - 10))
    write("")
    write("NEVER run a bare, suite-wide 'pytest --snapshot-update' -- it silently")
    write("rewrites every golden and destroys the net.")
    write("")
    write(
        "Policy (single source of truth): "
        "Tensile/Tests/unit/characterization/README.md "
        "-- 'Snapshot / golden discipline (governance)'"
    )
    write(bar)


@pytest.fixture(scope="session")
def cg_assembler():
    """The CPU-only assembler (amdclang++); shared across codegen suites."""
    from codegen_harness import get_assembler

    return get_assembler()


@pytest.fixture(scope="session")
def cg_isa_info_map():
    """The ISA capability map; shared across codegen suites."""
    from codegen_harness import get_isa_info_map

    return get_isa_info_map()
