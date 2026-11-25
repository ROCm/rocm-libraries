# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Pytest root config for the rocKE engine test tree. Puts the Python engine
# package root (rocke/platform/python) on sys.path so `import rocke` resolves
# without an external PYTHONPATH. Paths are derived from this file's location
# (relative), so the tree stays copy-able verbatim into another repo.
#
# source tree: this dir -> rocke/platform/tests, so parent -> rocke/platform

import sys
import warnings
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROCKE = _HERE.parent  # tests -> rocke/platform
_PYROOT = _ROCKE / "python"
if str(_PYROOT) not in sys.path:
    sys.path.insert(0, str(_PYROOT))

# A session started from a git hook inherits the hook's GIT_DIR and
# GIT_INDEX_FILE, which would point every git command a test runs, even in a
# scratch directory, at the repository being committed to.
from rocke.tests.git_environment import drop_inherited_git_repository_variables

drop_inherited_git_repository_variables()


def pytest_addoption(parser):
    # tools/test_pending_lanes.py: a scheduled run fails on any expired entry;
    # a pull-request run fails only on the ones its change touches.
    parser.addoption(
        "--strict-expiry",
        action="store_true",
        help="fail on every expired rocke.tests.pending_lanes entry",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Report a ``ROCKE_BACKEND=both`` coverage gap as a skip, never a pass.

    Under ``both`` the dispatcher refuses to substitute the Python result for a
    kernel the C++ engine could not lower, so an arch with no C++ ISA backend
    (see ``backend.CPP_UNPORTED_ARCHES``) raises ``BackendCoverageGap``. Turning
    it into a skip here keeps the gap *counted and named* in the run summary.
    Letting it pass -- which is what returning the Python IR would do -- would
    make the differential lane green on kernels it never compared.

    Only ``BackendCoverageGap`` is remapped. A ``BackendMismatch`` (the engines
    disagreed) and any other engine error stay failures.

    This rewrites the report rather than the raised exception because most of
    the suite is ``unittest.TestCase``: pytest's unittest integration captures
    the error inside ``runtest()`` and only republishes it onto ``call.excinfo``
    from its own (tryfirst, non-wrapper) makereport hook, so a
    ``pytest_runtest_call`` wrapper sees no exception at all and silently does
    nothing. Wrapping makereport is the one place that sees both styles.
    """
    outcome = yield
    if call.excinfo is None:
        return
    from rocke.core.backend import BackendCoverageGap

    if not isinstance(call.excinfo.value, BackendCoverageGap):
        return
    report = outcome.get_result()
    report.outcome = "skipped"
    report.longrepr = (str(item.path), item.location[1] + 1, str(call.excinfo.value))


def pytest_collection_modifyitems(config, items):
    """Skip the quarantined tests (``rocke.tests.quarantine``)."""
    from rocke.tests.quarantine import skip_quarantined_items

    skip_quarantined_items(items, _HERE, "platform/tests")


def pytest_collection_finish(session):
    """Before any test runs, call ``prepare_before_tests(items)`` on each
    collected test module that defines it, with that module's collected items.

    A module uses it for one-time work that would otherwise land inside the
    first test that needs it and count toward that test's time limit, such as
    a compile that takes a large share of the limit on a slow host (the library
    tree's prebuild in library/tests/conftest.py is the same idea for its
    kernels). The module keeps any error for the tests that need its result; an
    error that escapes is reported as a warning, never as a session error, so no
    unrelated test is affected. A ``--collect-only`` run prepares nothing. Under
    pytest-xdist each worker prepares for itself.
    """
    if session.config.option.collectonly:
        return
    by_module = {}
    for item in session.items:
        module = getattr(item, "module", None)
        if module is not None and hasattr(module, "prepare_before_tests"):
            by_module.setdefault(module, []).append(item)
    for module, items in by_module.items():
        try:
            module.prepare_before_tests(items)
        except Exception as error:  # noqa: BLE001 - must not fail unrelated tests
            warnings.warn(
                f"{module.__name__}.prepare_before_tests failed, so its tests do "
                f"that work themselves: {error!r}",
                stacklevel=1,
            )


# The IR parity harness' attention families are the one sanctioned platform ->
# library reach, so `kernels`/`builders` must resolve here as well. Probed
# against both layouts this tree runs in: staged under tests/library/ in an
# install (the destination TheRock's test-artifact globs capture), or the sibling
# library tree in a checkout.
for _lib_root in (_HERE / "library", _HERE.parents[1] / "library"):
    if (_lib_root / "kernels").is_dir():
        if str(_lib_root) not in sys.path:
            sys.path.insert(0, str(_lib_root))
        break

# An install has no platform root: it stages the IR-validity tool (which
# rocke.tests.ir_validity runs) under tests/arch_domain, the instance tests and
# their IR parity harness under tests/instances, and the parity-emitter tree
# under tests/instances/parity. Point the rocke.assets locators there. The
# IR-validity probe keys on the directory, so an install that lost the tool
# fails rather than skips.
from rocke.tests.staged_payload import point_locators_at_staged_payload

point_locators_at_staged_payload(_HERE)


def pytest_configure(config):
    """Apply the test session's garbage collector setting
    (``rocke.tests.session_gc``), declare the shard-cost and run-alone marks
    for runs that do not load the ``rocke.tests.ctest_shard`` and
    ``rocke.tests.ctest_workers`` plugins, and fail a sweep whose
    subtests all skipped (``rocke.tests.sweep_guard``)."""
    from rocke.tests.ctest_shard import register_shard_cost_marker
    from rocke.tests.ctest_workers import register_run_alone_marker
    from rocke.tests.session_gc import use_test_session_gc
    from rocke.tests.sweep_guard import register_sweep_guard

    use_test_session_gc()
    register_shard_cost_marker(config)
    register_run_alone_marker(config)
    register_sweep_guard(config)
