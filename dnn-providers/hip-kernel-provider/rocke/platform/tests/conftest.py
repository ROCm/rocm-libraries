# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Pytest root config for the rocKE engine test tree. Puts the Python engine
# package root (rocke/platform/python) on sys.path so `import rocke` resolves
# without an external PYTHONPATH. Paths are derived from this file's location
# (relative), so the tree stays copy-able verbatim into another repo.
#
# source tree: this dir -> rocke/platform/tests, so parent -> rocke/platform

import os
import sys
import warnings
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROCKE = _HERE.parent  # tests -> rocke/platform
_PYROOT = _ROCKE / "python"
if str(_PYROOT) not in sys.path:
    sys.path.insert(0, str(_PYROOT))


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

# rocke.tests.ir_validity compiles lowered IR with the platform's IR-validity
# tool, found through rocke.assets.platform_tools_dir() (tools/ in a checkout).
# An install has no tools/ tree: it stages a copy of the tool and the modules it
# imports under tests/arch_domain, so point the locator there unless the caller
# already chose a directory. The probe keys on the directory, so an install that
# lost the tool fails rather than skips.
_STAGED_TOOLS = _HERE / "arch_domain"
if _STAGED_TOOLS.is_dir():
    os.environ.setdefault("ROCKE_PLATFORM_TOOLS_DIR", str(_STAGED_TOOLS))
