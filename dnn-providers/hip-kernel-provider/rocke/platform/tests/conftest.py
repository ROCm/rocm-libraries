# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Pytest root config for the rocKE engine test tree. Puts the Python engine
# package root (rocke/platform/python) on sys.path so `import rocke` resolves
# without an external PYTHONPATH. Paths are derived from this file's location
# (relative), so the tree stays copy-able verbatim into another repo.
#
# source tree: this dir -> rocke/platform/tests, so parent -> rocke/platform

import re
import subprocess
import sys
from collections.abc import Callable, Iterable
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


@pytest.fixture
def provider_ctest_labels(tmp_path) -> Callable[[Iterable[str]], dict[str, set[str]]]:
    """Return a function that labels CTest names the way the provider build does.

    The provider's ``ROCKE_ENGINE_test_categories_external.yaml`` sits beside
    ``rocke/`` and the monorepo's shared parser applies it to the installed
    CTest file. The returned function feeds that parser one ``add_test`` line
    per name and maps each name to the labels it would get; a name no category
    matches maps to an empty set. A standalone rocKE tree has neither file, so
    the requesting test is skipped there.
    """
    provider = _ROCKE.parents[1]
    categories = provider / "ROCKE_ENGINE_test_categories_external.yaml"
    parser = provider.parents[1] / "shared/ctest/parse_ctest_categories.py"
    for path in (categories, parser):
        if not path.is_file():
            pytest.skip(f"no provider CTest categories to check: {path} does not exist")
    pytest.importorskip("yaml", reason="the shared category parser needs PyYAML")

    def labels_of(names: Iterable[str]) -> dict[str, set[str]]:
        ordered_names = list(names)
        # A subdirectory, so the requesting test's own tmp_path files are untouched.
        ctests = tmp_path / "provider_ctest_labels" / "CTestTestfile.cmake"
        ctests.parent.mkdir(exist_ok=True)
        ctests.write_text(
            "".join(f"add_test({name} python)\n" for name in ordered_names)
        )
        result = subprocess.run(
            [sys.executable, str(parser), str(categories), str(ctests)],
            check=True,
            capture_output=True,
            text=True,
        )
        labelled = {
            name: set(value.split(";"))
            for name, value in re.findall(
                r'set_tests_properties\("([^"]+)" PROPERTIES LABELS "([^"]*)"\)',
                result.stdout,
            )
        }
        return {name: labelled.get(name, set()) for name in ordered_names}

    return labels_of


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
