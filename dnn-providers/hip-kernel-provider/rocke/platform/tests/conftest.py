# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Pytest root config for the rocKE engine test tree. Puts the Python engine
# package root (rocke/platform/python) on sys.path so `import rocke` resolves
# without an external PYTHONPATH. Paths are derived from this file's location
# (relative), so the tree stays copy-able verbatim into another repo.
#
# source tree: this dir -> rocke/platform/tests, so parent -> rocke/platform

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROCKE = _HERE.parent  # tests -> rocke/platform
_PYROOT = _ROCKE / "python"
if str(_PYROOT) not in sys.path:
    sys.path.insert(0, str(_PYROOT))


def pytest_configure(config):
    """Initialise torch's HIP runtime before anything initialises rocKE's.

    rocKE binds the HIP runtime torch uses only when torch is already imported
    (``rocke.runtime.runtime_coexistence``); otherwise it loads a ROCm
    install's ``libamdhip64``. Modules of this tree probe the device through
    rocKE while they are collected (``get_device_arch(0)`` for a skip mark),
    long before a later module imports torch. That initialises the install's
    HSA runtime first, and torch's own HSA runtime, initialised after it in the
    same process, then finds no GPU: every torch device test fails with "no
    ROCm-capable device is detected" (a second HSA runtime initialised AFTER
    torch's does see the device). Initialising torch here, before collection,
    gives the whole session the single runtime a torch process has. Without
    torch installed nothing changes; a torch that fails to import or to
    initialise is left to the tests that need it.
    """
    import importlib.util

    try:
        if importlib.util.find_spec("torch") is None:
            return
        import torch

        torch.cuda.is_available()
    except Exception:  # noqa: BLE001 - see the docstring
        pass


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


@pytest.fixture(autouse=True)
def _gemm_selector_layer(request, monkeypatch):
    """Pin the RCR GEMM dispatcher to its rule layer unless a module opts in.

    ``dispatch_gemm_fp16`` / ``_bf16`` consult the learned selector
    (``rocke.dispatch.gemm.selector``, ``ROCKE_GEMM_SELECTOR``) after the
    registry's rules. Most dispatch and instance tests assert the RULE layer's
    decisions, so they run with the selector off; the selector's own
    tests set ``GEMM_SELECTOR_TESTS = True`` at module level and get the
    shipped default (a string value, e.g. ``"1"``, is exported as
    ``ROCKE_GEMM_SELECTOR``). ``ROCKE_GEMM_SELECTOR_TESTS=1`` in the environment keeps
    the selector on everywhere (to see which rule assertions it would change).
    """
    import os

    if os.environ.get("ROCKE_GEMM_SELECTOR_TESTS") == "1":
        yield
        return
    # the learned gemm_skinny configuration choice
    # (dispatch.gemm.skinny_select) is pinned off the same way unless a module
    # sets GEMM_SKINNY_SELECT_TESTS = True (the rule layer's table plans)
    if getattr(request.module, "GEMM_SKINNY_SELECT_TESTS", False) is True:
        monkeypatch.delenv("ROCKE_GEMM_SKINNY_SELECT", raising=False)
    else:
        monkeypatch.setenv("ROCKE_GEMM_SKINNY_SELECT", "0")
    mode = getattr(request.module, "GEMM_SELECTOR_TESTS", False)
    if mode is True:
        monkeypatch.delenv("ROCKE_GEMM_SELECTOR", raising=False)
    elif isinstance(mode, str):  # e.g. "1": on for every arch with a model
        monkeypatch.setenv("ROCKE_GEMM_SELECTOR", mode)
    else:
        monkeypatch.setenv("ROCKE_GEMM_SELECTOR", "0")
    yield


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
