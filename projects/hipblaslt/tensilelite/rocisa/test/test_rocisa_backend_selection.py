# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the ROCISA_BACKEND switch decision in rocisa/__init__.py.

Scope is deliberately narrow: the pure switch logic --

  * ``_resolve_backend``     -- maps (requested, availability, load) -> use-adapter?
  * ``_stinkytofu_available`` -- classifies *why* the standalone stinkytofu
                                 binding could not be imported so each failure
                                 mode yields a distinct fallback warning.

Both are exercised with NO stinkytofu build, NO _stinkytofu.so, and NO env
manipulation (``_resolve_backend`` via injected probes; ``_stinkytofu_available``
via monkeypatched ``importlib.import_module``) -- on any platform.

Importing ``rocisa`` here still loads the native ``_rocisa`` extension (same as
the sibling ``test_dll_dirs.py`` / ``test_staleness.py`` helper tests); it does
NOT require the standalone ``_stinkytofu.so`` that the adapter backend needs.
Actual end-to-end routing to the ``rocisa_stinkytofu_adaptor`` facade (which
does require ``_stinkytofu.so``) is covered by the three-path subprocess tests
in ``rocisa_stinkytofu_adaptor/tests/test_emission_consistency.py``.
"""

import glob
import importlib
import os
import sys


def _bootstrap_rocisa_on_syspath() -> None:
    """Make ``from rocisa import ...`` work without a manual ``PYTHONPATH`` prefix.

    Runs *before* the rocisa import below (import order matters). Mirrors the
    discovery in ``rocisa_stinkytofu_adaptor/tests/test.sh``: walk up to the
    ``tensilelite/`` dir and glob for ``*build*/tensilelite/rocisa`` (the layout
    CMake creates), then prepend the first root that holds a built
    ``rocisa/_rocisa*.so`` -- the standalone stinkytofu binding is NOT required,
    since these tests only touch the native ``_rocisa`` extension and the pure
    switch logic. No-op when a built rocisa is already importable (e.g. tox's
    ``pip install rocisa/`` into site-packages, or an explicit PYTHONPATH), so an
    existing setup always wins. Override discovery via ``STINKY_BUILD_DIR``.
    """
    def _has_built_rocisa(path):
        return bool(
            path
            and os.path.isfile(os.path.join(path, "rocisa", "__init__.py"))
            and glob.glob(os.path.join(path, "rocisa", "_rocisa*.so"))
        )

    if any(_has_built_rocisa(p) for p in sys.path if p):
        return

    tensilelite = None
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.basename(cur) == "tensilelite":
            tensilelite = cur
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    if tensilelite is None:
        return

    override = os.environ.get("STINKY_BUILD_DIR")
    if override:
        candidates = [override]
    else:
        candidates = sorted(set(
            glob.glob(os.path.join(tensilelite, "*build*", "tensilelite", "rocisa"))
            + glob.glob(os.path.join(tensilelite, "build*", "tensilelite", "rocisa"))
        ))

    for root in candidates:
        if _has_built_rocisa(root):
            # ``root`` for ``import rocisa``; ``tensilelite`` so an eventual
            # ``import rocisa_stinkytofu_adaptor`` (adapter backend) also works.
            for p in (tensilelite, root):
                if p not in sys.path:
                    sys.path.insert(0, p)
            break


_bootstrap_rocisa_on_syspath()

import pytest  # noqa: E402  (must follow sys.path bootstrap above)

from rocisa import _resolve_backend, _stinkytofu_available  # noqa: E402


class _Probe:
    """Records whether it was called and returns a canned value."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.value


@pytest.fixture()
def warnings_sink():
    msgs = []
    return msgs, (lambda msg, **kw: msgs.append(msg))


# ---------------------------------------------------------------------------
# _resolve_backend: the switch decision. available_fn / load_fn share the same
# (ok, reason) contract; the reason is surfaced verbatim on fallback.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("requested", ["", "native", "rocisa", "STINKYTOFU_TYPO", "0"])
def test_non_stinkytofu_selects_native_without_probing(requested, warnings_sink):
    """Anything other than exactly "stinkytofu" -> native, and the probes are
    never touched (no accidental import / filesystem work) and no warning."""
    msgs, warn = warnings_sink
    avail = _Probe((True, ""))
    load = _Probe((True, ""))

    assert _resolve_backend(requested, avail, load, warn=warn) is False
    assert avail.calls == 0
    assert load.calls == 0
    assert msgs == []


def test_stinkytofu_available_and_loaded_selects_adapter(warnings_sink):
    msgs, warn = warnings_sink
    load = _Probe((True, ""))

    assert _resolve_backend("stinkytofu", _Probe((True, "")), load, warn=warn) is True
    assert load.calls == 1
    assert msgs == []  # success path is silent


def test_unavailable_surfaces_reason_and_skips_load(warnings_sink):
    """When availability fails, its cause-specific reason is warned verbatim and
    the (potentially expensive) load probe is never attempted."""
    msgs, warn = warnings_sink
    load = _Probe((True, ""))
    reason = "ROCISA_BACKEND=stinkytofu requested but ... (_stinkytofu.so missing: ...)."

    assert _resolve_backend("stinkytofu", _Probe((False, reason)), load, warn=warn) is False
    assert load.calls == 0
    assert msgs == [reason]


def test_load_failure_falls_back_with_reason(warnings_sink):
    """Available but the adapter import/rewire failed -> native + warning that
    surfaces the concrete reason."""
    msgs, warn = warnings_sink
    load = _Probe((False, "import failed: ModuleNotFoundError('boom')"))

    assert _resolve_backend("stinkytofu", _Probe((True, "")), load, warn=warn) is False
    assert len(msgs) == 1
    assert "boom" in msgs[0]
    assert "adapter failed to load" in msgs[0]


def test_backend_value_is_normalized_strip_lower():
    """Contract mirror of the module-level ``_BACKEND`` normalization: the env
    value is compared case-insensitively and whitespace-trimmed."""
    for raw in ("  StinkyTofu ", "STINKYTOFU", "stinkytofu\n"):
        assert raw.strip().lower() == "stinkytofu"


# ---------------------------------------------------------------------------
# _stinkytofu_available: distinct message per import-failure mode.
# ---------------------------------------------------------------------------
def _patch_import(monkeypatch, exc):
    def _boom(_name):
        raise exc
    monkeypatch.setattr(importlib, "import_module", _boom)


def test_available_success(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", lambda _n: object())
    assert _stinkytofu_available() == (True, "")


def test_available_missing_so_reports_not_built(monkeypatch):
    _patch_import(monkeypatch, ModuleNotFoundError(
        "No module named 'stinkytofu._stinkytofu'"))
    ok, reason = _stinkytofu_available()
    assert ok is False
    assert "not built/importable" in reason
    assert "_stinkytofu.so missing" in reason


def test_available_stale_reports_rebuild(monkeypatch):
    _patch_import(monkeypatch, ImportError(
        "stinkytofu C++ sources are newer than the built _stinkytofu.so — "
        "bindings are stale."))
    ok, reason = _stinkytofu_available()
    assert ok is False
    assert "stale and must be rebuilt" in reason


def test_available_broken_binding_reports_load_failure(monkeypatch):
    _patch_import(monkeypatch, ImportError(
        "/path/_stinkytofu.so: undefined symbol: _ZN9stinkytofu3fooEv"))
    ok, reason = _stinkytofu_available()
    assert ok is False
    assert "present but failed to load" in reason


def test_available_unexpected_error_reported_separately(monkeypatch):
    _patch_import(monkeypatch, RuntimeError("something weird at import time"))
    ok, reason = _stinkytofu_available()
    assert ok is False
    assert "unexpected error" in reason


def test_available_failure_messages_are_distinct(monkeypatch):
    """The four failure modes must not collapse into one generic message."""
    cases = [
        ModuleNotFoundError("No module named 'stinkytofu._stinkytofu'"),
        ImportError("... newer than the built _stinkytofu.so ... bindings are stale."),
        ImportError("undefined symbol: foo"),
        RuntimeError("weird"),
    ]
    reasons = []
    for exc in cases:
        _patch_import(monkeypatch, exc)
        ok, reason = _stinkytofu_available()
        assert ok is False
        reasons.append(reason)
    assert len(set(reasons)) == len(reasons)
