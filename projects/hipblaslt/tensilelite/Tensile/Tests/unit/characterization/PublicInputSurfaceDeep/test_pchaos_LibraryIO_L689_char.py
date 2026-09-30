################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurfaceDeep characterization: ``if CU is None`` at
``Tensile/LibraryIO.py:689`` inside ``getCUCount``.

Branch d2f6f0df95db1dd2e280fe5b92edb75f8acc175d.

The predicate is a null-check on the value returned by
``os.environ.get("CU", None)``.  The single public input is the
environment variable ``CU``.

  * TRUE branch  (``CU is None``)  : env var ``CU`` is absent/unset.
    ``getCUCount`` enters the ``try`` block and runs a ``rocminfo`` subprocess
    to obtain the CU count from hardware.

  * FALSE branch (``CU is not None``): env var ``CU`` is set to a numeric
    string.  ``getCUCount`` skips the gpu-probe and uses the env value directly,
    returning ``int(CU)``.

Domain: {None (absent), "64" (set)}.  Tests pin ACTUAL observed behavior via
monkeypatching ``os.environ``.  CPU-only; no GPU hardware required for the
FALSE branch.  The TRUE branch cannot execute the full rocminfo probe without
hardware, so it is tested via the pure-helper predicate only.
"""

import os

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure-helper: mirror the predicate at LibraryIO.py:689
# ---------------------------------------------------------------------------


def _cu_is_none(env: dict) -> bool:
    """Mirror the predicate ``if CU is None`` at LibraryIO.py:689.

    Reads ``CU`` from *env* exactly as getCUCount does:
        CU = env.get("CU", None)
    Returns True when CU is None (env var absent), False otherwise.
    """
    CU = env.get("CU", None)
    return CU is None


# ---------------------------------------------------------------------------
# Pure-helper tests (no I/O, deterministic)
# ---------------------------------------------------------------------------


def test_cu_absent_predicate_is_true():
    """CU absent from env -> CU is None -> predicate True (TRUE branch L689)."""
    assert _cu_is_none({}) is True


def test_cu_set_to_64_predicate_is_false():
    """CU='64' in env -> CU is not None -> predicate False (FALSE branch L689)."""
    assert _cu_is_none({"CU": "64"}) is False


def test_cu_set_to_zero_string_predicate_is_false():
    """CU='0' in env (edge: falsy string) -> CU is not None -> predicate False."""
    assert _cu_is_none({"CU": "0"}) is False


def test_cu_set_to_any_numeric_string_predicate_is_false():
    """CU='128' in env -> predicate False regardless of numeric value."""
    assert _cu_is_none({"CU": "128"}) is False


# ---------------------------------------------------------------------------
# Integration pin: FALSE branch — getCUCount uses env var directly
# ---------------------------------------------------------------------------


def _import_getCUCount():
    """Import getCUCount from Tensile.LibraryIO."""
    import importlib
    M = importlib.import_module("Tensile.LibraryIO")
    return M.getCUCount


def test_false_branch_returns_env_cu_as_int(monkeypatch):
    """FALSE branch (CU='64' set): getCUCount returns int(64) without gpu probe.

    When CU is set, getCUCount skips the rocminfo subprocess entirely and
    returns int(CU).  No GPU hardware is needed for this path.
    """
    monkeypatch.setenv("CU", "64")
    getCUCount = _import_getCUCount()
    result = getCUCount()
    assert result == 64
    assert isinstance(result, int)


def test_false_branch_different_cu_value(monkeypatch):
    """FALSE branch (CU='128' set): getCUCount returns 128 without gpu probe."""
    monkeypatch.setenv("CU", "128")
    getCUCount = _import_getCUCount()
    result = getCUCount()
    assert result == 128


def test_false_branch_cu_string_coerced_to_int(monkeypatch):
    """FALSE branch: env CU value is coerced to int via int(CU) at line 704."""
    monkeypatch.setenv("CU", "60")
    getCUCount = _import_getCUCount()
    result = getCUCount()
    assert result == 60
    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# TRUE branch guard: CU absent from env — predicate fires; no GPU probe here
# (We cannot run the full subprocess path without GPU hardware, so we pin the
# predicate evaluation via the pure helper and verify that getCUCount's final
# guard at line 701 raises SystemExit when rocminfo is unavailable.)
# ---------------------------------------------------------------------------


def test_true_branch_predicate_pure(monkeypatch):
    """TRUE branch: CU absent -> _cu_is_none({}) is True (predicate fires)."""
    assert _cu_is_none({}) is True


def test_true_branch_no_cu_no_gpu_raises(monkeypatch):
    """TRUE branch (CU unset, rocminfo returns empty): getCUCount raises SystemExit.

    When CU is absent, getCUCount calls rocminfo.  We stub subprocess.run to
    return empty stdout (simulating a CPU-only environment), so the post-try
    guard at line 701 fires and printExit raises SystemExit.
    """
    import subprocess

    class _FakeResult:
        stdout = b""

    monkeypatch.delenv("CU", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeResult())
    getCUCount = _import_getCUCount()
    with pytest.raises(SystemExit):
        getCUCount()
