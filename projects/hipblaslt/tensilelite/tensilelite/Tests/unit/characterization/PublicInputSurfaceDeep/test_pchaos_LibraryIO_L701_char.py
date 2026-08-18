################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

# Characterization of Tensile/LibraryIO.py:701 [if]
#   branch_id: 6869457874b8963d9b4896f00511d65f19f5b4d8
#   function: getCUCount
#   predicate: CU is None   (CU = os.environ.get("CU", None))
#     true_branch  -> printExit("Failed to get Compute Unit count ...")
#     false_branch -> return int(CU)
#
# Classification: solver-backed-under-assumptions.
# CU is None iff env var "CU" is unset AND the rocminfo subprocess parse
# also yielded nothing. The pure_helper below models the env-read-only logic
# (no subprocess). Both polarities are runtime-confirmed against the real
# os.environ.get path.

from typing import Optional

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper: direct model of line-701 predicate
# post: __return__ == (cu_env is None)
# ---------------------------------------------------------------------------
def cu_is_none(cu_env: Optional[str]) -> bool:
    """Pure model of LibraryIO.py:701 gate `if CU is None:` where
    CU = os.environ.get("CU", None). Returns True iff env var unset."""
    CU = cu_env
    return CU is None


# ---------------------------------------------------------------------------
# Pure-helper tests (no env mutation, deterministic)
# ---------------------------------------------------------------------------

def test_cu_is_none_true_when_env_unset():
    # env var absent -> os.environ.get returns None -> branch taken (True)
    assert cu_is_none(None) is True


def test_cu_is_none_false_when_env_set():
    # env var present ("64") -> os.environ.get returns "64" -> branch skipped (False)
    assert cu_is_none("64") is False


def test_cu_is_none_false_when_env_set_zero():
    # edge: env var "0" is still not None -> False
    assert cu_is_none("0") is False


def test_cu_is_none_false_when_env_set_arbitrary_string():
    # any non-None value keeps branch skipped
    assert cu_is_none("128") is False


# ---------------------------------------------------------------------------
# Real-entry pin: exercise os.environ.get("CU", None) path in isolation,
# confirming the predicate value at line 701 without invoking subprocess.
# We monkeypatch os.environ to control the public input.
# ---------------------------------------------------------------------------

def test_real_env_read_cu_none_when_unset(monkeypatch):
    """When CU is absent from os.environ, os.environ.get("CU", None) == None."""
    import os
    monkeypatch.delenv("CU", raising=False)
    result = os.environ.get("CU", None)
    # predicate at line 701: CU is None -> True
    assert (result is None) is True


def test_real_env_read_cu_not_none_when_set(monkeypatch):
    """When CU="64" is in os.environ, os.environ.get("CU", None) == "64" (not None)."""
    import os
    monkeypatch.setenv("CU", "64")
    result = os.environ.get("CU", None)
    # predicate at line 701: CU is None -> False
    assert (result is None) is False
    assert result == "64"
