# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The backend lane helper fails when a lane's caller sees the wrong refusal.

``rocke.tests.engines.assert_backend_lane_rejects`` checks one refusal of
``lower_kernel_to_llvm`` in one row of the backend matrix. Its own checks are
only worth something if each one fails on the outcome it exists to reject, so
each test below gives it a real refusal and an expectation that one check must
catch, and matches that check's own failure text.

The refusal is the dispatcher's own: lowering without an ``arch`` raises
``ValueError`` before any engine runs, in every lane. That gives a refusal with
the right type and message but no fallback, which a fallback lane must reject.
"""

from __future__ import annotations

import re

import pytest
from rocke.core.ir import IRBuilder, KernelDef
from rocke.tests.engines import BACKEND_LANES, assert_backend_lane_rejects

_NO_ARCH_MESSAGE = (
    "lowering requires an explicit gfx target; pass arch=... (there is no default)"
)


def _empty_kernel() -> KernelDef:
    return IRBuilder("lane_helper_probe").kernel


def _assert_no_arch_refusal(
    monkeypatch: pytest.MonkeyPatch,
    lane: str,
    *,
    error_type: type[Exception] = ValueError,
    message: str = _NO_ARCH_MESSAGE,
) -> None:
    """Run the helper on a lowering without an ``arch``, expecting
    ``error_type`` with ``message``."""
    assert_backend_lane_rejects(
        monkeypatch,
        lane,
        _empty_kernel(),
        None,  # type: ignore[arg-type] -- the refusal under test
        error_type=error_type,
        message=message,
        llvm_flavor="llvm20",
    )


def test_matching_refusal_passes_in_the_python_lane(monkeypatch):
    _assert_no_arch_refusal(monkeypatch, "python")


def test_unknown_lane_is_refused(monkeypatch):
    with pytest.raises(ValueError) as error:
        _assert_no_arch_refusal(monkeypatch, "cpp")
    assert str(error.value) == f"lane='cpp', expected one of {BACKEND_LANES}"


def test_error_type_without_a_status_is_refused(monkeypatch):
    with pytest.raises(ValueError) as error:
        _assert_no_arch_refusal(monkeypatch, "python", error_type=RuntimeError)
    assert str(error.value) == (
        "error_type=RuntimeError, expected one of "
        "['KeyError', 'NotImplementedError', 'TypeError', 'ValueError']"
    )


def test_wrong_error_type_fails(monkeypatch):
    # Same message, other type: the exact type check fails, naming the
    # exception the caller saw.
    seen = re.escape(repr(ValueError(_NO_ARCH_MESSAGE)))
    with pytest.raises(AssertionError, match=f"^{seen}"):
        _assert_no_arch_refusal(monkeypatch, "python", error_type=TypeError)


def test_wrong_message_fails(monkeypatch):
    # Right type, other message: the message check fails, quoting the
    # message the caller saw.
    with pytest.raises(AssertionError, match=f"^{re.escape(_NO_ARCH_MESSAGE)}"):
        _assert_no_arch_refusal(monkeypatch, "python", message="another refusal")


@pytest.mark.parametrize("lane", ["installed-fallback", "cpp-permissive"])
def test_refusal_without_a_fallback_fails_in_a_fallback_lane(monkeypatch, lane):
    # The refusal matches in type and message; the fallback check fails,
    # showing the empty fallback record. cpp-permissive skips without the
    # rocke_engine binding.
    with pytest.raises(AssertionError, match=r"^\[\]($|\n)"):
        _assert_no_arch_refusal(monkeypatch, lane)
