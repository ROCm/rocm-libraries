# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The backend lane helper fails when a lane's caller sees the wrong refusal.

``rocke.tests.engines.assert_backend_lane_rejects`` checks one refusal of
``lower_kernel_to_llvm`` in one row of the backend matrix. Its own checks are
only worth something if each one fails on the outcome it exists to reject, so
each test below gives it a real refusal and an expectation that one check must
catch, and matches that check's own failure text.

Two real refusals are used. The dispatcher's own: lowering without an
``arch`` raises ``ValueError`` before any engine runs, in every lane. That
gives a refusal with the right type and message but no fallback, which a
fallback lane must reject, and a Python exception where the strict C++ lane
must see the C++ engine's. And the engines' own: an LDS vector store whose
``align`` is 24, which each engine refuses with the same message, so the
cpp-permissive lane falls back after the C++ engine refuses it.

A fallback in a lane that must not fall back is the one outcome no real
dispatcher produces, so that test alone stands in a dispatcher that records
one (``_dispatcher_that``) in the helper's module.
"""

from __future__ import annotations

import os
import re

import pytest
from rocke.core.ir import F32, IRBuilder, KernelDef, PtrType
from rocke.tests import engines
from rocke.tests.engines import BACKEND_LANES, assert_backend_lane_rejects

_NO_ARCH_MESSAGE = (
    "lowering requires an explicit gfx target; pass arch=... (there is no default)"
)
_STORE_ALIGN_MESSAGE = (
    "tile.smem_store_vN failed - align=24, expected a positive power of two"
)
_QUIET_FALLBACK = "ROCKE_CPP_QUIET_FALLBACK"


def _empty_kernel() -> KernelDef:
    return IRBuilder("lane_helper_probe").kernel


def _store_align_24_kernel() -> KernelDef:
    """A <4 x float> LDS store whose ``align`` attribute is 24: both engines
    refuse it with ``_STORE_ALIGN_MESSAGE``."""
    builder = IRBuilder("lane_helper_store_align")
    out = builder.param("out", PtrType(F32, "global"))
    zero = builder.const_i32(0)
    buffer = builder.smem_alloc(F32, [4], name_hint="f")
    vector = builder.global_load_vN(out, zero, F32, 4, align=16)
    builder.smem_store_vN(buffer, [zero], vector, 4)
    builder.ret()
    [store] = [op for op in builder.kernel.body.ops if op.name == "tile.smem_store_vN"]
    store.attrs["align"] = 24
    return builder.kernel


def _assert_store_align_refusal(
    monkeypatch: pytest.MonkeyPatch, lane: str, *, message: str = _STORE_ALIGN_MESSAGE
) -> None:
    """Run the helper on the store-align refusal, expecting ``ValueError`` with
    ``message``."""
    assert_backend_lane_rejects(
        monkeypatch,
        lane,
        _store_align_24_kernel(),
        "gfx942",
        error_type=ValueError,
        message=message,
        llvm_flavor="llvm20",
    )


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


def _dispatcher_that(
    monkeypatch: pytest.MonkeyPatch, *, refusal: Exception, fallback_reason: str
) -> None:
    """Replace the dispatcher the helper calls with one that records
    ``fallback_reason`` as a fallback and then raises ``refusal``."""
    fallbacks: list[tuple[str, str]] = []

    def lower(kernel: KernelDef, *, arch: str, llvm_flavor: str | None) -> str:
        fallbacks.append((kernel.name, fallback_reason))
        raise refusal

    monkeypatch.setattr(engines, "lower_kernel_to_llvm", lower)
    monkeypatch.setattr(engines, "cpp_fallbacks", lambda: list(fallbacks))


def test_a_python_exception_fails_in_the_strict_lane(monkeypatch):
    # The strict C++ lane must surface the C++ engine's RuntimeError; the
    # dispatcher's own ValueError fails the type check, naming it. Skips
    # without the rocke_engine binding.
    seen = re.escape(repr(ValueError(_NO_ARCH_MESSAGE)))
    with pytest.raises(AssertionError, match=f"^{seen}"):
        _assert_no_arch_refusal(monkeypatch, "cpp-strict")


def test_a_cpp_refusal_with_another_message_fails_in_the_strict_lane(monkeypatch):
    # The C++ engine's RuntimeError, but not with the expected message: the
    # suffix check fails, quoting the text the caller saw. Skips without the
    # rocke_engine binding.
    seen = re.escape(f"(status 1): {_STORE_ALIGN_MESSAGE}")
    with pytest.raises(AssertionError, match=f"^.*{seen}$"):
        _assert_store_align_refusal(monkeypatch, "cpp-strict", message="another")


@pytest.mark.parametrize("lane", ["python", "both"])
def test_a_stray_fallback_fails_in_a_lane_that_must_not_fall_back(monkeypatch, lane):
    # The refusal matches in type and message, but the dispatcher fell back
    # first: the no-fallback check fails, showing the fallback. ``both``
    # skips without the rocke_engine binding.
    _dispatcher_that(
        monkeypatch,
        refusal=ValueError(_NO_ARCH_MESSAGE),
        fallback_reason="BackendError: stray",
    )
    with pytest.raises(AssertionError, match=r"^\['BackendError: stray'\]"):
        _assert_no_arch_refusal(monkeypatch, lane)


@pytest.mark.parametrize("lane", ["installed-fallback", "cpp-permissive"])
def test_a_fallback_lane_silences_the_fallback_warning(monkeypatch, lane):
    # The lane falls back for real (no binding, or a C++ refusal) and the
    # helper passes; the fallback it expects is not news, so its warning is
    # silenced. cpp-permissive skips without the rocke_engine binding.
    monkeypatch.delenv(_QUIET_FALLBACK, raising=False)
    _assert_store_align_refusal(monkeypatch, lane)
    assert os.environ.get(_QUIET_FALLBACK) == "1"


@pytest.mark.parametrize("lane", ["python", "both"])
def test_a_lane_that_must_not_fall_back_clears_an_inherited_quiet_setting(
    monkeypatch, lane
):
    # A fallback here would be a bug, so its warning must stay audible even
    # when the caller's environment silences it. ``both`` skips without the
    # rocke_engine binding.
    monkeypatch.setenv(_QUIET_FALLBACK, "1")
    _assert_no_arch_refusal(monkeypatch, lane)
    assert _QUIET_FALLBACK not in os.environ
