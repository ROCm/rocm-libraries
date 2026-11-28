# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The exact text of a non-bool flag refusal, for the tests that pin it.

``rocke.helpers.spec.require_bool`` refuses a flag that is not exactly a
``bool`` with ``"{what} failed - {name}={value!r}, expected a bool"`` (``expected
a bool or None`` for a flag that may be unset). An ``(ok, reason)`` gate returns
the part after ``failed - ``. The tests spell the expected text through these
functions rather than restating it, so the form is written once on the test
side.
"""

from __future__ import annotations

__all__ = ["bool_flag_reason", "bool_flag_refusal"]


def bool_flag_reason(name: str, value: object, *, allow_none: bool = False) -> str:
    """The reason an ``(ok, reason)`` gate gives for a non-bool flag ``name``."""
    expected = "a bool or None" if allow_none else "a bool"
    return f"{name}={value!r}, expected {expected}"


def bool_flag_refusal(
    name: str, value: object, *, what: str, allow_none: bool = False
) -> str:
    """The ``ValueError`` message of ``what`` refusing the non-bool flag ``name``."""
    return f"{what} failed - {bool_flag_reason(name, value, allow_none=allow_none)}"
