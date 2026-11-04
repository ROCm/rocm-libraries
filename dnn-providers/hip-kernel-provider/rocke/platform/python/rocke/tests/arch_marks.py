# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: state why a test covers only some gfx targets.

A test that names one gfx target can hide a gap: the code under test may support
more targets. Decorate a test with ``arch_specific`` when the single target is
intended. The decorator records the target list and the reason on the test in the
attribute `ARCH_SPECIFIC_ATTR`. It does not skip the test and needs no pytest
marker registration.

    @arch_specific(("gfx950",), reason="mxfp6 scales exist on gfx950 only")
    def test_mxfp6_scale_layout(self): ...

The decorator works on a function, a method, and a class. It works with
``unittest.TestCase`` and plain pytest tests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from rocke.core.arch.target import known_arches

__all__ = ["ARCH_SPECIFIC_ATTR", "arch_specific"]

ARCH_SPECIFIC_ATTR = "__rocke_arch_specific__"

_T = TypeVar("_T")


def arch_specific(archs: Iterable[str], reason: str) -> Callable[[_T], _T]:
    """Mark a test as valid for ``archs`` only, with a ``reason``.

    Raises ``ValueError`` when ``archs`` is empty or names an arch that
    ``ArchTarget.from_gfx`` does not know, or when ``reason`` is blank.
    Raises ``TypeError`` when ``archs`` is a single string.
    """
    if isinstance(archs, str):
        raise TypeError(f"archs must be a list of gfx names, not a string: {archs!r}")
    names = tuple(archs)
    if not names:
        raise ValueError("arch_specific needs at least one arch")
    known = known_arches()
    unknown = [a for a in names if a not in known]
    if unknown:
        raise ValueError(f"unknown arch {unknown}; known arches: {', '.join(known)}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("arch_specific needs a non-empty reason")
    info = (names, reason.strip())

    def decorate(obj: _T) -> _T:
        setattr(obj, ARCH_SPECIFIC_ATTR, info)
        return obj

    return decorate
