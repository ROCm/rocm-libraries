# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host helpers of ``rocke.helpers.spec`` (CPU only, no IR):

* ``ceil_div`` / ``align_up``: exact for every sign of the numerator,
  integer-like inputs only;
* ``ceil_div_grid`` keeps its grids;
* ``non_default_fields``: declared defaults only, field order, ``skip``;
* ``salted_kernel_name``: the base alone without salt, otherwise the first
  10 hex digits of the SHA-1 of the name-sorted pairs (a pinned value: a
  kernel name is the code object's symbol and keys the IR goldens).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

_PLATFORM_PY = str(Path(__file__).resolve().parents[1] / "python")
if _PLATFORM_PY not in sys.path:
    sys.path.insert(0, _PLATFORM_PY)

from rocke.helpers.spec import (  # noqa: E402
    align_up,
    ceil_div,
    ceil_div_grid,
    non_default_fields,
    salted_kernel_name,
)


@pytest.mark.parametrize("a", range(-20, 21))
@pytest.mark.parametrize("b", (1, 2, 3, 7, 16, -3))
def test_ceil_div_is_the_exact_ceiling(a, b):
    want = a // b + (1 if a % b else 0)
    assert ceil_div(a, b) == want
    if b > 0 and a >= 0:
        assert ceil_div(a, b) == (a + b - 1) // b


def test_ceil_div_takes_integer_likes_only():
    np = pytest.importorskip("numpy")
    got = ceil_div(np.int64(100), np.int32(16))
    assert got == 7 and type(got) is int
    assert ceil_div(True, 1) == 1
    with pytest.raises(TypeError):
        ceil_div(100.0, 16)
    with pytest.raises(ZeroDivisionError):
        ceil_div(1, 0)


@pytest.mark.parametrize(
    "n,m,want", [(0, 64, 0), (1, 64, 64), (64, 64, 64), (65, 64, 128), (-5, 4, -4)]
)
def test_align_up(n, m, want):
    assert align_up(n, m) == want


def test_ceil_div_grid_unchanged():
    assert ceil_div_grid((100, 16), (128, 32)) == (7, 4, 1)
    assert ceil_div_grid((100, 16), (128, 32), (4, 1)) == (7, 4, 4)
    with pytest.raises(ValueError):
        ceil_div_grid((100, 0))


@dataclasses.dataclass(frozen=True)
class _Spec:
    shape: int
    a: int = 1
    b: str = "x"
    c: tuple = dataclasses.field(default_factory=tuple)
    name: str = "k"


def test_non_default_fields():
    assert non_default_fields(_Spec(3)) == ()
    s = _Spec(3, b="y", a=2, c=(1,), name="other")
    # field order; the required field and the default_factory field never
    assert non_default_fields(s) == (("a", 2), ("b", "y"), ("name", "other"))
    assert non_default_fields(s, skip=("name", "a")) == (("b", "y"),)


def test_salted_kernel_name():
    assert salted_kernel_name("k", []) == "k"
    assert salted_kernel_name("k", iter(())) == "k"
    pairs = [("b", (2, "x")), ("a", 1)]
    assert salted_kernel_name("k", pairs) == "k_420a851fb6"
    assert salted_kernel_name("k", reversed(pairs)) == "k_420a851fb6"
