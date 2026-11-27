# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host helpers of ``rocke.helpers.spec`` (CPU only, no IR):

* ``ceil_div`` / ``align_up``: exact for every sign of the numerator,
  integer-like inputs only;
* ``ceil_div_grid`` keeps its grids;
* ``non_default_fields``: declared defaults only, field order, ``skip``;
* ``salted_kernel_name``: the base alone without salt, otherwise the first
  10 hex digits of the SHA-1 of the name-sorted pairs (a pinned value: a
  kernel name is the code object's symbol and keys the IR goldens);
* ``bool_flag_reason`` / ``require_bool`` / ``require_bool_fields``: only an
  exact ``bool`` passes (or ``None`` too, with ``allow_none``), and a refusal
  is a ``ValueError`` in the error template that names the owner, the flag and
  its value.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

_PLATFORM_PY = str(Path(__file__).resolve().parents[1] / "python")
if _PLATFORM_PY not in sys.path:
    sys.path.insert(0, _PLATFORM_PY)

from rocke.helpers.spec import (  # noqa: E402
    align_up,
    bool_flag_reason,
    ceil_div,
    ceil_div_grid,
    non_default_fields,
    require_bool,
    require_bool_fields,
    salted_kernel_name,
)
from rocke.tests import bool_flags as expected_text  # noqa: E402


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


# Truthy and falsy stand-ins a coercion would read as a flag.
_NOT_BOOLS = ("false", "", 0, 1, 0.0, np.bool_(True), np.bool_(False))


@pytest.mark.parametrize("flag", (True, False))
def test_a_bool_passes(flag):
    assert bool_flag_reason("emit_lse", flag) is None
    require_bool("emit_lse", flag, what="Spec")


@pytest.mark.parametrize("value", (*_NOT_BOOLS, None))
def test_a_non_bool_is_refused_in_the_error_template(value):
    reason = f"emit_lse={value!r}, expected a bool"
    assert bool_flag_reason("emit_lse", value) == reason
    with pytest.raises(ValueError) as refusal:
        require_bool("emit_lse", value, what="Spec")
    assert str(refusal.value) == f"Spec failed - {reason}"
    assert str(refusal.value) == expected_text.bool_flag_refusal(
        "emit_lse", value, what="Spec"
    )


def test_allow_none_admits_none_only():
    assert bool_flag_reason("use_cfvst", None, allow_none=True) is None
    require_bool("use_cfvst", None, what="Spec", allow_none=True)
    for value in _NOT_BOOLS:
        reason = f"use_cfvst={value!r}, expected a bool or None"
        assert bool_flag_reason("use_cfvst", value, allow_none=True) == reason
        with pytest.raises(ValueError) as refusal:
            require_bool("use_cfvst", value, what="Spec", allow_none=True)
        assert str(refusal.value) == f"Spec failed - {reason}"
        assert str(refusal.value) == expected_text.bool_flag_refusal(
            "use_cfvst", value, what="Spec", allow_none=True
        )


@dataclasses.dataclass(frozen=True)
class _Flags:
    first: object = True
    second: object = False


class _SubFlags(_Flags):
    pass


def test_require_bool_fields_names_the_owner_class_and_the_first_bad_field():
    require_bool_fields(_Flags(), "first", "second")
    require_bool_fields(_Flags(first=None), "first", allow_none=True)
    with pytest.raises(ValueError) as refusal:
        require_bool_fields(_SubFlags(first=1, second="no"), "first", "second")
    assert str(refusal.value) == expected_text.bool_flag_refusal(
        "first", 1, what="_SubFlags"
    )
